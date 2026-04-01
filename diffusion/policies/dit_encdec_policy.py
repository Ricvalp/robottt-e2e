"""Encoder-decoder policy supporting diffusion and direct chunk regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion.models.dit import (
    DecoderTransformer,
    DecoderTransformerConfig,
    EncoderTransformer,
    EncoderTransformerConfig,
)


def _normalize_objective_type(objective_type: str) -> str:
    value = str(objective_type).strip().lower()
    aliases = {
        "diffusion": "diffusion",
        "direct_regression": "direct_regression",
        "regression": "direct_regression",
        "direct": "direct_regression",
    }
    if value not in aliases:
        raise ValueError(
            "Unsupported objective_type "
            f"'{objective_type}'. Expected one of: diffusion, direct_regression."
        )
    return aliases[value]


def _normalize_regression_loss(regression_loss: str) -> str:
    value = str(regression_loss).strip().lower()
    aliases = {
        "mse": "mse",
        "l2": "mse",
        "l1": "l1",
        "mae": "l1",
        "gaussian_nll_fixed": "gaussian_nll_fixed",
        "gaussian_nll": "gaussian_nll_fixed",
    }
    if value not in aliases:
        raise ValueError(
            "Unsupported regression_loss "
            f"'{regression_loss}'. Expected one of: mse/l2, l1/mae, gaussian_nll_fixed."
        )
    return aliases[value]


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        half_dim = max(1, dim // 2)
        inv_freq = base ** (
            -torch.arange(half_dim, dtype=torch.float32) / max(1, half_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim == 1:
            indices = indices.unsqueeze(1)
        values = indices.to(self.inv_freq.dtype)
        angles = values.unsqueeze(-1) * self.inv_freq
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            pad = torch.zeros(
                *emb.shape[:-1],
                self.dim - emb.shape[-1],
                device=emb.device,
                dtype=emb.dtype,
            )
            emb = torch.cat([emb, pad], dim=-1)
        return emb


def _normalize_prediction_type(prediction_type: str) -> str:
    value = str(prediction_type).strip().lower()
    aliases = {
        "epsilon": "epsilon",
        "eps": "epsilon",
        "sample": "sample",
        "x0": "sample",
        "v_prediction": "v_prediction",
        "v": "v_prediction",
    }
    if value not in aliases:
        raise ValueError(
            "Unsupported prediction_type "
            f"'{prediction_type}'. Expected one of: epsilon/eps, sample/x0, v_prediction/v."
        )
    return aliases[value]


def _diffusion_training_target(
    noise_scheduler: DDPMScheduler,
    *,
    x0: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    pred_type = _normalize_prediction_type(noise_scheduler.config.prediction_type)
    if pred_type == "epsilon":
        return noise
    if pred_type == "sample":
        return x0
    if pred_type == "v_prediction":
        if hasattr(noise_scheduler, "get_velocity"):
            return noise_scheduler.get_velocity(x0, noise, timesteps)

        alphas_cumprod = noise_scheduler.alphas_cumprod.to(
            device=x0.device, dtype=x0.dtype
        )
        alpha_t = alphas_cumprod[timesteps].sqrt()
        sigma_t = (1.0 - alphas_cumprod[timesteps]).sqrt()
        while alpha_t.ndim < x0.ndim:
            alpha_t = alpha_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)
        return alpha_t * noise - sigma_t * x0

    raise ValueError(f"Unsupported prediction type {pred_type}")


def _direct_regression_loss(
    regression_loss: str,
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    fixed_variance: float,
) -> torch.Tensor:
    normalized_loss = _normalize_regression_loss(regression_loss)
    if normalized_loss == "mse":
        return F.mse_loss(pred, target)
    if normalized_loss == "l1":
        return F.l1_loss(pred, target)
    if normalized_loss == "gaussian_nll_fixed":
        variance = torch.full_like(target, float(fixed_variance))
        return F.gaussian_nll_loss(pred, target, variance, full=True)
    raise ValueError(f"Unsupported regression loss {normalized_loss}")


@dataclass
class DiTEncDecDiffusionPolicyConfig:
    horizon: int
    point_feature_dim: int
    action_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5
    scalar_embedding_hidden_dim: int = 128
    time_embedding_base: float = 10000.0
    diffusion_embedding_base: float = 10000.0
    objective_type: str = "diffusion"
    regression_loss: str = "mse"
    regression_fixed_variance: float = 1.0
    prediction_type: str = "epsilon"
    num_inference_steps: int = 50
    noise_scheduler_kwargs: Dict[str, object] | None = None


class DiTEncDecDiffusionPolicy(nn.Module):
    def __init__(self, cfg: DiTEncDecDiffusionPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.objective_type = _normalize_objective_type(cfg.objective_type)
        self.regression_loss = _normalize_regression_loss(cfg.regression_loss)
        self.regression_fixed_variance = float(cfg.regression_fixed_variance)
        if self.regression_fixed_variance <= 0:
            raise ValueError("regression_fixed_variance must be positive.")

        encoder_transformer_cfg = EncoderTransformerConfig(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            mlp_dim=cfg.mlp_dim,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            activation=cfg.activation,
            layer_norm_eps=cfg.layer_norm_eps,
        )
        self.encoder_transformer = EncoderTransformer(encoder_transformer_cfg)

        decoder_transformer_cfg = DecoderTransformerConfig(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            mlp_dim=cfg.mlp_dim,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            activation=cfg.activation,
            layer_norm_eps=cfg.layer_norm_eps,
        )
        self.decoder_transformer = DecoderTransformer(decoder_transformer_cfg)

        if cfg.point_feature_dim <= 0:
            raise ValueError("point_feature_dim must be positive.")
        self.point_feature_proj = nn.Sequential(
            nn.Linear(cfg.point_feature_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.history_feature_proj = nn.Sequential(
            nn.Linear(cfg.point_feature_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.action_encoder = nn.Linear(cfg.action_dim, cfg.hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )

        self.world_time_embedder = SinusoidalTimeEmbedding(
            cfg.hidden_dim, base=cfg.time_embedding_base
        )
        self.diffusion_time_embedder = SinusoidalTimeEmbedding(
            cfg.hidden_dim, base=cfg.diffusion_embedding_base
        )
        self.diffusion_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        scheduler_kwargs = dict(cfg.noise_scheduler_kwargs or {})
        normalized_prediction_type = _normalize_prediction_type(cfg.prediction_type)
        existing_prediction_type = scheduler_kwargs.get("prediction_type")
        if existing_prediction_type is not None:
            existing_prediction_type = _normalize_prediction_type(existing_prediction_type)
            if existing_prediction_type != normalized_prediction_type:
                raise ValueError(
                    "prediction_type mismatch between cfg.prediction_type "
                    f"('{cfg.prediction_type}') and noise_scheduler_kwargs "
                    f"('{scheduler_kwargs['prediction_type']}')."
                )
        scheduler_kwargs["prediction_type"] = normalized_prediction_type
        self.scheduler = DDPMScheduler(**scheduler_kwargs)
        self.num_inference_steps = cfg.num_inference_steps

        self.context_time_indices = None
        action_idx = torch.arange(1, cfg.horizon + 1, dtype=torch.float32).unsqueeze(0)
        self.register_buffer("action_time_indices", action_idx, persistent=False)

    def _encode_context(self, points: torch.Tensor) -> torch.Tensor:
        batch_size, num_points = points.shape[:2]
        indices = torch.arange(
            -num_points + 1, 1, device=points.device, dtype=torch.float32
        )
        frame_time_emb = self.world_time_embedder(
            indices.unsqueeze(0).expand(batch_size, -1)
        )
        point_tokens = self.point_feature_proj(points)
        point_tokens = point_tokens + frame_time_emb

        return point_tokens

    def _encode_history(self, points: torch.Tensor) -> torch.Tensor:
        batch_size, num_points = points.shape[:2]
        indices = torch.arange(
            -num_points + 1, 1, device=points.device, dtype=torch.float32
        )
        frame_time_emb = self.world_time_embedder(
            indices.unsqueeze(0).expand(batch_size, -1)
        )
        point_tokens = self.history_feature_proj(points)
        point_tokens = point_tokens + frame_time_emb

        return point_tokens

    def _encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        tokens = self.action_encoder(actions)
        batch = tokens.shape[0]
        times = self.action_time_indices.to(
            device=actions.device, dtype=torch.float32
        ).expand(batch, -1)
        time_emb = self.world_time_embedder(times)
        return tokens + time_emb

    def _diffusion_condition(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.diffusion_time_embedder(timesteps.float().unsqueeze(1))[:, 0, :]
        return self.diffusion_proj(emb)

    def _predict_action_chunk(
        self,
        *,
        context: torch.Tensor,
        history: torch.Tensor,
        future_slots: torch.Tensor,
        query_mask: torch.Tensor,
        context_mask: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        context_tokens = self._encode_context(context)
        history_tokens = self._encode_history(history)
        action_tokens = self._encode_actions(future_slots)
        tokens = torch.cat([history_tokens, action_tokens], dim=1)

        memory = self.encoder_transformer(
            context_tokens, key_padding_mask=~context_mask
        )
        diffusion_cond = self._diffusion_condition(timesteps)
        decoded = self.decoder_transformer(
            tokens=tokens,
            tokens_kpm=~query_mask,
            memory=memory,
            encoder_kpm=~context_mask,
            diffusion_time_cond=diffusion_cond,
        )
        return self.output_head(decoded[:, -self.cfg.horizon :, :])

    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        context = batch["context"]
        history = batch["history"]
        actions = batch["actions"]
        x0 = actions
        query_mask = batch["query_mask"]
        context_mask = batch["context_mask"]

        if self.objective_type == "diffusion":
            noise = torch.randn_like(actions)
            timesteps = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (actions.shape[0],),
                device=actions.device,
                dtype=torch.long,
            )
            future_slots = self.scheduler.add_noise(x0, noise, timesteps)
            pred = self._predict_action_chunk(
                context=context,
                history=history,
                future_slots=future_slots,
                query_mask=query_mask,
                context_mask=context_mask,
                timesteps=timesteps,
            )
            target = _diffusion_training_target(
                self.scheduler,
                x0=x0,
                noise=noise,
                timesteps=timesteps,
            )
            loss = F.mse_loss(pred, target)
        else:
            timesteps = torch.zeros(
                (actions.shape[0],),
                device=actions.device,
                dtype=torch.long,
            )
            future_slots = torch.zeros_like(actions)
            pred = self._predict_action_chunk(
                context=context,
                history=history,
                future_slots=future_slots,
                query_mask=query_mask,
                context_mask=context_mask,
                timesteps=timesteps,
            )
            target = x0
            loss = _direct_regression_loss(
                self.regression_loss,
                pred=pred,
                target=target,
                fixed_variance=self.regression_fixed_variance,
            )

        metrics = {"loss": float(loss.detach().cpu())}
        return loss, metrics

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.compute_loss(batch)

    def sample_actions(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Run the reverse diffusion process to synthesize a horizon chunk.

        Args:
            context:
            context_mask:
            history:
            history_mask:
            generator: Optional ``torch.Generator`` for deterministic sampling.
            inference_steps: Optional override for the number of reverse-diffusion
                denoising steps. When omitted, ``self.num_inference_steps`` is used.

        Returns:
            Tensor with shape ``(B, horizon, action_dim)`` containing the
            denoised action tokens for the next horizon window.
        """

        device = context.device
        batch_size = context.shape[0]

        if self.objective_type == "direct_regression":
            timesteps = torch.zeros(
                (batch_size,),
                device=device,
                dtype=torch.long,
            )
            future_slots = torch.zeros(
                (batch_size, self.cfg.horizon, self.cfg.action_dim),
                device=device,
                dtype=history.dtype,
            )
            return self._predict_action_chunk(
                context=context,
                history=history,
                future_slots=future_slots,
                query_mask=history_mask,
                context_mask=context_mask,
                timesteps=timesteps,
            )

        sample = torch.randn(
            (batch_size, self.cfg.horizon, self.cfg.action_dim),
            generator=generator,
            device=device,
        )

        context_tokens = self._encode_context(context)
        history_tokens = self._encode_history(history)

        memory = self.encoder_transformer(
            context_tokens, key_padding_mask=~context_mask
        )

        num_inference_steps = (
            self.num_inference_steps
            if inference_steps is None
            else int(inference_steps)
        )
        if num_inference_steps <= 0:
            raise ValueError("inference_steps must be positive.")

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        for timestep in self.scheduler.timesteps:
            timesteps = torch.full(
                (batch_size,),
                timestep,
                device=device,
                dtype=torch.long,
            )

            action_tokens = self._encode_actions(sample)
            tokens = torch.cat([history_tokens, action_tokens], dim=1)
            diffusion_cond = self._diffusion_condition(timesteps)
            decoded = self.decoder_transformer(
                tokens,
                tokens_kpm=~history_mask,
                memory=memory,
                encoder_kpm=~context_mask,
                diffusion_time_cond=diffusion_cond,
            )

            model_pred = self.output_head(decoded[:, -self.cfg.horizon :, :])
            scheduler_step = self.scheduler.step(
                model_pred,
                timestep,
                sample,
                generator=generator,
            )
            sample = scheduler_step.prev_sample

        return sample


class MAMLDiTEncDecDiffusionPolicy(nn.Module):
    def __init__(self, cfg: DiTEncDecDiffusionPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.objective_type = _normalize_objective_type(cfg.objective_type)
        self.regression_loss = _normalize_regression_loss(cfg.regression_loss)
        self.regression_fixed_variance = float(cfg.regression_fixed_variance)
        if self.regression_fixed_variance <= 0:
            raise ValueError("regression_fixed_variance must be positive.")

        encoder_transformer_cfg = EncoderTransformerConfig(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            mlp_dim=cfg.mlp_dim,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            activation=cfg.activation,
            layer_norm_eps=cfg.layer_norm_eps,
        )
        self.encoder_transformer = EncoderTransformer(encoder_transformer_cfg)

        decoder_transformer_cfg = DecoderTransformerConfig(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            mlp_dim=cfg.mlp_dim,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            activation=cfg.activation,
            layer_norm_eps=cfg.layer_norm_eps,
        )
        self.decoder_transformer = DecoderTransformer(decoder_transformer_cfg)

        if cfg.point_feature_dim <= 0:
            raise ValueError("point_feature_dim must be positive.")
        self.point_feature_proj = nn.Sequential(
            nn.Linear(cfg.point_feature_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.history_feature_proj = nn.Sequential(
            nn.Linear(cfg.point_feature_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.action_encoder = nn.Linear(cfg.action_dim, cfg.hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )

        self.world_time_embedder = SinusoidalTimeEmbedding(
            cfg.hidden_dim, base=cfg.time_embedding_base
        )
        self.diffusion_time_embedder = SinusoidalTimeEmbedding(
            cfg.hidden_dim, base=cfg.diffusion_embedding_base
        )
        self.diffusion_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        scheduler_kwargs = dict(cfg.noise_scheduler_kwargs or {})
        normalized_prediction_type = _normalize_prediction_type(cfg.prediction_type)
        existing_prediction_type = scheduler_kwargs.get("prediction_type")
        if existing_prediction_type is not None:
            existing_prediction_type = _normalize_prediction_type(existing_prediction_type)
            if existing_prediction_type != normalized_prediction_type:
                raise ValueError(
                    "prediction_type mismatch between cfg.prediction_type "
                    f"('{cfg.prediction_type}') and noise_scheduler_kwargs "
                    f"('{scheduler_kwargs['prediction_type']}')."
                )
        scheduler_kwargs["prediction_type"] = normalized_prediction_type
        self.scheduler = DDPMScheduler(**scheduler_kwargs)
        self.num_inference_steps = cfg.num_inference_steps

        self.context_time_indices = None
        action_idx = torch.arange(1, cfg.horizon + 1, dtype=torch.float32).unsqueeze(0)
        self.register_buffer("action_time_indices", action_idx, persistent=False)

    def _encode_context(self, points: torch.Tensor) -> torch.Tensor:
        batch_size, num_points = points.shape[:2]
        indices = torch.arange(
            -num_points + 1, 1, device=points.device, dtype=torch.float32
        )
        frame_time_emb = self.world_time_embedder(
            indices.unsqueeze(0).expand(batch_size, -1)
        )
        point_tokens = self.point_feature_proj(points)
        point_tokens = point_tokens + frame_time_emb

        return point_tokens

    def _encode_history(self, points: torch.Tensor) -> torch.Tensor:
        batch_size, num_points = points.shape[:2]
        indices = torch.arange(
            -num_points + 1, 1, device=points.device, dtype=torch.float32
        )
        frame_time_emb = self.world_time_embedder(
            indices.unsqueeze(0).expand(batch_size, -1)
        )
        point_tokens = self.history_feature_proj(points)
        point_tokens = point_tokens + frame_time_emb

        return point_tokens

    def _encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        tokens = self.action_encoder(actions)
        batch = tokens.shape[0]
        times = self.action_time_indices.to(
            device=actions.device, dtype=torch.float32
        ).expand(batch, -1)
        time_emb = self.world_time_embedder(times)
        return tokens + time_emb

    def _diffusion_condition(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.diffusion_time_embedder(timesteps.float().unsqueeze(1))[:, 0, :]
        return self.diffusion_proj(emb)

    def _predict_action_chunk(
        self,
        *,
        context: torch.Tensor,
        history: torch.Tensor,
        future_slots: torch.Tensor,
        query_mask: torch.Tensor,
        context_mask: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        context_tokens = self._encode_context(context)
        history_tokens = self._encode_history(history)
        action_tokens = self._encode_actions(future_slots)
        tokens = torch.cat([history_tokens, action_tokens], dim=1)

        memory = self.encoder_transformer(
            context_tokens, key_padding_mask=~context_mask
        )
        diffusion_cond = self._diffusion_condition(timesteps)
        decoded = self.decoder_transformer(
            tokens=tokens,
            tokens_kpm=~query_mask,
            memory=memory,
            encoder_kpm=~context_mask,
            diffusion_time_cond=diffusion_cond,
        )
        return self.output_head(decoded[:, -self.cfg.horizon :, :])

    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        context = batch["context"]
        history = batch["history"]
        actions = batch["actions"]
        x0 = actions
        query_mask = batch["query_mask"]
        context_mask = batch["context_mask"]

        if self.objective_type == "diffusion":
            noise = batch.get("noise", None)
            if noise is None:
                noise = torch.randn_like(actions)

            timesteps = batch.get("timesteps", None)
            if timesteps is None:
                timesteps = torch.randint(
                    0,
                    self.scheduler.config.num_train_timesteps,
                    (actions.shape[0],),
                    device=actions.device,
                    dtype=torch.long,
                )

            future_slots = self.scheduler.add_noise(x0, noise, timesteps)
            pred = self._predict_action_chunk(
                context=context,
                history=history,
                future_slots=future_slots,
                query_mask=query_mask,
                context_mask=context_mask,
                timesteps=timesteps,
            )
            target = _diffusion_training_target(
                self.scheduler,
                x0=x0,
                noise=noise,
                timesteps=timesteps,
            )
            loss = F.mse_loss(pred, target)
        else:
            timesteps = torch.zeros(
                (actions.shape[0],),
                device=actions.device,
                dtype=torch.long,
            )
            future_slots = torch.zeros_like(actions)
            pred = self._predict_action_chunk(
                context=context,
                history=history,
                future_slots=future_slots,
                query_mask=query_mask,
                context_mask=context_mask,
                timesteps=timesteps,
            )
            target = x0
            loss = _direct_regression_loss(
                self.regression_loss,
                pred=pred,
                target=target,
                fixed_variance=self.regression_fixed_variance,
            )

        metrics = {"loss": float(loss.detach().cpu())}
        return loss, metrics

    def loss_only(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss, _ = self.compute_loss(batch)
        return loss

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.loss_only(batch)

    def sample_actions(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Run the reverse diffusion process to synthesize a horizon chunk.

        Args:
            context:
            context_mask:
            history:
            history_mask:
            generator: Optional ``torch.Generator`` for deterministic sampling.
            inference_steps: Optional override for the number of reverse-diffusion
                denoising steps. When omitted, ``self.num_inference_steps`` is used.

        Returns:
            Tensor with shape ``(B, horizon, action_dim)`` containing the
            denoised action tokens for the next horizon window.
        """

        device = context.device
        batch_size = context.shape[0]

        if self.objective_type == "direct_regression":
            timesteps = torch.zeros(
                (batch_size,),
                device=device,
                dtype=torch.long,
            )
            future_slots = torch.zeros(
                (batch_size, self.cfg.horizon, self.cfg.action_dim),
                device=device,
                dtype=history.dtype,
            )
            return self._predict_action_chunk(
                context=context,
                history=history,
                future_slots=future_slots,
                query_mask=history_mask,
                context_mask=context_mask,
                timesteps=timesteps,
            )

        sample = torch.randn(
            (batch_size, self.cfg.horizon, self.cfg.action_dim),
            generator=generator,
            device=device,
        )

        context_tokens = self._encode_context(context)
        history_tokens = self._encode_history(history)

        memory = self.encoder_transformer(
            context_tokens, key_padding_mask=~context_mask
        )

        num_inference_steps = (
            self.num_inference_steps
            if inference_steps is None
            else int(inference_steps)
        )
        if num_inference_steps <= 0:
            raise ValueError("inference_steps must be positive.")

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        for timestep in self.scheduler.timesteps:
            timesteps = torch.full(
                (batch_size,),
                timestep,
                device=device,
                dtype=torch.long,
            )

            action_tokens = self._encode_actions(sample)
            tokens = torch.cat([history_tokens, action_tokens], dim=1)
            diffusion_cond = self._diffusion_condition(timesteps)
            decoded = self.decoder_transformer(
                tokens,
                tokens_kpm=~history_mask,
                memory=memory,
                encoder_kpm=~context_mask,
                diffusion_time_cond=diffusion_cond,
            )

            model_pred = self.output_head(decoded[:, -self.cfg.horizon :, :])
            scheduler_step = self.scheduler.step(
                model_pred,
                timestep,
                sample,
                generator=generator,
            )
            sample = scheduler_step.prev_sample

        return sample


__all__ = ["DiTEncDecDiffusionPolicy", "DiTEncDecDiffusionPolicyConfig", "MAMLDiTEncDecDiffusionPolicy"]


if __name__ == "__main__":

    config = DiTEncDecDiffusionPolicyConfig(
        horizon=8,
        point_feature_dim=7,
        action_dim=7,
        hidden_dim=64,
        num_layers=4,
        num_heads=4,
        mlp_dim=128,
    )
    dit = DiTEncDecDiffusionPolicy(config)
    batch = {
        "points": torch.randn(32, 16, 7),
        "actions": torch.randn(32, 8, 7),
        "mask": torch.ones(32, 24, dtype=torch.bool),
    }

    loss, metrics = dit.compute_loss(batch)
    print(f"Loss: {loss.item()}, Metrics: {metrics}")
