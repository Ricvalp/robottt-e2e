"""
Helpers for preparing QuickDraw episodes for diffusion-policy training.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from .preprocess import ProcessedSketch

__all__ = [
    "ContextQueryInContextDiffusionCollator",
    "MAMLDiffusionCollator",
]


class ContextQueryInContextDiffusionCollator:
    """
    In-context diffusion collator that separates context and query points for and encoder-decoder architecture.
    """

    def __init__(self, horizon: int, seed: int = 0) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        points_batch: List[torch.Tensor] = []
        actions_batch: List[torch.Tensor] = []
        contexts_batch: List[torch.Tensor] = []
        query_lengths: List[int] = []
        points_lengths: List[int] = []
        contexts_lengths: List[int] = []

        for sample in batch:
            tokens = sample["tokens"]

            reset_idx = (tokens[:, 5] == 1.0).nonzero(as_tuple=True)[0]
            start_idx = int(self.rng.integers(reset_idx + 1, tokens.shape[0]))

            tokens = torch.cat(
                [tokens[:, :5], tokens[:, 6:]], dim=-1
            )  # Remove reset token from tokens

            context = tokens[:reset_idx].clone()
            points = tokens[reset_idx + 1 : start_idx + 1].clone()
            actions = tokens[start_idx + 1 : start_idx + 1 + self.horizon].clone()

            if actions.shape[0] < self.horizon:
                actions = self._pad_actions(actions)

            points_batch.append(points)
            actions_batch.append(actions)
            contexts_batch.append(context)
            query_lengths.append(points.shape[0] + actions.shape[0])
            points_lengths.append(points.shape[0])
            contexts_lengths.append(context.shape[0])

        if not points_batch:
            raise ValueError("No valid samples for diffusion collator.")

        max_query_len = max(query_lengths)
        max_context_len = max(contexts_lengths)

        batch_size = len(points_batch)
        feature_dim = points_batch[0].shape[-1]

        points = torch.zeros(
            batch_size, max_query_len, feature_dim, dtype=torch.float32
        )
        actions = torch.zeros(
            batch_size, self.horizon, feature_dim, dtype=torch.float32
        )
        contexts = torch.zeros(
            batch_size, max_context_len, feature_dim, dtype=torch.float32
        )

        context_mask = torch.zeros(batch_size, max_context_len, dtype=torch.bool)
        query_mask = torch.zeros(
            batch_size, max_query_len + self.horizon, dtype=torch.bool
        )

        for idx, (pts, context, points_len, query_len, contexts_len) in enumerate(
            zip(
                points_batch,
                contexts_batch,
                points_lengths,
                query_lengths,
                contexts_lengths,
            )
        ):
            points[idx, -points_len:] = pts
            # actions[idx, :] = acts
            query_mask[idx, -query_len:] = True

            contexts[idx, -contexts_len:] = context
            context_mask[idx, -contexts_len:] = True

        actions = torch.stack(actions_batch, dim=0)

        return {
            "history": points,
            "actions": actions,
            "query_mask": query_mask,
            "context": contexts,
            "context_mask": context_mask,
        }

    def _pad_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Pads actions that are shorter than the horizon with end-tokens."""
        pad_len = self.horizon - actions.shape[0]
        padding = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).tile((pad_len, 1))

        return torch.cat([actions, padding])


class MAMLDiffusionCollator:
    """
    Diffusion collator for MAML episodes.
    """

    def __init__(self, token_dim: int, dtype: np.dtype = np.float32, coordinate_mode: str = "delta") -> None:
        self.token_dim = token_dim
        self.dtype = dtype
        self.coordinate_mode = coordinate_mode

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        tasks = []
        for item in batch:
            
            tasks.append(
                {
                    "context_episodes": [self._sketch_to_tokens(sketch) for sketch in item['task']['context_episodes']],
                    "query_episode": self._sketch_to_tokens(item['task']['query_episode']),
                }
            )
        return tasks

    def _sketch_to_tokens(self, sketch: ProcessedSketch) -> np.ndarray:
        """
        Convert a processed sketch into its `(dx, dy, pen-up, pen-down, sep, reset, stop)` tokens.
        """
        tokens = np.zeros((sketch.length, self.token_dim), dtype=self.dtype)
        if self.coordinate_mode == "delta":
            tokens[:, 0:2] = sketch.deltas
        elif self.coordinate_mode == "absolute":
            tokens[:, 0:2] = sketch.absolute
        else:
            raise ValueError(
                f"Unsupported coordinate_mode '{self.coordinate_mode}'. "
                "Expected 'delta' or 'absolute'."
            )
        tokens[:, 2] = sketch.pen
        tokens[:, 3] = 1 - sketch.pen
        return tokens
    








