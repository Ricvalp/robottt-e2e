from ml_collections import ConfigDict


def get_config() -> ConfigDict:

    cfg = ConfigDict()

    cfg.run = ConfigDict()
    cfg.run.seed = 42
    cfg.run.device = "gpu"

    cfg.data = ConfigDict()
    cfg.data.image_size = 32
    cfg.data.batch_size = 64
    cfg.data.shuffle_buffer = 10000
    cfg.data.holdout_per_class = 200
    cfg.data.use_full_dataset = False
    cfg.data.leave_out_digit = 9
    cfg.data.leaveout_eval_holdout = 10000

    cfg.model = ConfigDict()
    cfg.model.in_channels = 1
    cfg.model.base_channels = 64
    cfg.model.channel_mults = (1, 2, 4)
    cfg.model.num_res_blocks = 2
    cfg.model.dropout = 0.0
    cfg.model.attn_resolutions = (16,)
    cfg.model.num_heads = 4

    cfg.diffusion = ConfigDict()
    cfg.diffusion.train_steps = 1000
    cfg.diffusion.beta_start = 1e-4
    cfg.diffusion.beta_end = 2e-2

    cfg.training = ConfigDict()
    cfg.training.epochs = 500
    cfg.training.steps_per_epoch = 500
    cfg.training.outer_lr = 2e-4
    cfg.training.inner_lr = 1e-3
    cfg.training.inner_steps = 2
    cfg.training.weight_decay = 0.0
    cfg.training.grad_clip = 1.0
    cfg.training.log_every = 50
    cfg.training.sample_every_epochs = 5
    cfg.training.checkpoint_every_epochs = 5

    cfg.eval = ConfigDict()
    cfg.eval.inner_steps = 1
    cfg.eval.inner_lr = 1e-3

    cfg.fast_params = ConfigDict()
    cfg.fast_params.selector = "up_down_mid_head"

    cfg.sample = ConfigDict()
    cfg.sample.num_images = 16
    cfg.sample.steps = 100
    cfg.sample.dir = "playground_jax/maml_samples"

    cfg.counting = ConfigDict()
    cfg.counting.use = True
    cfg.counting.num_samples = 3000
    cfg.counting.batch_size = 256
    cfg.counting.classifier_ckpt = "playground_jax/classifier_checkpoints/classifier_epoch_010.pt"

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = "playground_jax/maml_checkpoints"
    cfg.checkpoint.resume = ""

    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = "ddpm-maml-mnist-jax"
    cfg.wandb.entity = "ricvalp"
    cfg.wandb.run_name = None

    return cfg
