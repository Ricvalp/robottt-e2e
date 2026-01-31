from ml_collections import ConfigDict


def get_config() -> ConfigDict:

    cfg = ConfigDict()

    cfg.run = ConfigDict()
    cfg.run.seed = 42
    cfg.run.device = "gpu"  # "gpu" or "cpu"

    cfg.data = ConfigDict()
    cfg.data.dataset = "mnist"
    cfg.data.image_size = 32
    cfg.data.batch_size = 64
    cfg.data.num_workers = 4
    cfg.data.shuffle_buffer = 10000

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
    cfg.training.epochs = 400
    cfg.training.lr = 2e-4
    cfg.training.weight_decay = 0.0
    cfg.training.grad_clip = 1.0
    cfg.training.log_every = 100
    cfg.training.sample_every_epochs = 10
    cfg.training.checkpoint_every_epochs = 10

    cfg.sample = ConfigDict()
    cfg.sample.num_images = 16
    cfg.sample.steps = 100
    cfg.sample.dir = "playground_jax/samples"

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = "playground_jax/checkpoints"
    cfg.checkpoint.resume = ""

    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = "ddpm-mnist-jax"
    cfg.wandb.entity = "ricvalp"
    cfg.wandb.run_name = None

    return cfg
