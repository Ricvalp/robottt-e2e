from ml_collections import ConfigDict


def get_config() -> ConfigDict:

    cfg = ConfigDict()

    cfg.run = ConfigDict()
    cfg.run.seed = 42
    cfg.run.device = "cuda"  # "cuda" or "cpu"

    cfg.data = ConfigDict()
    cfg.data.dataset = "mnist"  # "mnist", "cifar10", or "cifar100"
    cfg.data.root = "playground/data"
    cfg.data.image_size = 32
    cfg.data.batch_size = 1
    cfg.data.num_workers = 4
    cfg.data.download = True
    cfg.data.shuffle = True

    cfg.model = ConfigDict()
    cfg.model.in_channels = None  # auto-set from dataset if None
    cfg.model.base_channels = 64
    cfg.model.channel_mults = (1, 2, 4)
    cfg.model.num_res_blocks = 2
    cfg.model.dropout = 0.0
    cfg.model.attn_resolutions = (16,)
    cfg.model.num_heads = 4

    cfg.diffusion = ConfigDict()
    cfg.diffusion.train_steps = 1000  # time scaling for continuous diffusion
    cfg.diffusion.beta_start = 1e-4
    cfg.diffusion.beta_end = 2e-2

    cfg.training = ConfigDict()
    cfg.training.epochs = 400
    cfg.training.lr = 2e-4
    cfg.training.weight_decay = 0.0
    cfg.training.grad_clip = 1.0
    cfg.training.use_amp = False
    cfg.training.log_every = 1
    cfg.training.sample_every_epochs = 100
    cfg.training.checkpoint_every_epochs = 1

    cfg.sample = ConfigDict()
    cfg.sample.num_images = 16
    cfg.sample.batch_size = 16
    cfg.sample.steps = 1000  # inference steps
    cfg.sample.dir = "playground/samples"

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = "playground/checkpoints"
    cfg.checkpoint.resume = ""

    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = "ddpm-mnist"
    cfg.wandb.entity = "ricvalp"
    cfg.wandb.run_name = None
    cfg.wandb.dir = None

    return cfg
