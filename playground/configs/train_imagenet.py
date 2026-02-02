from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()

    cfg.run = ConfigDict()
    cfg.run.seed = 42
    cfg.run.device = "cuda"

    cfg.data = ConfigDict()
    cfg.data.train_dir = "/data/imagenet/train"  # UPDATE THIS PATH
    cfg.data.val_dir = "/data/imagenet/val"      # UPDATE THIS PATH
    cfg.data.image_size = 64
    cfg.data.batch_size = 64  # Per GPU
    cfg.data.num_workers = 8

    cfg.model = ConfigDict()
    cfg.model.in_channels = 3
    cfg.model.base_channels = 128
    cfg.model.channel_mults = [1, 2, 3, 4]
    cfg.model.num_res_blocks = 2
    cfg.model.dropout = 0.1
    cfg.model.attn_resolutions = [16, 8]
    cfg.model.num_heads = 4
    cfg.model.num_classes = 1000
    cfg.model.cond_dim = 512

    cfg.diffusion = ConfigDict()
    cfg.diffusion.beta_start = 1e-4
    cfg.diffusion.beta_end = 0.02
    cfg.diffusion.train_steps = 1000
    cfg.diffusion.cfg_drop_prob = 0.1

    cfg.training = ConfigDict()
    cfg.training.epochs = 300
    cfg.training.lr = 1e-4
    cfg.training.weight_decay = 0.0
    cfg.training.grad_clip = 1.0
    cfg.training.use_amp = True
    cfg.training.sample_every_epochs = 10
    cfg.training.log_every = 100
    cfg.training.checkpoint_every_epochs = 20
    cfg.training.fid_every_epochs = 50

    cfg.sample = ConfigDict()
    cfg.sample.num_images = 24
    cfg.sample.steps = 100
    cfg.sample.cfg_scale = 3.0

    cfg.fid = ConfigDict()
    cfg.fid.enabled = False  # Enable when ready to compute FID
    cfg.fid.num_samples = 1000
    cfg.fid.batch_size = 256
    cfg.fid.stats_file = "playground/fid_stats_classifier.json"

    cfg.wandb = ConfigDict()
    cfg.wandb.use = False
    cfg.wandb.project = "ddpm-imagenet64-class-cond"
    cfg.wandb.entity = None
    cfg.wandb.run_name = "ddpm-imagenet64"
    cfg.wandb.dir = "."

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = "playground/checkpoints_imagenet"

    return cfg
