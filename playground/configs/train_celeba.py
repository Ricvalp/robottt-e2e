import os
from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()

    cfg.run = ConfigDict()
    cfg.run.seed = 42
    cfg.run.device = "cuda"

    cfg.data = ConfigDict()
    cfg.data.root = os.environ.get("PLAYGROUND_DATA_ROOT", "data")
    cfg.data.download = True
    cfg.data.image_size = 64
    cfg.data.batch_size = 64  # Per GPU
    cfg.data.num_workers = 8

    cfg.model = ConfigDict()
    cfg.model.in_channels = 3
    cfg.model.base_channels = 128
    cfg.model.channel_mults = [1, 2, 2, 2]
    cfg.model.num_res_blocks = 2
    cfg.model.dropout = 0.1
    cfg.model.attn_resolutions = [16, 8]
    cfg.model.num_heads = 4
    cfg.model.cond_dim = 256

    cfg.diffusion = ConfigDict()
    cfg.diffusion.log_snr_max = 5.0
    cfg.diffusion.log_snr_min = -15.0
    

    cfg.training = ConfigDict()
    cfg.training.epochs = 200
    cfg.training.lr = 2e-4
    cfg.training.weight_decay = 0.0
    cfg.training.grad_clip = 1.0
    cfg.training.use_amp = True
    cfg.training.sample_every_epochs = 5
    cfg.training.log_every = 100
    cfg.training.checkpoint_every_epochs = 20
    cfg.training.fid_every_epochs = 50

    cfg.sample = ConfigDict()
    cfg.sample.num_images = 16
    cfg.sample.steps = 100

    cfg.fid = ConfigDict()
    cfg.fid.enabled = False  # Enable when ready to compute FID
    cfg.fid.num_samples = 1000
    cfg.fid.batch_size = 256
    cfg.fid.stats_file = os.environ.get("FID_STATS_FILE", "fid_stats_classifier.json")

    cfg.wandb = ConfigDict()
    cfg.wandb.use = False
    cfg.wandb.project = "ddpm-celeba64-uncond"
    cfg.wandb.entity = None
    cfg.wandb.run_name = "ddpm-celeba64"
    cfg.wandb.dir = "."

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = os.path.join(os.environ.get("PLAYGROUND_CHECKPOINT_DIR", "checkpoints"), "celeba")

    return cfg
