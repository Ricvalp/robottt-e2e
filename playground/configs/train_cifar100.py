from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()

    cfg.run = ConfigDict()
    cfg.run.seed = 42
    cfg.run.device = "cuda"

    cfg.data = ConfigDict()
    cfg.data.root = "data"
    cfg.data.download = True
    cfg.data.image_size = 32
    cfg.data.batch_size = 64
    cfg.data.num_workers = 4

    cfg.model = ConfigDict()
    cfg.model.in_channels = 3
    cfg.model.base_channels = 128
    cfg.model.channel_mults = [1, 2, 2, 2]
    cfg.model.num_res_blocks = 2
    cfg.model.dropout = 0.1
    cfg.model.attn_resolutions = [16, 8]
    cfg.model.num_heads = 4
    cfg.model.num_classes = 100
    cfg.model.cond_dim = 256

    cfg.diffusion = ConfigDict()
    cfg.diffusion.log_snr_max = 5.0
    cfg.diffusion.log_snr_min = -15.0
    
    cfg.diffusion.cfg_drop_prob = 0.1

    cfg.training = ConfigDict()
    cfg.training.epochs = 500
    cfg.training.steps_per_epoch = 1000
    cfg.training.lr = 2e-4
    cfg.training.weight_decay = 0.0
    cfg.training.grad_clip = 1.0
    cfg.training.use_amp = True
    cfg.training.sample_every_epochs = 5
    cfg.training.log_every = 50
    cfg.training.checkpoint_every_epochs = 20
    cfg.training.fid_every_epochs = 1

    cfg.sample = ConfigDict()
    cfg.sample.num_images = 16
    cfg.sample.steps = 100
    cfg.sample.cfg_scale = 3.0

    cfg.fid = ConfigDict()
    cfg.fid.enabled = True
    cfg.fid.num_samples = 1000
    cfg.fid.batch_size = 512  # Larger batch for faster FID sample generation
    cfg.fid.classifier_checkpoint = "playground/classifier_checkpoints/cifar100/cifar100_classifier.pt"
    cfg.fid.stats_file = "playground/fid_stats_classifier.json"
    cfg.fid.classifier_width = 64  # Must match trained classifier
    cfg.fid.classifier_num_blocks = 4  # Must match trained classifier

    cfg.wandb = ConfigDict()
    cfg.wandb.use = False
    cfg.wandb.project = "ddpm-cifar100-class-cond"
    cfg.wandb.entity = None
    cfg.wandb.run_name = "ddpm-cifar100-class-cond"
    cfg.wandb.dir = "."

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = "playground/checkpoints_cifar100"

    return cfg
