from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()

    cfg.run = ConfigDict()
    cfg.run.seed = 42
    cfg.run.device = "cuda"

    cfg.data = ConfigDict()
    cfg.data.root = "data"  # CelebA will be downloaded here
    cfg.data.download = True
    cfg.data.image_size = 64
    cfg.data.batch_size = 128  # Per GPU
    cfg.data.num_workers = 8

    cfg.model = ConfigDict()
    cfg.model.width = 64
    cfg.model.num_blocks = 4
    cfg.model.dropout = 0.1
    cfg.model.num_attributes = 40  # CelebA has 40 binary attributes
    cfg.model.in_channels = 3

    cfg.training = ConfigDict()
    cfg.training.epochs = 50
    cfg.training.lr = 1e-3
    cfg.training.weight_decay = 1e-4
    cfg.training.grad_clip = 1.0
    cfg.training.log_every = 100
    cfg.training.checkpoint_every_epochs = 20

    cfg.eval = ConfigDict()
    cfg.eval.batch_size = 256

    cfg.wandb = ConfigDict()
    cfg.wandb.use = False
    cfg.wandb.project = "celeba-classifier"
    cfg.wandb.entity = None
    cfg.wandb.run_name = "celeba-classifier"
    cfg.wandb.dir = "."

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = "playground/classifier_checkpoints/celeba"

    return cfg
