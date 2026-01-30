from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()

    cfg.run = ConfigDict()
    cfg.run.seed = 42
    cfg.run.device = "cuda"  # or "cpu"

    cfg.data = ConfigDict()
    cfg.data.root = "playground/data"
    cfg.data.image_size = 32
    cfg.data.batch_size = 256
    cfg.data.num_workers = 4
    cfg.data.download = True

    cfg.model = ConfigDict()
    cfg.model.width = 64
    cfg.model.num_blocks = 3
    cfg.model.dropout = 0.1

    cfg.training = ConfigDict()
    cfg.training.epochs = 15
    cfg.training.lr = 1e-3
    cfg.training.weight_decay = 1e-4
    cfg.training.grad_clip = 1.0
    cfg.training.log_every = 100
    cfg.training.checkpoint_every_epochs = 5

    cfg.eval = ConfigDict()
    cfg.eval.batch_size = 512

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = "playground/classifier_checkpoints"
    cfg.checkpoint.resume = ""

    cfg.wandb = ConfigDict()
    cfg.wandb.use = False
    cfg.wandb.project = "mnist-classifier"
    cfg.wandb.entity = None
    cfg.wandb.run_name = None
    cfg.wandb.dir = None

    return cfg
