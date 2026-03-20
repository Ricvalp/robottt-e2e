import os

from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:
    output_parent_dir = os.environ.get(
        "QRD_OUTPUT_PARENT_DIR",
        "/mnt/external_storage/robotics/quick_robot_draw/runs/outputs",
    )
    checkpoint_parent_dir = os.environ.get(
        "QRD_CHECKPOINT_PARENT_DIR",
        "/mnt/external_storage/robotics/quick_robot_draw/runs/checkpoints",
    )
    wandb_project = os.environ.get("WANDB_PROJECT", "qrd-pretrain")
    wandb_entity = os.environ.get("WANDB_ENTITY", "ricvalp")

    cfg = ConfigDict()
    cfg.seed = 42
    cfg.data_dir = os.path.join(output_parent_dir, "metrics_cache")
    cfg.checkpoint_dir = os.path.join(checkpoint_parent_dir, "metrics")
    cfg.batch_size = 256
    cfg.num_workers = 4
    cfg.learning_rate = 1e-3
    cfg.num_epochs = 10
    cfg.val_steps_per_epoch = 1000
    cfg.save_interval = 10000

    cfg.wandb_logging = ConfigDict()
    cfg.wandb_logging.use = False
    cfg.wandb_logging.project = wandb_project
    cfg.wandb_logging.entity = wandb_entity
    cfg.wandb_logging.log_interval = 200
    cfg.wandb_logging.log_all = False

    cfg.fid = ConfigDict()
    cfg.fid.checkpoint_filename = "resnet18_step100000.pt"

    return cfg
