import os

from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:
    cache_root = os.environ.get(
        "QRD_CACHE_ROOT",
        "/mnt/external_storage/robotics/quick_robot_draw/data",
    )
    output_parent_dir = os.environ.get(
        "QRD_OUTPUT_PARENT_DIR",
        "/mnt/external_storage/robotics/quick_robot_draw/runs/outputs",
    )
    checkpoint_parent_dir = os.environ.get(
        "QRD_CHECKPOINT_PARENT_DIR",
        "/mnt/external_storage/robotics/quick_robot_draw/runs/checkpoints",
    )

    cfg = ConfigDict()

    cfg.seed = 37
    cfg.dataset_path = cache_root
    cfg.output_dir = os.path.join(output_parent_dir, "metrics_cache")
    cfg.family_to_label_path = "metrics/family_to_label.yaml"
    cfg.checkpoint_dir = os.path.join(checkpoint_parent_dir, "metrics")
    cfg.shard_size = 100000
    cfg.split_fracs = {"train": 0.8, "val": 0.1}

    cfg.rasterizer_config = ConfigDict()
    cfg.rasterizer_config.img_size = 64
    cfg.rasterizer_config.antialias = 2
    cfg.rasterizer_config.line_width = 2.0
    cfg.rasterizer_config.background_value = 0.0
    cfg.rasterizer_config.stroke_value = 1.0
    cfg.rasterizer_config.normalize_inputs = False

    return cfg
