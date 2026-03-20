import os

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cache_root = os.environ.get(
        "QRD_CACHE_ROOT",
        "/mnt/external_storage/robotics/quick_robot_draw/data",
    )
    index_root = os.environ.get(
        "QRD_INDEX_ROOT",
        "/mnt/external_storage/robotics/quick_robot_draw/index",
    )

    cfg = ConfigDict()
    cfg.dataset_path = cache_root
    cfg.out_dir = index_root
    cfg.checkpoint_path = "metrics/checkpoints/resnet18_step40000.pt"
    cfg.family_to_label_path = "metrics/family_to_label.yaml"
    cfg.num_workers = 16
    cfg.backend = "lmdb"

    return cfg
