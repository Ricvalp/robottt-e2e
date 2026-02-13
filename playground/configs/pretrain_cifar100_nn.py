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
    cfg.data.image_size = 32
    cfg.data.batch_size = 256
    cfg.data.cond_batch_size = 8
    cfg.data.num_workers = 4

    # Nearest-neighbor dataset settings
    cfg.data.train_split = "train"
    cfg.data.eval_split = "test"
    cfg.data.index_dir = os.path.join(
        os.environ.get("PLAYGROUND_DATA_ROOT", "data"),
        "cifar100_faiss",
    )
    cfg.data.index_path_train = ""
    cfg.data.meta_path_train = ""
    cfg.data.index_path_eval = ""
    cfg.data.meta_path_eval = ""
    cfg.data.hf_cache_dir = ""
    cfg.data.build_index_if_missing = True
    cfg.data.random_query_train = True
    cfg.data.random_query_eval = True
    cfg.data.return_metadata_train = False
    cfg.data.return_metadata_eval = True

    # Keep architecture aligned with conditional MAML finetuning scripts.
    cfg.model = ConfigDict()
    cfg.model.in_channels = 3
    cfg.model.base_channels = 64
    cfg.model.channel_mults = [1, 2, 4]
    cfg.model.num_res_blocks = 2
    cfg.model.dropout = 0.1
    cfg.model.attn_resolutions = [16]
    cfg.model.cross_attn_resolutions = [32]
    cfg.model.num_heads = 4
    cfg.model.cond_dim = 256

    cfg.diffusion = ConfigDict()
    cfg.diffusion.log_snr_max = 5.0
    cfg.diffusion.log_snr_min = -15.0

    cfg.training = ConfigDict()
    cfg.training.epochs = 300
    cfg.training.steps_per_epoch = 2000
    cfg.training.lr = 2e-4
    cfg.training.weight_decay = 0.0
    cfg.training.grad_clip = 1.0
    cfg.training.use_amp = True
    cfg.training.log_every = 50
    cfg.training.sample_every_epochs = 1
    cfg.training.fid_every_epochs = 1
    cfg.training.checkpoint_every_epochs = 10
    cfg.training.reset_optimizer_on_resume = False

    cfg.sample = ConfigDict()
    cfg.sample.steps = 100
    cfg.sample.num_images = 16
    cfg.sample.eval_batch_size = 64

    cfg.fid = ConfigDict()
    cfg.fid.enabled = True
    cfg.fid.num_samples = 1000
    cfg.fid.batch_size = 256
    cfg.fid.feature_batch_size = 256
    cfg.fid.inception_feature_batch_size = 256
    cfg.fid.x0_log_images = 8
    cfg.fid.reference_batch_size = 256
    cfg.fid.reference_max_samples = 0
    cfg.fid.reference_split = "train"
    cfg.fid.dataset_key = "cifar100_cond_pretrain_nn"
    cfg.fid.inception_dataset_key = "cifar100_cond_pretrain_nn_inception"
    cfg.fid.classifier_checkpoint = os.environ.get(
        "CIFAR100_CLASSIFIER_CKPT",
        os.path.join(
            os.environ.get("PLAYGROUND_CHECKPOINT_DIR", "playground/checkpoints"),
            "classifier_checkpoints",
            "cifar100",
            "cifar100_classifier.pt",
        ),
    )
    cfg.fid.classifier_width = 64
    cfg.fid.classifier_num_blocks = 4
    cfg.fid.stats_file = os.environ.get("FID_STATS_FILE", "fid_stats_classifier.json")

    # Stored into checkpoints for easier downstream MAML finetuning consistency checks.
    cfg.fast_params = ConfigDict()
    cfg.fast_params.selector = "up_down_mid_head_gn"

    cfg.wandb = ConfigDict()
    cfg.wandb.use = False
    cfg.wandb.project = "ddpm-cifar100-conditional-pretrain-nn"
    cfg.wandb.entity = "ricvalp"
    cfg.wandb.run_name = None
    cfg.wandb.dir = "."

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = os.path.join(
        os.environ.get("PLAYGROUND_CHECKPOINT_DIR", "playground/checkpoints"),
        "pretrain_cond_cifar100_nn",
    )
    cfg.checkpoint.resume = ""

    return cfg
