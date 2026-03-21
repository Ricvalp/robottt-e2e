import os

from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:
    cache_root = os.environ.get(
        "QRD_CACHE_ROOT",
        "/mnt/external_storage/robotics/quick_robot_draw/data",
    )
    index_root = os.environ.get(
        "QRD_INDEX_ROOT",
        "/mnt/external_storage/robotics/quick_robot_draw/index",
    )
    checkpoint_parent_dir = os.environ.get(
        "QRD_CHECKPOINT_PARENT_DIR",
        "/mnt/external_storage/robotics/quick_robot_draw/runs/checkpoints",
    )
    profile_trace_dir = os.environ.get(
        "QRD_PROFILE_TRACE_DIR",
        "/mnt/external_storage/robotics/quick_robot_draw/runs/profiles",
    )
    wandb_project = os.environ.get("WANDB_PROJECT", "qrd-pretrain")
    wandb_entity = os.environ.get("WANDB_ENTITY", "ricvalp")

    cfg = ConfigDict()
    cfg.run = ConfigDict()
    cfg.run.seed = 2026
    cfg.run.device = "cuda"

    cfg.data = ConfigDict()
    cfg.data.root = cache_root
    cfg.data.split = "train"
    cfg.data.backend = "lmdb"
    cfg.data.K = 4
    cfg.data.max_seq_len = 480
    cfg.data.max_query_len = 60
    cfg.data.max_context_len = 400
    cfg.data.coordinate_mode = "absolute"
    cfg.data.index_dir = os.path.join(index_root, "faiss_index")
    cfg.data.ids_dir = os.path.join(index_root, "ids_family")
    cfg.data.families_cache_path = "all_families.txt"

    cfg.loader = ConfigDict()
    cfg.loader.batch_size = 256
    cfg.loader.num_workers = 16

    cfg.training = ConfigDict()
    cfg.training.epochs = 100
    cfg.training.lr = 1e-4
    cfg.training.weight_decay = 0.0

    cfg.training.warmup_cosine_annealing = ConfigDict()
    cfg.training.warmup_cosine_annealing.use = False
    cfg.training.warmup_cosine_annealing.warmup_steps = 5000
    cfg.training.warmup_cosine_annealing.T_max = 20000
    cfg.training.warmup_cosine_annealing.max_lr = 1e-3
    cfg.training.warmup_cosine_annealing.min_lr = 1e-5

    cfg.training.cosine_annealing = ConfigDict()
    cfg.training.cosine_annealing.use = False
    cfg.training.cosine_annealing.T_max = 20000
    cfg.training.cosine_annealing.eta_min = 1e-6

    cfg.logging = ConfigDict()
    cfg.logging.loss_log_every = 100

    cfg.model = ConfigDict()
    cfg.model.input_dim = 6
    cfg.model.output_dim = 6
    cfg.model.num_train_timesteps = 1000
    cfg.model.beta_start = 1e-4
    cfg.model.beta_end = 2e-2
    cfg.model.beta_schedule = "scaled_linear"
    cfg.model.prediction_type = "v_prediction"
    cfg.model.hidden_dim = 512
    cfg.model.num_layers = 4
    cfg.model.num_heads = 4
    cfg.model.mlp_dim = 1024
    cfg.model.dropout = 0.0
    cfg.model.attention_dropout = 0.0
    cfg.model.horizon = 8

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = os.path.join(checkpoint_parent_dir, "encoder_decoder")
    cfg.checkpoint.save_interval = 1
    cfg.checkpoint.latest_filename = "latest.pt"
    cfg.checkpoint.save_latest_every_steps = None
    cfg.checkpoint.auto_resume = False
    cfg.checkpoint.resume_from = None

    cfg.eval = ConfigDict()
    cfg.eval.samples = 16
    cfg.eval.seed = 42
    cfg.eval.num_inference_steps = 1000
    cfg.eval.eval_on_train = False

    cfg.profiling = ConfigDict()
    cfg.profiling.use = False
    cfg.profiling.trace_dir = os.path.join(profile_trace_dir, "encoder_decoder")
    cfg.profiling.trace_filename = os.environ.get(
        "QRD_PRETRAIN_PROFILE_TRACE_FILE",
        "pretrain_trace.json",
    )

    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = wandb_project
    cfg.wandb.entity = wandb_entity
    cfg.wandb.log_interval = 200
    cfg.wandb.log_samples_interval = 5000
    cfg.wandb.log_all = False

    return cfg
