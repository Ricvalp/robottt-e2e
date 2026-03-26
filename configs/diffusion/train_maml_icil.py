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
    trace_filename = os.environ.get(
        "QRD_PROFILE_TRACE_FILE",
        os.environ.get("QRD_TTT_PROFILE_TRACE_FILE", "trace.json"),
    )
    wandb_project = os.environ.get("WANDB_PROJECT", "qrd-pretrain")
    wandb_entity = os.environ.get("WANDB_ENTITY", "ricvalp")

    cfg = ConfigDict()

    cfg.run = ConfigDict()
    cfg.run.seed = 2026
    cfg.run.device = "cuda"  # or "cpu"

    cfg.data = ConfigDict()
    cfg.data.root = cache_root
    cfg.data.split = "train"
    cfg.data.backend = "lmdb"
    cfg.data.K = 0
    cfg.data.max_seq_len = 480
    cfg.data.coordinate_mode = "absolute"
    cfg.data.index_dir = os.path.join(index_root, "faiss_index")
    cfg.data.ids_dir = os.path.join(index_root, "ids_family")
    cfg.data.families_cache_path = "all_families.txt"

    cfg.loader = ConfigDict()
    cfg.loader.batch_size = 32
    cfg.loader.num_workers = 4

    cfg.finetune = ConfigDict()
    cfg.finetune.pretrained_checkpoint = ""
    cfg.finetune.strict_load = True

    cfg.maml = ConfigDict()
    cfg.maml.inner_steps = 1
    cfg.maml.inner_lr = 3e-4
    cfg.maml.outer_lr = 1e-5
    cfg.maml.max_grad_norm = 1.0
    cfg.maml.last_frac_fast = 0.25
    cfg.maml.include_ada_fast = True
    cfg.maml.include_final_norm_fast = True
    cfg.maml.num_loo_per_task = 2
    cfg.maml.outer_context_size = 0
    cfg.maml.reuse_diffusion_noise = True
    cfg.maml.math_attention = True

    cfg.outer = ConfigDict()
    cfg.outer.train_encoder = False
    cfg.outer.train_decoder = True
    cfg.outer.train_input_projections = True
    cfg.outer.train_output_head = True
    cfg.outer.train_diffusion_conditioning = True

    cfg.training = ConfigDict()
    cfg.training.epochs = 100
    cfg.training.weight_decay = 1e-4

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
    cfg.logging.log_loss_every = 10

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
    cfg.checkpoint.save_latest_every_steps = 0

    cfg.eval = ConfigDict()
    cfg.eval.samples = 8
    cfg.eval.seed = 42
    cfg.eval.num_inference_steps = 0
    cfg.eval.eval_on_train = False

    cfg.profiling = ConfigDict()
    cfg.profiling.use = False
    cfg.profiling.trace_dir = os.path.join(profile_trace_dir, "encoder_decoder")
    cfg.profiling.trace_filename = trace_filename

    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = wandb_project
    cfg.wandb.entity = wandb_entity
    cfg.wandb.samples_log_interval = 500
    cfg.wandb.log_all = False

    return cfg
