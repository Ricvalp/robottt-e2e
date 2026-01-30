from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()

    cfg.run = ConfigDict()
    cfg.run.seed = 42
    cfg.run.device = "cuda"  # "cuda" or "cpu"

    cfg.data = ConfigDict()
    cfg.data.image_size = 32
    cfg.data.batch_size = 64
    cfg.data.num_workers = 4
    cfg.data.holdout_per_class = 200
    cfg.data.use_full_dataset = True
    cfg.data.leave_out_digit = 9
    cfg.data.leaveout_eval_holdout = 500

    cfg.model = ConfigDict()
    cfg.model.base_channels = 64
    cfg.model.channel_mults = (1, 2, 4)
    cfg.model.num_res_blocks = 2
    cfg.model.dropout = 0.0
    cfg.model.attn_resolutions = (16,)
    cfg.model.num_heads = 4

    cfg.diffusion = ConfigDict()
    cfg.diffusion.train_steps = 1000
    cfg.diffusion.beta_start = 1e-4
    cfg.diffusion.beta_end = 2e-2

    cfg.training = ConfigDict()
    cfg.training.epochs = 100
    cfg.training.steps_per_epoch = 250
    cfg.training.outer_lr = 2e-4
    cfg.training.inner_lr = 1e-2
    cfg.training.inner_steps = 3
    cfg.training.grad_clip = 1.0
    cfg.training.use_amp = False
    cfg.training.log_every = 50
    cfg.training.sample_every_epochs = 1
    cfg.training.checkpoint_every_epochs = 1

    cfg.eval = ConfigDict()
    cfg.eval.inner_steps = 1
    cfg.eval.inner_lr = 1e-3

    cfg.fast_params = ConfigDict()
    cfg.fast_params.selector = "up_down_mid_full"  # default fast set

    cfg.sample = ConfigDict()
    cfg.sample.num_images = 16
    cfg.sample.steps = 100
    cfg.sample.dir = "playground_jax/samples"

    cfg.counting = ConfigDict()
    cfg.counting.use = True
    cfg.counting.num_samples = 3000
    cfg.counting.batch_size = 256
    cfg.counting.classifier_ckpt = "playground_jax/classifier_checkpoints/ckpt.npz"

    cfg.fid = ConfigDict()
    cfg.fid.use = True
    cfg.fid.num_samples = 3000
    cfg.fid.batch_size = 256
    cfg.fid.stats_path = "playground_jax/fid_stats_classifier.json"

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = "playground_jax/checkpoints"
    cfg.checkpoint.resume = ""

    cfg.wandb = ConfigDict()
    cfg.wandb.use = False
    cfg.wandb.project = "ddpm-maml-mnist-jax"
    cfg.wandb.entity = None
    cfg.wandb.run_name = None
    cfg.wandb.dir = None

    return cfg
