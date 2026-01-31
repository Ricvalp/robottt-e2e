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
    cfg.data.batch_size = 32            # inner-loop batch
    cfg.data.cond_batch_size = 32       # conditioning batch (same digit, different instances)
    cfg.data.holdout_per_class = 500
    cfg.data.use_full_dataset = False    # if False, leave_out_digit is excluded from training
    cfg.data.leave_out_digit = 9
    cfg.data.leaveout_eval_holdout = 10000
    cfg.data.num_workers = 4

    cfg.model = ConfigDict()
    cfg.model.in_channels = 1
    cfg.model.base_channels = 64
    cfg.model.channel_mults = [1, 2, 4]
    cfg.model.num_res_blocks = 2
    cfg.model.dropout = 0.1
    cfg.model.attn_resolutions = [16]
    cfg.model.cross_attn_resolutions = [32]
    cfg.model.num_heads = 4
    cfg.model.cond_dim = 256

    cfg.diffusion = ConfigDict()
    cfg.diffusion.beta_start = 1e-4
    cfg.diffusion.beta_end = 0.02
    cfg.diffusion.train_steps = 1000

    cfg.training = ConfigDict()
    cfg.training.epochs = 500
    cfg.training.steps_per_epoch = 1000
    cfg.training.inner_steps = 1
    cfg.training.inner_lr = 1e-3
    cfg.training.outer_lr = 2e-4
    cfg.training.weight_decay = 0.0
    cfg.training.grad_clip = 1.0
    cfg.training.use_amp = True
    cfg.training.sample_every_epochs = 1
    cfg.training.log_every = 50
    cfg.training.checkpoint_every_epochs = 10

    cfg.eval = ConfigDict()
    cfg.eval.inner_steps = 1
    cfg.eval.inner_lr = 1e-3

    cfg.wandb = ConfigDict()
    cfg.wandb.use = False
    cfg.wandb.project = "ddpm-maml-mnist-cond"
    cfg.wandb.entity = None
    cfg.wandb.run_name = "ddpm-maml-mnist-cond"
    cfg.wandb.dir = "."

    cfg.fast_params = ConfigDict()
    cfg.fast_params.selector = "up_down_mid_head_gn"

    cfg.sample = ConfigDict()
    cfg.sample.num_images = 16
    cfg.sample.steps = 100
    cfg.sample.dir = "outputs_cond"

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = "playground/checkpoints_cond"

    return cfg
