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
    cfg.data.batch_size = 32            # inner-loop batch per task (reduced for memory with N tasks)
    cfg.data.cond_batch_size = 8       # conditioning batch per task
    cfg.data.holdout_per_class = 50     # CIFAR-100 has 500 train samples per class
    cfg.data.use_full_dataset = False   # if False, leave_out_classes are excluded from training
    cfg.data.leave_out_classes = [99]   # classes to leave out for evaluation
    cfg.data.leaveout_eval_holdout = 500
    cfg.data.num_workers = 4

    cfg.model = ConfigDict()
    cfg.model.in_channels = 3           # RGB images
    cfg.model.base_channels = 64
    cfg.model.channel_mults = [1, 2, 4]
    cfg.model.num_res_blocks = 2
    cfg.model.dropout = 0.1
    cfg.model.attn_resolutions = [16]
    cfg.model.num_heads = 4
    # Note: No cond_dim or cross_attn_resolutions - this model uses channel concat

    cfg.diffusion = ConfigDict()
    cfg.diffusion.log_snr_max = 5.0
    cfg.diffusion.log_snr_min = -15.0

    cfg.training = ConfigDict()
    cfg.training.epochs = 500
    cfg.training.steps_per_epoch = 2000
    cfg.training.num_tasks = 8          # number of tasks per outer step for variance reduction
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
    cfg.eval.inner_steps = 2
    cfg.eval.inner_lr = 1e-3

    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = "ddpm-maml-cifar100-cond-concat"
    cfg.wandb.entity = "ricvalp"
    cfg.wandb.run_name = "ddpm-maml-cifar100-cond-concat"
    cfg.wandb.dir = "."

    cfg.fast_params = ConfigDict()
    cfg.fast_params.selector = "up_down_mid_head_gn"

    cfg.sample = ConfigDict()
    cfg.sample.num_images = 32
    cfg.sample.steps = 100
    cfg.sample.dir = os.path.join(os.environ.get("PLAYGROUND_OUTPUT_DIR", "."), "outputs_cond_cifar100_concat")

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = os.path.join(os.environ.get("PLAYGROUND_CHECKPOINT_DIR", "checkpoints"), "cond_cifar100_concat")

    return cfg
