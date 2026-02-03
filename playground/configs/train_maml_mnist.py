import os
from ml_collections import ConfigDict


def get_config() -> ConfigDict:

    cfg = ConfigDict()

    cfg.run = ConfigDict()
    cfg.run.seed = 42
    cfg.run.device = "cuda"  # or "cpu"

    cfg.data = ConfigDict()
    cfg.data.root = os.environ.get("PLAYGROUND_DATA_ROOT", "data")
    cfg.data.image_size = 32
    cfg.data.batch_size = 64
    cfg.data.num_workers = 4
    cfg.data.download = True
    cfg.data.holdout_per_class = 200  # images held out per class for eval/adaptation
    cfg.data.use_full_dataset = False  # if False, leave out one digit from training
    cfg.data.leave_out_digit = 9      # digit excluded from training when use_full_dataset=False
    cfg.data.leaveout_eval_holdout = 10000  # number of images of leave_out_digit used for eval/adaptation

    cfg.model = ConfigDict()
    cfg.model.in_channels = 1
    cfg.model.base_channels = 64
    cfg.model.channel_mults = (1, 2, 4)
    cfg.model.num_res_blocks = 2
    cfg.model.dropout = 0.0
    cfg.model.attn_resolutions = (16,)
    cfg.model.num_heads = 4

    cfg.diffusion = ConfigDict()
    
    cfg.diffusion.log_snr_max = 5.0
    cfg.diffusion.beta_end = 2e-2

    cfg.training = ConfigDict()
    cfg.training.epochs = 500
    cfg.training.steps_per_epoch = 500
    cfg.training.outer_lr = 2e-4
    cfg.training.inner_lr = 1e-3
    cfg.training.inner_steps = 2
    cfg.training.weight_decay = 0.0
    cfg.training.grad_clip = 1.0
    cfg.training.use_amp = False
    cfg.training.log_every = 50
    cfg.training.sample_every_epochs = 5
    cfg.training.checkpoint_every_epochs = 5

    cfg.eval = ConfigDict()
    cfg.eval.inner_steps = 1
    cfg.eval.inner_lr = 1e-3

    cfg.fast_params = ConfigDict()
    cfg.fast_params.selector = "up_down_mid_head"  # only time MLPs in up blocks

    cfg.sample = ConfigDict()
    cfg.sample.num_images = 16
    cfg.sample.steps = 100
    cfg.sample.dir = os.path.join(os.environ.get("PLAYGROUND_OUTPUT_DIR", "."), "maml_samples")

    cfg.counting = ConfigDict()
    cfg.counting.use = True
    cfg.counting.num_samples = 3000
    cfg.counting.batch_size = 256
    cfg.counting.classifier_ckpt = os.environ.get("MNIST_CLASSIFIER_CKPT",
        os.path.join(os.environ.get("PLAYGROUND_CHECKPOINT_DIR", "checkpoints"), "classifier_mnist", "classifier_epoch_010.pt"))

    cfg.fid = ConfigDict()
    cfg.fid.use = True
    cfg.fid.num_samples = 3000
    cfg.fid.batch_size = 256
    cfg.fid.stats_path = os.environ.get("FID_STATS_FILE", "fid_stats_classifier.json")

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = os.path.join(os.environ.get("PLAYGROUND_CHECKPOINT_DIR", "checkpoints"), "maml_mnist")
    cfg.checkpoint.resume = ""

    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = "ddpm-maml-mnist"
    cfg.wandb.entity = "ricvalp"
    cfg.wandb.run_name = None
    cfg.wandb.dir = None

    return cfg
