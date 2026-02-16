import os
from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()

    cfg.run = ConfigDict()
    cfg.run.seed = 42
    cfg.run.device = "cuda"

    cfg.ddp = ConfigDict()
    cfg.ddp.backend = "nccl"
    cfg.ddp.find_unused_parameters = False

    cfg.data = ConfigDict()
    cfg.data.train_dir = os.environ.get("IMAGENET_TRAIN_DIR", "/mnt/external_storage/torchvision_ImageFolder/train")
    cfg.data.eval_dir = os.environ.get("IMAGENET_VAL_DIR", "/mnt/external_storage/torchvision_ImageFolder/val")
    cfg.data.image_size = 64
    cfg.data.batch_size = 24  # per GPU
    cfg.data.cond_batch_size = 8
    cfg.data.num_workers = 0  # avoid duplicating large FAISS index in worker processes

    emb_dir = os.environ.get("IMAGENET_EMBEDDINGS_DIR", "/mnt/external_storage/imagenet_nn/embeddings")
    faiss_dir = os.environ.get("IMAGENET_FAISS_DIR", "/mnt/external_storage/imagenet_nn/faiss")

    cfg.data.train_embeddings = os.path.join(emb_dir, "imagenet_train_dinov2_vitb14_embeddings.npy")
    cfg.data.eval_embeddings = os.path.join(emb_dir, "imagenet_val_dinov2_vitb14_embeddings.npy")

    cfg.data.train_faiss_index = os.path.join(faiss_dir, "imagenet_train_dinov2_vitb14_ivf.index")
    cfg.data.eval_faiss_index = os.path.join(faiss_dir, "imagenet_val_dinov2_vitb14_ivf.index")
    cfg.data.train_faiss_meta = cfg.data.train_faiss_index + ".meta.json"
    cfg.data.eval_faiss_meta = cfg.data.eval_faiss_index + ".meta.json"

    cfg.data.random_query_train = True
    cfg.data.random_query_eval = True
    cfg.data.return_metadata_train = False
    cfg.data.return_metadata_eval = True
    cfg.data.faiss_nprobe_train = 64
    cfg.data.faiss_nprobe_eval = 64
    cfg.data.train_random_flip_prob = 0.5

    cfg.model = ConfigDict()
    cfg.model.in_channels = 3
    cfg.model.base_channels = 192
    cfg.model.channel_mults = [1, 2, 3, 4]
    cfg.model.num_res_blocks = 2
    cfg.model.dropout = 0.1
    cfg.model.attn_resolutions = [16, 8]
    cfg.model.cross_attn_resolutions = [32, 16]
    cfg.model.num_heads = 4
    cfg.model.cond_dim = 512

    cfg.diffusion = ConfigDict()
    cfg.diffusion.log_snr_max = 5.0
    cfg.diffusion.log_snr_min = -15.0
    cfg.diffusion.p_uncond = 0.1

    cfg.training = ConfigDict()
    cfg.training.epochs = 300
    cfg.training.steps_per_epoch = 5000
    cfg.training.lr = 1e-4
    cfg.training.weight_decay = 0.0
    cfg.training.grad_clip = 1.0
    cfg.training.use_amp = True
    cfg.training.log_every = 100
    cfg.training.sample_every_epochs = 1
    cfg.training.fid_every_epochs = 1
    cfg.training.checkpoint_every_epochs = 5
    cfg.training.reset_optimizer_on_resume = False

    cfg.sample = ConfigDict()
    cfg.sample.steps = 100
    cfg.sample.num_images = 16
    cfg.sample.eval_batch_size = 64

    cfg.fid = ConfigDict()
    cfg.fid.enabled = True
    cfg.fid.backend = "pytorch_fid"
    cfg.fid.num_samples = 1000
    cfg.fid.batch_size = 128
    cfg.fid.feature_batch_size = 64
    cfg.fid.pytorch_fid_batch_size = 64
    cfg.fid.pytorch_fid_num_workers = 0
    cfg.fid.pytorch_fid_dims = 2048
    cfg.fid.keep_generated_samples = False
    cfg.fid.generated_samples_dir = os.path.join(
        os.environ.get("PLAYGROUND_OUTPUT_DIR", "playground/outputs"),
        "fid_generated_imagenet_nn",
    )
    cfg.fid.pytorch_fid_stats_file = os.environ.get(
        "IMAGENET_PYTORCH_FID_STATS_FILE",
        os.path.join(
            os.environ.get("PLAYGROUND_OUTPUT_DIR", "playground/outputs"),
            "fid_stats_inception_pytorch_fid.npz",
        ),
    )
    cfg.fid.x0_log_images = 8
    cfg.fid.reference_batch_size = 128
    cfg.fid.reference_max_samples = 50000
    cfg.fid.reference_dir = cfg.data.eval_dir

    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = "ddpm-imagenet-conditional-pretrain-nn"
    cfg.wandb.entity = "ricvalp"
    cfg.wandb.run_name = None
    cfg.wandb.dir = "."

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = os.path.join(
        os.environ.get("PLAYGROUND_CHECKPOINT_DIR", "playground/checkpoints"),
        "pretrain_cond_imagenet_nn",
    )
    cfg.checkpoint.resume = ""

    return cfg
