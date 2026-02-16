import os
from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()

    cfg.run = ConfigDict()
    cfg.run.seed = 42
    cfg.run.device = "cuda"

    cfg.ddp = ConfigDict()
    cfg.ddp.backend = "nccl"

    cfg.data = ConfigDict()
    cfg.data.train_dir = os.environ.get("IMAGENET_TRAIN_DIR", "/mnt/external_storage/torchvision_ImageFolder/train")
    cfg.data.eval_dir = os.environ.get("IMAGENET_VAL_DIR", "/mnt/external_storage/torchvision_ImageFolder/val")
    cfg.data.image_size = 64
    cfg.data.cond_batch_size = 8
    cfg.data.num_workers = 0  # keep low because each worker may hold a FAISS index copy

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
    cfg.data.return_metadata_train = True
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
    cfg.training.epochs = 200
    cfg.training.steps_per_epoch = 2000
    cfg.training.num_tasks = 8  # meta-batch size (tasks per outer step, per GPU)
    cfg.training.inner_steps = 1
    cfg.training.inner_lr = 1e-4
    cfg.training.outer_lr = 1e-4
    cfg.training.weight_decay = 0.0
    cfg.training.grad_clip = 1.0
    cfg.training.use_amp = True
    cfg.training.sample_every_epochs = 1
    cfg.training.log_every = 20
    cfg.training.checkpoint_every_epochs = 5
    cfg.training.reset_optimizer_on_resume = False

    cfg.eval = ConfigDict()
    cfg.eval.inner_steps = 2
    cfg.eval.inner_lr = 1e-4

    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = "ddpm-maml-imagenet-cond-nn"
    cfg.wandb.entity = "ricvalp"
    cfg.wandb.run_name = None
    cfg.wandb.dir = "."

    cfg.fast_params = ConfigDict()
    cfg.fast_params.selector = "up_down_mid_head_gn"

    cfg.pretrained = ConfigDict()
    cfg.pretrained.use = True
    cfg.pretrained.strict = True
    cfg.pretrained.checkpoint = os.environ.get(
        "IMAGENET_PRETRAIN_CKPT",
        os.path.join(
            os.environ.get("PLAYGROUND_CHECKPOINT_DIR", "playground/checkpoints"),
            "pretrain_cond_imagenet_nn",
            "pretrain_imagenet_nn_best_fid.pt",
        ),
    )

    cfg.sample = ConfigDict()
    cfg.sample.num_images = 16
    cfg.sample.steps = 100
    cfg.sample.eval_batch_size = 16
    cfg.sample.dir = os.path.join(os.environ.get("PLAYGROUND_OUTPUT_DIR", "."), "outputs_cond_imagenet_nn")

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = os.path.join(
        os.environ.get("PLAYGROUND_CHECKPOINT_DIR", "playground/checkpoints"),
        "cond_imagenet_nn_vmap",
    )
    cfg.checkpoint.resume = ""

    return cfg
