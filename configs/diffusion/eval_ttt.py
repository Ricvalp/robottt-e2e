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
    output_parent_dir = os.environ.get(
        "QRD_OUTPUT_PARENT_DIR",
        "/mnt/external_storage/robotics/quick_robot_draw/runs/outputs",
    )
    checkpoint_parent_dir = os.environ.get(
        "QRD_CHECKPOINT_PARENT_DIR",
        "/mnt/external_storage/robotics/quick_robot_draw/runs/checkpoints",
    )
    resnet_checkpoint_parent_dir = os.environ.get(
        "QRD_RESNET_CHECKPOINT_PARENT_DIR",
        "metrics/checkpoints",
    )

    cfg = ConfigDict()

    cfg.run = ConfigDict()
    cfg.run.seed = 2026
    cfg.run.device = "cuda"

    cfg.data = ConfigDict()
    cfg.data.root = cache_root
    cfg.data.backend = "lmdb"
    cfg.data.K = 0
    cfg.data.max_seq_len = 0
    cfg.data.coordinate_mode = ""
    cfg.data.index_dir = os.path.join(index_root, "faiss_index")
    cfg.data.ids_dir = os.path.join(index_root, "ids_family")
    cfg.data.families_cache_path = "all_families.txt"

    cfg.loader = ConfigDict()
    cfg.loader.num_workers = 4

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = os.path.join(checkpoint_parent_dir, "encoder_decoder")
    cfg.checkpoint.path = ""
    cfg.checkpoint.run_name = ""
    cfg.checkpoint.epoch = 0
    cfg.checkpoint.latest_filename = "latest.pt"

    cfg.ttt = ConfigDict()
    cfg.ttt.inner_steps = 10
    cfg.ttt.inner_lr = 3e-4
    cfg.ttt.max_grad_norm = 1.0
    cfg.ttt.last_frac_fast = 0.25
    cfg.ttt.include_decoder_mlp_fast = True
    cfg.ttt.include_ada_fast = True
    cfg.ttt.include_final_norm_fast = True
    cfg.ttt.include_decoder_self_attention_fast = False
    cfg.ttt.include_decoder_cross_attention_fast = False
    cfg.ttt.include_encoder_fast = False
    cfg.ttt.include_input_projections_fast = False
    cfg.ttt.include_output_head_fast = False
    cfg.ttt.include_diffusion_conditioning_fast = False
    cfg.ttt.num_loo_per_task = 2
    cfg.ttt.outer_context_size = 0
    cfg.ttt.task_chunk_size = 8
    cfg.ttt.reuse_diffusion_noise = True
    cfg.ttt.math_attention = True

    cfg.eval = ConfigDict()
    cfg.eval.tasks = (
        # "empty_sketches",
        # "partial_sketches",
        # "many_samples",
        "fid",
        "fid_no_adaptation",
    )
    cfg.eval.samples = 8
    cfg.eval.seed = 42
    cfg.eval.qualitative_split = "val"
    cfg.eval.num_many_samples = 16
    cfg.eval.max_tokens = 0
    cfg.eval.inference_steps = 10

    cfg.eval.fid = ConfigDict()
    cfg.eval.fid.num_samples = 8192
    cfg.eval.fid.feature_batch_size = 128
    cfg.eval.fid.splits = ("train", "val")
    cfg.eval.fid.resnet_checkpoint_path = os.path.join(
        resnet_checkpoint_parent_dir, "resnet18_step90000.pt"
    )

    cfg.logging = ConfigDict()
    cfg.logging.output_parent_dir = output_parent_dir
    cfg.logging.dir = ""

    return cfg
