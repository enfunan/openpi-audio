"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import logging
import os
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
from openpi.models_pytorch.lora_pytorch import GRAD_MODE_FLOW_MATCHING, LoRAConfig, inject_lora
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


class ExponentialMovingAverage:
    """Maintains exponential moving averages of trainable model parameters.

    Shadow params are kept in float32 on GPU (same device as model) for fast
    updates. Only moved to CPU during checkpoint save.
    """

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.num_updates = 0
        # Pre-build (name, param, shadow) triples for fast iteration
        raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        self._params = []
        self.shadow = {}
        for name, param in raw_model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.float().clone()  # same device as param
                self._params.append((name, param, self.shadow[name]))

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for name, param, shadow in self._params:
            # In-place update on GPU — no CPU transfer
            shadow.mul_(decay).add_(param.data.float(), alpha=1 - decay)

    def state_dict(self):
        # Move shadow to CPU for serialization
        return {
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
            "decay": self.decay,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        # Move loaded shadow back to same device as current shadow
        for name in self.shadow:
            if name in state_dict["shadow"]:
                self.shadow[name].copy_(state_dict["shadow"][name])
        # Rebuild _params references
        self._params = [(n, p, self.shadow[n]) for n, p, _ in self._params]

    def apply_shadow_to_model(self, model):
        """Copy EMA weights into a model state dict for saving/inference."""
        raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        model_sd = raw_model.state_dict()
        for name, shadow in self.shadow.items():
            if name in model_sd:
                model_sd[name] = shadow.to(dtype=model_sd[name].dtype)
        return model_sd


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config, ema=None):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save EMA weights for inference (preferred over raw training weights)
        if ema is not None:
            ema_sd = ema.apply_shadow_to_model(model)
            # Cast to bfloat16 for compact inference checkpoint
            ema_sd = {k: v.to(torch.bfloat16) for k, v in ema_sd.items()}
            safetensors.torch.save_file(ema_sd, tmp_ckpt_dir / "model_ema.safetensors")
            torch.save(ema.state_dict(), tmp_ckpt_dir / "ema_state.pt")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
            "ema_enabled": ema is not None,
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device, ema=None, skip_optimizer=False):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            # Re-tie shared weights that safetensors deduplicates during save.
            # PaliGemma ties embed_tokens.weight ↔ lm_head.weight; safetensors
            # only saves one copy, so the other is stale after load_model.
            sd = model_to_load.state_dict()
            ptrs = {}
            retied = []
            for k, v in sd.items():
                p = v.data_ptr()
                if p in ptrs:
                    retied.append((ptrs[p], k))
                else:
                    ptrs[p] = k
            if not retied:
                # No ties found — weights were de-duplicated by safetensors.
                # Copy lm_head → embed_tokens (or any shared pair).
                pali = getattr(model_to_load, "paligemma_with_expert", None)
                if pali is not None:
                    lm = getattr(pali.paligemma, "language_model", None) or getattr(pali.paligemma.model, "language_model", None)
                    lm_head = getattr(pali.paligemma, "lm_head", None)
                    if lm is not None and lm_head is not None and hasattr(lm, "embed_tokens"):
                        lm.embed_tokens.weight = lm_head.weight
                        logging.info("Re-tied embed_tokens.weight ← lm_head.weight after checkpoint load")
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        if skip_optimizer:
            logging.info("Skipping optimizer state load (--reset-optimizer)")
        else:
            logging.info("Loading optimizer state...")
            optimizer_path = ckpt_dir / "optimizer.pt"

            if optimizer_path.exists():
                optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
                logging.info("Loaded optimizer state from pt format")
            else:
                raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        # Load EMA state if available
        if ema is not None:
            ema_path = ckpt_dir / "ema_state.pt"
            if ema_path.exists():
                ema.load_state_dict(torch.load(ema_path, map_location="cpu", weights_only=False))
                logging.info(f"Loaded EMA state (num_updates={ema.num_updates})")
            else:
                logging.info("No EMA state found in checkpoint, starting fresh EMA")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    # Set CUDA allocator config before any CUDA calls
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")

    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    if effective_batch_size == 0:
        raise ValueError(
            f"batch_size ({config.batch_size}) < world_size ({world_size}). "
            f"Use --batch-size={world_size} or higher."
        )
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    # --- Audio / Split LoRA setup ---
    audio_enabled = getattr(model_cfg, "audio_enabled", False)
    training_stage = getattr(model_cfg, "training_stage", "default")
    rehearsal_lambda = getattr(model_cfg, "rehearsal_lambda", 0.01)
    distill_lambda = getattr(model_cfg, "distill_lambda", 0.0)
    is_stage2_load = audio_enabled and training_stage == "default"

    # Stage 1: load base weights BEFORE LoRA injection (checkpoint has original Linear keys)
    # Stage 2: load AFTER LoRA injection (checkpoint has LoRA A/B weights from Stage 1)
    if config.pytorch_weight_path is not None and not is_stage2_load:
        logging.info(f"Loading base weights from: {config.pytorch_weight_path}")
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        pretrained_sd = safetensors.torch.load_file(model_path)
        missing, unexpected = model.load_state_dict(pretrained_sd, strict=False)
        if missing:
            logging.info(f"Keys in model but not in checkpoint ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logging.warning(f"Unexpected keys in checkpoint ({len(unexpected)}): {unexpected[:5]}...")
        logging.info(f"Loaded {len(pretrained_sd)} weights from {config.pytorch_weight_path}")

    if audio_enabled:
        lora_rank = 16
        task_cfg = LoRAConfig(rank=lora_rank, alpha=float(lora_rank))
        audio_cfg = LoRAConfig(rank=lora_rank, alpha=float(lora_rank))
        inject_lora(
            model.paligemma_with_expert.paligemma.language_model,
            model.paligemma_with_expert.gemma_expert.model,
            task_cfg=task_cfg,
            audio_cfg=audio_cfg,
        )
        model = model.to(device)  # ensure new LoRA params are on device

        # Freeze base model weights (LoRA params stay trainable)
        for name, param in model.named_parameters():
            if "base_linear" in name:
                param.requires_grad_(False)

        # Freeze Whisper encoder
        if hasattr(model, "whisper_encoder"):
            model.whisper_encoder.requires_grad_(False)

        # Log trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logging.info(f"Split LoRA injected: {trainable:,} trainable / {total:,} total params")

    # Stage 2: load Stage 1 checkpoint AFTER LoRA injection (has LoRA + audio projector weights)
    if config.pytorch_weight_path is not None and is_stage2_load:
        logging.info(f"Loading Stage 1 checkpoint for Stage 2: {config.pytorch_weight_path}")
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        pretrained_sd = safetensors.torch.load_file(model_path)
        missing, unexpected = model.load_state_dict(pretrained_sd, strict=False)
        if missing:
            logging.info(f"Keys in model but not in Stage 1 checkpoint ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logging.warning(f"Unexpected keys in Stage 1 checkpoint ({len(unexpected)}): {unexpected[:5]}...")
        logging.info(f"Loaded {len(pretrained_sd)} Stage 1 weights (incl. LoRA + audio projector)")

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory/compute optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if use_ddp:
        # Stage 2 dual backward (flow matching + ASR) uses different graphs per step,
        # so static_graph must be False. Stage 1 can use static_graph for efficiency.
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            gradient_as_bucket_view=True,
            static_graph=not is_stage2_load,
        )

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Build parameter groups with stage-dependent LR multipliers
    base_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    is_stage2 = training_stage == "default" and audio_enabled
    audio_proj_params = []
    audio_lora_params = []
    other_params = []
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        if "audio_projector" in name or "attention_pooling" in name:
            audio_proj_params.append(param)
        elif is_stage2 and ("audio_A" in name or "audio_B" in name):
            audio_lora_params.append(param)
        else:
            other_params.append(param)

    # Stage 1: projector 50×, everything else 1×
    # Stage 2: projector 10×, audio LoRA 0.25×, task LoRA 1×
    proj_mult = 10.0 if is_stage2 else 50.0
    if audio_proj_params:
        param_groups = [
            {"params": other_params, "lr": peak_lr},
            {"params": audio_proj_params, "lr": peak_lr * proj_mult},
        ]
        if audio_lora_params:
            param_groups.append({"params": audio_lora_params, "lr": peak_lr * 0.25})
            logging.info(f"Stage 2 param groups: {len(other_params)} task/other at 1× LR, {len(audio_proj_params)} proj/pool at {proj_mult}× LR, {len(audio_lora_params)} audio LoRA at 0.25× LR")
        else:
            logging.info(f"Audio param groups: {len(audio_proj_params)} proj/pool params at {proj_mult}× LR, {len(other_params)} others at base LR")
    else:
        param_groups = [{"params": other_params, "lr": peak_lr}]

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        param_groups,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Initialize EMA before resume so checkpoint can restore EMA state
    ema = None
    if config.ema_decay is not None and is_main:
        ema = ExponentialMovingAverage(model, decay=config.ema_decay)
        ema_mem_gb = sum(p.numel() * 4 for p in model.parameters() if p.requires_grad) / 1e9
        logging.info(f"EMA enabled: decay={config.ema_decay}, shadow memory ~{ema_mem_gb:.1f}GB (GPU float32)")

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device, ema=ema, skip_optimizer=config.reset_optimizer)
        if config.reset_optimizer:
            logging.info(f"Reset optimizer and step counter (loaded weights from step {global_step})")
            global_step = 0
        else:
            logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info(f"EMA: {'enabled, decay=' + str(config.ema_decay) if ema else 'disabled'}")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, observation)  # noqa: PLW2901
            actions = actions.to(device=device, dtype=torch.float32)  # noqa: PLW2901

            # Update LR with per-group multipliers
            # Group 0: task LoRA / other (1×)
            # Group 1: audio projector/pooling (proj_mult×)
            # Group 2 (Stage 2 only): audio LoRA (0.25×)
            base_lr = lr_schedule(global_step)
            for pg_idx, pg in enumerate(optim.param_groups):
                if pg_idx == 1 and audio_enabled:
                    pg["lr"] = base_lr * proj_mult
                elif pg_idx == 2 and is_stage2:
                    pg["lr"] = base_lr * 0.25
                else:
                    pg["lr"] = base_lr

            # Determine if we need dual gradient (Stage 2 rehearsal)
            has_rehearsal = (
                audio_enabled and training_stage == "default"
                and rehearsal_lambda > 0
            )
            has_distill = (
                audio_enabled and training_stage == "default"
                and distill_lambda > 0
            )
            loss_asr_val = 0.0
            loss_distill_val = 0.0
            nan_skipped = False

            if training_stage == "asr_alignment":
                # Stage 1: single ASR forward pass
                losses = model(observation, actions, training_stage="asr_alignment")
                if isinstance(losses, list | tuple):
                    losses = torch.stack(losses)
                elif not isinstance(losses, torch.Tensor):
                    losses = torch.tensor(losses, device=device, dtype=torch.float32)
                loss = losses.mean()
                loss.backward()

            elif has_rehearsal:
                # Stage 2: dual/triple gradient (flow matching + ASR rehearsal + optional distillation)
                inner_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

                # First backward: flow matching (detach audio LoRA)
                # Use return_intermediates to get noise/time for distillation
                if use_ddp:
                    model.require_backward_grad_sync = False
                result = model(observation, actions, gradient_mode=GRAD_MODE_FLOW_MATCHING,
                               return_intermediates=has_distill)
                if has_distill:
                    losses, fm_noise, fm_time = result
                else:
                    losses = result
                if isinstance(losses, list | tuple):
                    losses = torch.stack(losses)
                elif not isinstance(losses, torch.Tensor):
                    losses = torch.tensor(losses, device=device, dtype=torch.float32)
                loss_fm = losses.mean()
                loss_fm.backward()

                # Save flow matching gradients
                grads_fm = {}
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        grads_fm[n] = p.grad.clone()
                optim.zero_grad(set_to_none=True)

                # Optional: distillation backward (teacher = text oracle via base weights)
                if has_distill:
                    if use_ddp:
                        model.require_backward_grad_sync = False
                    # Teacher: text oracle velocity (no grad, LoRA bypassed)
                    v_teacher = inner_model.compute_teacher_velocity(
                        observation, actions, fm_noise, fm_time)
                    # Student: audio velocity (same noise/time, LoRA active)
                    student_losses, _, _ = model(
                        observation, actions, noise=fm_noise, time=fm_time,
                        gradient_mode=GRAD_MODE_FLOW_MATCHING,
                        return_intermediates=True)
                    # We need v_t from student — recompute forward to get velocity
                    # Actually, the MSE loss already captures the difference.
                    # Instead, compute distillation as MSE between student and teacher velocities.
                    # Re-run student forward to get v_t directly:
                    obs_inner = inner_model._preprocess_observation(observation, train=True)
                    images_d = list(obs_inner.images.values())
                    img_masks_d = list(obs_inner.image_masks.values())
                    time_exp = fm_time[:, None, None]
                    x_t_d = time_exp * fm_noise + (1 - time_exp) * actions
                    prefix_embs, prefix_pad_masks, prefix_att_masks, audio_pos_mask = inner_model.embed_prefix(
                        images_d, img_masks_d, obs_inner.tokenized_prompt, obs_inner.tokenized_prompt_mask,
                        audio_whisper_hidden=obs_inner.audio_whisper_hidden,
                        audio=getattr(obs_inner, "audio", None),
                        audio_mask=obs_inner.audio_mask,
                    )
                    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond_d = inner_model.embed_suffix(
                        obs_inner.state, x_t_d, fm_time)
                    # Cast to bfloat16 if model weights are bf16 (matches regular forward())
                    if (
                        inner_model.paligemma_with_expert.paligemma.language_model
                        .layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
                    ):
                        prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
                        suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
                    pad_masks_d = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
                    att_masks_d = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
                    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
                    att_2d_d = make_att_2d_masks(pad_masks_d, att_masks_d)
                    pos_ids_d = torch.cumsum(pad_masks_d, dim=1) - 1
                    att_4d_d = inner_model._prepare_attention_masks_4d(att_2d_d)
                    # gradient_mode=None: both task + audio LoRA get distill gradients.
                    # This is intentional — distillation teaches the audio pathway to
                    # produce text-oracle-like velocities, so audio LoRA MUST be updated.
                    (_, suff_out_d), _ = inner_model.paligemma_with_expert.forward(
                        attention_mask=att_4d_d, position_ids=pos_ids_d,
                        past_key_values=None, inputs_embeds=[prefix_embs, suffix_embs],
                        use_cache=False, adarms_cond=[None, adarms_cond_d],
                        audio_mask=audio_pos_mask, gradient_mode=None,
                    )
                    suff_out_d = suff_out_d[:, -inner_model.config.action_horizon:]
                    v_student = inner_model.action_out_proj(suff_out_d.float())
                    loss_distill = F.mse_loss(v_student, v_teacher.detach())
                    loss_distill.backward()
                    loss_distill_val = loss_distill.item()

                    # Save distillation gradients and combine with flow matching
                    for n, p in model.named_parameters():
                        if p.grad is not None and n in grads_fm:
                            grads_fm[n] = grads_fm[n] + distill_lambda * p.grad
                        # If only distill grad exists (unlikely), add it
                        elif p.grad is not None:
                            grads_fm[n] = distill_lambda * p.grad.clone()
                    optim.zero_grad(set_to_none=True)
                    del fm_noise, fm_time

                # ASR rehearsal backward (detach task LoRA)
                if use_ddp:
                    model.require_backward_grad_sync = True
                loss_asr = inner_model.compute_asr_loss_for_rehearsal(observation)
                loss_asr.backward()
                loss_asr_val = loss_asr.item()

                # Combine gradients: grad = grad_fm [+ α·grad_distill] + λ·grad_asr
                for n, p in model.named_parameters():
                    if p.grad is not None and n in grads_fm:
                        p.grad = grads_fm[n] + rehearsal_lambda * p.grad
                    elif n in grads_fm:
                        p.grad = grads_fm[n]
                del grads_fm
                loss = loss_fm

            else:
                # Standard flow matching (no audio or no rehearsal)
                losses = model(observation, actions)
                if isinstance(losses, list | tuple):
                    losses = torch.stack(losses)
                elif not isinstance(losses, torch.Tensor):
                    losses = torch.tensor(losses, device=device, dtype=torch.float32)
                loss = losses.mean()
                loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping (trainable params only)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.optimizer.clip_gradient_norm)

            # NaN gradient guard
            if not torch.isfinite(grad_norm):
                logging.warning(f"Step {global_step}: NaN/Inf grad_norm ({grad_norm:.4f}), skipping")
                optim.zero_grad(set_to_none=True)
                nan_skipped = True
            else:
                # Optimizer step
                optim.step()
                optim.zero_grad(set_to_none=True)
                # Update EMA shadow params
                if ema is not None:
                    ema.update(model)

            # Collect stats
            if is_main:
                info = {
                    "loss": loss.item(),
                    "learning_rate": optim.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "nan_skipped": 1.0 if nan_skipped else 0.0,
                }
                if has_rehearsal:
                    info["loss_asr"] = loss_asr_val
                if has_distill:
                    info["loss_distill"] = loss_distill_val
                infos.append(info)

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                avg_nan_skipped = sum(info.get("nan_skipped", 0.0) for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)

                log_parts = [f"step={global_step}", f"loss={avg_loss:.4f}", f"lr={avg_lr:.2e}"]
                if avg_grad_norm is not None:
                    log_parts.append(f"grad_norm={avg_grad_norm:.2f}")
                if avg_nan_skipped > 0:
                    log_parts.append(f"nan_skipped={avg_nan_skipped:.2f}")

                avg_loss_asr = None
                if any("loss_asr" in info for info in infos):
                    asr_vals = [info["loss_asr"] for info in infos if "loss_asr" in info]
                    if asr_vals:
                        avg_loss_asr = sum(asr_vals) / len(asr_vals)
                        log_parts.append(f"loss_asr={avg_loss_asr:.4f}")

                avg_loss_distill = None
                if any("loss_distill" in info for info in infos):
                    distill_vals = [info["loss_distill"] for info in infos if "loss_distill" in info]
                    if distill_vals:
                        avg_loss_distill = sum(distill_vals) / len(distill_vals)
                        log_parts.append(f"loss_distill={avg_loss_distill:.4f}")

                log_parts.append(f"time={elapsed:.1f}s")
                logging.info(" ".join(log_parts))

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                        "nan_skipped": avg_nan_skipped,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    if avg_loss_asr is not None:
                        log_payload["loss_asr"] = avg_loss_asr
                    if avg_loss_distill is not None:
                        log_payload["loss_distill"] = avg_loss_distill
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config, ema=ema)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
