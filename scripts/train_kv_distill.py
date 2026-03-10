"""
KV Cache Distillation training for audio-conditioned π0.5.

Stage 1: Aligns student (audio) KV cache to teacher (text) KV cache.
  - Teacher: frozen base model, text input, no LoRA (GRAD_MODE_BYPASS)
  - Student: Perceiver Resampler + PaliGemma LoRA, audio fills text slot
  - Loss: per-position MSE on image KVs + pooled MSE on text/audio KVs

Stage 2: End-to-end action fine-tuning (if Stage 1 eval insufficient).
  - Adds Action Expert LoRA, full ODE rollout
  - Loss: flow matching + optional KV alignment regularization

Usage (single GPU):
  OPENPI_NO_COMPILE=1 python scripts/train_kv_distill.py \\
    --config pi05_kv_distill_stage1 --exp-name kv_stage1_run1

Usage (multi-GPU):
  OPENPI_NO_COMPILE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  torchrun --standalone --nnodes=1 --nproc_per_node=8 \\
    scripts/train_kv_distill.py --config pi05_kv_distill_stage1 --exp-name kv_stage1_run1
"""

import argparse
import dataclasses
import gc
import logging
import os
import shutil
import time

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch as pi0_mod
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.models_pytorch.kv_distill_loss import compute_kv_distill_loss
from openpi.models_pytorch.lora_pytorch import (
    GRAD_MODE_BYPASS,
    LoRAConfig,
    inject_lora,
)


def init_logging():
    class CustomFormatter(logging.Formatter):
        _level_map = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}
        def format(self, record):
            record.levelname = self._level_map.get(record.levelname, record.levelname)
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


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed, local_rank):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def save_checkpoint(model, optimizer, step, checkpoint_dir, data_config):
    final_dir = checkpoint_dir / f"{step}"
    tmp_dir = checkpoint_dir / f"tmp_{step}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    safetensors.torch.save_model(model_to_save, tmp_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    torch.save({"global_step": step, "timestamp": time.time()}, tmp_dir / "metadata.pt")

    if data_config is not None:
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_dir / "assets" / data_config.asset_id, norm_stats)

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    logging.info(f"Saved checkpoint at step {step} → {final_dir}")


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    steps = [int(d.name) for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not steps:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
    latest = max(steps)
    ckpt = checkpoint_dir / f"{latest}"
    model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    safetensors.torch.load_model(model_to_load, ckpt / "model.safetensors", device=str(device))
    optimizer.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location=device, weights_only=False))
    meta = torch.load(ckpt / "metadata.pt", map_location=device, weights_only=False)
    logging.info(f"Resumed from step {meta['global_step']}")
    return meta["global_step"]


def train_stage1(args):
    """Stage 1: KV cache alignment training."""
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(42, local_rank)

    config = _config.get_config(args.config)
    model_cfg = config.model
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    # Create checkpoint dir
    checkpoint_dir = config.checkpoint_dir
    if args.exp_name:
        checkpoint_dir = checkpoint_dir.parent / args.exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    model = pi0_mod.PI0Pytorch(model_cfg).to(device)

    # Load base weights BEFORE LoRA injection
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading base weights from {config.pytorch_weight_path}")
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(model, model_path)

    # Inject LoRA
    lora_targets = getattr(model_cfg, "lora_targets", "qkv")
    if lora_targets == "qkv":
        paligemma_targets = ("q_proj", "k_proj", "v_proj")
    else:
        paligemma_targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    task_cfg = LoRAConfig(rank=16, alpha=16.0)
    audio_cfg = LoRAConfig(rank=16, alpha=16.0)

    inject_lora(
        model.paligemma_with_expert.paligemma.language_model,
        action_expert_model=None,  # No action expert LoRA in Stage 1
        task_cfg=task_cfg,
        audio_cfg=audio_cfg,
        paligemma_targets=paligemma_targets,
        expert_targets=(),
    )

    # Restore float32 for LoRA params after bfloat16 cast
    model.paligemma_with_expert.to_bfloat16_for_selected_params(model_cfg.dtype)

    # Freeze everything except Perceiver Resampler + LoRA
    trainable_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in ["perceiver_resampler", "task_A", "task_B", "audio_A", "audio_B"]):
            param.requires_grad_(True)
            trainable_params.append(param)
        else:
            param.requires_grad_(False)

    # Separate param groups: Perceiver at 50× LR, LoRA at 1×
    perceiver_params = [p for n, p in model.named_parameters() if "perceiver_resampler" in n and p.requires_grad]
    lora_params = [p for n, p in model.named_parameters()
                   if any(k in n for k in ["task_A", "task_B", "audio_A", "audio_B"]) and p.requires_grad]

    peak_lr = config.lr_schedule.peak_lr
    param_groups = [
        {"params": lora_params, "lr": peak_lr},
        {"params": perceiver_params, "lr": peak_lr * 50.0},
    ]

    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    if is_main:
        logging.info(f"Trainable: {num_trainable:,} / {num_total:,} params ({100*num_trainable/num_total:.1f}%)")
        logging.info(f"Perceiver params: {sum(p.numel() for p in perceiver_params):,}")
        logging.info(f"LoRA params: {sum(p.numel() for p in lora_params):,}")

    # DDP
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True, gradient_as_bucket_view=True,
        )

    inner_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    # Optimizer
    optim = torch.optim.AdamW(
        param_groups, lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps, weight_decay=config.optimizer.weight_decay,
    )

    # LR schedule
    warmup_steps = config.lr_schedule.warmup_steps
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    def lr_schedule(step):
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    # Resume
    global_step = 0
    if args.resume and checkpoint_dir.exists():
        global_step = load_checkpoint(model, optim, checkpoint_dir, device)

    # Data
    loader, data_config = _data.create_data_loader(config, framework="pytorch", shuffle=True), None
    try:
        loader, data_config = loader, loader.data_config() if hasattr(loader, "data_config") else None
    except Exception:
        pass

    # Wandb
    if is_main and not args.no_wandb:
        wandb.init(name=args.exp_name or config.name, project="kv-distill")
    else:
        wandb.init(mode="disabled")

    alpha = getattr(model_cfg, "kv_distill_alpha", 1.0)
    text_slot_len = model_cfg.max_token_len  # 200 for pi05

    model.train()
    pbar = tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="KV Distill Stage 1",
                     disable=not is_main)

    import jax  # for tree_map on observations

    while global_step < config.num_train_steps:
        for observation, actions in loader:
            if global_step >= config.num_train_steps:
                break

            observation = jax.tree.map(lambda x: x.to(device), observation)

            # Update LR
            current_lr = lr_schedule(global_step)
            for pg in optim.param_groups:
                pg["lr"] = current_lr * (pg["lr"] / peak_lr)  # preserve per-group multiplier ratio

            # Preprocess
            obs = inner_model._preprocess_observation(observation, train=True)
            images = list(obs.images.values())
            img_masks = list(obs.image_masks.values())

            # === Teacher forward (no grad, no LoRA) ===
            with torch.no_grad():
                teacher_embs, teacher_pad, teacher_att = inner_model.embed_prefix(
                    images, img_masks, obs.tokenized_prompt, obs.tokenized_prompt_mask,
                )
                teacher_pos_ids = torch.cumsum(teacher_pad, dim=1) - 1
                teacher_kv = inner_model.forward_prefix_get_kv(
                    teacher_embs, teacher_pad, teacher_att,
                    position_ids=teacher_pos_ids,
                    gradient_mode=GRAD_MODE_BYPASS,
                )

            # === Student forward (grad through LoRA + Perceiver) ===
            audio_wh = getattr(obs, "audio_whisper_hidden", None)
            if audio_wh is None and getattr(obs, "audio", None) is not None:
                with torch.no_grad():
                    audio_wh = inner_model.whisper_encoder(obs.audio).last_hidden_state

            student_embs, student_pad, student_att, audio_pos_mask = inner_model.embed_prefix_audio(
                images, img_masks, audio_wh,
                getattr(obs, "audio_mask", None), text_slot_len,
            )

            student_kv = inner_model.forward_prefix_get_kv(
                student_embs, student_pad, student_att,
                position_ids=teacher_pos_ids,  # Use teacher's positions for RoPE match
                audio_position_mask=audio_pos_mask,
                gradient_mode=None,  # LoRA active
            )

            # === Loss ===
            num_image_tokens = sum(m.shape[1] for m in [img_masks[0][:, None].expand(img_masks[0].shape[0], 256)] * 3)
            # Simpler: 3 images × 256 tokens = 768
            loss, metrics = compute_kv_distill_loss(
                teacher_kv, student_kv,
                teacher_valid_mask=teacher_pad,
                student_valid_mask=student_pad,
                num_image_tokens=768,
                alpha=alpha,
                num_layers=len(inner_model.paligemma_with_expert.paligemma.language_model.layers),
            )

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            if torch.isfinite(grad_norm):
                optim.step()
            else:
                logging.warning(f"Step {global_step}: NaN/Inf grad_norm, skipping")
            optim.zero_grad(set_to_none=True)

            # Logging
            if is_main and global_step % config.log_interval == 0:
                log_data = {"step": global_step, "lr": current_lr, "grad_norm": float(grad_norm)}
                log_data.update(metrics)
                wandb.log(log_data, step=global_step)
                logging.info(
                    f"step={global_step} total={metrics['kv/total_loss']:.4f} "
                    f"img={metrics['kv/image_loss']:.4f} sem={metrics['kv/semantic_loss']:.4f} "
                    f"lr={current_lr:.2e} gn={float(grad_norm):.2f}"
                )

            # Save
            if is_main and ((global_step > 0 and global_step % config.save_interval == 0)
                            or global_step == config.num_train_steps - 1):
                save_checkpoint(
                    model, optim, global_step, checkpoint_dir, data_config,
                )

            global_step += 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"loss": f"{metrics['kv/total_loss']:.4f}", "step": global_step})

    if pbar:
        pbar.close()
    if is_main:
        wandb.finish()
    cleanup_ddp()


def main():
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config name from _CONFIGS")
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = _config.get_config(args.config)
    stage = getattr(config.model, "training_stage", "kv_distill")

    if stage in ("kv_distill", "kv_distill_stage1"):
        train_stage1(args)
    else:
        raise ValueError(f"Unknown training_stage: {stage}. Use 'kv_distill' for Stage 1.")


if __name__ == "__main__":
    main()
