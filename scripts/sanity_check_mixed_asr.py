#!/usr/bin/env python3
"""Sanity check for MixedASRDataset: ratio verification + sample inspection.

Checks:
1. Print 10 consecutive samples with source, audio_path, prompt
2. Count LibriSpeech vs DROID over 100 samples
3. Verify fresh projector + LoRA weights (NOT from failed checkpoint)
"""

import sys
import os

# Suppress JAX GPU preallocation for this small script
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.05"

import numpy as np


def check_dataset():
    """Parts 1 & 2: dataset sampling check."""
    from openpi.training.mixed_asr_dataset import MixedASRDataset

    LIBRISPEECH_DIR = "/home/user1/workspace/VLA/data/librispeech/LibriSpeech/train-clean-360"
    DROID_TTS_DIR = "/home/user1/workspace/VLA/data/tts/droid_train"

    ds = MixedASRDataset(
        librispeech_dir=LIBRISPEECH_DIR,
        droid_tts_dir=DROID_TTS_DIR,
        librispeech_ratio=0.25,
    )

    print(f"Dataset size: {len(ds)}")
    print(f"  LibriSpeech: {len(ds._librispeech)}")
    print(f"  DROID TTS:   {len(ds._droid)}")
    print()

    # Part 1: Print 10 consecutive samples
    print("=" * 80)
    print("10 consecutive samples:")
    print("=" * 80)
    for i in range(10):
        sample = ds[i]
        audio_path = sample["audio_path"]
        prompt = sample["prompt"]
        source = "LibriSpeech" if "librispeech" in audio_path.lower() or "LibriSpeech" in audio_path else "DROID TTS"
        print(f"  [{i:2d}] {source:12s} | {audio_path[-60:]:60s} | {prompt[:50]}")
    print()

    # Part 2: Count over 1000 samples
    print("=" * 80)
    print("Ratio check over 1000 samples:")
    print("=" * 80)
    libri_count = 0
    droid_count = 0
    for i in range(1000):
        sample = ds[i]
        audio_path = sample["audio_path"]
        if "librispeech" in audio_path.lower() or "LibriSpeech" in audio_path:
            libri_count += 1
        else:
            droid_count += 1

    total = libri_count + droid_count
    print(f"  LibriSpeech: {libri_count}/{total} ({libri_count/total:.1%})")
    print(f"  DROID TTS:   {droid_count}/{total} ({droid_count/total:.1%})")
    print(f"  Target:      25.0% / 75.0%")
    deviation = abs(libri_count / total - 0.25)
    if deviation < 0.05:
        print(f"  Status: OK (deviation {deviation:.1%} < 5%)")
    else:
        print(f"  Status: WARNING (deviation {deviation:.1%} >= 5%)")
    print()


def check_weights():
    """Part 3: Verify fresh projector + LoRA initialization.

    Instead of instantiating the full model (OOM on single GPU), we check what keys
    the weight loader provides. If audio_projector and LoRA are NOT in the loaded
    checkpoint, they must come from random init (model.create). This is definitive.
    """
    import jax
    from flax import traverse_util
    from openpi.training.config import get_config

    config = get_config("pi05_audio_mixed_asr")

    print("=" * 80)
    print("Weight initialization check:")
    print("=" * 80)
    print(f"  Config: {config.name}")
    print(f"  Weight loader: {config.weight_loader}")
    print()

    # Use eval_shape to get param shapes without materializing
    print("  Getting model param shapes via eval_shape...")
    rng = jax.random.key(42)

    import flax.nnx as nnx
    def get_shapes(rng):
        model = config.model.create(rng)
        _, state = nnx.split(model)
        return state.to_pure_dict()

    params_shape = jax.eval_shape(get_shapes, rng)

    # Show model param structure for audio_projector and LoRA
    flat_shape = traverse_util.flatten_dict(params_shape)
    proj_shape_keys = [k for k in flat_shape if "audio_projector" in str(k)]
    lora_shape_keys = [k for k in flat_shape if "lora" in str(k)]

    print(f"\n  Model has {len(proj_shape_keys)} audio_projector params:")
    for k in sorted(proj_shape_keys, key=str):
        v = flat_shape[k]
        shape = v.shape if hasattr(v, 'shape') else '?'
        print(f"    {'.'.join(str(x) for x in k)}: {shape}")

    print(f"\n  Model has {len(lora_shape_keys)} LoRA params:")
    for k in sorted(lora_shape_keys, key=str)[:6]:
        v = flat_shape[k]
        shape = v.shape if hasattr(v, 'shape') else '?'
        print(f"    {'.'.join(str(x) for x in k)}: {shape}")
    if len(lora_shape_keys) > 6:
        print(f"    ... ({len(lora_shape_keys)} total)")

    # Now load the checkpoint and check what it contains
    print("\n  Loading checkpoint weights (base Pi0.5 + Whisper)...")
    loaded_params = config.weight_loader.load(params_shape)

    # Remove ShapeDtypeStruct entries (unloaded params)
    loaded_flat = {
        k: v for k, v in traverse_util.flatten_dict(loaded_params).items()
        if not isinstance(v, jax.ShapeDtypeStruct)
    }

    proj_loaded = [k for k in loaded_flat if "audio_projector" in str(k)]
    lora_loaded = [k for k in loaded_flat if "lora" in str(k)]

    print(f"\n  Checkpoint contains {len(loaded_flat)} materialized params")
    print(f"  Checkpoint contains {len(proj_loaded)} audio_projector params")
    print(f"  Checkpoint contains {len(lora_loaded)} LoRA params")

    if proj_loaded:
        print("\n  WARNING: audio_projector keys found in checkpoint:")
        for k in sorted(proj_loaded, key=str):
            v = loaded_flat[k]
            arr = np.array(v).astype(np.float32)
            print(f"    {'.'.join(str(x) for x in k)}: shape={v.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}")
    if lora_loaded:
        print("\n  WARNING: LoRA keys found in checkpoint:")
        for k in sorted(lora_loaded, key=str)[:4]:
            print(f"    {'.'.join(str(x) for x in k)}")

    # Summary
    print()
    print("  " + "=" * 60)
    if len(proj_loaded) == 0:
        print("  CONFIRMED: audio_projector NOT in checkpoint")
        print("  → Will be FRESH RANDOM INIT from model.create()")
    else:
        print("  WARNING: audio_projector loaded from checkpoint!")

    if len(lora_loaded) == 0:
        print("  CONFIRMED: LoRA NOT in checkpoint")
        print("  → Will be FRESH INIT (zero B, random A) from model.create()")
    else:
        print("  WARNING: LoRA loaded from checkpoint!")
    print("  " + "=" * 60)


if __name__ == "__main__":
    print("MixedASRDataset Sanity Check")
    print()

    if "--weights-only" not in sys.argv:
        check_dataset()

    if "--dataset-only" not in sys.argv:
        check_weights()
