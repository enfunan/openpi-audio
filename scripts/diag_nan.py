"""Diagnose NaN in Stage 3 training.

Checks:
1. Stage 2 checkpoint for NaN/Inf in raw weights
2. Merged model params (after weight loading + random init) for NaN/Inf
3. Single forward pass (no gradients) for NaN in output
"""

import sys
import os

# Limit GPU memory usage
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import orbax.checkpoint as ocp
from flax import traverse_util
from flax.nnx.traversals import unflatten_mapping

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.shared.download as download


def log(msg):
    sys.stdout.write(str(msg) + "\n")
    sys.stdout.flush()


# ============================================================
# CHECK 1: Raw Stage 2 checkpoint for NaN/Inf
# ============================================================
CKPT = "checkpoints/pi05_audio_mixed_asr/mixed_asr/14999/params"
log(f"=== CHECK 1: Raw checkpoint NaN/Inf scan ===")
log(f"Loading {CKPT} as numpy...")
raw_params = _model.restore_params(CKPT, restore_type=np.ndarray)
flat_raw = traverse_util.flatten_dict(raw_params, sep="/")

nan_count = 0
inf_count = 0
total_params = 0
for key, val in sorted(flat_raw.items()):
    total_params += 1
    has_nan = np.any(np.isnan(val))
    has_inf = np.any(np.isinf(val))
    if has_nan or has_inf:
        nan_count += has_nan
        inf_count += has_inf
        log(f"  BAD: {key}, shape={val.shape}, dtype={val.dtype}, "
            f"nan={np.sum(np.isnan(val))}, inf={np.sum(np.isinf(val))}")

if nan_count == 0 and inf_count == 0:
    log(f"  PASS: All {total_params} parameter arrays clean (no NaN/Inf)")
else:
    log(f"  FAIL: {nan_count} NaN arrays, {inf_count} Inf arrays out of {total_params}")

# Print a few stats for key components
log("\n  Key parameter norms:")
for key in sorted(flat_raw.keys()):
    if any(k in key for k in ["audio_projector", "lora_a", "lora_b", "output_scale"]):
        val = flat_raw[key]
        log(f"    {key}: shape={val.shape}, norm={np.linalg.norm(val.astype(np.float32)):.4f}, "
            f"min={np.min(val):.6f}, max={np.max(val):.6f}")


# ============================================================
# CHECK 2: Create Stage 3 model, merge weights, check for NaN/Inf
# ============================================================
log(f"\n=== CHECK 2: Merged model params NaN/Inf scan ===")

# Stage 3 config (same as in config.py)
config = pi0_config.Pi0Config(
    pi05=True,
    audio_enabled=True,
    training_stage="robot_task",
    action_horizon=10,
    discrete_state_input=False,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
)

log("Creating model shape...")
model_shape = nnx.eval_shape(config.create, jax.random.key(0))
graphdef, ref_state = nnx.split(model_shape)

# Get reference params as pure dict (for _merge_params)
ref_params = ref_state.to_pure_dict()
flat_ref = traverse_util.flatten_dict(ref_params, sep="/")
log(f"  Reference model has {len(flat_ref)} parameter arrays")

# Count params by component
components = {"audio_projector": 0, "whisper_encoder": 0, "lora": 0,
              "PaliGemma/img": 0, "PaliGemma/llm": 0, "other": 0}
for key in flat_ref:
    matched = False
    for comp in ["audio_projector", "whisper_encoder", "lora", "PaliGemma/img", "PaliGemma/llm"]:
        if comp in key:
            components[comp] += 1
            matched = True
            break
    if not matched:
        components["other"] += 1
log(f"  Component counts: {components}")

# Now do the merge (same as CheckpointWeightLoader.load)
log("Merging checkpoint with reference params...")
from openpi.training.weight_loaders import _merge_params
merged_params = _merge_params(
    raw_params, ref_params,
    missing_regex=".*lora.*|.*whisper_encoder.*|.*audio_projector.*|.*alignment_pooler.*"
)
flat_merged = traverse_util.flatten_dict(merged_params, sep="/")
log(f"  Merged params: {len(flat_merged)} arrays")

# Check which keys came from checkpoint vs reference
from_ckpt = set(traverse_util.flatten_dict(raw_params, sep="/").keys()) & set(flat_merged.keys())
from_ref = set(flat_merged.keys()) - from_ckpt
log(f"  From checkpoint: {len(from_ckpt)}, from reference (random init): {len(from_ref)}")
if from_ref:
    log(f"  Reference-sourced keys (sample):")
    for key in sorted(from_ref)[:20]:
        log(f"    {key}")

# Check merged params for NaN/Inf
nan_count = 0
inf_count = 0
for key, val in sorted(flat_merged.items()):
    # val might be jax.ShapeDtypeStruct (from eval_shape) — check
    if isinstance(val, (np.ndarray, jnp.ndarray)):
        has_nan = np.any(np.isnan(val)) if isinstance(val, np.ndarray) else bool(jnp.any(jnp.isnan(val)))
        has_inf = np.any(np.isinf(val)) if isinstance(val, np.ndarray) else bool(jnp.any(jnp.isinf(val)))
        if has_nan or has_inf:
            nan_count += 1
            log(f"  BAD: {key}, shape={val.shape}, dtype={val.dtype}")
    elif hasattr(val, 'shape'):
        # It's a ShapeDtypeStruct — no actual values to check
        log(f"  WARN: {key} is ShapeDtypeStruct (no values), shape={val.shape}")

if nan_count == 0:
    log(f"  PASS: All merged params clean")
else:
    log(f"  FAIL: {nan_count} bad arrays in merged params")


# ============================================================
# CHECK 3: Forward pass (no gradients) with merged model
# ============================================================
log(f"\n=== CHECK 3: Single forward pass (no gradients) ===")

log("Materializing model...")
# Create model with real random init (gives actual JAX arrays for all params)
model = config.create(jax.random.key(42))

# Merge RAW checkpoint params (not merged_params which has ShapeDtypeStruct)
# This mimics what train.py does: real model + checkpoint overlay
graphdef2, state2 = nnx.split(model)
flat_state = state2.flat_state()
for kp, v in traverse_util.flatten_dict(raw_params).items():
    if kp in flat_state:
        val = jnp.asarray(v) if isinstance(v, np.ndarray) else v
        flat_state[kp] = flat_state[kp].replace(val) if hasattr(flat_state[kp], 'replace') else val
    else:
        alt_kp = tuple(int(k) if isinstance(k, str) and k.isdigit() else k for k in kp)
        if alt_kp in flat_state:
            val = jnp.asarray(v) if isinstance(v, np.ndarray) else v
            flat_state[alt_kp] = flat_state[alt_kp].replace(val) if hasattr(flat_state[alt_kp], 'replace') else val
state2.update(unflatten_mapping(flat_state))
model = nnx.merge(graphdef2, state2)
model.eval()
log("  Model loaded and set to eval mode")
log(f"  AE LoRA params are from random init (model.create), not checkpoint")

# Check param norms after loading
params_state = nnx.state(model)
flat_params_final = params_state.flat_state()
nan_final = 0
for kp, v in flat_params_final.items():
    val = v.value if hasattr(v, 'value') else v
    if jnp.any(jnp.isnan(val)):
        nan_final += 1
        path_str = "/".join(str(p) for p in kp)
        log(f"  NaN in loaded model param: {path_str}, shape={val.shape}")
if nan_final == 0:
    log(f"  PASS: All {len(flat_params_final)} loaded model params clean")

# Create dummy observation (text + image, no audio)
log("\nCreating dummy observation...")
from openpi.models.model import Observation
batch_size = 2

# Minimal dummy data matching LIBERO format (3 cameras)
cam_names = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
dummy_obs = Observation(
    images={name: jnp.ones((batch_size, 224, 224, 3), dtype=jnp.float32) * 0.5 for name in cam_names},
    image_masks={name: jnp.ones((batch_size,), dtype=jnp.bool_) for name in cam_names},
    state=jnp.zeros((batch_size, 32), dtype=jnp.float32),
    tokenized_prompt=jnp.ones((batch_size, 200), dtype=jnp.int32),  # all token ID 1
    tokenized_prompt_mask=jnp.ones((batch_size, 200), dtype=jnp.bool_),
    audio=jnp.zeros((batch_size, 80, 3000), dtype=jnp.float32),
    audio_mask=jnp.zeros((batch_size,), dtype=jnp.bool_),  # no audio
)
dummy_actions = jnp.zeros((batch_size, 10, 32), dtype=jnp.float32)

log("Running forward pass (compute_loss, train=False)...")
try:
    rng = jax.random.key(0)
    loss = model.compute_loss(rng, dummy_obs, dummy_actions, train=False)
    loss_val = jax.device_get(loss)
    log(f"  Loss shape: {loss_val.shape}")
    log(f"  Loss values: {loss_val}")
    log(f"  Loss mean: {np.mean(loss_val):.6f}")
    log(f"  Any NaN: {np.any(np.isnan(loss_val))}")
    log(f"  Any Inf: {np.any(np.isinf(loss_val))}")
    if np.any(np.isnan(loss_val)):
        log("  FAIL: Forward pass produces NaN!")
    elif np.any(np.isinf(loss_val)):
        log("  FAIL: Forward pass produces Inf!")
    else:
        log("  PASS: Forward pass clean")
except Exception as e:
    log(f"  ERROR: Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

# Also try with audio_mask=True (zero audio but mask says present)
log("\nRunning forward pass with audio_mask=True (zero audio, mask on)...")
try:
    dummy_obs_audio = Observation(
        images=dummy_obs.images,
        image_masks=dummy_obs.image_masks,
        state=dummy_obs.state,
        tokenized_prompt=dummy_obs.tokenized_prompt,
        tokenized_prompt_mask=dummy_obs.tokenized_prompt_mask,
        audio=dummy_obs.audio,
        audio_mask=jnp.ones((batch_size,), dtype=jnp.bool_),  # audio mask ON
    )
    loss2 = model.compute_loss(rng, dummy_obs_audio, dummy_actions, train=False)
    loss2_val = jax.device_get(loss2)
    log(f"  Loss mean: {np.mean(loss2_val):.6f}")
    log(f"  Any NaN: {np.any(np.isnan(loss2_val))}")
    if np.any(np.isnan(loss2_val)):
        log("  FAIL: Forward pass with audio_mask=True produces NaN!")
    else:
        log("  PASS: Forward pass with audio_mask=True clean")
except Exception as e:
    log(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

# Try a gradient computation too (mimics one training step)
log("\nRunning gradient computation (value_and_grad)...")
try:
    def loss_fn(model):
        return jnp.mean(model.compute_loss(rng, dummy_obs, dummy_actions, train=True))

    loss_val, grads = nnx.value_and_grad(loss_fn)(model)
    loss_scalar = jax.device_get(loss_val)
    grad_norm = jax.device_get(jnp.sqrt(sum(jnp.sum(jnp.square(v.value)) for v in jax.tree.leaves(grads))))
    log(f"  Loss: {loss_scalar:.6f}")
    log(f"  Grad norm: {grad_norm:.6f}")
    log(f"  Loss NaN: {np.isnan(loss_scalar)}")
    log(f"  Grad NaN: {np.isnan(grad_norm)}")
    if np.isnan(loss_scalar) or np.isnan(grad_norm):
        log("  FAIL: Gradient computation produces NaN!")
    else:
        log("  PASS: Gradient computation clean")
except Exception as e:
    log(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

log("\n=== DIAGNOSIS COMPLETE ===")
