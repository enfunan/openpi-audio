"""Diagnose SigLIP per-token norms and check for NaN-producing batches."""

import sys
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
from flax import traverse_util
from flax.nnx.traversals import unflatten_mapping

from openpi.models import model as _model
from openpi.models import pi0_config


def log(msg):
    sys.stdout.write(str(msg) + "\n")
    sys.stdout.flush()


# Load Stage 3 model (same as training)
CKPT = "checkpoints/pi05_audio_mixed_asr/mixed_asr/14999/params"
config = pi0_config.Pi0Config(
    pi05=True, audio_enabled=True, training_stage="robot_task",
    action_horizon=10, discrete_state_input=False,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
)

log("Loading model...")
model = config.create(jax.random.key(42))
raw_params = _model.restore_params(CKPT, restore_type=np.ndarray)
graphdef, state = nnx.split(model)
flat_state = state.flat_state()
for kp, v in traverse_util.flatten_dict(raw_params).items():
    if kp in flat_state:
        val = jnp.asarray(v)
        flat_state[kp] = flat_state[kp].replace(val) if hasattr(flat_state[kp], 'replace') else val
    else:
        alt_kp = tuple(int(k) if isinstance(k, str) and k.isdigit() else k for k in kp)
        if alt_kp in flat_state:
            val = jnp.asarray(v)
            flat_state[alt_kp] = flat_state[alt_kp].replace(val) if hasattr(flat_state[alt_kp], 'replace') else val
state.update(unflatten_mapping(flat_state))
model = nnx.merge(graphdef, state)
model.eval()

# Also cast frozen params to bfloat16 (same as train.py:115)
from openpi.shared import nnx_utils
from openpi.training import config as _config
stage3_config = _config.get_config("pi05_audio_stage3_libero")
params_state = nnx.state(model)
params_state = nnx_utils.state_map(
    params_state, stage3_config.freeze_filter,
    lambda p: p.replace(p.value.astype(jnp.bfloat16))
)
graphdef2, _ = nnx.split(model)
model = nnx.merge(graphdef2, params_state)
model.eval()
log("Model loaded with frozen params in bfloat16")

# ============================================================
# CHECK 1: SigLIP per-token norms on dummy images
# ============================================================
log("\n=== CHECK 1: SigLIP per-token norms (dummy images) ===")

for val in [0.0, 0.5, 1.0]:
    dummy_img = jnp.full((1, 224, 224, 3), val, dtype=jnp.float32)
    img_tokens, _ = model.PaliGemma.img(dummy_img, train=False)
    per_token_norm = jnp.linalg.norm(img_tokens.astype(jnp.float32), axis=-1)  # (1, N)
    log(f"  img={val:.1f}: shape={img_tokens.shape}, dtype={img_tokens.dtype}, "
        f"per_token mean={jnp.mean(per_token_norm):.1f}, "
        f"max={jnp.max(per_token_norm):.1f}, "
        f"min={jnp.min(per_token_norm):.1f}, "
        f"any_nan={bool(jnp.any(jnp.isnan(img_tokens)))}")


# ============================================================
# CHECK 2: Text embedding per-token norms
# ============================================================
log("\n=== CHECK 2: Text embedding per-token norms ===")
dummy_tokens = jnp.ones((1, 50), dtype=jnp.int32) * 100  # arbitrary token ID
text_emb = model.PaliGemma.llm(dummy_tokens, method="embed")
text_per_token_norm = jnp.linalg.norm(text_emb.astype(jnp.float32), axis=-1)
log(f"  text: shape={text_emb.shape}, dtype={text_emb.dtype}, "
    f"per_token mean={jnp.mean(text_per_token_norm):.1f}, "
    f"max={jnp.max(text_per_token_norm):.1f}")


# ============================================================
# CHECK 3: Audio token norms (zero audio)
# ============================================================
log("\n=== CHECK 3: Audio token norms (zero audio) ===")
dummy_audio = jnp.zeros((1, 80, 3000), dtype=jnp.float32)
audio_hidden = jax.lax.stop_gradient(
    model.whisper_encoder(dummy_audio, deterministic=True)
)
audio_tokens = model.audio_projector(audio_hidden)
audio_per_token_norm = jnp.linalg.norm(audio_tokens.astype(jnp.float32), axis=-1)
log(f"  audio: shape={audio_tokens.shape}, dtype={audio_tokens.dtype}, "
    f"per_token mean={jnp.mean(audio_per_token_norm):.1f}, "
    f"max={jnp.max(audio_per_token_norm):.1f}")


# ============================================================
# CHECK 4: Load real LIBERO data and check multiple batches
# ============================================================
log("\n=== CHECK 4: SigLIP on real LIBERO data (first 10 batches) ===")

import openpi.training.data_loader as _data_loader
import pathlib

data_config = stage3_config.data.create(
    pathlib.Path("assets/pi05_audio_stage3_libero"),
    stage3_config.model,
)
data_loader = _data_loader.create_data_loader(
    stage3_config,
    sharding=None,
    shuffle=True,
)
data_iter = iter(data_loader)

for batch_idx in range(10):
    batch = next(data_iter)
    obs, actions = batch

    # Check raw image data
    for cam_name in obs.images:
        img = obs.images[cam_name]
        img_f32 = img.astype(np.float32) if hasattr(img, 'astype') else np.array(img, dtype=np.float32)

        # Check for NaN in raw image data
        has_nan = np.any(np.isnan(img_f32))
        has_inf = np.any(np.isinf(img_f32))

        if has_nan or has_inf:
            log(f"  BATCH {batch_idx} {cam_name}: RAW IMAGE HAS NaN={has_nan} Inf={has_inf}!")

    # Run SigLIP on each camera
    all_clean = True
    for cam_name in obs.images:
        img_jax = jnp.asarray(obs.images[cam_name])
        img_tokens, _ = model.PaliGemma.img(img_jax, train=False)
        per_tok = jnp.linalg.norm(img_tokens.astype(jnp.float32), axis=-1)
        has_nan = bool(jnp.any(jnp.isnan(img_tokens)))
        mean_norm = float(jnp.mean(per_tok))
        max_norm = float(jnp.max(per_tok))

        status = "NaN!" if has_nan else "OK"
        if has_nan:
            all_clean = False
        log(f"  batch={batch_idx} {cam_name}: per_token mean={mean_norm:.1f} max={max_norm:.1f} [{status}]")

    if not all_clean:
        log(f"  *** BATCH {batch_idx} HAS NaN IN SIGLIP OUTPUT ***")

    # Also check state/actions for NaN
    state_nan = np.any(np.isnan(np.array(obs.state)))
    actions_nan = np.any(np.isnan(np.array(actions)))
    if state_nan or actions_nan:
        log(f"  batch={batch_idx}: state_nan={state_nan}, actions_nan={actions_nan}")

log("\n=== DONE ===")
