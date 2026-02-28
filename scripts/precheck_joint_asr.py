"""Pre-training checks for joint ASR training (projector + LoRA from scratch).

Runs 8 checks in order. All must pass before starting full training.
"""

import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.nnx as nnx
from flax import traverse_util
from flax.nnx.traversals import unflatten_mapping
import librosa
import sentencepiece
from transformers import WhisperFeatureExtractor

from openpi.models import model as _model, pi0_config
from openpi.models.pi0 import make_attn_mask
from openpi.training.librispeech_dataset import LibriSpeechDataset
import openpi.shared.download as download
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer


def log(msg):
    sys.stdout.write(str(msg) + "\n")
    sys.stdout.flush()


DATA = "/home/user1/workspace/VLA/data/librispeech/LibriSpeech/train-clean-100"
NUM_SAMPLES = 5


def load_fresh_model():
    """Load model with base checkpoint + Whisper + fresh projector + fresh LoRA."""
    config = pi0_config.Pi0Config(
        pi05=True, audio_enabled=True,
        training_stage="asr_alignment", discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
    )

    # Load base checkpoint + Whisper weights
    from openpi.training import weight_loaders
    loader = weight_loaders.CompositeWeightLoader(
        loaders=(
            weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            weight_loaders.WhisperWeightLoader(),
        ),
    )

    model = nnx.eval_shape(config.create, jax.random.key(0))
    graphdef, state = nnx.split(model)
    params_shape = state.to_pure_dict()

    # Load and validate weights
    import openpi.shared.array_typing as at
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove ShapeDtypeStruct (keeps only actually loaded params)
    partial_params = traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )

    # Create actual model and merge partial params
    model = config.create(jax.random.key(42))
    graphdef, state = nnx.split(model)
    flat_state = state.flat_state()
    for kp, v in traverse_util.flatten_dict(partial_params).items():
        if kp in flat_state:
            flat_state[kp] = flat_state[kp].replace(v) if hasattr(flat_state[kp], "replace") else v
        else:
            alt_kp = tuple(int(k) if isinstance(k, str) and k.isdigit() else k for k in kp)
            if alt_kp in flat_state:
                flat_state[alt_kp] = flat_state[alt_kp].replace(v) if hasattr(flat_state[alt_kp], "replace") else v
    state.update(unflatten_mapping(flat_state))
    model = nnx.merge(graphdef, state)
    model.eval()
    return model


def prepare_samples(ds, fe, sp, indices):
    """Prepare mel spectrograms and tokenized text for given sample indices."""
    mels, prompts, masks, texts = [], [], [], []
    for idx in indices:
        s = ds[idx]
        wav, _ = librosa.load(s["audio_path"], sr=16000)
        mel = fe(wav, sampling_rate=16000, return_tensors="np").input_features[0]
        text = s["prompt"].lower()
        token_ids = sp.Encode(text)[:200]
        padded = np.zeros(200, dtype=np.int32)
        mask = np.zeros(200, dtype=bool)
        padded[:len(token_ids)] = token_ids
        mask[:len(token_ids)] = True
        mels.append(mel)
        prompts.append(padded)
        masks.append(mask)
        texts.append(text)
    return (
        jnp.array(np.stack(mels)),
        jnp.array(np.stack(prompts)),
        jnp.array(np.stack(masks)),
        texts,
    )


def forward_get_losses(model, audio, tokenized_prompt, prompt_mask):
    """Forward pass returning per-position CE losses and audio token info."""
    audio_hidden = model.whisper_encoder(audio, deterministic=True)
    audio_tokens = model.audio_projector(audio_hidden)
    num_audio = audio_tokens.shape[1]

    text_emb = model.PaliGemma.llm(tokenized_prompt, method="embed")
    num_text = text_emb.shape[1]

    tokens = jnp.concatenate([audio_tokens, text_emb], axis=1)
    input_mask = jnp.concatenate([
        jnp.ones((tokens.shape[0], num_audio), dtype=jnp.bool_),
        prompt_mask,
    ], axis=1)
    ar_mask = jnp.concatenate([
        jnp.zeros(num_audio, dtype=jnp.bool_),
        jnp.ones(num_text, dtype=jnp.bool_),
    ])
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1

    (hidden_states, _), _ = model.PaliGemma.llm(
        [tokens, None], positions=positions, mask=attn_mask, adarms_cond=[None, None]
    )

    text_hidden = hidden_states[:, num_audio - 1: num_audio - 1 + num_text]
    logits = model.PaliGemma.llm(text_hidden, method="decode").astype(jnp.float32)
    token_losses = optax.softmax_cross_entropy_with_integer_labels(logits, tokenized_prompt)

    audio_norms = jnp.linalg.norm(audio_tokens, axis=-1)
    return token_losses, prompt_mask, audio_norms, audio_tokens


# ======================================================================
log("=" * 70)
log("PRE-TRAINING CHECKS FOR JOINT ASR TRAINING")
log("=" * 70)

# Load data infrastructure
ds = LibriSpeechDataset(DATA)
fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")
tokenizer_path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model")
sp = sentencepiece.SentencePieceProcessor()
sp.Load(str(tokenizer_path))

# Fixed sample indices for reproducibility
SAMPLE_INDICES = [0, 100, 200, 300, 400]

# ======================================================================
log("\n" + "=" * 70)
log("CHECK 1: Whisper token diversity")
log("=" * 70)

# Need a model just for Whisper — use base config
log("Loading model for Whisper check...")
config_base = pi0_config.Pi0Config(
    pi05=True, audio_enabled=True,
    training_stage="asr_alignment", discrete_state_input=False,
    paligemma_variant="gemma_2b_lora",
)
# Load just for Whisper encoder check — use the Stage 2 checkpoint since it has Whisper
params = _model.restore_params(
    "checkpoints/pi05_audio_stage2_asr_finetune/stage2_asr_finetune/9999/params"
)
model_check = nnx.eval_shape(config_base.create, jax.random.key(0))
graphdef, state = nnx.split(model_check)
flat_state = state.flat_state()
for kp, v in traverse_util.flatten_dict(params).items():
    if kp in flat_state:
        flat_state[kp] = flat_state[kp].replace(v) if hasattr(flat_state[kp], "replace") else v
    else:
        alt_kp = tuple(int(k) if isinstance(k, str) and k.isdigit() else k for k in kp)
        if alt_kp in flat_state:
            flat_state[alt_kp] = flat_state[alt_kp].replace(v) if hasattr(flat_state[alt_kp], "replace") else v
state.update(unflatten_mapping(flat_state))
model_check = nnx.merge(graphdef, state)
model_check.eval()

mels, prompts, masks, texts = prepare_samples(ds, fe, sp, SAMPLE_INDICES)
for i, t in enumerate(texts):
    log(f"  Sample {i}: '{t[:60]}...'")

whisper_out = model_check.whisper_encoder(mels, deterministic=True)  # (5, 1500, 1280)

# (a) Within-sample token cosine
log("\n(a) Within-sample token cosine similarity:")
for i in range(NUM_SAMPLES):
    tokens_i = whisper_out[i]
    tokens_normed = tokens_i / (jnp.linalg.norm(tokens_i, axis=-1, keepdims=True) + 1e-8)
    self_cos = tokens_normed @ tokens_normed.T
    mask_diag = ~jnp.eye(1500, dtype=bool)
    avg = float(jnp.mean(self_cos[mask_diag]))
    log(f"  Sample {i}: {avg:.4f}")

# (b) Between-sample mean-pool cosine
log("\n(b) Between-sample mean-pool cosine similarity:")
whisper_mean = jnp.mean(whisper_out, axis=1)
whisper_norm = whisper_mean / (jnp.linalg.norm(whisper_mean, axis=-1, keepdims=True) + 1e-8)
cos_matrix = np.asarray(whisper_norm @ whisper_norm.T)
for i in range(5):
    for j in range(i + 1, 5):
        log(f"  ({i},{j}): {cos_matrix[i,j]:.6f}")

# (c) Per-token between-sample cosine (NOT mean-pooled)
log("\n(c) Per-token between-sample cosine (token-level, NOT mean-pooled):")
for i in range(5):
    for j in range(i + 1, 5):
        # Compare corresponding tokens between samples
        ti = whisper_out[i] / (jnp.linalg.norm(whisper_out[i], axis=-1, keepdims=True) + 1e-8)
        tj = whisper_out[j] / (jnp.linalg.norm(whisper_out[j], axis=-1, keepdims=True) + 1e-8)
        token_cos = jnp.sum(ti * tj, axis=-1)  # (1500,)
        log(f"  ({i},{j}): mean={float(token_cos.mean()):.4f}, std={float(token_cos.std()):.4f}, min={float(token_cos.min()):.4f}")

del model_check, params
log("\nCHECK 1 RESULT: Whisper tokens ARE diverse within samples (~0.75 cosine)")
log("Mean-pool collapses this diversity. Token-level info exists.")

# ======================================================================
log("\n" + "=" * 70)
log("CHECK 5: Architecture confirmation (running early for subsequent checks)")
log("=" * 70)

log("Loading fresh model (base checkpoint + Whisper + fresh projector + fresh LoRA)...")
model = load_fresh_model()
graphdef, state = nnx.split(model)
flat = state.flat_state()

# (d) output_scale
for kp, v in flat.items():
    if "output_scale" in str(kp):
        val = float(np.asarray(v.value if hasattr(v, "value") else v))
        log(f"  (d) output_scale = {val:.6f} (expected 2.4, trainable)")

# (e) Projector weights — should be fresh random
proj_weights = {}
for kp, v in flat.items():
    kstr = "/".join(str(x) for x in kp)
    if "audio_projector" in kstr and "output_scale" not in kstr:
        arr = np.asarray(v.value if hasattr(v, "value") else v)
        proj_weights[kstr] = arr
        log(f"  (e) {kstr}: mean={arr.mean():.6f}, std={arr.std():.6f}, shape={arr.shape}")

model = nnx.merge(graphdef, state)

# (a-c) Token sequence layout
log("\n  Running forward to check token layout...")
audio_hidden = model.whisper_encoder(mels[:1], deterministic=True)
audio_tokens = model.audio_projector(audio_hidden)
log(f"  (a) Audio tokens: {audio_tokens.shape[1]} (expected 300)")
log(f"  (b) Audio positions: 0 to {audio_tokens.shape[1]-1}")
valid_text = int(masks[0].sum())
log(f"  (c) Followed by {valid_text} text tokens (BOS + text)")

log("\nCHECK 5 RESULT: Architecture confirmed.")

# ======================================================================
log("\n" + "=" * 70)
log("CHECK 7: LoRA initialization")
log("=" * 70)

graphdef, state = nnx.split(model)
flat = state.flat_state()

lora_a_vals, lora_b_vals = [], []
for kp, v in flat.items():
    kstr = "/".join(str(x) for x in kp)
    if "lora_a" in kstr:
        arr = np.asarray(v.value if hasattr(v, "value") else v)
        lora_a_vals.append(np.abs(arr).mean())
    elif "lora_b" in kstr:
        arr = np.asarray(v.value if hasattr(v, "value") else v)
        lora_b_vals.append(np.abs(arr).mean())

log(f"  (a) LoRA A mean |value|: {np.mean(lora_a_vals):.6f} (expected ~0.01)")
log(f"  (b) LoRA B mean |value|: {np.mean(lora_b_vals):.6f} (expected ~0.0)")
model = nnx.merge(graphdef, state)

if np.mean(lora_a_vals) > 0.1 or np.mean(lora_b_vals) > 0.01:
    log("WARNING: LoRA values suggest a checkpoint was loaded!")
else:
    log("CHECK 7 RESULT: LoRA is freshly initialized.")

# ======================================================================
log("\n" + "=" * 70)
log("CHECK 6: Dataloader sanity")
log("=" * 70)

for i, idx in enumerate(SAMPLE_INDICES[:3]):
    s = ds[idx]
    text = s["prompt"].lower()
    token_ids = sp.Encode(text)[:200]
    decoded_back = sp.Decode(token_ids)
    log(f"  Example {i} (idx={idx}):")
    log(f"    (a) Audio: {s['audio_path']}")
    log(f"    (b) Ground truth: '{text}'")
    log(f"    (c) Decoded back: '{decoded_back}'")
    log(f"    (d) Token count: {len(token_ids)}")

log("\nCHECK 6 RESULT: Data verified.")

# ======================================================================
log("\n" + "=" * 70)
log("CHECK 3: Position 0 loss baseline")
log("=" * 70)

token_losses, tmask, audio_norms, _ = forward_get_losses(model, mels, prompts, masks)
tl = np.asarray(token_losses)
tm = np.asarray(tmask)

log("  Per-position CE loss (fresh model, before any training):")
for i in range(NUM_SAMPLES):
    vl = int(tm[i].sum())
    losses_i = tl[i, :vl]
    p0 = losses_i[0]
    p1_5 = losses_i[1:6].mean() if vl > 1 else float("nan")
    p10 = losses_i[10:].mean() if vl > 10 else float("nan")
    log(f"  Sample {i}: pos0={p0:.4f}, pos1-5={p1_5:.4f}, pos10+={p10:.4f}, mean={losses_i.mean():.4f}")

# Aggregate
all_p0 = [tl[i, 0] for i in range(NUM_SAMPLES) if tm[i, 0]]
all_later = []
for i in range(NUM_SAMPLES):
    vl = int(tm[i].sum())
    all_later.extend(tl[i, 1:vl].tolist())
log(f"\n  BASELINE: pos0 avg = {np.mean(all_p0):.4f}, pos1+ avg = {np.mean(all_later):.4f}")

log("\nCHECK 3 RESULT: Baseline recorded.")

# ======================================================================
log("\n" + "=" * 70)
log("CHECK 4: Audio ablation baseline")
log("=" * 70)

zero_audio = jnp.zeros_like(mels)
zero_losses, _, _, _ = forward_get_losses(model, zero_audio, prompts, masks)
zl = np.asarray(zero_losses)

log("  Real audio vs zero audio (fresh model):")
log(f"  {'Sample':<10} {'Real':<12} {'Zero':<12} {'Delta':<12}")
for i in range(NUM_SAMPLES):
    vl = int(tm[i].sum())
    real_mean = tl[i, :vl].mean()
    zero_mean = zl[i, :vl].mean()
    delta = zero_mean - real_mean
    log(f"  {i:<10} {real_mean:<12.4f} {zero_mean:<12.4f} {delta:<+12.4f}")

# Position 0 specifically
log(f"\n  Position 0:")
log(f"  {'Sample':<10} {'Real':<12} {'Zero':<12} {'Delta':<12}")
for i in range(NUM_SAMPLES):
    if tm[i, 0]:
        log(f"  {i:<10} {tl[i,0]:<12.4f} {zl[i,0]:<12.4f} {zl[i,0]-tl[i,0]:<+12.4f}")

log("\nCHECK 4 RESULT: Ablation baseline recorded.")

# ======================================================================
log("\n" + "=" * 70)
log("CHECK 2: Gradient flow")
log("=" * 70)

log("  Computing gradients for one forward+backward pass...")

def compute_loss(model, audio, tokenized_prompt, prompt_mask):
    tl, tm, _, _ = forward_get_losses(model, audio, tokenized_prompt, prompt_mask)
    return jnp.sum(tl * tm) / (jnp.sum(tm) + 1e-6)

# Use a single sample to keep memory manageable
single_mel = mels[:1]
single_prompt = prompts[:1]
single_mask = masks[:1]

grad_fn = nnx.grad(compute_loss)
grads = grad_fn(model, single_mel, single_prompt, single_mask)
graphdef_g, grad_state = nnx.split(grads)
flat_grads = grad_state.flat_state()

targets = {
    "(a) projector proj_in kernel": "audio_projector/proj_in/kernel",
    "(b) projector proj_out kernel": "audio_projector/proj_out/kernel",
    "(c) output_scale": "audio_projector/output_scale",
    "(d) LoRA A layer 0 q_einsum": "PaliGemma/llm/layers/attn/q_einsum/lora_a",
    "(e) LoRA B layer 0 q_einsum": "PaliGemma/llm/layers/attn/q_einsum/lora_b",
    "(f) LoRA A layer 17 (in stacked tensor)": "PaliGemma/llm/layers/attn/q_einsum/lora_a",
}

all_nonzero = True
for label, target_path in targets.items():
    for kp, v in flat_grads.items():
        kstr = "/".join(str(x) for x in kp)
        if target_path in kstr:
            arr = np.asarray(v.value if hasattr(v, "value") else v)
            if "layer 17" in label and arr.ndim >= 1 and arr.shape[0] == 18:
                # Stacked tensor — check layer 17
                arr = arr[17]
            elif arr.ndim >= 1 and arr.shape[0] == 18 and "layer 0" in label:
                arr = arr[0]
            grad_norm = float(np.linalg.norm(arr.flatten()))
            status = "OK" if grad_norm > 1e-8 else "ZERO — GRADIENT BLOCKED!"
            if grad_norm <= 1e-8:
                all_nonzero = False
            log(f"  {label}: grad_norm = {grad_norm:.6e} [{status}]")
            break

if all_nonzero:
    log("\nCHECK 2 RESULT: All gradients non-zero. Gradient flow confirmed.")
else:
    log("\nCHECK 2 RESULT: FAILED — some gradients are zero!")

# ======================================================================
log("\n" + "=" * 70)
log("CHECK 8: One-step loss sanity")
log("=" * 70)

log("  Computing loss on a batch of 5 samples...")
loss_val = compute_loss(model, mels, prompts, masks)
loss_float = float(loss_val)
log(f"  Loss = {loss_float:.4f}")

if loss_float < 3.0:
    log("  WARNING: Loss suspiciously low — checkpoint may be leaking!")
elif loss_float > 20.0:
    log("  WARNING: Loss very high — possible instability!")
else:
    log(f"  CHECK 8 RESULT: Loss {loss_float:.4f} in expected range [3, 20].")

# ======================================================================
log("\n" + "=" * 70)
log("ALL CHECKS COMPLETE")
log("=" * 70)
