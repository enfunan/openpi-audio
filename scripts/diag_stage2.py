"""Stage 2 post-training diagnostics.

1. Per-position CE loss (position 0 vs later)
2. Audio token norms and output_scale value
3. Audio ablation: real audio vs zero audio
"""

import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.nnx as nnx
from flax import traverse_util
from flax.nnx.traversals import unflatten_mapping

from openpi.models import model as _model, pi0_config
from openpi.models.pi0 import make_attn_mask
from openpi.training.librispeech_dataset import LibriSpeechDataset
from transformers import WhisperFeatureExtractor
import openpi.shared.download as download


def log(msg):
    sys.stdout.write(str(msg) + "\n")
    sys.stdout.flush()


def load_model(config, checkpoint_path):
    params = _model.restore_params(checkpoint_path)
    model = nnx.eval_shape(config.create, jax.random.key(0))
    graphdef, state = nnx.split(model)
    flat_state = state.flat_state()
    for kp, v in traverse_util.flatten_dict(params).items():
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


CKPT = "checkpoints/pi05_audio_stage2_asr_finetune/stage2_asr_finetune/9999/params"
DATA = "/home/user1/workspace/VLA/data/librispeech/LibriSpeech/train-clean-100"
NUM_SAMPLES = 5
BATCH = NUM_SAMPLES

config = pi0_config.Pi0Config(
    pi05=True, audio_enabled=True,
    training_stage="asr_alignment", discrete_state_input=False,
    paligemma_variant="gemma_2b_lora",
)

log("Loading model...")
model = load_model(config, CKPT)

# --- Diagnostic 2: output_scale and audio token norms ---
log("\n=== DIAGNOSTIC 2: output_scale and audio token norms ===")
graphdef, state = nnx.split(model)
flat = state.flat_state()
for kp, v in flat.items():
    if "output_scale" in str(kp):
        val = v.value if hasattr(v, "value") else v
        log(f"  output_scale key: {kp}")
        log(f"  output_scale value: {float(np.asarray(val)):.6f} (init was 2.4)")
model = nnx.merge(graphdef, state)

# --- Load data ---
log("\nLoading data...")
ds = LibriSpeechDataset(DATA)
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")
tokenizer_path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model")
import sentencepiece
sp = sentencepiece.SentencePieceProcessor()
sp.Load(str(tokenizer_path))

samples = []
for i in range(NUM_SAMPLES):
    s = ds[i * 100]  # spread out samples
    import librosa
    audio_wav, _ = librosa.load(s["audio_path"], sr=16000)
    mel = feature_extractor(audio_wav, sampling_rate=16000, return_tensors="np")
    mel = mel.input_features[0]  # (80, 3000)

    text = s["prompt"].lower()
    token_ids = sp.Encode(text)
    token_ids = token_ids[:200]
    padded = np.zeros(200, dtype=np.int32)
    mask = np.zeros(200, dtype=bool)
    padded[:len(token_ids)] = token_ids
    mask[:len(token_ids)] = True
    samples.append((mel, padded, mask, text))
    log(f"  Sample {i}: '{text[:60]}...' ({len(token_ids)} tokens)")

mels = jnp.array(np.stack([s[0] for s in samples]))         # (B, 80, 3000)
prompts = jnp.array(np.stack([s[1] for s in samples]))       # (B, 200)
prompt_masks = jnp.array(np.stack([s[2] for s in samples]))  # (B, 200)


def forward_and_get_losses(model, audio, tokenized_prompt, prompt_mask):
    """Run ASR forward pass and return per-position CE losses."""
    # Audio path
    audio_hidden = model.whisper_encoder(audio, deterministic=True)
    audio_tokens = model.audio_projector(audio_hidden)  # (B, 300, D)
    num_audio = audio_tokens.shape[1]

    # Text path
    text_emb = model.PaliGemma.llm(tokenized_prompt, method="embed")
    num_text = text_emb.shape[1]

    # Concatenate
    tokens = jnp.concatenate([audio_tokens, text_emb], axis=1)

    # Attention mask
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

    # Forward
    (hidden_states, _), _ = model.PaliGemma.llm(
        [tokens, None], positions=positions, mask=attn_mask, adarms_cond=[None, None]
    )

    # Per-position logits and losses
    text_hidden = hidden_states[:, num_audio - 1: num_audio - 1 + num_text]
    logits = model.PaliGemma.llm(text_hidden, method="decode").astype(jnp.float32)
    token_losses = optax.softmax_cross_entropy_with_integer_labels(logits, tokenized_prompt)

    # Also get audio token norms
    audio_norms = jnp.linalg.norm(audio_tokens, axis=-1)  # (B, 300)

    return token_losses, prompt_mask, audio_norms, logits


# --- Run with real audio ---
log("\nRunning forward pass with REAL audio...")
token_losses, tmask, audio_norms, logits = forward_and_get_losses(model, mels, prompts, prompt_masks)

# Diagnostic 2 continued: audio token norms
log("\n=== DIAGNOSTIC 2 (cont): Audio token norms ===")
for i in range(BATCH):
    norms = np.asarray(audio_norms[i])
    log(f"  Sample {i}: mean={norms.mean():.1f}, std={norms.std():.1f}, min={norms.min():.1f}, max={norms.max():.1f}")

# --- Diagnostic 1: per-position CE loss ---
log("\n=== DIAGNOSTIC 1: Per-position CE loss ===")
tl = np.asarray(token_losses)
tm = np.asarray(tmask)

# Per-sample breakdown
for i in range(BATCH):
    valid_len = int(tm[i].sum())
    losses_i = tl[i, :valid_len]
    log(f"  Sample {i} ({valid_len} tokens):")
    log(f"    Pos 0 (first word, conditioned on audio only): {losses_i[0]:.4f}")
    log(f"    Pos 1-4: {losses_i[1:5].mean():.4f}")
    log(f"    Pos 5-19: {losses_i[5:min(20,valid_len)].mean():.4f}")
    log(f"    Pos 20+: {losses_i[20:].mean():.4f}" if valid_len > 20 else "    Pos 20+: N/A")
    log(f"    Mean all: {losses_i.mean():.4f}")

# Aggregate per-position stats
max_len = int(tm.sum(axis=1).max())
pos_losses = []
for pos in range(min(max_len, 50)):
    vals = []
    for i in range(BATCH):
        if tm[i, pos]:
            vals.append(tl[i, pos])
    if vals:
        pos_losses.append((pos, np.mean(vals)))

log("\n  Avg loss by position (first 30):")
for pos, loss in pos_losses[:30]:
    bar = "#" * int(loss * 3)
    log(f"    pos {pos:3d}: {loss:.3f} {bar}")

# --- Diagnostic 3: audio ablation (zero audio) ---
log("\n=== DIAGNOSTIC 3: Audio ablation (real vs zero) ===")
zero_audio = jnp.zeros_like(mels)
log("Running forward pass with ZERO audio...")
zero_losses, _, _, _ = forward_and_get_losses(model, zero_audio, prompts, prompt_masks)

zl = np.asarray(zero_losses)

log("\n  Per-sample mean CE loss:")
log(f"  {'Sample':<10} {'Real Audio':<15} {'Zero Audio':<15} {'Delta':<15} {'Verdict'}")
for i in range(BATCH):
    valid_len = int(tm[i].sum())
    real_mean = tl[i, :valid_len].mean()
    zero_mean = zl[i, :valid_len].mean()
    delta = zero_mean - real_mean
    verdict = "AUDIO HELPS" if delta > 0.1 else "AUDIO IGNORED" if abs(delta) < 0.1 else "AUDIO HURTS?"
    log(f"  {i:<10} {real_mean:<15.4f} {zero_mean:<15.4f} {delta:<+15.4f} {verdict}")

# Position 0 specifically
log("\n  Position 0 CE loss (conditioned ONLY on audio):")
log(f"  {'Sample':<10} {'Real Audio':<15} {'Zero Audio':<15} {'Delta'}")
for i in range(BATCH):
    if tm[i, 0]:
        real_p0 = tl[i, 0]
        zero_p0 = zl[i, 0]
        log(f"  {i:<10} {real_p0:<15.4f} {zero_p0:<15.4f} {zero_p0 - real_p0:<+15.4f}")

log("\nDone.")
