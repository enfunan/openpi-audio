"""Diagnose embedding alignment quality after Stage 1 (v2) training.

Checks:
1. Distribution match: audio projector output norms/stats vs text embedding norms/stats
2. Cosine similarity: mean-pooled audio vs text (should be >0.8)
3. Per-sample variation: audio embeddings should differ across samples (no collapse)
4. Downstream test: plug audio into full Gemma forward pass, check text predictions
"""

import sys
import jax
import jax.numpy as jnp
import librosa
import numpy as np
import orbax.checkpoint as ocp
import sentencepiece
import flax.nnx as nnx
from flax import traverse_util
from flax.nnx.traversals import unflatten_mapping
from transformers import WhisperFeatureExtractor

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import make_attn_mask
from openpi.training.librispeech_dataset import LibriSpeechDataset
import openpi.shared.download as download


def log(msg):
    sys.stdout.write(str(msg) + "\n")
    sys.stdout.flush()


def load_model(config, checkpoint_path):
    params = _model.restore_params(checkpoint_path)
    model = nnx.eval_shape(config.create, jax.random.key(0))
    graphdef, state = nnx.split(model)
    params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
    flat_state = state.flat_state()
    for kp, v in traverse_util.flatten_dict(params).items():
        if kp in flat_state:
            flat_state[kp] = flat_state[kp].replace(v) if hasattr(flat_state[kp], 'replace') else v
        else:
            int_kp = tuple(int(k) if isinstance(k, str) and k.isdigit() else k for k in kp)
            if int_kp in flat_state:
                flat_state[int_kp] = flat_state[int_kp].replace(v) if hasattr(flat_state[int_kp], 'replace') else v
    state.update(unflatten_mapping(flat_state))
    model = nnx.merge(graphdef, state)
    model.eval()
    return model


# --- Configuration ---
CKPT = "checkpoints/pi05_audio_stage1_embed_align/stage1_embed_align_5k/4999/params"
DATA = "/home/user1/workspace/VLA/data/librispeech/LibriSpeech/train-clean-100"

config = pi0_config.Pi0Config(
    pi05=True, audio_enabled=True,
    training_stage="embedding_alignment", discrete_state_input=False,
)

log("Loading model...")
model = load_model(config, CKPT)
log("Model loaded.")

sp_path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
with sp_path.open("rb") as f:
    sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())
fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")

dataset = LibriSpeechDataset(data_dir=DATA)
indices = [0, 100, 500, 1000, 5000]

# Collect per-sample stats
audio_means_list = []
text_means_list = []
audio_norms_list = []
text_norms_list = []

log(f"\n{'='*70}")
log("PART 1: Distribution match & cosine similarity")
log(f"{'='*70}")

for idx in indices:
    sample = dataset[idx]
    text_gt = sample["prompt"]

    # Audio path
    waveform, _ = librosa.load(sample["audio_path"], sr=16000)
    mel = fe(waveform, sampling_rate=16000, return_tensors="np").input_features[0]
    mel_batch = jnp.array(mel[None].astype(np.float32))

    audio_hidden = jax.lax.stop_gradient(model.whisper_encoder(mel_batch, deterministic=True))
    audio_tokens = model.audio_projector(audio_hidden)  # (1, 300, D)

    # Text path
    cleaned = text_gt.strip().replace("_", " ").replace("\n", " ")
    gt_ids = sp.encode(cleaned, add_bos=True) + sp.encode("\n")
    gt_ids = gt_ids[:200]
    tok_arr = jnp.array([gt_ids], dtype=jnp.int32)
    text_emb = jax.lax.stop_gradient(model.PaliGemma.llm(tok_arr, method="embed"))  # (1, T, D)

    # Per-token norms
    audio_token_norms = jnp.linalg.norm(audio_tokens[0], axis=-1)  # (300,)
    text_token_norms = jnp.linalg.norm(text_emb[0], axis=-1)  # (T,)

    # Mean-pool
    audio_mean = jnp.mean(audio_tokens[0], axis=0)  # (D,)
    text_mean = jnp.mean(text_emb[0], axis=0)  # (D,)

    # Cosine similarity
    cos_sim = float(jnp.sum(audio_mean * text_mean) / (jnp.linalg.norm(audio_mean) * jnp.linalg.norm(text_mean) + 1e-8))

    # MSE
    mse = float(jnp.mean(jnp.square(audio_mean - text_mean)))

    log(f"\nSample {idx}: '{text_gt[:60]}...'")
    log(f"  Audio token norms: mean={float(jnp.mean(audio_token_norms)):.2f}, std={float(jnp.std(audio_token_norms)):.2f}, min={float(jnp.min(audio_token_norms)):.2f}, max={float(jnp.max(audio_token_norms)):.2f}")
    log(f"  Text  token norms: mean={float(jnp.mean(text_token_norms)):.2f}, std={float(jnp.std(text_token_norms)):.2f}, min={float(jnp.min(text_token_norms)):.2f}, max={float(jnp.max(text_token_norms)):.2f}")
    log(f"  Mean-pool audio norm: {float(jnp.linalg.norm(audio_mean)):.2f}")
    log(f"  Mean-pool text  norm: {float(jnp.linalg.norm(text_mean)):.2f}")
    log(f"  Cosine similarity:    {cos_sim:.4f}")
    log(f"  MSE:                  {mse:.4f}")

    audio_means_list.append(np.array(audio_mean))
    text_means_list.append(np.array(text_mean))
    audio_norms_list.append(float(jnp.mean(audio_token_norms)))
    text_norms_list.append(float(jnp.mean(text_token_norms)))

log(f"\n{'='*70}")
log("PART 2: Per-sample variation (collapse check)")
log(f"{'='*70}")

audio_means_arr = np.stack(audio_means_list)  # (N, D)
# Pairwise cosine similarities between audio embeddings
n = len(audio_means_list)
log("\nAudio-audio cosine similarity matrix:")
for i in range(n):
    row = []
    for j in range(n):
        cos = float(np.sum(audio_means_arr[i] * audio_means_arr[j]) / (np.linalg.norm(audio_means_arr[i]) * np.linalg.norm(audio_means_arr[j]) + 1e-8))
        row.append(f"{cos:.3f}")
    log(f"  [{', '.join(row)}]")

# Mean pairwise cosine similarity (off-diagonal)
off_diag_cos = []
for i in range(n):
    for j in range(i + 1, n):
        cos = float(np.sum(audio_means_arr[i] * audio_means_arr[j]) / (np.linalg.norm(audio_means_arr[i]) * np.linalg.norm(audio_means_arr[j]) + 1e-8))
        off_diag_cos.append(cos)
log(f"\nMean off-diagonal audio-audio cosine sim: {np.mean(off_diag_cos):.4f}")
log(f"  (Ideal: < 0.9 means no collapse; < 0.7 means good diversity)")

log(f"\n{'='*70}")
log("PART 3: Downstream test — plug audio into Gemma, check predictions")
log(f"{'='*70}")

for idx in indices[:3]:
    sample = dataset[idx]
    text_gt = sample["prompt"]

    waveform, _ = librosa.load(sample["audio_path"], sr=16000)
    mel = fe(waveform, sampling_rate=16000, return_tensors="np").input_features[0]
    mel_batch = jnp.array(mel[None].astype(np.float32))

    audio_hidden = jax.lax.stop_gradient(model.whisper_encoder(mel_batch, deterministic=True))
    audio_tokens = model.audio_projector(audio_hidden)
    num_audio = audio_tokens.shape[1]

    # Start with just BOS after audio, then greedy decode 10 tokens
    bos_id = sp.bos_id()
    generated_ids = [bos_id]

    log(f"\nSample {idx}: GT = '{text_gt[:60]}...'")
    log(f"Greedy decode (10 tokens):")

    for step in range(10):
        tok_arr = jnp.array([generated_ids], dtype=jnp.int32)
        text_emb = jax.lax.stop_gradient(model.PaliGemma.llm(tok_arr, method="embed"))
        nt = text_emb.shape[1]

        tokens = jnp.concatenate([audio_tokens, text_emb], axis=1)
        input_mask = jnp.ones((1, num_audio + nt), dtype=jnp.bool_)
        ar_mask = jnp.concatenate([
            jnp.zeros(num_audio, dtype=jnp.bool_),
            jnp.ones(nt, dtype=jnp.bool_),
        ])
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.arange(num_audio + nt)[None]

        (h, _), _ = model.PaliGemma.llm(
            [tokens, None], positions=positions, mask=attn_mask, adarms_cond=[None, None]
        )
        last_h = h[:, -1:, :]
        logits = model.PaliGemma.llm(last_h, method="decode").astype(jnp.float32)[0, 0]
        probs = jax.nn.softmax(logits)
        next_id = int(jnp.argmax(logits))
        next_word = sp.decode([next_id])
        next_prob = float(probs[next_id])
        generated_ids.append(next_id)
        log(f"  step {step}: '{next_word}' (p={next_prob:.4f})")

    decoded = sp.decode(generated_ids[1:])
    log(f"  Full: {decoded}")

log(f"\n{'='*70}")
log("SUMMARY")
log(f"{'='*70}")
log(f"Audio token norm range: {min(audio_norms_list):.2f} - {max(audio_norms_list):.2f}")
log(f"Text  token norm range: {min(text_norms_list):.2f} - {max(text_norms_list):.2f}")
log(f"Mean off-diagonal audio-audio cos sim: {np.mean(off_diag_cos):.4f}")
log(f"\nExpected outcomes:")
log(f"  - Audio norms should be ~{np.mean(text_norms_list):.0f} (matching text norms)")
log(f"  - Cosine sim (audio vs text) should be > 0.8")
log(f"  - Off-diagonal audio cos sim should be < 0.9 (no collapse)")
log(f"  - Greedy decode should vary across samples")
log("\nDone.")
