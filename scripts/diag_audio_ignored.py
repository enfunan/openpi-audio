"""Diagnostic: Is the model ignoring audio tokens?

Compare loss with real audio vs zero audio vs random audio.
If losses are similar, the model learned to ignore audio entirely.
"""

import sys
import jax
import jax.numpy as jnp
import librosa
import numpy as np
import orbax.checkpoint as ocp
import optax
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


def compute_ce_loss(model, audio_tokens, tokenized_prompt, tokenized_prompt_mask):
    """Replicate compute_alignment_loss but with provided audio_tokens."""
    num_audio = audio_tokens.shape[1]

    text_emb = jax.lax.stop_gradient(
        model.PaliGemma.llm(tokenized_prompt, method="embed")
    )
    num_text = text_emb.shape[1]

    tokens = jnp.concatenate([audio_tokens, text_emb], axis=1)

    input_mask = jnp.concatenate([
        jnp.ones((tokens.shape[0], num_audio), dtype=jnp.bool_),
        tokenized_prompt_mask,
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

    text_hidden = hidden_states[:, num_audio - 1 : num_audio - 1 + num_text]
    logits = model.PaliGemma.llm(text_hidden, method="decode").astype(jnp.float32)

    token_losses = optax.softmax_cross_entropy_with_integer_labels(logits, tokenized_prompt)
    per_token = token_losses * tokenized_prompt_mask
    return jnp.sum(per_token) / (jnp.sum(tokenized_prompt_mask) + 1e-6), per_token, tokenized_prompt_mask


CKPT = "checkpoints/pi05_audio_stage1_asr/stage1_asr_ce_5k/4999/params"
DATA = "/home/user1/workspace/VLA/data/librispeech/LibriSpeech/train-clean-100"
NUM_SAMPLES = 10

config = pi0_config.Pi0Config(
    pi05=True, audio_enabled=True,
    training_stage="asr_alignment", discrete_state_input=False,
)

log("Loading model...")
model = load_model(config, CKPT)
log("Model loaded.")

sp_path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
with sp_path.open("rb") as f:
    sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())
fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")

dataset = LibriSpeechDataset(data_dir=DATA)
rng = np.random.default_rng(42)
indices = rng.choice(len(dataset), size=NUM_SAMPLES, replace=False)

max_len = config.max_token_len  # 200

losses_real = []
losses_zero = []
losses_rand = []

for i, idx in enumerate(indices):
    sample = dataset[idx]
    text = sample["prompt"]

    # Audio → mel → Whisper → projector (real audio)
    waveform, _ = librosa.load(sample["audio_path"], sr=16000)
    mel = fe(waveform, sampling_rate=16000, return_tensors="np").input_features[0]
    mel_batch = jnp.array(mel[None].astype(np.float32))

    audio_hidden = jax.lax.stop_gradient(model.whisper_encoder(mel_batch, deterministic=True))
    real_audio_tokens = model.audio_projector(audio_hidden)  # (1, 300, 2048)

    # Tokenize text
    cleaned = text.strip().replace("_", " ").replace("\n", " ")
    tokens = sp.encode(cleaned, add_bos=True) + sp.encode("\n")
    if len(tokens) < max_len:
        mask = [True] * len(tokens) + [False] * (max_len - len(tokens))
        tokens = tokens + [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
        mask = [True] * max_len
    tok_arr = jnp.array([tokens], dtype=jnp.int32)
    mask_arr = jnp.array([mask], dtype=jnp.bool_)

    # Loss with real audio
    loss_real, _, _ = compute_ce_loss(model, real_audio_tokens, tok_arr, mask_arr)

    # Loss with ZERO audio tokens
    zero_audio = jnp.zeros_like(real_audio_tokens)
    loss_zero, _, _ = compute_ce_loss(model, zero_audio, tok_arr, mask_arr)

    # Loss with RANDOM audio tokens (same scale as real)
    scale = float(jnp.std(real_audio_tokens))
    rand_audio = jax.random.normal(jax.random.key(i), real_audio_tokens.shape) * scale
    loss_rand, _, _ = compute_ce_loss(model, rand_audio, tok_arr, mask_arr)

    lr = float(loss_real)
    lz = float(loss_zero)
    lrand = float(loss_rand)
    losses_real.append(lr)
    losses_zero.append(lz)
    losses_rand.append(lrand)

    log(f"[{i+1}/{NUM_SAMPLES}] real={lr:.4f}  zero={lz:.4f}  rand={lrand:.4f}  text={text[:60]}...")

log(f"\n{'='*60}")
log(f"DIAGNOSTIC: Is audio being used?")
log(f"{'='*60}")
log(f"Mean loss (real audio):   {np.mean(losses_real):.4f}")
log(f"Mean loss (zero audio):   {np.mean(losses_zero):.4f}")
log(f"Mean loss (random audio): {np.mean(losses_rand):.4f}")
log(f"")
if abs(np.mean(losses_real) - np.mean(losses_zero)) < 0.1:
    log("CONCLUSION: Model is IGNORING audio (real ≈ zero). Projector collapsed.")
else:
    log(f"CONCLUSION: Model IS using audio (diff = {np.mean(losses_real) - np.mean(losses_zero):.4f})")
