"""Diagnose why greedy decoding produces identical output for all samples."""

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


import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--checkpoint", default="checkpoints/pi05_audio_mixed_asr/mixed_asr/500/params")
_parser.add_argument("--data-dir", default="/home/user1/workspace/VLA/data/librispeech/LibriSpeech/train-clean-360")
_args = _parser.parse_args()
CKPT = _args.checkpoint
DATA = _args.data_dir

config = pi0_config.Pi0Config(
    pi05=True, audio_enabled=True,
    training_stage="asr_alignment", discrete_state_input=False,
    paligemma_variant="gemma_2b_lora",
)

log("Loading model...")
model = load_model(config, CKPT)
log("Model loaded.")

sp_path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
with sp_path.open("rb") as f:
    sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())
fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")

dataset = LibriSpeechDataset(data_dir=DATA)
indices = [0, 100, 500, 1000, 2000]  # 5 different samples

for idx in indices:
    sample = dataset[idx]
    text_gt = sample["prompt"]

    waveform, _ = librosa.load(sample["audio_path"], sr=16000)
    mel = fe(waveform, sampling_rate=16000, return_tensors="np").input_features[0]
    mel_batch = jnp.array(mel[None].astype(np.float32))

    audio_hidden = jax.lax.stop_gradient(model.whisper_encoder(mel_batch, deterministic=True))
    audio_tokens = model.audio_projector(audio_hidden)

    # Stats on audio tokens
    at_mean = float(jnp.mean(audio_tokens))
    at_std = float(jnp.std(audio_tokens))
    at_norm = float(jnp.mean(jnp.linalg.norm(audio_tokens[0], axis=-1)))
    log(f"\n{'='*60}")
    log(f"Sample {idx}: GT = {text_gt[:80]}...")
    log(f"Audio token stats: mean={at_mean:.4f}, std={at_std:.4f}, norm={at_norm:.2f}")

    # Teacher-forced: what does the model predict at each position?
    # Use the GT tokens and look at what the model predicts
    cleaned = text_gt.strip().replace("_", " ").replace("\n", " ")
    gt_tokens = sp.encode(cleaned, add_bos=True) + sp.encode("\n")
    gt_tokens = gt_tokens[:20]  # first 20 tokens only

    tok_arr = jnp.array([gt_tokens], dtype=jnp.int32)
    text_emb = jax.lax.stop_gradient(model.PaliGemma.llm(tok_arr, method="embed"))
    num_audio = audio_tokens.shape[1]
    num_text = text_emb.shape[1]

    tokens = jnp.concatenate([audio_tokens, text_emb], axis=1)
    input_mask = jnp.ones((1, num_audio + num_text), dtype=jnp.bool_)
    ar_mask = jnp.concatenate([
        jnp.zeros(num_audio, dtype=jnp.bool_),
        jnp.ones(num_text, dtype=jnp.bool_),
    ])
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.arange(num_audio + num_text)[None]

    (hidden, _), _ = model.PaliGemma.llm(
        [tokens, None], positions=positions, mask=attn_mask, adarms_cond=[None, None]
    )

    # Check predictions at key positions
    log(f"\nTeacher-forced predictions (position → predicted vs target):")
    for pos_offset in range(min(10, num_text)):
        pos = num_audio - 1 + pos_offset  # start from last audio position
        h = hidden[:, pos:pos+1, :]
        logits = model.PaliGemma.llm(h, method="decode").astype(jnp.float32)[0, 0]
        probs = jax.nn.softmax(logits)

        target_id = gt_tokens[pos_offset]
        target_word = sp.decode([target_id])
        target_prob = float(probs[target_id])

        top5_ids = jnp.argsort(-probs)[:5]
        top5 = [(int(tid), sp.decode([int(tid)]), float(probs[int(tid)])) for tid in top5_ids]

        input_desc = f"audio[{pos}]" if pos_offset == 0 else f"'{sp.decode([gt_tokens[pos_offset-1]])}'"
        log(f"  pos {pos} (input={input_desc}) → target='{target_word}'(p={target_prob:.4f})")
        log(f"    top5: {[(w, f'{p:.4f}') for _, w, p in top5]}")

    # Now do greedy decode for first 10 tokens
    log(f"\nGreedy decode (first 10 tokens):")
    bos_id = sp.bos_id()
    generated_ids = [bos_id]
    for step in range(10):
        tok_arr = jnp.array([generated_ids], dtype=jnp.int32)
        text_emb = jax.lax.stop_gradient(model.PaliGemma.llm(tok_arr, method="embed"))
        nt = text_emb.shape[1]
        na = audio_tokens.shape[1]

        toks = jnp.concatenate([audio_tokens, text_emb], axis=1)
        im = jnp.ones((1, na + nt), dtype=jnp.bool_)
        arm = jnp.concatenate([jnp.zeros(na, dtype=jnp.bool_), jnp.ones(nt, dtype=jnp.bool_)])
        am = make_attn_mask(im, arm)
        pos = jnp.arange(na + nt)[None]

        (h, _), _ = model.PaliGemma.llm([toks, None], positions=pos, mask=am, adarms_cond=[None, None])
        last_h = h[:, -1:, :]
        logits = model.PaliGemma.llm(last_h, method="decode").astype(jnp.float32)[0, 0]
        probs = jax.nn.softmax(logits)
        next_id = int(jnp.argmax(logits))
        next_word = sp.decode([next_id])
        next_prob = float(probs[next_id])
        generated_ids.append(next_id)
        log(f"  step {step}: '{next_word}' (id={next_id}, p={next_prob:.4f})")

    decoded = sp.decode(generated_ids[1:])
    log(f"  Full: {decoded}")

# ===== Audio Ablation Test =====
log(f"\n{'='*60}")
log("AUDIO ABLATION TEST: real audio vs zero audio")
log(f"{'='*60}")

ablation_indices = [0, 100, 500, 1000, 2000]
for idx in ablation_indices:
    sample = dataset[idx]
    text_gt = sample["prompt"]

    waveform, _ = librosa.load(sample["audio_path"], sr=16000)
    mel = fe(waveform, sampling_rate=16000, return_tensors="np").input_features[0]
    mel_batch = jnp.array(mel[None].astype(np.float32))

    # Real audio tokens
    audio_hidden = jax.lax.stop_gradient(model.whisper_encoder(mel_batch, deterministic=True))
    real_audio_tokens = model.audio_projector(audio_hidden)

    # Zero audio tokens
    zero_audio_tokens = jnp.zeros_like(real_audio_tokens)

    # Prepare GT tokens for loss computation
    cleaned = text_gt.strip().replace("_", " ").replace("\n", " ")
    gt_tokens = sp.encode(cleaned, add_bos=True) + sp.encode("\n")
    gt_tokens = gt_tokens[:20]

    def compute_loss_with_audio(audio_toks):
        tok_arr = jnp.array([gt_tokens], dtype=jnp.int32)
        text_emb = jax.lax.stop_gradient(model.PaliGemma.llm(tok_arr, method="embed"))
        na = audio_toks.shape[1]
        nt = text_emb.shape[1]
        tokens = jnp.concatenate([audio_toks, text_emb], axis=1)
        im = jnp.ones((1, na + nt), dtype=jnp.bool_)
        arm = jnp.concatenate([jnp.zeros(na, dtype=jnp.bool_), jnp.ones(nt, dtype=jnp.bool_)])
        am = make_attn_mask(im, arm)
        pos = jnp.arange(na + nt)[None]
        (h, _), _ = model.PaliGemma.llm([tokens, None], positions=pos, mask=am, adarms_cond=[None, None])

        # Compute CE loss at each text position
        losses = []
        for t in range(nt - 1):
            p = na + t
            logits = model.PaliGemma.llm(h[:, p:p+1, :], method="decode").astype(jnp.float32)[0, 0]
            log_probs = jax.nn.log_softmax(logits)
            target = gt_tokens[t + 1]
            losses.append(-float(log_probs[target]))
        return losses

    real_losses = compute_loss_with_audio(real_audio_tokens)
    zero_losses = compute_loss_with_audio(zero_audio_tokens)

    real_mean = np.mean(real_losses)
    zero_mean = np.mean(zero_losses)
    delta = zero_mean - real_mean

    # Position 0 (first text prediction after audio)
    pos0_real = real_losses[0] if real_losses else float('nan')
    pos0_zero = zero_losses[0] if zero_losses else float('nan')
    pos0_delta = pos0_zero - pos0_real

    log(f"\nSample {idx}: '{text_gt[:60]}...'")
    log(f"  Mean loss:  real={real_mean:.4f}  zero={zero_mean:.4f}  delta={delta:+.4f}")
    log(f"  Pos 0 loss: real={pos0_real:.4f}  zero={pos0_zero:.4f}  delta={pos0_delta:+.4f}")

log("\nDone.")
