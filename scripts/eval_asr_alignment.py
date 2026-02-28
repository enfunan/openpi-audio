"""Evaluate Stage 1 audio projector alignment quality.

Measures how well the audio projector maps Whisper features into PaliGemma's
text embedding space via:
1. Cosine similarity between audio and text embeddings for the same utterance
2. Retrieval accuracy: given an audio embedding, find the correct text among N candidates

Batches all samples for fast evaluation across all GPUs.

Usage:
    python scripts/eval_asr_alignment.py \
        --checkpoint checkpoints/pi05_audio_stage1_asr/stage1_asr_ce/899/params \
        --data-dir /path/to/LibriSpeech/train-clean-100 \
        --num-samples 200
"""

import argparse
import sys
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import librosa
import numpy as np
import orbax.checkpoint as ocp
import sentencepiece
from flax import traverse_util
from flax.nnx.traversals import unflatten_mapping
from transformers import WhisperFeatureExtractor

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.training.librispeech_dataset import LibriSpeechDataset
import openpi.shared.download as download


def log(msg):
    sys.stdout.write(str(msg) + "\n")
    sys.stdout.flush()


def load_tokenizer_sp():
    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        return sentencepiece.SentencePieceProcessor(model_proto=f.read())


def load_model(config, checkpoint_path):
    """Load model from checkpoint, working around Flax int-key bug."""
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


def compute_retrieval_metrics(audio_vecs, text_vecs):
    """Compute retrieval accuracy: for each audio, rank all texts by cosine sim."""
    audio_norm = audio_vecs / (np.linalg.norm(audio_vecs, axis=1, keepdims=True) + 1e-8)
    text_norm = text_vecs / (np.linalg.norm(text_vecs, axis=1, keepdims=True) + 1e-8)
    sim_matrix = audio_norm @ text_norm.T  # (N, N)

    N = sim_matrix.shape[0]
    rankings = np.argsort(-sim_matrix, axis=1)

    top1 = top5 = top10 = 0
    mrr_sum = 0.0
    for i in range(N):
        rank = int(np.where(rankings[i] == i)[0][0])
        if rank == 0: top1 += 1
        if rank < 5: top5 += 1
        if rank < 10: top10 += 1
        mrr_sum += 1.0 / (rank + 1)

    return {
        "top1": top1 / N, "top5": top5 / N, "top10": top10 / N,
        "mrr": mrr_sum / N,
        "diag_sims": np.diag(sim_matrix),  # per-sample cosine similarities
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 alignment quality")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = pi0_config.Pi0Config(
        pi05=True, audio_enabled=True,
        training_stage="asr_alignment", discrete_state_input=False,
    )

    log("Loading trained model...")
    model = load_model(config, args.checkpoint)
    log("Model loaded.")

    sp = load_tokenizer_sp()
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")

    dataset = LibriSpeechDataset(data_dir=args.data_dir)
    log(f"Dataset: {len(dataset)} samples")

    rng_np = np.random.default_rng(args.seed)
    indices = rng_np.choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False)
    indices.sort()
    N = len(indices)

    # --- Step 1: Preprocess all samples on CPU ---
    log(f"Preprocessing {N} audio + text samples...")
    t0 = time.time()

    all_mels = []
    all_tokens = []
    all_masks = []
    all_texts = []

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        text = sample["prompt"]
        all_texts.append(text)

        # Audio → mel
        waveform, _ = librosa.load(sample["audio_path"], sr=16000)
        features = fe(waveform, sampling_rate=16000, return_tensors="np")
        all_mels.append(features.input_features[0].astype(np.float32))

        # Text → tokens
        cleaned = text.strip().replace("_", " ").replace("\n", " ")
        tokens = sp.encode(cleaned, add_bos=True) + sp.encode("\n")
        max_len = config.max_token_len
        if len(tokens) < max_len:
            mask = [True] * len(tokens) + [False] * (max_len - len(tokens))
            tokens = tokens + [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
            mask = [True] * max_len
        all_tokens.append(np.array(tokens, dtype=np.int32))
        all_masks.append(np.array(mask, dtype=bool))

        if (i + 1) % 50 == 0:
            log(f"  Preprocessed {i+1}/{N}")

    log(f"Preprocessing done in {time.time()-t0:.1f}s")

    # Stack into arrays
    mels_np = np.stack(all_mels)       # (N, 80, 3000)
    tokens_np = np.stack(all_tokens)   # (N, max_len)
    masks_np = np.stack(all_masks)     # (N, max_len)

    # --- Step 2: Batched forward passes on GPU ---
    log(f"Computing embeddings in batches of {args.batch_size}...")

    all_audio_vecs = []
    all_text_vecs = []
    BS = args.batch_size
    t1 = time.time()

    for start in range(0, N, BS):
        end = min(start + BS, N)
        batch_mels = jnp.array(mels_np[start:end])
        batch_tokens = jnp.array(tokens_np[start:end])
        batch_masks = jnp.array(masks_np[start:end])

        # Audio embeddings: mel → Whisper (frozen) → MLP → mean pool
        audio_hidden = jax.lax.stop_gradient(
            model.whisper_encoder(batch_mels, deterministic=True)
        )
        audio_tokens = model.audio_projector(audio_hidden)  # (B, 300, D)
        audio_vecs = jnp.mean(audio_tokens, axis=1)  # (B, D)

        # Text embeddings: tokens → PaliGemma embed (frozen) → masked mean pool
        text_emb = jax.lax.stop_gradient(
            model.PaliGemma.llm(batch_tokens, method="embed")
        )  # (B, T, D)
        mask_expanded = batch_masks[:, :, None]  # (B, T, 1)
        text_vecs = jnp.sum(text_emb * mask_expanded, axis=1) / (jnp.sum(mask_expanded, axis=(1, 2), keepdims=True).squeeze(-1) + 1e-6)  # (B, D)

        all_audio_vecs.append(np.array(jax.device_get(audio_vecs)))
        all_text_vecs.append(np.array(jax.device_get(text_vecs)))

        log(f"  Batch {start//BS + 1}/{(N + BS - 1)//BS} done ({end}/{N} samples)")

    log(f"Embedding computation done in {time.time()-t1:.1f}s")

    audio_vecs_all = np.concatenate(all_audio_vecs, axis=0)  # (N, D)
    text_vecs_all = np.concatenate(all_text_vecs, axis=0)    # (N, D)

    # --- Step 3: Compute metrics ---
    metrics = compute_retrieval_metrics(audio_vecs_all, text_vecs_all)
    diag_sims = metrics["diag_sims"]

    log(f"\n{'='*70}")
    log(f"STAGE 1 ALIGNMENT EVALUATION ({N} samples)")
    log(f"{'='*70}")

    log(f"\n--- Cosine Similarity (audio vs text embedding, same utterance) ---")
    log(f"  Mean:   {np.mean(diag_sims):.4f}")
    log(f"  Median: {np.median(diag_sims):.4f}")
    log(f"  Std:    {np.std(diag_sims):.4f}")
    log(f"  Min:    {np.min(diag_sims):.4f}")
    log(f"  Max:    {np.max(diag_sims):.4f}")

    log(f"\n--- Retrieval Accuracy (audio → text, {N} candidates) ---")
    log(f"  Top-1:  {metrics['top1']:.1%}")
    log(f"  Top-5:  {metrics['top5']:.1%}")
    log(f"  Top-10: {metrics['top10']:.1%}")
    log(f"  MRR:    {metrics['mrr']:.4f}")

    log(f"\n--- Reference (random chance with {N} candidates) ---")
    log(f"  Top-1:  {1/N:.1%}")
    log(f"  Top-5:  {5/N:.1%}")
    log(f"  Top-10: {10/N:.1%}")

    log(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
