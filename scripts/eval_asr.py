"""Evaluate Stage 1 ASR alignment checkpoint via teacher-forced decoding.

Loads the trained audio projector checkpoint, runs teacher-forced forward
passes on LibriSpeech samples, and reports cross-entropy loss, token accuracy,
and predicted vs ground truth text.

Teacher-forced = feed ground truth text, check if model predicts the correct
next token at each position. Only 1 forward pass per sample (fast).

Usage:
    python scripts/eval_asr.py \
        --checkpoint checkpoints/pi05_audio_stage1_asr/stage1_asr_ce/899/params \
        --data-dir /path/to/LibriSpeech/train-clean-100 \
        --num-samples 50
"""

import argparse
import logging
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import librosa
import numpy as np
import optax
import orbax.checkpoint as ocp
import sentencepiece
from flax import traverse_util
from flax.nnx.traversals import unflatten_mapping
from transformers import WhisperFeatureExtractor

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import make_attn_mask
from openpi.training.librispeech_dataset import LibriSpeechDataset
import openpi.shared.download as download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_tokenizer_sp() -> sentencepiece.SentencePieceProcessor:
    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        return sentencepiece.SentencePieceProcessor(model_proto=f.read())


def load_model(config, checkpoint_path):
    """Load model from checkpoint, working around Flax int-key bug."""
    params = _model.restore_params(checkpoint_path)
    model = nnx.eval_shape(config.create, jax.random.key(0))
    graphdef, state = nnx.split(model)
    params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)

    # Flax's replace_by_pure_dict converts string-digit keys ('5' -> 5)
    # but the model state from Linen bridge keeps them as strings.
    # Manually merge to handle both key formats.
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


def teacher_forced_eval(model, observation, sp):
    """Teacher-forced evaluation: single forward pass, get loss + predictions.

    Replicates compute_alignment_loss but also extracts predicted token IDs.
    """
    # Audio path
    audio_hidden = jax.lax.stop_gradient(
        model.whisper_encoder(observation.audio, deterministic=True)
    )
    audio_tokens = model.audio_projector(audio_hidden)
    num_audio = audio_tokens.shape[1]

    # Text path
    text_emb = jax.lax.stop_gradient(
        model.PaliGemma.llm(observation.tokenized_prompt, method="embed")
    )
    num_text = text_emb.shape[1]

    # Concat [audio | text]
    tokens = jnp.concatenate([audio_tokens, text_emb], axis=1)

    # Attention mask: prefix-LM
    input_mask = jnp.concatenate([
        jnp.ones((tokens.shape[0], num_audio), dtype=jnp.bool_),
        observation.tokenized_prompt_mask,
    ], axis=1)
    ar_mask = jnp.concatenate([
        jnp.zeros(num_audio, dtype=jnp.bool_),
        jnp.ones(num_text, dtype=jnp.bool_),
    ])
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1

    # Forward through Gemma
    (hidden_states, _), _ = model.PaliGemma.llm(
        [tokens, None], positions=positions, mask=attn_mask, adarms_cond=[None, None]
    )

    # Extract text prediction positions
    text_hidden = hidden_states[:, num_audio - 1 : num_audio - 1 + num_text]
    logits = model.PaliGemma.llm(text_hidden, method="decode").astype(jnp.float32)

    # Loss
    text_targets = observation.tokenized_prompt
    text_mask = observation.tokenized_prompt_mask
    token_losses = optax.softmax_cross_entropy_with_integer_labels(logits, text_targets)
    loss = float(jnp.sum(token_losses * text_mask) / (jnp.sum(text_mask) + 1e-6))

    # Token accuracy
    predicted_ids = jnp.argmax(logits, axis=-1)  # (B, T)
    correct = (predicted_ids == text_targets) & text_mask
    accuracy = float(jnp.sum(correct)) / float(jnp.sum(text_mask) + 1e-6)

    # Decode predicted text (first sample in batch)
    pred_ids = np.array(predicted_ids[0])
    mask = np.array(text_mask[0])
    valid_pred_ids = pred_ids[mask].tolist()
    predicted_text = sp.decode(valid_pred_ids)

    return loss, accuracy, predicted_text


def build_observation(mel, tokens, token_mask, action_dim, action_horizon):
    """Build a batch-1 Observation."""
    dummy_img = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)
    dummy_img_mask = jnp.array([False])
    return _model.Observation(
        images={
            "base_0_rgb": dummy_img,
            "left_wrist_0_rgb": dummy_img,
            "right_wrist_0_rgb": dummy_img,
        },
        image_masks={
            "base_0_rgb": dummy_img_mask,
            "left_wrist_0_rgb": dummy_img_mask,
            "right_wrist_0_rgb": dummy_img_mask,
        },
        state=jnp.zeros((1, action_dim), dtype=jnp.float32),
        tokenized_prompt=jnp.array(tokens[None]),
        tokenized_prompt_mask=jnp.array(token_mask[None]),
        audio=jnp.array(mel[None]),
        audio_mask=jnp.array([True]),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 ASR checkpoint (teacher-forced)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("Loading model and checkpoint...")
    config = pi0_config.Pi0Config(
        pi05=True,
        audio_enabled=True,
        training_stage="asr_alignment",
        discrete_state_input=False,
    )
    model = load_model(config, args.checkpoint)

    sp = load_tokenizer_sp()
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")

    dataset = LibriSpeechDataset(data_dir=args.data_dir)
    logger.info(f"Dataset: {len(dataset)} samples")

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False)
    indices.sort()

    def tokenize_prompt(text: str):
        cleaned = text.strip().replace("_", " ").replace("\n", " ")
        tokens = sp.encode(cleaned, add_bos=True) + sp.encode("\n")
        max_len = config.max_token_len
        if len(tokens) < max_len:
            mask = [True] * len(tokens) + [False] * (max_len - len(tokens))
            tokens = tokens + [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
            mask = [True] * max_len
        return np.array(tokens, dtype=np.int32), np.array(mask, dtype=bool)

    logger.info(f"Evaluating {len(indices)} samples (teacher-forced)...")
    logger.info("First sample will be slow (JIT compilation)...")

    losses = []
    accuracies = []

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        ground_truth = sample["prompt"]
        audio_path = sample["audio_path"]

        # Preprocess audio
        waveform, _ = librosa.load(audio_path, sr=16000)
        features = fe(waveform, sampling_rate=16000, return_tensors="np")
        mel = features.input_features[0].astype(np.float32)

        # Tokenize ground truth
        tokens, token_mask = tokenize_prompt(ground_truth)

        # Build observation
        obs = build_observation(mel, tokens, token_mask, config.action_dim, config.action_horizon)

        # Teacher-forced eval (1 forward pass)
        loss, accuracy, predicted_text = teacher_forced_eval(model, obs, sp)
        losses.append(loss)
        accuracies.append(accuracy)

        # Print sample results
        print(f"\n--- Sample {i+1}/{len(indices)} (idx={idx}) | Loss: {loss:.3f} | Acc: {accuracy:.1%} ---")
        print(f"  File:  {pathlib.Path(audio_path).name}")
        print(f"  Truth: {ground_truth}")
        print(f"  Pred:  {predicted_text}")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY ({len(indices)} samples)")
    print(f"  Mean CE loss:      {np.mean(losses):.4f}")
    print(f"  Median CE loss:    {np.median(losses):.4f}")
    print(f"  Mean token acc:    {np.mean(accuracies):.1%}")
    print(f"  Median token acc:  {np.median(accuracies):.1%}")
    print(f"  Min/Max loss:      {np.min(losses):.4f} / {np.max(losses):.4f}")
    print(f"  Min/Max acc:       {np.min(accuracies):.1%} / {np.max(accuracies):.1%}")


if __name__ == "__main__":
    main()
