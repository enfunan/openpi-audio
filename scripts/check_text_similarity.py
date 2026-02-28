"""Quick check: how similar are mean-pooled text embeddings across different samples?
If cosine > 0.95, contrastive loss on mean-pools can't work.
If cosine < 0.8, contrastive loss should work with right hyperparams.
"""
import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece

import openpi.shared.download as download
from openpi.models import model as _model, pi0_config
from openpi.training.librispeech_dataset import LibriSpeechDataset

import flax.nnx as nnx
from flax import traverse_util
from flax.nnx.traversals import unflatten_mapping
import orbax.checkpoint as ocp

DATA = "/home/user1/workspace/VLA/data/librispeech/LibriSpeech/train-clean-100"
CKPT = "checkpoints/pi05_audio_stage1_embed_align/stage1_embed_align_5k/4999/params"

config = pi0_config.Pi0Config(
    pi05=True, audio_enabled=True,
    training_stage="embedding_alignment", discrete_state_input=False,
)

print("Loading model (for embedding table only)...")
params = _model.restore_params(CKPT)
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
print("Model loaded.")

sp_path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
with sp_path.open("rb") as f:
    sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())

dataset = LibriSpeechDataset(data_dir=DATA)

# Get text embeddings for 20 diverse samples
indices = list(range(0, 1000, 50))  # 20 samples, spaced apart
text_means = []

for idx in indices:
    sample = dataset[idx]
    text = sample["prompt"].strip().replace("_", " ").replace("\n", " ")
    ids = sp.encode(text, add_bos=True) + sp.encode("\n")
    ids = ids[:200]
    tok_arr = jnp.array([ids], dtype=jnp.int32)
    text_emb = jax.lax.stop_gradient(model.PaliGemma.llm(tok_arr, method="embed"))  # (1, T, D)

    # Mean-pool (no padding mask needed since all tokens are valid)
    text_mean = jnp.mean(text_emb[0], axis=0)  # (D,)
    text_means.append(np.array(text_mean))
    print(f"Sample {idx}: norm={float(jnp.linalg.norm(text_mean)):.2f}, text='{text[:50]}...'")

text_means_arr = np.stack(text_means)
n = len(text_means)

# Compute pairwise cosine similarity
print(f"\n{'='*60}")
print("Text-text cosine similarity (20 samples)")
print(f"{'='*60}")

cosines = []
for i in range(n):
    for j in range(i+1, n):
        cos = float(np.dot(text_means_arr[i], text_means_arr[j]) /
                     (np.linalg.norm(text_means_arr[i]) * np.linalg.norm(text_means_arr[j]) + 1e-8))
        cosines.append(cos)

print(f"Pairwise cosine similarities ({len(cosines)} pairs):")
print(f"  Mean:   {np.mean(cosines):.4f}")
print(f"  Std:    {np.std(cosines):.4f}")
print(f"  Min:    {np.min(cosines):.4f}")
print(f"  Max:    {np.max(cosines):.4f}")
print(f"  Median: {np.median(cosines):.4f}")

# Histogram
bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
hist, _ = np.histogram(cosines, bins=bins)
print(f"\nDistribution:")
for i, count in enumerate(hist):
    print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {count}")

print(f"\nVerdict:")
if np.mean(cosines) > 0.95:
    print("  Text embeddings are TOO SIMILAR (>0.95). Contrastive loss on mean-pools won't work.")
    print("  Recommendation: Skip to Stage 2 or use per-token alignment.")
elif np.mean(cosines) > 0.85:
    print("  Text embeddings are MODERATELY SIMILAR (0.85-0.95). Contrastive loss might work with high weight + low temperature.")
elif np.mean(cosines) > 0.7:
    print("  Text embeddings have DECENT DIVERSITY (0.7-0.85). Contrastive loss should work.")
else:
    print("  Text embeddings have GOOD DIVERSITY (<0.7). Contrastive loss should work easily.")
