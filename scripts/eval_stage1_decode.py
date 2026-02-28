"""Quick greedy decode test: does the audio projector produce useful tokens?"""

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


CKPT = "checkpoints/pi05_audio_stage1_asr/stage1_asr_ce_5k/4999/params"
DATA = "/home/user1/workspace/VLA/data/librispeech/LibriSpeech/train-clean-100"
NUM_SAMPLES = 5
MAX_TOKENS = 80

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

for idx in indices:
    sample = dataset[idx]
    text_gt = sample["prompt"]

    # Audio -> mel -> whisper -> projector
    waveform, _ = librosa.load(sample["audio_path"], sr=16000)
    mel = fe(waveform, sampling_rate=16000, return_tensors="np").input_features[0]
    mel_batch = jnp.array(mel[None])  # (1, 80, 3000)

    audio_hidden = jax.lax.stop_gradient(model.whisper_encoder(mel_batch, deterministic=True))
    audio_tokens = model.audio_projector(audio_hidden)  # (1, 300, 2048)

    # Start with BOS token
    bos_id = sp.bos_id()
    generated_ids = [bos_id]

    for _ in range(MAX_TOKENS):
        # Embed current text tokens
        tok_arr = jnp.array([generated_ids])[None]  # need shape check
        tok_arr = jnp.array([generated_ids], dtype=jnp.int32)  # (1, T)
        text_emb = jax.lax.stop_gradient(model.PaliGemma.llm(tok_arr, method="embed"))  # (1, T, 2048)
        num_text = text_emb.shape[1]
        num_audio = audio_tokens.shape[1]

        # Concat [audio | text]
        tokens = jnp.concatenate([audio_tokens, text_emb], axis=1)

        # Attention mask
        input_mask = jnp.ones((1, num_audio + num_text), dtype=jnp.bool_)
        ar_mask = jnp.concatenate([
            jnp.zeros(num_audio, dtype=jnp.bool_),
            jnp.ones(num_text, dtype=jnp.bool_),
        ])
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.arange(num_audio + num_text)[None]

        # Forward
        (hidden, _), _ = model.PaliGemma.llm(
            [tokens, None], positions=positions, mask=attn_mask, adarms_cond=[None, None]
        )

        # Get logits for last position only
        last_hidden = hidden[:, -1:, :]  # (1, 1, 2048)
        logits = model.PaliGemma.llm(last_hidden, method="decode").astype(jnp.float32)  # (1, 1, V)
        next_id = int(jnp.argmax(logits[0, 0]))

        if next_id == sp.eos_id() or next_id == 0:
            break
        generated_ids.append(next_id)

    decoded = sp.decode(generated_ids[1:])  # skip BOS
    log(f"\n--- Sample {idx} ---")
    log(f"GT:   {text_gt}")
    log(f"PRED: {decoded}")

log("\nDone.")
