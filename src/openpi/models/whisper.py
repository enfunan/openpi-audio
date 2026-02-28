"""Whisper encoder and audio projector modules for speech-conditioned VLA.

Provides a frozen Whisper encoder (Flax Linen) and a learned downsampling
projector that converts encoder hidden states into tokens compatible with
PaliGemma's embedding space, following the VLAS architecture.
"""

import logging

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger("openpi")

# Whisper-large-v2 encoder output dimension.
WHISPER_HIDDEN_DIM = 1280
# Whisper encoder sequence length for 30s audio (3000 mel frames -> 1500 encoder frames).
WHISPER_SEQ_LEN = 1500


def _get_whisper_config(variant: str):
    """Load a WhisperConfig from HuggingFace (config only, no weights)."""
    from transformers import WhisperConfig

    return WhisperConfig.from_pretrained(variant)


class WhisperEncoder(nn.Module):
    """Wraps HuggingFace's FlaxWhisperEncoder as a Flax Linen submodule.

    Input: mel spectrogram (batch, 80, 3000) float32
    Output: encoder hidden states (batch, 1500, 1280)
    """

    variant: str = "openai/whisper-large-v2"

    def setup(self):
        from transformers.models.whisper.modeling_flax_whisper import FlaxWhisperEncoder as _HFEncoder

        config = _get_whisper_config(self.variant)
        self.encoder = _HFEncoder(config, dtype=jnp.float32)

    def __call__(self, mel: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # mel: (batch, 80, 3000)
        # FlaxWhisperEncoder expects input_features with same shape.
        outputs = self.encoder(input_features=mel, deterministic=deterministic)
        # outputs[0] is last_hidden_state: (batch, 1500, 1280)
        return outputs[0]


class DownsampleAudioProjector(nn.Module):
    """Downsamples and projects Whisper encoder output to PaliGemma embedding space.

    Following VLAS: reshape (B, 1500, 1280) -> (B, 300, 6400) via reduce_factor=5,
    then project to target embedding width via two Dense layers with GELU.

    Input: (batch, 1500, 1280)
    Output: (batch, 300, embed_dim)
    """

    reduce_factor: int = 5
    embed_dim: int = 2048  # PaliGemma embedding width

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        batch, seq_len, hidden_dim = hidden_states.shape
        new_seq_len = seq_len // self.reduce_factor
        merged_dim = hidden_dim * self.reduce_factor

        # Temporal downsampling via reshape: (B, 1500, 1280) -> (B, 300, 6400)
        x = jnp.reshape(hidden_states, (batch, new_seq_len, merged_dim))

        # Two-layer MLP projection: 6400 -> embed_dim -> embed_dim
        x = nn.Dense(self.embed_dim, name="proj_in")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.embed_dim, name="proj_out")(x)
        return x


def load_whisper_params(variant: str = "openai/whisper-large-v2") -> dict:
    """Load pretrained Whisper encoder parameters from HuggingFace.

    Returns a nested dict of numpy arrays containing the encoder weights,
    structured to match the WhisperEncoder Linen module's parameter tree.
    The params are placed under an "encoder" key to match the submodule name.
    """
    from transformers import FlaxWhisperForConditionalGeneration

    logger.info(f"Loading Whisper encoder params from {variant}")
    model = FlaxWhisperForConditionalGeneration.from_pretrained(variant, from_pt=True)

    # Extract only encoder params from the full model params tree.
    # HF structure: params["model"]["encoder"] -> {conv1, conv2, embed_positions, layers, layer_norm}
    raw_encoder_params = model.params["model"]["encoder"]

    # Wrap under "encoder" to match our WhisperEncoder.encoder submodule name.
    encoder_params = {"encoder": raw_encoder_params}

    # Convert to numpy arrays for consistency with other weight loaders.
    return jax.tree.map(lambda x: np.asarray(x), encoder_params)
