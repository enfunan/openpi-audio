"""Whisper encoder wrapper for audio feature extraction.

Wraps the HuggingFace Whisper encoder to produce audio hidden states
that can be projected into the Gemma embedding space.

For real variants (whisper-large-v3, etc.), uses HuggingFace's FlaxWhisperEncoder
directly so that pretrained weights load without name mapping.

For the "test" variant, uses a lightweight reimplementation for CPU testing.
"""

import logging

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Whisper encoder hidden dimensions by variant
_WHISPER_DIMS = {
    "openai/whisper-large-v2": 1280,
    "openai/whisper-large-v3": 1280,
    "openai/whisper-medium": 1024,
    "openai/whisper-small": 768,
    "openai/whisper-base": 512,
    "test": 64,
}

# Number of encoder output frames for 30s audio
_WHISPER_FRAMES = 1500


def _get_whisper_config(variant: str):
    """Load a WhisperConfig from HuggingFace (config only, no weights)."""
    from transformers import WhisperConfig
    return WhisperConfig.from_pretrained(variant)


class WhisperEncoder(nn.Module):
    """Whisper encoder that maps mel spectrograms to hidden states.

    The encoder is always frozen (stop_gradient applied to outputs).
    Weights are loaded externally via weight_loaders.

    For real variants: wraps HuggingFace's FlaxWhisperEncoder.
    For "test" variant: uses a small reimplementation for CPU testing.

    Input:  mel spectrogram (B, 128, 3000)
    Output: hidden states   (B, 1500, hidden_dim)
    """

    variant: str = "openai/whisper-large-v3"

    @property
    def hidden_dim(self) -> int:
        return _WHISPER_DIMS[self.variant]

    @property
    def num_frames(self) -> int:
        return _WHISPER_FRAMES

    def setup(self):
        if self.variant == "test":
            # Lightweight test encoder - no HF dependency
            pass
        else:
            from transformers.models.whisper.modeling_flax_whisper import FlaxWhisperEncoder as _HFEncoder
            config = _get_whisper_config(self.variant)
            self.encoder = _HFEncoder(config, dtype=jnp.float32)

    def __call__(self, mel: at.Array, deterministic: bool = True) -> at.Array:
        if self.variant == "test":
            return jax.lax.stop_gradient(self._test_forward(mel, deterministic))
        # Real variant: use HuggingFace encoder
        outputs = self.encoder(input_features=mel, deterministic=deterministic)
        return jax.lax.stop_gradient(outputs[0])

    @nn.compact
    def _test_forward(self, mel: at.Array, deterministic: bool = True) -> at.Array:
        """Lightweight encoder for CPU testing."""
        hidden_dim = self.hidden_dim
        x = jnp.transpose(mel, (0, 2, 1))  # (B, 3000, 80)

        x = nn.Conv(features=hidden_dim, kernel_size=(3,), strides=(1,),
                     padding="SAME", name="conv1")(x)
        x = nn.gelu(x)
        x = nn.Conv(features=hidden_dim, kernel_size=(3,), strides=(2,),
                     padding="SAME", name="conv2")(x)
        x = nn.gelu(x)

        pos_embed = self.param("embed_positions",
                               nn.initializers.normal(stddev=0.02),
                               (self.num_frames, hidden_dim))
        x = x + pos_embed[None, :x.shape[1], :]

        for i in range(2):  # 2 layers for test
            residual = x
            x = nn.LayerNorm(name=f"layers_{i}_self_attn_layer_norm")(x)
            x = nn.SelfAttention(num_heads=4, qkv_features=hidden_dim,
                                  out_features=hidden_dim, deterministic=deterministic,
                                  name=f"layers_{i}_self_attn")(x)
            x = residual + x
            residual = x
            x = nn.LayerNorm(name=f"layers_{i}_final_layer_norm")(x)
            x = nn.Dense(hidden_dim * 4, name=f"layers_{i}_fc1")(x)
            x = nn.gelu(x)
            x = nn.Dense(hidden_dim, name=f"layers_{i}_fc2")(x)
            x = residual + x

        return nn.LayerNorm(name="layer_norm")(x)


def load_whisper_params(variant: str = "openai/whisper-large-v3") -> dict:
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
