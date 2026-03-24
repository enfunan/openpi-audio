"""Audio projector and attention pooling modules.

Maps Whisper encoder hidden states to Gemma embedding space with temporal
downsampling (5:1) and attention-based pooling (300→32 tokens).
"""

import flax.linen as nn
import jax.numpy as jnp

import openpi.shared.array_typing as at


class AudioProjector(nn.Module):
    """Projects Whisper hidden states to Gemma embedding dimension.

    Performs 5:1 temporal downsampling by reshaping adjacent frames,
    then projects via a two-layer MLP with GELU activation.

    Input:  (B, 1500, whisper_dim)  e.g. whisper_dim=1280
    Output: (B, 300, output_dim)    e.g. output_dim=2048
    """

    output_dim: int = 2048
    temporal_factor: int = 5

    @nn.compact
    def __call__(self, x: at.Array) -> at.Array:
        """Forward pass.

        Args:
            x: (B, 1500, whisper_dim) Whisper encoder hidden states

        Returns:
            (B, 300, output_dim) projected audio features
        """
        batch_size, seq_len, hidden_dim = x.shape

        # 5:1 temporal downsampling via reshape
        # (B, 1500, 1280) -> (B, 300, 6400)
        new_seq_len = seq_len // self.temporal_factor
        x = x[:, :new_seq_len * self.temporal_factor, :]  # trim to divisible length
        x = x.reshape(batch_size, new_seq_len, hidden_dim * self.temporal_factor)

        # Two-layer MLP projection
        x = nn.Dense(self.output_dim, name="proj_in")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.output_dim, name="proj_out")(x)

        return x


class AttentionPooling(nn.Module):
    """Cross-attention pooling that compresses audio tokens.

    Uses learnable query tokens to attend to projected audio features,
    compressing 300 tokens down to num_queries tokens.

    Input:  (B, 300, dim)
    Output: (B, num_queries, dim)
    """

    num_queries: int = 32
    dim: int = 2048
    num_heads: int = 8

    @nn.compact
    def __call__(self, x: at.Array) -> at.Array:
        """Forward pass.

        Args:
            x: (B, 300, dim) projected audio features

        Returns:
            (B, num_queries, dim) pooled audio tokens
        """
        batch_size = x.shape[0]
        head_dim = self.dim // self.num_heads

        # Learnable query tokens: (num_queries, dim)
        queries = self.param(
            "queries",
            nn.initializers.normal(stddev=0.02),
            (self.num_queries, self.dim),
        )
        # Broadcast to batch: (B, num_queries, dim)
        q = jnp.broadcast_to(queries[None], (batch_size, self.num_queries, self.dim))

        # Project queries, keys, values
        q = nn.Dense(self.dim, name="q_proj")(q)  # (B, num_queries, dim)
        k = nn.Dense(self.dim, name="k_proj")(x)   # (B, 300, dim)
        v = nn.Dense(self.dim, name="v_proj")(x)   # (B, 300, dim)

        # Reshape for multi-head attention
        # (B, seq, dim) -> (B, num_heads, seq, head_dim)
        q = q.reshape(batch_size, self.num_queries, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, k.shape[1], self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, v.shape[1], self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = head_dim ** -0.5
        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn_weights = nn.softmax(attn_weights, axis=-1)

        # Apply attention to values
        out = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)

        # Reshape back: (B, num_heads, num_queries, head_dim) -> (B, num_queries, dim)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, self.num_queries, self.dim)

        # Output projection
        out = nn.Dense(self.dim, name="out_proj")(out)

        return out
