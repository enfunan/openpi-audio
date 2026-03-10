"""Perceiver Resampler for audio token compression.

Whisper encoder (frozen, 1500 frames @ 1280-dim) → input projection → N cross-attention
layers → num_queries output tokens at model dimension.  Replaces the AudioProjector +
AttentionPooling pipeline from v1 with a cleaner architecture that directly compresses
Whisper frames via learned queries.
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F  # noqa: N812


class PerceiverResamplerLayer(nn.Module):
    """Single cross-attention layer with FFN and pre-norm residual connections."""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, queries: Tensor, kv_input: Tensor) -> Tensor:
        # Cross-attention: queries attend to whisper frames
        attn_out, _ = self.cross_attn(query=queries, key=kv_input, value=kv_input)
        queries = self.norm1(queries + attn_out)

        # FFN with residual
        ffn_out = self.ffn(queries)
        queries = self.norm2(queries + ffn_out)

        return queries


class PerceiverResampler(nn.Module):
    """Compress Whisper encoder output into a fixed number of audio tokens.

    Architecture:
        - Input projection: Linear(whisper_dim → dim)
        - Learned queries: (num_queries, dim)
        - N cross-attention layers, each with:
            MultiheadAttention(dim, num_heads) + LayerNorm + FFN(dim→ffn_dim→dim) + LayerNorm

    Input:  (B, 1500, whisper_dim)  — from frozen Whisper-large-v3 encoder
    Output: (B, num_queries, dim)   — compressed audio tokens (e.g. 32 × 2048)
    """

    def __init__(
        self,
        num_queries: int = 32,
        dim: int = 2048,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_dim: int = 8192,
        whisper_dim: int = 1280,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim

        # Project Whisper hidden states to model dimension
        self.input_proj = nn.Linear(whisper_dim, dim)

        # Learned query tokens
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)

        # Cross-attention layers
        self.layers = nn.ModuleList(
            [PerceiverResamplerLayer(dim, num_heads, ffn_dim) for _ in range(num_layers)]
        )

    def forward(self, whisper_hidden: Tensor) -> Tensor:
        """
        Args:
            whisper_hidden: (B, 1500, whisper_dim) from frozen Whisper encoder.

        Returns:
            (B, num_queries, dim) compressed audio tokens.
        """
        B = whisper_hidden.shape[0]

        # Project to model dimension
        kv_input = self.input_proj(whisper_hidden)  # (B, 1500, dim)

        # Expand learned queries for the batch
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, num_queries, dim)

        # Process through cross-attention layers
        for layer in self.layers:
            queries = layer(queries, kv_input)

        return queries  # (B, num_queries, dim)
