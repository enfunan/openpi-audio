import math
import re

import flax.linen as nn
import flax.struct as struct
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at

# Gradient mode encoding for JAX-compatible tracing through nn.scan.
# Strings can't be traced through JAX, so we encode as integers.
GRAD_MODE_NONE = 0           # No gradient isolation
GRAD_MODE_FLOW_MATCHING = 1  # Stop gradient on audio LoRA
GRAD_MODE_ASR = 2            # Stop gradient on task LoRA


def encode_gradient_mode(mode: str | None) -> jnp.ndarray:
    """Convert string gradient_mode to jnp.int32 for JAX tracing."""
    if mode is None:
        return jnp.int32(GRAD_MODE_NONE)
    if mode == "flow_matching":
        return jnp.int32(GRAD_MODE_FLOW_MATCHING)
    if mode == "asr":
        return jnp.int32(GRAD_MODE_ASR)
    raise ValueError(f"Unknown gradient_mode: {mode}")


@struct.dataclass
class LoRAConfig:
    """Configuration for LoRA."""

    # LoRA rank.
    rank: int
    # LoRA scaling factor.
    alpha: float = 1.0
    # Initialization function for LoRA parameters.
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    # Enable rank-stabilized LoRA: https://arxiv.org/pdf/2312.03732
    rslora: bool = False
    # Axes in the weight to apply LoRA to. Should typically be the last two axes.
    axes: tuple[int, int] = (-2, -1)
    # Axis label which is used by LoRA in einsum equations. Must not be present in the original equation.
    label: str = "L"

    @property
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


class Einsum(nn.Module):
    """Einsum with LoRA support, including split (dual) LoRA for audio/task separation.

    When both `lora_config` and `audio_lora_config` are set, operates in split LoRA mode:
    - `lora_config` weights handle non-audio (task) positions
    - `audio_lora_config` weights handle audio positions
    - Position selection via `audio_mask`, gradient isolation via `gradient_mode`
    """

    # Shape of the weight.
    shape: tuple[int, ...]
    # Initialization function for the weight.
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # If not None, apply LoRA to the weight (task LoRA in split mode).
    lora_config: LoRAConfig | None = None
    # If not None, apply a second LoRA for audio positions (split LoRA mode).
    audio_lora_config: LoRAConfig | None = None

    def setup(self):
        self.w = self.param("w", self.init_fn, self.shape)

        if config := self.lora_config:
            # Setup task LoRA parameters.
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            self.w_a = self.param("lora_a", config.init_fn, shape_a)
            self.w_b = self.param("lora_b", config.init_fn, shape_b)

        if config := self.audio_lora_config:
            # Setup audio LoRA parameters.
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            self.audio_w_a = self.param("audio_lora_a", config.init_fn, shape_a)
            self.audio_w_b = self.param("audio_lora_b", config.init_fn, shape_b)

    @nn.compact
    def __call__(self, eqn: str, x, audio_mask=None, gradient_mode=None):
        dtype = x.dtype  # original dtype, could be half-precision
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        if self.lora_config and self.audio_lora_config and audio_mask is not None:
            # Split LoRA mode: separate task and audio LoRA with gradient isolation.
            config = self.lora_config
            audio_config = self.audio_lora_config
            eqn_a, eqn_b = self._make_lora_eqns(eqn, config)

            # Task LoRA output
            task_lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            task_lora = jnp.einsum(eqn_b, task_lora, self.w_b.astype(dtype))
            task_lora = task_lora * config.scaling_value

            # Audio LoRA output
            audio_eqn_a, audio_eqn_b = self._make_lora_eqns(eqn, audio_config)
            audio_lora = jnp.einsum(audio_eqn_a, x, self.audio_w_a.astype(dtype))
            audio_lora = jnp.einsum(audio_eqn_b, audio_lora, self.audio_w_b.astype(dtype))
            audio_lora = audio_lora * audio_config.scaling_value

            # Gradient isolation via stop_gradient + jnp.where (JAX-traceable).
            audio_lora_stopped = jax.lax.stop_gradient(audio_lora)
            task_lora_stopped = jax.lax.stop_gradient(task_lora)

            if gradient_mode is not None:
                is_flow = (gradient_mode == GRAD_MODE_FLOW_MATCHING)
                is_asr = (gradient_mode == GRAD_MODE_ASR)
                audio_lora = jnp.where(is_flow, audio_lora_stopped, audio_lora)
                task_lora = jnp.where(is_asr, task_lora_stopped, task_lora)

            # Position-based selection: audio_mask is (B, S) bool.
            # Reshape mask to match LoRA output shape by finding B,S positions in the
            # einsum output (e.g. "2BSKH" has B at axis 1, S at axis 2).
            mask = self._expand_mask_for_eqn(audio_mask, eqn, task_lora.ndim)
            lora_out = jnp.where(mask, audio_lora, task_lora)
            result = result + lora_out

        elif config := self.lora_config:
            # Single LoRA fallback (backward compatible).
            eqn_a, eqn_b = self._make_lora_eqns(eqn, config)
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            result = result + lora * config.scaling_value

        return result

    @staticmethod
    def _expand_mask_for_eqn(mask: at.Array, eqn: str, target_ndim: int) -> at.Array:
        """Expand (B, S) mask to match the einsum output shape.

        The einsum equation tells us where B and S appear in the output.
        E.g. "BTD,NDH->BTNH" has B at axis 0, T at axis 1 → mask shape (B, T, 1, 1).
             "BSD,2KDH->2BSKH" has B at axis 1, S at axis 2 → mask shape (1, B, S, 1, 1).
        """
        lhs = eqn.split(",")[0]  # e.g. "BSD" or "BTD" or "BTNH"
        out = eqn.split("->")[1]  # e.g. "BTNH" or "2BSKH" or "BTD"
        b_label = lhs[0]  # First dim of input is batch
        s_label = lhs[1]  # Second dim of input is sequence

        b_pos = out.index(b_label)
        s_pos = out.index(s_label)

        # Build target shape: 1 everywhere except at B and S positions
        shape = [1] * target_ndim
        shape[b_pos] = mask.shape[0]
        shape[s_pos] = mask.shape[1]
        return mask.reshape(shape)

    def _make_lora_eqns(self, eqn: str, config: LoRAConfig | None = None) -> tuple[str, str]:
        if config is None:
            config = self.lora_config
        label = config.label
        if label in eqn:
            raise ValueError(f"{label} already in eqn: {eqn}")
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()

        a_label, b_label = (rhs[x] for x in config.axes)

        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"

        return eqn_a, eqn_b


class FeedForward(nn.Module):
    """Feed forward module with optional split LoRA support."""

    features: int
    hidden_dim: int
    # If not None, apply LoRA to the weight (task LoRA in split mode).
    lora_config: LoRAConfig | None = None
    # If not None, apply a second LoRA for audio positions (split LoRA mode).
    audio_lora_config: LoRAConfig | None = None

    def setup(self):
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        # Task LoRA weights
        self.w_gating_lora = None
        self.w_linear_lora = None
        if self.lora_config:
            self.w_gating_lora = (
                self.param("gating_einsum_lora_a", self.lora_config.init_fn, (2, self.features, self.lora_config.rank)),
                self.param(
                    "gating_einsum_lora_b", self.lora_config.init_fn, (2, self.lora_config.rank, self.hidden_dim)
                ),
            )
            self.w_linear_lora = (
                self.param("linear_lora_a", self.lora_config.init_fn, (self.hidden_dim, self.lora_config.rank)),
                self.param("linear_lora_b", self.lora_config.init_fn, (self.lora_config.rank, self.features)),
            )
        # Audio LoRA weights
        self.w_gating_audio_lora = None
        self.w_linear_audio_lora = None
        if self.audio_lora_config:
            cfg = self.audio_lora_config
            self.w_gating_audio_lora = (
                self.param("gating_einsum_audio_lora_a", cfg.init_fn, (2, self.features, cfg.rank)),
                self.param("gating_einsum_audio_lora_b", cfg.init_fn, (2, cfg.rank, self.hidden_dim)),
            )
            self.w_linear_audio_lora = (
                self.param("linear_audio_lora_a", cfg.init_fn, (self.hidden_dim, cfg.rank)),
                self.param("linear_audio_lora_b", cfg.init_fn, (cfg.rank, self.features)),
            )

    @nn.compact
    def __call__(self, x, audio_mask=None, gradient_mode=None):
        dtype = x.dtype  # original dtype, could be half-precision
        ff_gate = self._dot(
            x,
            self.w_gating[0],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
            None if self.w_gating_audio_lora is None else (self.w_gating_audio_lora[0][0], self.w_gating_audio_lora[1][0]),
            audio_mask,
            gradient_mode,
        )
        gate_value = nn.gelu(ff_gate)

        ff1 = self._dot(
            x,
            self.w_gating[1],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
            None if self.w_gating_audio_lora is None else (self.w_gating_audio_lora[0][1], self.w_gating_audio_lora[1][1]),
            audio_mask,
            gradient_mode,
        )
        activations = gate_value * ff1

        outputs = self._dot(
            activations,
            self.w_linear,
            self.w_linear_lora,
            self.w_linear_audio_lora,
            audio_mask,
            gradient_mode,
        )
        assert outputs.dtype == dtype
        return outputs

    def _dot(
        self,
        x: at.Array,
        w: at.Array,
        lora_weights: tuple[at.Array, at.Array] | None,
        audio_lora_weights: tuple[at.Array, at.Array] | None = None,
        audio_mask: at.Array | None = None,
        gradient_mode=None,
    ) -> at.Array:
        base = jnp.dot(x, w.astype(x.dtype))

        if lora_weights is not None and audio_lora_weights is not None and audio_mask is not None:
            # Split LoRA mode
            task_lora = jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))
            audio_lora = jnp.dot(jnp.dot(x, audio_lora_weights[0].astype(x.dtype)), audio_lora_weights[1].astype(x.dtype))

            # Gradient isolation via stop_gradient + jnp.where (JAX-traceable)
            audio_lora_stopped = jax.lax.stop_gradient(audio_lora)
            task_lora_stopped = jax.lax.stop_gradient(task_lora)

            if gradient_mode is not None:
                is_flow = (gradient_mode == GRAD_MODE_FLOW_MATCHING)
                is_asr = (gradient_mode == GRAD_MODE_ASR)
                audio_lora = jnp.where(is_flow, audio_lora_stopped, audio_lora)
                task_lora = jnp.where(is_asr, task_lora_stopped, task_lora)

            # Position-based selection: audio_mask is (B, S), output is (B, S, D)
            mask = audio_mask[..., None]  # (B, S, 1)
            return base + jnp.where(mask, audio_lora, task_lora)

        if lora_weights is not None:
            return base + jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))

        return base
