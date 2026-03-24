import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Audio VLA v2 options.
    audio_enabled: bool = False
    whisper_variant: str = "openai/whisper-large-v3"
    audio_num_tokens: int = 32
    # ASR rehearsal loss weight (Stage 2).
    rehearsal_lambda: float = 0.01
    # Distillation loss weight (Stage 2): force audio velocity to match text oracle velocity.
    distill_lambda: float = 0.0
    # Training stage: "default" (flow matching) or "asr_alignment" (Stage 1 ASR only).
    training_stage: str = "default"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        audio_spec = None
        audio_whisper_hidden_spec = None
        audio_mask_spec = None
        if self.audio_enabled:
            audio_spec = jax.ShapeDtypeStruct([batch_size, 128, 3000], jnp.float32)
            # Precomputed Whisper encoder output: (B, 1500, 1280)
            audio_whisper_hidden_spec = jax.ShapeDtypeStruct([batch_size, 1500, 1280], jnp.float32)
            audio_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        # Original prompt fields for Stage 2 ASR rehearsal (when text is cleared).
        original_tokenized_prompt_spec = None
        original_tokenized_prompt_mask_spec = None
        asr_target_tokens_spec = None
        asr_target_mask_spec = None
        if self.audio_enabled:
            asr_target_tokens_spec = jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32)
            asr_target_mask_spec = jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool)
            original_tokenized_prompt_spec = jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32)
            original_tokenized_prompt_mask_spec = jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                audio=audio_spec,
                audio_whisper_hidden=audio_whisper_hidden_spec,
                audio_mask=audio_mask_spec,
                asr_target_tokens=asr_target_tokens_spec,
                asr_target_mask=asr_target_mask_spec,
                original_tokenized_prompt=original_tokenized_prompt_spec,
                original_tokenized_prompt_mask=original_tokenized_prompt_mask_spec,
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )

        if self.audio_enabled:
            # Always freeze whisper encoder weights.
            filters.append(nnx.Not(nnx_utils.PathRegex(".*whisper.*")))

        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
