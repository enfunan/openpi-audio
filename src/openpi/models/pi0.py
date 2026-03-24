import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.audio_enabled = config.audio_enabled
        self.training_stage = config.training_stage
        self.rehearsal_lambda = config.rehearsal_lambda

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Audio components
        if self.audio_enabled:
            from openpi.models.audio_projector import AttentionPooling as _AttentionPooling
            from openpi.models.audio_projector import AudioProjector as _AudioProjector
            from openpi.models.whisper import WhisperEncoder as _WhisperEncoder

            self.audio_num_tokens = config.audio_num_tokens

            whisper_module = _WhisperEncoder(variant=config.whisper_variant)
            self.whisper_encoder = nnx_bridge.ToNNX(whisper_module)
            # Initialize whisper with dummy mel spectrogram
            self.whisper_encoder.lazy_init(
                jnp.zeros((1, 128, 3000)),
                rngs=rngs,
            )

            whisper_dim = whisper_module.hidden_dim
            self.audio_projector = nnx_bridge.ToNNX(
                _AudioProjector(output_dim=paligemma_config.width)
            )
            self.audio_projector.lazy_init(
                jnp.zeros((1, 1500, whisper_dim)),
                rngs=rngs,
            )

            self.attention_pooling = nnx_bridge.ToNNX(
                _AttentionPooling(
                    num_queries=config.audio_num_tokens,
                    dim=paligemma_config.width,
                )
            )
            self.attention_pooling.lazy_init(
                jnp.zeros((1, 300, paligemma_config.width)),
                rngs=rngs,
            )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation, *, gradient_mode: str | None = None, causal_text: bool = False,
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"], at.Bool[at.Array, "b s"] | None]:
        """Embed prefix tokens (images + audio + text).

        Args:
            gradient_mode: Controls split LoRA stop_gradient routing.
            causal_text: If True, text tokens use causal attention (ar_mask=True).
                Used for ASR loss so text can't attend to future text tokens.

        Returns:
            tokens: (B, S, D) embedded tokens
            input_mask: (B, S) validity mask
            ar_mask: (S,) autoregressive mask
            audio_position_mask: (B, S) bool mask, True at audio token positions (None if no audio)
        """
        input_mask = []
        ar_mask = []
        tokens = []
        audio_position_mask_parts = []
        num_image_tokens = 0

        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]
            num_image_tokens += image_tokens.shape[1]

            if self.audio_enabled:
                # Image positions are NOT audio positions
                audio_position_mask_parts.append(
                    jnp.zeros((obs.images[name].shape[0], image_tokens.shape[1]), dtype=jnp.bool_)
                )

        # embed audio (symmetric prefix — always present when audio_enabled)
        if self.audio_enabled:
            # Use precomputed Whisper hidden states if available, else run live
            if obs.audio_whisper_hidden is not None:
                audio_hidden = obs.audio_whisper_hidden  # (B, 1500, 1280) precomputed
            else:
                audio_hidden = self.whisper_encoder(obs.audio)  # (B, 1500, 1280) live
            audio_tokens = self.audio_projector(audio_hidden)  # (B, 300, 2048)
            audio_tokens = self.attention_pooling(audio_tokens)  # (B, num_queries, 2048)

            # Zero out audio tokens for text-only samples (audio_mask=False)
            if obs.audio_mask is not None:
                audio_tokens = audio_tokens * obs.audio_mask[:, None, None]

            tokens.append(audio_tokens)
            audio_seq_len = audio_tokens.shape[1]

            # Audio mask for input: True if audio is valid
            if obs.audio_mask is not None:
                input_mask.append(
                    einops.repeat(obs.audio_mask, "b -> b s", s=audio_seq_len)
                )
            else:
                input_mask.append(
                    jnp.ones((audio_tokens.shape[0], audio_seq_len), dtype=jnp.bool_)
                )

            # Audio tokens use bidirectional attention (same as image/text)
            ar_mask += [False] * audio_seq_len

            # Audio positions ARE audio positions
            audio_position_mask_parts.append(
                jnp.ones((audio_tokens.shape[0], audio_seq_len), dtype=jnp.bool_)
            )

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            if causal_text:
                # Causal: each text token attends to image/audio + previous text only.
                # Used for ASR loss (autoregressive next-token prediction).
                ar_mask += [True] * tokenized_inputs.shape[1]
            else:
                # Bidirectional: full attention between image, audio, and language.
                # Used for flow matching (standard prefix-LM behavior).
                ar_mask += [False] * tokenized_inputs.shape[1]

            if self.audio_enabled:
                # Text positions are NOT audio positions
                audio_position_mask_parts.append(
                    jnp.zeros((tokenized_inputs.shape[0], tokenized_inputs.shape[1]), dtype=jnp.bool_)
                )

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        # Build audio position mask for split LoRA
        audio_position_mask = None
        if self.audio_enabled and audio_position_mask_parts:
            audio_position_mask = jnp.concatenate(audio_position_mask_parts, axis=1)

        return tokens, input_mask, ar_mask, audio_position_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def compute_alignment_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, *, train: bool = False, gradient_mode: str | None = None,
    ) -> at.Float[at.Array, ""]:
        """Compute autoregressive ASR cross-entropy loss.

        Uses causal attention for text positions and shifted next-token prediction,
        matching standard LM training (and VLAS's approach). Text tokens are kept
        as input (teacher forcing); the causal mask prevents looking ahead.

        Used for Stage 1 (standalone) and Stage 2 (rehearsal).
        """
        preprocess_rng = rng
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # Embed prefix with causal text: text positions use autoregressive attention.
        # Image + audio tokens remain bidirectional; text tokens attend to
        # image/audio + previous text only (can't see future text).
        prefix_tokens, prefix_mask, prefix_ar_mask, audio_position_mask = self.embed_prefix(
            observation, gradient_mode=gradient_mode, causal_text=True,
        )

        # Forward through LLM (prefix only, no suffix/actions needed for ASR)
        attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        (prefix_out, _), _ = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=attn_mask,
            positions=positions,
            audio_mask=audio_position_mask,
            gradient_mode=gradient_mode,
        )

        # Decode prefix output to vocabulary logits
        logits = self.PaliGemma.llm(prefix_out, method="decode")  # (B, S, vocab_size)

        # Autoregressive next-token prediction on text positions.
        if observation.asr_target_tokens is not None and observation.asr_target_mask is not None:
            target_tokens = observation.asr_target_tokens  # (B, L)
            target_mask = observation.asr_target_mask  # (B, L)

            # Text logits are at the end of prefix_out
            text_len = target_tokens.shape[1]
            text_logits = logits[:, -text_len:, :]  # (B, L, vocab_size)

            # Shifted prediction: logits at position i predict token_{i+1}.
            # Drop last logit (nothing to predict) and first target (predicted by last audio pos).
            shifted_logits = text_logits[:, :-1, :]  # (B, L-1, vocab_size)
            shifted_targets = target_tokens[:, 1:]  # (B, L-1)
            shifted_mask = target_mask[:, 1:]  # (B, L-1)

            # Cross-entropy loss
            log_probs = jax.nn.log_softmax(shifted_logits, axis=-1)
            token_losses = -jnp.take_along_axis(log_probs, shifted_targets[..., None], axis=-1).squeeze(-1)
            masked_losses = token_losses * shifted_mask
            loss = jnp.sum(masked_losses) / jnp.maximum(jnp.sum(shifted_mask), 1.0)
            return loss

        # Fallback: if no ASR targets, return zero loss
        return jnp.float32(0.0)

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        # Stage 1: ASR alignment only (no robot actions)
        # Uses gradient_mode="asr" so task LoRA is stop_gradient'd — only audio LoRA
        # receives ASR gradients. Task LoRA stays at random init, to be trained from
        # scratch in Stage 2 via flow matching gradients.
        if self.training_stage == "asr_alignment":
            asr_loss = self.compute_alignment_loss(rng, observation, train=train, gradient_mode="asr")
            # Return as (B, 1) to match expected shape
            batch_size = next(iter(observation.images.values())).shape[0]
            return jnp.broadcast_to(asr_loss, (batch_size, 1))

        # Default: flow matching loss
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Determine gradient_mode for split LoRA
        gradient_mode = "flow_matching" if self.audio_enabled else None

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask, audio_position_mask = self.embed_prefix(
            observation, gradient_mode=gradient_mode,
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # For split LoRA: audio_position_mask covers prefix tokens only (B, prefix_len).
        # Expert 0 (PaliGemma) processes prefix tokens and uses split LoRA.
        # Expert 1 (action expert) processes suffix tokens and doesn't use audio LoRA.
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            audio_mask=audio_position_mask,
            gradient_mode=gradient_mode,
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def compute_asr_loss_for_rehearsal(
        self, rng: at.KeyArrayLike, observation: _model.Observation, *, train: bool = False,
    ) -> at.Float[at.Array, ""]:
        """Compute ASR loss with gradient_mode='asr' for Stage 2 rehearsal.

        This ensures task LoRA is stop_gradient'd while audio LoRA receives gradients.
        Called separately from compute_loss in the training loop.

        When ``original_tokenized_prompt`` is available (Stage 2 with
        clear_prompt=True), swaps it into ``tokenized_prompt`` so the LLM
        prefix sees the full instruction text during the ASR forward pass.
        The flow matching pass uses the cleared (empty) prompt, but the ASR
        pass needs the original text for meaningful next-token prediction.
        """
        # If original prompt tokens exist, use them as the prefix text for
        # the ASR forward pass. This is needed because in Stage 2 with
        # clear_prompt=True, tokenized_prompt is empty.
        if observation.original_tokenized_prompt is not None:
            observation = _model.Observation(
                images=observation.images,
                image_masks=observation.image_masks,
                state=observation.state,
                tokenized_prompt=observation.original_tokenized_prompt,
                tokenized_prompt_mask=observation.original_tokenized_prompt_mask,
                audio=observation.audio,
                audio_whisper_hidden=observation.audio_whisper_hidden,
                audio_mask=observation.audio_mask,
                asr_target_tokens=observation.asr_target_tokens,
                asr_target_mask=observation.asr_target_mask,
                original_tokenized_prompt=observation.original_tokenized_prompt,
                original_tokenized_prompt_mask=observation.original_tokenized_prompt_mask,
                token_ar_mask=observation.token_ar_mask,
                token_loss_mask=observation.token_loss_mask,
            )
        return self.compute_alignment_loss(rng, observation, train=train, gradient_mode="asr")

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask, audio_position_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
            audio_mask=audio_position_mask,
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                # No audio_mask needed during inference with KV cache (prefix already computed)
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
