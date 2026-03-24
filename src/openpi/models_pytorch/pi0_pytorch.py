import logging
import math
import os

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.lora_pytorch import GRAD_MODE_ASR, GRAD_MODE_BYPASS, GRAD_MODE_FLOW_MATCHING
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class AudioProjectorPT(nn.Module):
    """Projects Whisper hidden states to Gemma embedding dimension.

    5:1 temporal downsampling via reshape, then 2-layer MLP.
    (B, 1500, whisper_dim) → (B, 300, output_dim)
    """

    def __init__(self, whisper_dim: int = 1280, output_dim: int = 2048, temporal_factor: int = 5):
        super().__init__()
        self.temporal_factor = temporal_factor
        hidden_dim = whisper_dim * temporal_factor  # 6400
        self.proj_in = nn.Linear(hidden_dim, output_dim)
        self.proj_out = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        new_S = S // self.temporal_factor
        x = x[:, :new_S * self.temporal_factor, :]
        x = x.reshape(B, new_S, D * self.temporal_factor)
        x = F.gelu(self.proj_in(x))
        return self.proj_out(x)


class AttentionPoolingPT(nn.Module):
    """Cross-attention pooling: (B, 300, D) → (B, num_queries, D)."""

    def __init__(self, num_queries: int = 32, dim: int = 2048, num_heads: int = 8):
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)
        q = self.q_proj(q)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(B, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, k.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, v.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(1, 2).reshape(B, self.num_queries, self.dim)
        return self.out_proj(out)


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # Audio components
        self.audio_enabled = getattr(config, "audio_enabled", False)
        self.training_stage = getattr(config, "training_stage", "default")
        self.rehearsal_lambda = getattr(config, "rehearsal_lambda", 0.01)

        if self.audio_enabled:
            self.audio_num_tokens = getattr(config, "audio_num_tokens", 32)
            paligemma_width = paligemma_config.width  # e.g. 2048
            whisper_dim = 1280  # whisper-large-v3 hidden dim

            # Whisper encoder (frozen — loaded separately via weight loader)
            from transformers import WhisperModel
            whisper_variant = getattr(config, "whisper_variant", "openai/whisper-large-v3")
            self.whisper_encoder = WhisperModel.from_pretrained(whisper_variant).encoder
            self.whisper_encoder.requires_grad_(False)

            # Audio projector: (B, 1500, 1280) → (B, 300, paligemma_width)
            self.audio_projector = AudioProjectorPT(whisper_dim=whisper_dim, output_dim=paligemma_width)

            # Attention pooling: (B, 300, D) → (B, num_queries, D)
            self.attention_pooling = AttentionPoolingPT(
                num_queries=self.audio_num_tokens, dim=paligemma_width,
            )

        torch.set_float32_matmul_precision("high")
        if os.environ.get("OPENPI_NO_COMPILE", "") != "1":
            self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        # Use the model's dtype so SDPA bias matches query dtype
        model_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        min_val = torch.finfo(model_dtype).min
        return torch.where(att_2d_masks_4d, 0.0, min_val).to(dtype=model_dtype)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return observation

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks,
        audio_whisper_hidden=None, audio=None, audio_mask=None, causal_text=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Embed images (+ audio) + language tokens for PaliGemma transformer processing.

        Returns:
            embs, pad_masks, att_masks, audio_position_mask
            audio_position_mask is (B, S) bool, True at audio token positions. None if no audio.
        """
        # If we have mel spectrogram but no precomputed whisper hidden, run encoder
        if self.audio_enabled and audio_whisper_hidden is None and audio is not None:
            with torch.no_grad():
                audio_whisper_hidden = self.whisper_encoder(audio).last_hidden_state  # (B, 1500, 1280)
        embs = []
        pad_masks = []
        att_masks = []
        audio_position_mask_parts = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

            if self.audio_enabled:
                audio_position_mask_parts.append(
                    torch.zeros(bsize, num_img_embs, dtype=torch.bool, device=img.device)
                )

        # Process audio (symmetric prefix — always present when audio_enabled)
        if self.audio_enabled and audio_whisper_hidden is not None:
            audio_hidden = audio_whisper_hidden  # (B, 1500, 1280) precomputed
            audio_tokens = self.audio_projector(audio_hidden)  # (B, 300, D)
            audio_tokens = self.attention_pooling(audio_tokens)  # (B, num_queries, D)

            # Zero out audio tokens where audio_mask is False (text-only samples)
            if audio_mask is not None:
                audio_tokens = audio_tokens * audio_mask[:, None, None].float()

            embs.append(audio_tokens)
            audio_seq_len = audio_tokens.shape[1]

            if audio_mask is not None:
                pad_masks.append(audio_mask[:, None].expand(bsize, audio_seq_len))
            else:
                pad_masks.append(torch.ones(bsize, audio_seq_len, dtype=torch.bool, device=audio_tokens.device))

            att_masks += [0] * audio_seq_len
            audio_position_mask_parts.append(
                torch.ones(bsize, audio_seq_len, dtype=torch.bool, device=audio_tokens.device)
            )

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        if causal_text:
            att_masks += [1] * num_lang_embs
        else:
            att_masks += [0] * num_lang_embs

        if self.audio_enabled:
            audio_position_mask_parts.append(
                torch.zeros(bsize, num_lang_embs, dtype=torch.bool, device=lang_emb.device)
            )

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, att_masks.shape[0])

        audio_position_mask = None
        if self.audio_enabled and audio_position_mask_parts:
            audio_position_mask = torch.cat(audio_position_mask_parts, dim=1)

        return embs, pad_masks, att_masks, audio_position_mask

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def compute_alignment_loss(self, observation, gradient_mode=None):
        """Compute autoregressive ASR cross-entropy loss (Stage 1 or rehearsal).

        Uses causal attention for text positions and shifted next-token prediction.
        """
        obs = self._preprocess_observation(observation, train=self.training)
        images = list(obs.images.values())
        img_masks = list(obs.image_masks.values())

        prefix_embs, prefix_pad_masks, prefix_att_masks, audio_position_mask = self.embed_prefix(
            images, img_masks, obs.tokenized_prompt, obs.tokenized_prompt_mask,
            audio_whisper_hidden=obs.audio_whisper_hidden, audio=getattr(obs, "audio", None),
            audio_mask=obs.audio_mask, causal_text=True,
        )

        # Cast if needed
        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Forward through PaliGemma only (prefix-only path with LoRA routing)
        (prefix_out, _), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            adarms_cond=[None, None],
            audio_mask=audio_position_mask,
            gradient_mode=gradient_mode,
        )

        # Decode to vocabulary logits
        logits = self.paligemma_with_expert.decode_to_vocab(prefix_out)  # (B, S, vocab_size)

        # Compute shifted next-token prediction loss on text positions
        asr_target_tokens = obs.asr_target_tokens
        asr_target_mask = obs.asr_target_mask
        if asr_target_tokens is None or asr_target_mask is None:
            return torch.tensor(0.0, device=prefix_embs.device)

        text_len = asr_target_tokens.shape[1]
        text_logits = logits[:, -text_len:, :]  # (B, L, vocab_size)

        # Shifted prediction: logits[i] predicts token[i+1]
        shifted_logits = text_logits[:, :-1, :].float()  # (B, L-1, V)
        shifted_targets = asr_target_tokens[:, 1:]  # (B, L-1)
        shifted_mask = asr_target_mask[:, 1:].float()  # (B, L-1)

        log_probs = F.log_softmax(shifted_logits, dim=-1)
        token_losses = -torch.gather(log_probs, 2, shifted_targets.unsqueeze(-1).long()).squeeze(-1)
        masked_losses = token_losses * shifted_mask
        loss = masked_losses.sum() / shifted_mask.sum().clamp(min=1.0)
        return loss

    def compute_asr_loss_for_rehearsal(self, observation):
        """Compute ASR loss with gradient_mode=ASR for Stage 2 rehearsal.

        Swaps original_tokenized_prompt back if available (cleared in Stage 2).
        """
        obs = observation
        # If original prompt tokens exist, swap them in for the ASR pass
        if getattr(obs, "original_tokenized_prompt", None) is not None:
            class _SwappedObs:
                pass
            swapped = _SwappedObs()
            for attr in dir(obs):
                if not attr.startswith("_"):
                    setattr(swapped, attr, getattr(obs, attr))
            swapped.tokenized_prompt = obs.original_tokenized_prompt
            swapped.tokenized_prompt_mask = obs.original_tokenized_prompt_mask
            obs = swapped
        return self.compute_alignment_loss(obs, gradient_mode=GRAD_MODE_ASR)

    @torch.no_grad()
    def compute_teacher_velocity(self, observation, actions, noise, time):
        """Teacher forward: run text oracle (base weights only, LoRA bypassed).

        Uses original_tokenized_prompt (text) instead of audio, and sets
        gradient_mode=GRAD_MODE_BYPASS so all LoRA layers return base_linear(x).
        Returns the predicted velocity v_t (detached).
        """
        # Build a shallow copy of observation with text prompt restored and audio cleared
        class _TeacherObs:
            pass
        tobs = _TeacherObs()
        for attr in dir(observation):
            if not attr.startswith("_"):
                try:
                    setattr(tobs, attr, getattr(observation, attr))
                except Exception:
                    pass
        # Swap in original text prompt
        if getattr(observation, "original_tokenized_prompt", None) is not None:
            tobs.tokenized_prompt = observation.original_tokenized_prompt
            tobs.tokenized_prompt_mask = observation.original_tokenized_prompt_mask
        # Clear audio so embed_prefix uses text path only
        tobs.audio_mask = None
        tobs.audio_whisper_hidden = None
        if hasattr(tobs, "audio"):
            tobs.audio = None

        obs = self._preprocess_observation(tobs, train=True)
        images = list(obs.images.values())
        img_masks = list(obs.image_masks.values())

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, audio_position_mask = self.embed_prefix(
            images, img_masks, obs.tokenized_prompt, obs.tokenized_prompt_mask,
            audio_whisper_hidden=None, audio=None, audio_mask=None,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(obs.state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            audio_mask=None,
            gradient_mode=GRAD_MODE_BYPASS,
        )

        suffix_out = suffix_out[:, -self.config.action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t

    def forward(self, observation, actions, noise=None, time=None,
                training_stage=None, gradient_mode=None,
                return_intermediates=False) -> Tensor:
        """Training forward pass. Returns per-element MSE loss or ASR scalar loss.

        If return_intermediates=True, returns (loss, noise, time) so the caller
        can reuse the same noise/time for distillation teacher forward.
        """
        # Allow training_stage override from caller (training loop), else use config
        stage = training_stage or self.training_stage

        # Stage 1: ASR alignment only
        if stage == "asr_alignment":
            asr_loss = self.compute_alignment_loss(observation, gradient_mode=GRAD_MODE_ASR)
            batch_size = observation.state.shape[0] if hasattr(observation, "state") else actions.shape[0]
            return asr_loss.expand(batch_size, 1)

        # Default: flow matching loss
        obs = self._preprocess_observation(observation, train=True)
        images = list(obs.images.values())
        img_masks = list(obs.image_masks.values())

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Determine gradient_mode for split LoRA
        gm = GRAD_MODE_FLOW_MATCHING if self.audio_enabled else gradient_mode

        prefix_embs, prefix_pad_masks, prefix_att_masks, audio_position_mask = self.embed_prefix(
            images, img_masks, obs.tokenized_prompt, obs.tokenized_prompt_mask,
            audio_whisper_hidden=obs.audio_whisper_hidden, audio=getattr(obs, "audio", None),
            audio_mask=obs.audio_mask,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(obs.state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                audio_mask=audio_position_mask,
                gradient_mode=gm,
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        loss = F.mse_loss(u_t, v_t, reduction="none")
        if return_intermediates:
            return loss, noise, time
        return loss

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        obs = self._preprocess_observation(observation, train=False)
        images = list(obs.images.values())
        img_masks = list(obs.image_masks.values())
        state = obs.state

        prefix_embs, prefix_pad_masks, prefix_att_masks, audio_position_mask = self.embed_prefix(
            images, img_masks, obs.tokenized_prompt, obs.tokenized_prompt_mask,
            audio_whisper_hidden=getattr(obs, "audio_whisper_hidden", None),
            audio=getattr(obs, "audio", None),
            audio_mask=getattr(obs, "audio_mask", None),
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            audio_mask=audio_position_mask,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
