# π0.5 Architecture & Audio KV Cache Distillation

A detailed reference for the π0.5 model structure, inference flow, training,
and the audio extension via KV cache distillation.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Model Components](#2-model-components)
3. [Tensor Shapes & Dimensions](#3-tensor-shapes--dimensions)
4. [Inputs (Observation)](#4-inputs-observation)
5. [Prefix Embedding (Image + Language)](#5-prefix-embedding-image--language)
6. [Suffix Embedding (State + Actions + Timestep)](#6-suffix-embedding-state--actions--timestep)
7. [Joint Forward Pass (Training)](#7-joint-forward-pass-training)
8. [Inference (Action Sampling)](#8-inference-action-sampling)
9. [Flow Matching (Denoising ODE)](#9-flow-matching-denoising-ode)
10. [Attention Masking](#10-attention-masking)
11. [Audio Extension: KV Cache Distillation](#11-audio-extension-kv-cache-distillation)
12. [Split LoRA](#12-split-lora)
13. [Training Stages](#13-training-stages)
14. [File Map](#14-file-map)

---

## 1. High-Level Overview

π0.5 is a vision-language-action (VLA) model that generates robot actions
conditioned on camera images and a language instruction. It is built by
combining two pretrained LLMs:

```
┌─────────────────────────────────────────────────────────────┐
│  PaliGemma (3B VLM)          │  Action Expert (300M Gemma)  │
│  - SigLIP vision tower       │  - adaRMSNorm (timestep)     │
│  - Gemma 2B language model   │  - Gemma 300M decoder        │
│  - 18 layers, 2048 width     │  - 18 layers, 1024 width     │
│  - 8 heads, head_dim=256     │  - 8 heads, head_dim=256     │
│  - 1 KV head (GQA)           │  - 1 KV head (GQA)           │
└─────────────────────────────────────────────────────────────┘
```

At each transformer layer, both models' Q/K/V are concatenated and attend
jointly, then results are split back. This is the "dual-expert" architecture.

Key difference from π0 (non-0.5):
- **π0**: State is a continuous embedding in the suffix (fed to expert).
- **π0.5**: State is discretized into language tokens in the prefix. The
  action expert uses **adaRMSNorm** to inject the flow-matching timestep
  (rather than concatenating it with actions).

---

## 2. Model Components

### 2.1 PaliGemma (VLM backbone)

**`PaliGemmaForConditionalGeneration`** from HuggingFace, with patched Gemma
decoder layers (`transformers_replace/models/gemma/modeling_gemma.py`).

| Sub-module | Role |
|---|---|
| `vision_tower` (SigLIP) | Encodes 224×224 images → 256 patch embeddings each |
| `multi_modal_projector` | Linear projection from SigLIP dim → Gemma dim (2048) |
| `language_model` (Gemma 2B) | 18-layer transformer, processes prefix (images+text) |
| `embed_tokens` | Token embedding table (vocab=257,152) |

### 2.2 Action Expert (Gemma 300M)

**`GemmaForCausalLM`** with `model.embed_tokens = None` (no token embedding;
action embeddings are projected externally).

| Sub-module | Role |
|---|---|
| `model.layers` (×18) | Decoder layers with adaRMSNorm |
| adaRMSNorm | Injects timestep → (scale, shift, gate) modulation |

### 2.3 Action Projection MLPs

```python
action_in_proj:  Linear(32 → 1024)      # action_dim → expert_width
action_out_proj: Linear(1024 → 32)      # expert_width → action_dim
time_mlp_in:     Linear(1024 → 1024)    # timestep sinusoidal → adaRMS cond
time_mlp_out:    Linear(1024 → 1024)
```

### 2.4 Audio Components (our addition)

| Module | Shape | Role |
|---|---|---|
| Whisper encoder (frozen) | `(B, 1500, 1280)` | Audio feature extraction |
| Perceiver Resampler | `(B, 1500, 1280) → (B, 32, 2048)` | Compress to 32 audio tokens |
| Split LoRA (task + audio) | Applied to Q/K/V projections | Modality-specific adaptation |

---

## 3. Tensor Shapes & Dimensions

### Gemma 2B (PaliGemma language model)

| Param | Value |
|---|---|
| `width` (hidden_size) | 2048 |
| `depth` (num_hidden_layers) | 18 |
| `mlp_dim` (intermediate_size) | 16,384 |
| `num_heads` | 8 |
| `num_kv_heads` | 1 (GQA: grouped-query attention) |
| `head_dim` | 256 |
| vocab_size | 257,152 |

### Gemma 300M (Action Expert)

| Param | Value |
|---|---|
| `width` | 1024 |
| `depth` | 18 |
| `mlp_dim` | 4096 |
| `num_heads` | 8 |
| `num_kv_heads` | 1 |
| `head_dim` | 256 |

### SigLIP Vision Tower

| Param | Value |
|---|---|
| Input resolution | 224 × 224 |
| Patches | 14×14 = 196... projected to 256 tokens |
| Output dim | 2048 (after `multi_modal_projector`) |

### Sequence Layout (π0.5 default — LIBERO)

```
Prefix (PaliGemma):
  [img0: 256] [img1: 256] [img2: 256] [text: 200] = 968 tokens

Suffix (Action Expert):
  [actions: 10] = 10 tokens       (action_horizon=10 for LIBERO)

Total at training: 978 tokens
```

The 3 images correspond to:
- `base_0_rgb` (base camera)
- `left_wrist_0_rgb` (left wrist camera)
- `right_wrist_0_rgb` (right wrist camera)

Each 224×224 image → SigLIP → 256 patch tokens at dim 2048.

---

## 4. Inputs (Observation)

The model receives a `SimpleProcessedObservation` (defined in
`preprocessing_pytorch.py`):

```python
@dataclass
class SimpleProcessedObservation:
    images: dict[str, Tensor]        # {"base_0_rgb": (B,224,224,3), ...}
    image_masks: dict[str, Tensor]   # {"base_0_rgb": (B,) bool, ...}
    state: Tensor                    # (B, 32) robot joint state
    tokenized_prompt: Tensor         # (B, 200) int32 token IDs
    tokenized_prompt_mask: Tensor    # (B, 200) bool (True=valid, False=pad)
    token_ar_mask: Tensor            # (B, 200) autoregressive mask flags
    token_loss_mask: Tensor          # (B, 200) loss mask

    # Audio fields (None if not audio-enabled):
    audio: Tensor | None             # (B, 480000) raw waveform at 16kHz
    audio_whisper_hidden: Tensor | None  # (B, 1500, 1280) precomputed
    audio_mask: Tensor | None        # (B,) bool — True if audio present
```

### Image preprocessing

1. Images arrive as `(B, H, W, 3)` float32, range [0, 1]
2. Resized with padding to 224×224 if needed
3. Fed to SigLIP vision tower → 256 embeddings per image

### Language preprocessing

1. Text instruction tokenized with Gemma tokenizer (vocab 257,152)
2. Padded/truncated to `max_token_len` (200 for π0.5)
3. `tokenized_prompt_mask[i] = True` for real tokens, `False` for padding

### State

For π0.5, state is discretized and embedded as part of the language tokens
(the `discrete_state_input=True` path). The raw `state` tensor is only used
in the suffix for π0 (non-0.5).

---

## 5. Prefix Embedding (Image + Language)

**Method**: `PI0Pytorch.embed_prefix(images, img_masks, lang_tokens, lang_masks)`

```
Step 1: Embed images through SigLIP
    For each of 3 cameras:
        img (B, 224, 224, 3) → SigLIP → (B, 256, 2048)

Step 2: Embed language tokens
    lang_tokens (B, 200) → embed_tokens → (B, 200, 2048)
    Scale by √2048

Step 3: Concatenate
    prefix_embs = [img0, img1, img2, lang] → (B, 968, 2048)

Step 4: Build masks
    pad_masks:  (B, 968) bool — True for valid tokens
    att_masks:  (B, 968) — all zeros (full bidirectional attention within prefix)
```

The pad_masks are:
- Image positions: `img_mask[:, None].expand(B, 256)` — True if that camera is present
- Language positions: `lang_masks` — True for non-padding tokens

The att_masks are all 0 → prefix tokens attend bidirectionally to each other.

---

## 6. Suffix Embedding (State + Actions + Timestep)

**Method**: `PI0Pytorch.embed_suffix(state, noisy_actions, timestep)`

For π0.5:

```
Step 1: Sinusoidal timestep encoding
    timestep (B,) → sin/cos positional → (B, 1024)

Step 2: Timestep MLP (for adaRMSNorm conditioning)
    time_emb → Linear(1024,1024) → SiLU → Linear(1024,1024) → SiLU
    → adarms_cond (B, 1024)

Step 3: Project noisy actions
    x_t (B, 10, 32) → action_in_proj → (B, 10, 1024)
    This becomes action_time_emb (no concatenation with time for π0.5)

Step 4: Build masks
    pad_masks: (B, 10) all True
    att_masks: [1, 0, 0, ..., 0]  — causal: first action token starts new block
```

For π0 (non-0.5): state is projected and prepended, timestep is concatenated
with action embeddings.

The timestep is injected via **adaRMSNorm** in every action expert layer:
```
RMSNorm(x) → (1 + scale) * x + shift, gated by gate
where (scale, shift, gate) = Linear(adarms_cond)
```

---

## 7. Joint Forward Pass (Training)

**Method**: `PI0Pytorch.forward(observation, actions)`

```
1. Preprocess observation → images, masks, tokens, state
2. Sample noise ε ~ N(0,1), time t ~ Beta(1.5, 1.0)*0.999 + 0.001
3. Interpolate: x_t = t*ε + (1-t)*actions     (noisy actions)
4. Target:     u_t = ε - actions               (velocity field)

5. Build prefix: embed_prefix → (B, 968, 2048)
6. Build suffix: embed_suffix → (B, 10, 1024), adarms_cond

7. Cast to model dtype (bfloat16 typically)

8. Concatenate:
     full_embs:  [prefix: 968, suffix: 10]
     pad_masks:  [prefix_pad, suffix_pad]  → (B, 978)
     att_masks:  [prefix_att, suffix_att]  → (B, 978)

9. Build 2D attention mask → 4D attention mask
     position_ids = cumsum(pad_masks) - 1

10. Joint forward through PaliGemmaWithExpertModel:
     For each of 18 layers:
       a. LayerNorm both prefix and suffix hidden states
       b. Compute Q/K/V for PaliGemma (prefix) and Expert (suffix)
       c. Concatenate Q/K/V across sequence dimension
       d. Apply RoPE with shared position_ids
       e. Compute joint attention (all tokens see each other per mask)
       f. Split attention output back to prefix/suffix portions
       g. O projection, residual connection
       h. FFN with gated residual
       i. adaRMSNorm for Expert layers (timestep conditioning)
     Final norm for each model

11. Extract suffix output: last action_horizon positions → (B, 10, 1024)
12. Project to action space: action_out_proj → v_t (B, 10, 32)
13. Loss = MSE(u_t, v_t)  per-element
```

### The Dual-Expert Joint Attention

This is the key architectural insight. At each layer:

```
PaliGemma processes:  prefix tokens (968)
Action Expert processes: suffix tokens (10)

But attention is JOINT:
  Q = [Q_prefix ; Q_suffix]     (978 queries)
  K = [K_prefix ; K_suffix]     (978 keys)
  V = [V_prefix ; V_suffix]     (978 values)

  Attention = softmax(Q @ K^T / √d) @ V

Then split:
  prefix_out = Attention[:, :968, :]
  suffix_out = Attention[:, 968:, :]
```

This means action tokens can attend to image and language tokens (and vice
versa, subject to the attention mask). The two models share attention but
maintain separate FFN weights.

---

## 8. Inference (Action Sampling)

**Method**: `PI0Pytorch.sample_actions(device, observation, num_steps=10)`

Inference uses a two-phase approach:

### Phase 1: Prefix KV Cache (runs once)

```
1. Embed prefix: images + language → (B, 968, 2048)
2. Build attention mask and position_ids
3. Run PaliGemma-only forward with use_cache=True
   → DynamicCache with KV for all 18 layers
   Each layer: K,V shape = (B, 1, 968, 256)  [1 KV head, head_dim=256]
```

This KV cache encodes the scene (images) and instruction (text).
It is computed once and reused for all denoising steps.

### Phase 2: ODE Denoising Loop (runs num_steps times)

```
Initialize: x_t = noise ~ N(0,1),  shape (B, 10, 32)
            time = 1.0
            dt = -1/num_steps = -0.1

For step in range(num_steps):
    v_t = denoise_step(state, prefix_pad_masks, past_key_values, x_t, time)
    x_t = x_t + dt * v_t      (Euler step)
    time += dt                 (1.0 → 0.9 → ... → 0.0)

Return x_t  (the denoised actions)
```

### denoise_step detail

```
1. Embed suffix: state + x_t + timestep → (B, 10, 1024), adarms_cond
2. Build suffix attention mask:
   - suffix tokens can see all prefix tokens (via KV cache)
   - suffix tokens have causal attention among themselves
3. Run Action Expert-only forward:
   inputs_embeds = [None, suffix_embs]   ← prefix=None triggers suffix-only
   past_key_values = prefix KV cache     ← provides prefix context
4. Extract last action_horizon tokens → (B, 10, 1024)
5. Project: action_out_proj → v_t (B, 10, 32)
```

### Why two phases?

The prefix (968 tokens × 18 layers) is expensive. Computing its KV cache
once and reusing it across 10 denoising steps saves ~10× compute. The suffix
(10 tokens) is cheap to run each step.

---

## 9. Flow Matching (Denoising ODE)

π0.5 uses **Conditional Flow Matching** (not diffusion) for action generation.

### Training

The model learns a velocity field `v(x_t, t)` where:
- `x_t = t * noise + (1-t) * actions` — linear interpolation
- Target: `u_t = noise - actions` — the straight-line velocity
- Loss: `MSE(v_t, u_t)` — match the predicted velocity to the true velocity

Time `t` is sampled from `Beta(1.5, 1.0) * 0.999 + 0.001`, biased toward
`t ≈ 1` (noisier states) where the signal is harder to predict.

### Inference

Start from pure noise (`t=1`) and integrate the ODE backward to `t=0`:

```
dx/dt = v(x_t, t)

Euler integration: x_{t+dt} = x_t + dt * v(x_t, t)
```

With `num_steps=10`: `dt = -0.1`, stepping from `t=1.0` to `t=0.0`.

The final `x_0` is the predicted action trajectory: `(B, 10, 32)` =
10 future timesteps × 32 joint dimensions.

---

## 10. Attention Masking

### att_masks encoding

The `att_masks` tensor uses an integer encoding where:
- `0` = this token is in the same attention group as the previous token
- `1` = this token starts a new causal block

**cumsum trick**: `cumsum(att_masks)` gives group IDs. Token `i` can attend
to token `j` if `group[j] ≤ group[i]`.

### Prefix attention (all bidirectional)

```
att_masks = [0, 0, ..., 0]  (968 zeros)
                              ↓
cumsum =    [0, 0, ..., 0]   all same group → full attention
```

All prefix tokens (images + text) can attend to each other bidirectionally.

### Suffix attention (causal blocks)

```
att_masks = [1, 0, 0, ..., 0]   (1 followed by 9 zeros)
                                  ↓
cumsum =    [1, 1, 1, ..., 1]    all same group, but > prefix group
```

Suffix tokens can attend to all prefix tokens (group 0 ≤ 1) and to each
other. The `1` at the start prevents prefix tokens from attending to suffix.

### 2D → 4D mask construction

```python
att_2d_masks = (cumsum[:, None, :] <= cumsum[:, :, None]) & pad_masks_2d
# Shape: (B, total_seq, total_seq)

att_2d_masks_4d = att_2d_masks[:, None, :, :]
# Shape: (B, 1, total_seq, total_seq)
# Broadcast over attention heads

# Convert to additive mask:
mask = where(att_2d_masks_4d, 0.0, -2.3819763e38)
```

---

## 11. Audio Extension: KV Cache Distillation

### Motivation

The base model understands text instructions. We want to add speech/audio
input **without retraining from scratch**. The key insight: if the audio
pathway produces similar KV caches as the text pathway, the action expert
will generate similar actions.

### Architecture

```
TEACHER (frozen, text input):
  [img×3: 768] [text: 200] → PaliGemma → KV cache (teacher)
                                          No LoRA (GRAD_MODE_BYPASS)

STUDENT (trainable, audio input):
  [img×3: 768] [audio:32 + pad:168] → PaliGemma + LoRA → KV cache (student)
      ↑                                     ↑
  Same images                    Perceiver Resampler + LoRA
```

Both produce prefix KV caches of shape `(18 layers, B, 1 KV head, 968, 256)`.

### Audio Pipeline

```
Raw audio (30s, 16kHz) → Whisper-large-v3 encoder (frozen)
→ (B, 1500, 1280) Whisper frames

→ Perceiver Resampler:
    input_proj: Linear(1280 → 2048)
    Learned queries: (32, 2048)
    2× cross-attention layers:
        MultiheadAttention(2048, 8 heads) + LayerNorm + FFN(2048→8192→2048) + LayerNorm
→ (B, 32, 2048) audio tokens

→ Placed in text slot: [audio:32] [zero_pad:168]
→ Total prefix: [img:768] [audio+pad:200] = 968 tokens (same as teacher)
```

### embed_prefix_audio

```python
def embed_prefix_audio(images, img_masks, audio_whisper_hidden, audio_mask, text_slot_len):
    # Images: identical to teacher (3×256 = 768 tokens)
    # Audio tokens: perceiver_resampler(whisper_hidden) → (B, 32, 2048)
    # Pad to fill text slot: cat([audio_32, zeros_168]) → (B, 200, 2048)
    # Concatenate: [img:768, audio_slot:200] → (B, 968, 2048)
    # Returns: embs, pad_masks, att_masks, audio_position_mask
```

The `audio_position_mask` (B, 968) is True at positions 768..799 (the 32
audio token positions). This tells the LoRA module which positions should
use the audio LoRA branch vs the task LoRA branch.

### forward_prefix_get_kv

Runs PaliGemma prefix-only forward with `use_cache=True` to extract the
KV cache (HuggingFace `DynamicCache`):

```python
def forward_prefix_get_kv(prefix_embs, prefix_pad_masks, prefix_att_masks, ...):
    # Temporarily disable gradient_checkpointing (incompatible with use_cache)
    # Run paligemma_with_expert.forward(inputs_embeds=[prefix_embs, None], use_cache=True)
    # Returns: DynamicCache[layer_idx] → (K, V) each (B, 1, 968, 256)
```

### KV Distillation Loss

```python
compute_kv_distill_loss(teacher_kv, student_kv, ...)
```

**Image region** (positions 0–767): Per-position MSE.
Both teacher and student see the same images, but their KVs differ because
self-attention mixes in text vs audio context. We want these to match.

```
image_loss = MSE(teacher_kv[:, :, :768, :], student_kv[:, :, :768, :])
```

**Semantic region** (positions 768–967): Pooled MSE.
Teacher has ~10-40 valid text tokens; student has 32 audio tokens.
Per-position alignment is impossible (different content), so we pool:

```
teacher_pooled = masked_mean(teacher_kv[:, :, 768:, :], teacher_valid_mask)
student_pooled = masked_mean(student_kv[:, :, 768:, :], student_valid_mask)
semantic_loss = MSE(student_pooled, teacher_pooled)
```

**Total**: `loss = image_loss + α * semantic_loss`

---

## 12. Split LoRA

LoRA is applied to PaliGemma's linear projections (Q, K, V and optionally
O, gate, up, down). Each projection gets **two** LoRA branches:

```python
class LoRALinear:
    base_linear: nn.Linear          # Frozen base weights
    task_A: Linear(in, rank)        # Task LoRA down-projection
    task_B: Linear(rank, out)       # Task LoRA up-projection (init=0)
    audio_A: Linear(in, rank)       # Audio LoRA down-projection
    audio_B: Linear(rank, out)      # Audio LoRA up-projection (init=0)

def forward(x, audio_mask=None, gradient_mode=None):
    base_out = base_linear(x)                    # Always computed
    task_out = task_B(task_A(x_f32)) * scaling   # Task LoRA delta
    audio_out = audio_B(audio_A(x_f32)) * scaling # Audio LoRA delta

    # Position routing: audio positions use audio LoRA, others use task LoRA
    lora_out = where(audio_mask, audio_out, task_out)
    return base_out + lora_out
```

### Gradient modes

| Mode | Constant | Effect |
|---|---|---|
| None / NONE | 0 | Both branches receive gradients |
| FLOW_MATCHING | 1 | `audio_out.detach()` — only task LoRA trained |
| ASR | 2 | `task_out.detach()` — only audio LoRA trained |
| BYPASS | 3 | Skip LoRA entirely, return `base_out` only (teacher) |

### Numerical stability

All LoRA parameters (A, B matrices) are kept in **float32** even when the
base model runs in bfloat16. The input is cast to float32 before the LoRA
computation, and the output is cast back to the base dtype. This prevents
NaN gradients that occur with bfloat16 LoRA.

---

## 13. Training Stages

### Stage 1: KV Cache Alignment (`training_stage="kv_distill"`)

**Goal**: Teach the Perceiver Resampler and audio LoRA to produce KV caches
that match the text teacher.

**Trainable parameters**:
- Perceiver Resampler (~all params, LR = 50× base)
- PaliGemma LoRA (task_A, task_B, audio_A, audio_B on Q/K/V, LR = 1× base)

**Frozen**: Everything else (PaliGemma base, Action Expert, Whisper encoder)

**Training loop** (per step):
```
1. Teacher forward (no grad, GRAD_MODE_BYPASS):
   embed_prefix(images, text) → forward_prefix_get_kv → teacher_kv

2. Student forward (with grad):
   Whisper(audio) → Perceiver Resampler → embed_prefix_audio
   → forward_prefix_get_kv (LoRA active) → student_kv

3. Loss = compute_kv_distill_loss(teacher_kv, student_kv)
4. Backprop through student only
```

**Config**: `pi05_kv_distill_stage1`
- Batch size: 96
- LR: 1e-4 (cosine decay, 500 warmup)
- Steps: 10,000
- LoRA targets: Q, K, V only
- No EMA

### Stage 2: End-to-End Action Fine-tuning (`training_stage="kv_distill_stage2"`)

**Goal**: Fine-tune the full pipeline including Action Expert LoRA.

**Additional trainable parameters**:
- Action Expert LoRA (all projections)
- All PaliGemma projections (Q, K, V, O, gate, up, down)

**Loss**: `flow_matching_loss + β * kv_alignment_loss`

**Config**: `pi05_kv_distill_stage2`
- Batch size: 64
- LR: 5e-5
- Steps: 20,000
- LoRA targets: all projections
- EMA: 0.999

### Inference Modes

After training, the model supports two modes:

**Text mode** (original behavior):
```python
model.sample_actions(device, obs, audio_mode=False)
# LoRA is bypassed (GRAD_MODE_BYPASS) → exact same as base model
```

**Audio mode** (new capability):
```python
model.sample_actions(device, obs, audio_mode=True)
# Audio → Whisper → Perceiver Resampler → prefix with LoRA active
# Prefix KV cache computed once, then ODE denoising loop as usual
```

---

## 14. File Map

### Core model

| File | Description |
|---|---|
| `src/openpi/models/pi0_config.py` | Config dataclass (Pi0Config) |
| `src/openpi/models/gemma.py` | Gemma variant configs (dimensions, depths) |
| `src/openpi/models_pytorch/pi0_pytorch.py` | Main model: PI0Pytorch |
| `src/openpi/models_pytorch/gemma_pytorch.py` | PaliGemmaWithExpertModel (dual-expert) |
| `src/openpi/models_pytorch/preprocessing_pytorch.py` | Observation preprocessing |
| `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py` | Patched HF Gemma (adaRMSNorm, LoRA kwargs) |

### Audio extension (our additions)

| File | Description |
|---|---|
| `src/openpi/models_pytorch/perceiver_resampler.py` | Perceiver Resampler module |
| `src/openpi/models_pytorch/lora_pytorch.py` | Split LoRA (LoRALinear, inject_lora) |
| `src/openpi/models_pytorch/kv_distill_loss.py` | KV cache distillation loss |
| `scripts/train_kv_distill.py` | Stage 1 training loop |
| `scripts/smoke_test_kv_distill.py` | Smoke test (5/5 pass on CPU) |

### Training configs

| Config name | Stage | Key settings |
|---|---|---|
| `pi05_kv_distill_stage1` | KV alignment | bs=96, lr=1e-4, 10k steps, QKV LoRA |
| `pi05_kv_distill_stage2` | Action fine-tune | bs=64, lr=5e-5, 20k steps, all LoRA |

---

## Appendix: Dimension Quick Reference

```
B = batch size
S = sequence length (968 prefix + 10 suffix = 978 total)

PaliGemma hidden:    (B, S, 2048)
Action Expert hidden: (B, S, 1024)
Head dim:            256
Num attention heads: 8
Num KV heads:        1  (GQA)

Image tokens:        3 × 256 = 768
Text tokens:         200 (max_token_len for π0.5)
Audio tokens:        32 (audio_num_tokens)
Action horizon:      10 (for LIBERO)
Action dim:          32

Whisper frames:      1500 × 1280
Perceiver output:    32 × 2048

KV cache per layer:  (B, 1, 968, 256)  — [batch, kv_heads, seq, head_dim]
Total KV cache:      18 layers × 2 (K+V) × (B, 1, 968, 256)
```
