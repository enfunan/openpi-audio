# Stage 3 Pre-Flight Audit

## 1. Training Objective

**Loss**: Conditional Flow Matching (CFM) with linear optimal transport paths.

```
L = E_{t,x₀,x₁} [ ||v_θ(x_t, t, c) - u_t||² ]
```

- **x₀** = ground-truth actions (quantile-normalized), shape (B, 10, 32)
- **x₁ ~ N(0, I)** = isotropic Gaussian noise
- **x_t = t·x₁ + (1-t)·x₀** = linear interpolation (OT path)
- **u_t = x₁ - x₀** = target velocity (constant in t)
- **v_θ** = model's predicted velocity from action expert output projection
- **t ~ Beta(1.5, 1) · 0.999 + 0.001** ∈ [0.001, 1.0] — biased toward noisy states
- **c** = conditioning context (images, audio OR text, proprioceptive state)

Loss is **unweighted MSE** averaged over action_dim (32), action_horizon (10), and batch (32).

**Inference**: 10-step forward Euler ODE from t=1 (noise) to t=0 (clean), with KV cache reuse across denoising steps.

**Normalization**: Quantile normalization (`use_quantile_norm=True` for Pi0.5):
```
x_norm = (x - q01) / (q99 - q01) · 2 - 1
```
Maps 1st-99th percentile to [-1, +1]. Same `norm_stats.json` and quantile flag used at train and inference. **Consistent.**

**Actions**: 7-dim (6 joint deltas + 1 gripper), zero-padded to 32. Noise sampled over all 32 dims. For padding dims, u_t = noise (trivially learnable).

## 2. Data Pipeline

### Audio/Text Mixing

- **Ratio**: 60% audio / 40% text via `AudioTextMixingTransform(audio_ratio=0.6)`
- **Mutual exclusivity**: When audio assigned, `data["prompt"] = np.asarray("")` — model gets only audio for task understanding (VLAS design). Confirmed in `transforms.py` line 454.
- **CRITICAL**: `tts_cache_dir` is NOT set in the config (defaults to `""`). Must be passed via CLI: `--data.tts-cache-dir=...`. If omitted, training silently runs 100% text-only with no warning.

### TTS Coverage

- 112 instructions in manifest, all 40 LIBERO training tasks covered
- 20 speaker voices per instruction = 2,240 audio files
- If a prompt is not in manifest, sample silently falls back to text-only (never triggers for LIBERO)

### Dataset Statistics

- **Total frames**: 273,465 across 1,693 episodes
- **Total tasks**: 40 (libero_spatial + libero_object)
- **Task distribution**: 29-50 episodes per task (~1.7x range, mildly non-uniform)
- **Shuffling**: Per-frame (not per-episode) — each batch draws from diverse episodes/tasks
- **Steps per epoch**: 273,465 / 32 = 8,545
- **Epochs in 30k steps**: 3.51 — each frame seen ~3.5 times

## 3. Trainable Components

| Component | Init Source | LR Scale | Effective Peak LR | Rationale |
|-----------|-----------|----------|-------------------|-----------|
| **AE LoRA** (rank 32) | Fresh random (stddev=0.01) | 1.0x | 2e-5 | Fresh init, must learn robot actions from scratch |
| **Gemma LoRA** (rank 16) | Stage 2 checkpoint (15k ASR) | 0.5x | 1e-5 | Carries audio representations — adapt but preserve |
| **Audio projector** (3 layers + scale) | Stage 2 checkpoint | 0.1x | 2e-6 | Well-trained in Stages 1-2, gentle fine-tune only |
| **Action head** (in/out proj) | Stage 2 random init (never trained) | 1.0x | 2e-5 | Effectively fresh — never used in Stage 2 ASR path |
| **Time MLP** (2 layers) | Stage 2 random init (never trained) | 1.0x | 2e-5 | Same as action head — bypassed in Stage 2 |

### Key Insight

Action head and time MLP exist in the Stage 2 checkpoint but were **never used** during Stage 2 training (`compute_alignment_loss` bypasses `embed_suffix`). Their values are lecun_normal random init from Stage 2's RNG seed. Functionally equivalent to fresh random init.

### Gemma LoRA Regex

```
r".*/attn/(q_einsum|kv_einsum|attn_vec_einsum)/lora_[ab]|.*/mlp/.*lora.*"
```

Verified correct:
- Matches Gemma (expert 0): `q_einsum/lora_a`, `mlp/gating_einsum_lora_a`, etc.
- Does NOT match action expert (expert 1): `q_einsum_1/lora_a`, `mlp_1/gating_einsum_lora_a`
- Naming convention: `_name(name, i)` returns `name` for i=0, `f"{name}_{i}"` for i>0
- `PathRegex` uses `fullmatch` semantics

## 4. Warmup Schedule

Config: `warmup_steps=1,000`, linear warmup, cosine decay.

| Step | Base LR | AE LoRA (1.0x) | Gemma LoRA (0.5x) | Projector (0.1x) |
|------|---------|----------------|--------------------|--------------------|
| 0 | ~0 | ~0 | ~0 | ~0 |
| 100 | 2.0e-6 | 2.0e-6 | 1.0e-6 | 2.0e-7 |
| 500 | 1.0e-5 | 1.0e-5 | 5.0e-6 | 1.0e-6 |
| 1,000 | 2.0e-5 | 2.0e-5 | 1.0e-5 | 2.0e-6 |
| 5,000 | 1.9e-5 | 1.9e-5 | 9.6e-6 | 1.9e-6 |
| 15,000 | 1.1e-5 | 1.1e-5 | 5.7e-6 | 1.1e-6 |
| 29,999 | 2.0e-6 | 2.0e-6 | 1.0e-6 | 2.0e-7 |

Formula: `init_value = peak_lr / (warmup_steps + 1)`, linear ramp to peak, then cosine decay over remaining steps.

Previously `warmup_steps=5,000` was a defensive response to NaN. Reverted to 1,000 because the NaN fix is in `lora.py` (float32 LoRA), not the warmup schedule. 1,000 matches the Aloha Stage 3 config and official LIBERO configs.

## 5. Checkpoint and Weight Loading

### Load Flow

1. Stage 3 model shape computed (includes AE LoRA params not in Stage 2)
2. `CheckpointWeightLoader` loads Stage 2 checkpoint (`mixed_asr/14999/params`)
3. `_merge_params` with `missing_regex=".*lora.*|.*whisper_encoder.*|.*audio_projector.*|.*alignment_pooler.*"`:
   - Checkpoint keys present in model → loaded as np.ndarray
   - Missing keys matching regex → kept as ShapeDtypeStruct placeholders
4. `check_pytree_equality` validates complete tree structure
5. Placeholders stripped; only actual arrays passed to `init()`
6. Model created with fresh random init, then checkpoint arrays merged in
7. AE LoRA (not in checkpoint) stays at fresh random init

### Verification

- Stage 2 checkpoint verified present: 6.0 GB, Orbax format, step 14999
- No shape mismatch: `check_pytree_equality` catches any missing keys
- AE LoRA correctly fresh-initialized (10 params matched by `.*lora.*` regex)
- Validated by 200-step test run (loaded without errors)

## 6. Known Risks

### Risk 1: Overfitting (3.5 epochs)

960k frame presentations over 273k frames. Each frame seen ~3.5 times. Moderate but not extreme. LoRA constrains capacity (rank 16/32).

**Monitor**: Loss curve inflection. If loss keeps decreasing past 15k without eval improvement → overfitting.

### Risk 2: Audio conditioning degradation

Gemma LoRA at 0.5x LR will drift toward robot task objectives, potentially overwriting Stage 2 ASR alignment.

**Monitor**: Run audio ablation (`diag_audio_ignored.py`) at checkpoints 5k, 15k, 30k. Stage 2 baseline: +5.70 mean delta. If delta drops to ~0 → audio conditioning lost.

### Risk 3: Eval train/eval mismatch — FIXED

`examples/libero/main.py` in audio mode was sending both audio AND text prompt. Training uses mutual exclusivity (prompt cleared when audio assigned).

**Fix applied**: `element["prompt"] = ""` when audio is injected, matching `AudioTextMixingTransform` behavior.

### Risk 4: 40-task training set

LeRobot `physical-intelligence/libero` contains only 40 tasks (libero_spatial + libero_object). Eval on libero_goal or libero_10 tests zero-shot generalization, not fine-tuning.

**Primary metrics**: libero_spatial + libero_object success rates (trained tasks).

### Risk 5: No NaN early stopping

Training loop has no NaN detection. Float32 LoRA fix validated over 200 steps gives high confidence. `log_interval=50` provides monitoring.

## 7. Final Config

### Files Modified

**`src/openpi/models/lora.py`** — float32 LoRA in Einsum + FeedForward. **KEEP.** Core NaN fix, validated 200 steps.

**`src/openpi/training/config.py`** — 3 changes vs original:

| Change | Value | Verdict | Why |
|--------|-------|---------|-----|
| Gemma LoRA LR scale | 0.5x (new) | **KEEP** | Preserve Stage 2 audio knowledge. First knob to tune. |
| log_interval | 50 (was default 100) | **KEEP** | Better monitoring for new training stage. |
| Comments | Simplified | **KEEP** | Accurate, concise. |

Unchanged from original: `warmup_steps=1,000`, `peak_lr=2e-5`, `batch_size=32`, `num_train_steps=30,000`, `save_interval=1,000`.

**`examples/libero/main.py`** — Clear text prompt in audio eval mode. **KEEP.** Matches training mutual exclusivity.

### Launch Command

```bash
cd /home/user1/workspace/VLA/openpi && \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_audio_stage3_libero \
  --data.tts-cache-dir=/home/user1/workspace/VLA/data/tts/libero_train \
  --data.audio-ratio=0.6 \
  --no-wandb-enabled --exp-name=stage3_libero
```

### Estimated Runtime

- ~1.7s/step × 30,000 steps = ~14.2 hours + checkpoint overhead ≈ **14.5 hours**

### Monitoring Checkpoints

| Step | Check |
|------|-------|
| 50 | Loss not NaN, grad_norm stable |
| 1,000 | First checkpoint saved, loss < 0.08 |
| 5,000 | Audio ablation — delta should be > 0 |
| 15,000 | Mid-training eval (libero_spatial, 10 trials) |
| 30,000 | Final eval (libero_spatial + libero_object, 50 trials, text + audio modes) |
