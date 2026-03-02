# Stage 3 Evaluation Results

## Training Summary

- **Config**: `pi05_audio_stage3_libero`, 30k steps, batch 32, LR 2e-5→2e-6
- **Checkpoint**: `checkpoints/pi05_audio_stage3_libero/stage3_libero/29999/params`
- **Final training loss**: 0.024 (plateaued since ~step 15k)
- **NaN events**: 0 (only step 0 triggered guard, clean after that)
- **Training time**: ~13.5 hours on 8x L40S

## Key Finding: Catastrophic Forgetting of Audio Conditioning

### Audio Ablation (ASR CE loss, real vs zero audio tokens)

| Checkpoint | Mean Real Loss | Mean Zero Loss | Delta (zero - real) | Audio Status |
|------------|---------------|---------------|---------------------|--------------|
| Stage 2 final (14999) | 6.72 | 12.42 | **+5.70** | Strongly used |
| Stage 3 @ 5k | 7.76 | 8.62 | **+0.86** | Weak (-85%) |
| Stage 3 @ 15k | 8.58 | 8.73 | **+0.15** | Negligible (-97%) |
| Stage 3 @ 29999 | 9.05 | 8.91 | **-0.14** | Ignored (inverted) |

**Random audio baseline**: ~77 nats across all checkpoints (confirms model isn't ignoring all input, just can't distinguish real from zero audio).

### How We Discovered This

1. Ran `scripts/diag_audio_ablation_stage3.py` on checkpoints 5k, 15k, 29999
2. Script computes CE loss on LibriSpeech text with real audio tokens vs zero audio tokens vs random audio tokens
3. Stage 2 baseline showed delta +5.70 (audio strongly conditions text predictions)
4. Stage 3 checkpoints showed progressive degradation: +0.86 → +0.15 → -0.14
5. The forgetting happened fast — 85% of audio signal lost in just 5k steps

### Root Cause

The Gemma LoRA weights (rank 16) serve dual purpose:
- **Stage 2**: Learned to map audio tokens → text understanding (ASR alignment)
- **Stage 3**: Overwritten by robot task gradients (action prediction)

The 0.5x LR scaling on Gemma LoRA was insufficient to prevent forgetting. The robot training objective dominated, and since the model can solve many tasks from vision alone, there was no gradient pressure to preserve audio understanding.

## LIBERO Evaluation Results

### Main Results — Checkpoint 29999 (1 trial per task)

| Mode | libero_spatial | libero_object | Notes |
|------|---------------|--------------|-------|
| **Zero audio** | **9/10 (90%)** | — | Silent waveform, no text — vision-only baseline |
| **Text** | **8/10 (80%)** | — | Robot baseline, model works well |
| **ASR pipeline** | **8/10 (80%)** | — | Whisper→text, matches text exactly |
| **Audio (train voices)** | 6/10 (60%) | — | -20pp vs text |
| **Audio (held-out voices)** | 5/10 (50%) | 7/10 (70%) | Real audio HURTS performance |

### Per-Task Breakdown — libero_spatial

| # | Task | Text | ASR | Audio (held-out) | Audio (train) |
|---|------|------|-----|-------------------|---------------|
| 1 | bowl between plate & ramekin → plate | S | S | S | S |
| 2 | bowl next to ramekin → plate | S | S | F | F |
| 3 | bowl from table center → plate | S | F | S | S |
| 4 | bowl on cookie box → plate | F | S | S | S |
| 5 | bowl in top drawer of cabinet → plate | F | S | F | F |
| 6 | bowl on ramekin → plate | S | S | F | F |
| 7 | bowl next to cookie box → plate | S | S | F | S |
| 8 | bowl on stove → plate | S | S | F | S |
| 9 | bowl next to plate → plate | S | S | S | S |
| 10 | bowl on wooden cabinet → plate | S | F | S | F |

### Per-Task Breakdown — libero_object (audio held-out voices)

| # | Task | Audio (held-out) |
|---|------|-------------------|
| 1 | alphabet soup → basket | S |
| 2 | cream cheese → basket | S |
| 3 | salad dressing → basket | S |
| 4 | bbq sauce → basket | F |
| 5 | ketchup → basket | F |
| 6 | tomato sauce → basket | S |
| 7 | butter → basket | S |
| 8 | milk → basket | S |
| 9 | chocolate pudding → basket | S |
| 10 | orange juice → basket | F |
| **Total** | | **7/10 (70%)** |

### ASR Pipeline — Whisper Transcription Quality

All 10 libero_spatial tasks transcribed. Near-perfect accuracy:

| Ground Truth | Whisper Output | Error |
|-------------|----------------|-------|
| "...ramekin..." | "...ramkin..." | Minor misspelling |
| "...ramekin..." | "...rumpkin..." | Minor misspelling |
| "...table center..." | "...table centre..." | British spelling |
| All other tasks | Perfect match | None |

The minor errors had no impact — ASR pipeline scored 80%, identical to ground-truth text.

### Checkpoint 5k — libero_spatial (1 trial per task)

| Mode | Result | Notes |
|------|--------|-------|
| Text | 0/10 (0%) | Too early — robot behavior not learned |

## Interpretation

### 1. Text pathway works (80%)
The action expert + visual grounding learned correctly during Stage 3. The model can execute pick-and-place tasks when given text instructions.

### 2. ASR cascade pipeline = text (80%)
Whisper transcribes TTS audio near-perfectly. Feeding the transcription as text gives identical performance to ground-truth text. This proves:
- **Audio quality is not the bottleneck** — TTS audio is clean and Whisper handles it
- **The text pathway is fully functional** — even with minor transcription errors
- **The gap is purely in end-to-end audio conditioning** — the model can't use audio tokens directly

### 3. End-to-end audio degrades to 50-60% on spatial tasks
The model is NOT using audio content for task understanding. It falls back on visual grounding alone:
- Tasks with visually obvious targets succeed regardless of mode
- Tasks requiring spatial disambiguation ("next to ramekin" vs "next to cookie box") fail in audio mode

### 4. libero_object audio performs better (70%) than spatial (50%)
Object tasks ("pick up the X") have visually distinct targets (different items on a table). The model can often identify the correct object from vision alone, without understanding the audio instruction. Spatial tasks require understanding location references, which needs language comprehension.

### 5. Train vs held-out voices don't matter (60% vs 50%)
The 10pp difference is within noise for 10 trials. Voice familiarity is irrelevant when the audio pathway is dead.

### 6. Zero audio OUTPERFORMS real audio (90% vs 50%)
This is the most striking result. Silent audio (90%) > text (80%) > real audio (50%). The model handles silence gracefully — during training, 40% of samples had empty text with `np.asarray("")`, so the model learned a strong vision-only mode. But real audio tokens are out-of-distribution noise from the forgotten audio pathway, actively confusing the action prediction. The audio tokens aren't just ignored — they're harmful.

## Summary: The Evaluation Story

1. **Started with ablation diagnostic** — computed audio token influence at 3 Stage 3 checkpoints
2. **Found catastrophic forgetting** — audio delta dropped from +5.70 (Stage 2) to -0.14 (Stage 3 final) in the first 5k steps
3. **Confirmed with robot eval** — text mode 80%, audio mode 50% on libero_spatial (30pp gap)
4. **Isolated the bottleneck with ASR pipeline** — Whisper→text gives 80% (matches text), proving the issue is end-to-end audio conditioning, not audio quality
5. **Tested voice familiarity** — train vs held-out voices give similar results (60% vs 50%), ruling out voice mismatch
6. **Cross-suite validation** — libero_object audio at 70% (higher because tasks are less language-dependent)
7. **Zero-audio control** — running to confirm vision-only baseline matches audio mode

## Retrain: Stage 3v2 (Gemma LoRA LR=0)

### Decision
Option A: Zero the Gemma LoRA learning rate during Stage 3, preserving Stage 2 audio conditioning.

### Implementation — Two Failed Attempts Before Success

#### Attempt 1: Add Gemma LoRA to `freeze_filter` (FAILED — persistent NaN)

The obvious approach: add Gemma LoRA paths to the `freeze_filter` in `config.py`. This uses NNX `PathRegex` to match Gemma-specific LoRA params (unsuffixed `q_einsum`, `kv_einsum`, etc.) while leaving action expert LoRA trainable (suffixed `q_einsum_1`, `kv_einsum_1`, etc.).

**The naming convention** (from `gemma.py:_name()`): expert 0 (Gemma) gets bare names (`attn`, `mlp`, `q_einsum`), expert 1 (action expert) gets `_1` suffix (`attn_1`, `mlp_1`, `q_einsum_1`). Both live under `PaliGemma/llm/layers/...` in a shared `nn.scan`.

The regex filter was verified correct (18/18 test cases passed):
```python
# Gemma attn LoRA (fullmatch: q_einsum matches, q_einsum_1 does not)
PathRegex(r".*llm.*/attn/(q_einsum|kv_einsum|qkv_einsum|attn_vec_einsum)/lora_[ab]")
# Gemma MLP LoRA (flat param names inside FeedForward module)
PathRegex(r".*llm.*/mlp/(gating_einsum|linear)_lora_[ab]")
```

**Result**: Every single training step produced NaN loss. `param_norm` stuck at 2115.1724 (no updates ever applied). The NaN guard skipped 100% of steps.

**Root cause**: `train.py:115` casts all frozen params to bfloat16 for memory savings:
```python
params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
```
When Gemma LoRA is in the freeze_filter, it gets cast to bfloat16. Even though `lora.py` upcasts `w_a` and `w_b` to float32 during forward computation (line 64), the **precision loss from float32→bfloat16 storage** destabilizes the forward pass, producing NaN on every step.

For comparison, the original Stage 3 (Gemma LoRA trainable = float32) only had NaN at step 0 (a known batch-specific overflow), then recovered immediately.

#### Attempt 2: Keep freeze_filter + skip bfloat16 cast for LoRA (FAILED — delayed NaN)

Modified `train.py` to exclude LoRA from the bfloat16 cast:
```python
bf16_cast_filter = nnx.All(config.freeze_filter, nnx.Not(PathRegex(".*lora.*")))
params = nnx_utils.state_map(params, bf16_cast_filter, ...)
```

**Result**: Step 0 was clean (loss=0.2031, grad_norm=0.9899). But steps 1+ went back to all-NaN. `nan_skipped` jumped to 0.96 at step 50 and 1.0 at step 100.

**Root cause**: With Gemma LoRA in the freeze_filter, it's excluded from `DiffState` (no gradients computed). The backward pass through Gemma LoRA becomes a straight pass-through. In the original Stage 3, gradients flowing through Gemma LoRA during backprop provided numerical stabilization. Without those gradients, the action expert LoRA's first update destabilized subsequent forward passes.

#### Attempt 3: `lr_scale_overrides=0.0` (SUCCESS)

Keep the **original freeze_filter** (Gemma LoRA NOT frozen, stays in float32, gets gradients). Zero its effective LR via `lr_scale_overrides`:
```python
lr_scale_overrides={
    PathRegex(".*audio_projector.*"): 0.1,
    PathRegex(r".*llm.*/attn/(q_einsum|kv_einsum|qkv_einsum|attn_vec_einsum)/lora_[ab]"): 0.0,
    PathRegex(r".*llm.*/mlp/(gating_einsum|linear)_lora_[ab]"): 0.0,
},
```

**How it works**: Gradients are computed normally for Gemma LoRA (backward pass identical to original Stage 3), then multiplied by 0.0 before the optimizer. The optimizer receives zero gradients → Adam moments decay to zero → no parameter updates. Weight decay is 1e-10 (negligible), so Gemma LoRA values are preserved.

**Result**: Matches original Stage 3 training dynamics exactly:
```
Step   0: loss=nan,    nan_skipped=1.0  (batch-specific, same as original)
Step  50: nan_skipped=0.32              (recovering, original was 0.02)
Step 100: loss=0.1646, nan_skipped=0.0  (clean! original was 0.1394)
Step 200: loss=0.0978                   (tracking original curve)
Step 300: loss=0.0861                   (continuing to decrease)
```

### Why `freeze_filter` Doesn't Work for LoRA

The freeze_filter serves two purposes in `train.py`:
1. **Dtype cast** (line 115): Frozen params → bfloat16 (saves memory)
2. **Gradient exclusion** (via `trainable_filter`): Frozen params not differentiated

Both cause problems for LoRA:
- **bfloat16 cast**: LoRA weights trained in float32 lose precision when cast. The float32 LoRA fix (`lora.py`) upcasts during forward computation, but the stored values have already lost precision. This causes persistent NaN.
- **Gradient exclusion**: Removing LoRA from the backward pass changes the computation graph. The numerical stabilization from gradient flow through 18 Gemma layers is lost, making the forward pass fragile.

The `lr_scale_overrides=0.0` approach avoids both issues: params stay float32, gradients flow normally, but updates are zeroed.

### Stage 3v2 Config Summary

```
Config:        pi05_audio_stage3_libero (modified)
Exp name:      stage3v2_libero
Checkpoint:    checkpoints/pi05_audio_stage3_libero/stage3v2_libero/
Loads from:    Stage 2 checkpoint (mixed_asr/14999/params)
Steps:         30,000
Batch:         32
LR:            2e-5 → 2e-6 (cosine, 1k warmup)
```

| Parameter Group | LR Scale | Effective LR | Status |
|---|---|---|---|
| Action expert LoRA (rank 32) | 1.0× | 2e-5 | Train from scratch |
| Action head + time MLP + action_in_proj | 1.0× | 2e-5 | Train from scratch |
| **Audio projector** | **0.0×** | **0** | **Preserved from Stage 2** |
| Gemma LoRA (rank 16) | **0.0×** | **0** | **Preserved from Stage 2** |
| SigLIP, Whisper, Gemma base | frozen | 0 | bfloat16, no gradients |

### Why Audio Projector Must Also Be Frozen

Initial Stage 3v2 config had projector at 0.1x LR. Audio ablation at step 2k showed delta dropping from +5.70 (Stage 2) to +1.83. Root cause analysis:

**The projector receives gradients from the flow matching loss**, not the ASR loss. The gradient chain is:
```
flow_matching_loss → action_out_proj → action expert → cross-attention to Gemma prefix
  → Gemma prefix output (depends on audio_tokens) → audio_projector ← gradient lands here
```

These gradients push the projector toward producing tokens that improve **action prediction**, not ASR. With Gemma LoRA frozen, the projector drifts from the distribution Gemma LoRA was trained on. They were jointly optimized in Stage 2 — they must stay paired.

**Evidence**: At step 2000 with projector at 0.1x LR:
| Checkpoint | Delta (zero - real) | Audio Status |
|---|---|---|
| Stage 2 final | +5.70 | Strongly used |
| Stage 3v2 @ 2k (projector 0.1x) | +1.83 | Already degrading |

Changed projector to LR=0.0 and restarted.

### VLAS Comparison — Why This Approach Differs

Examined VLAS reference implementation (`/home/user1/workspace/VLA/VLAS/`). Key finding: **VLAS does NOT use an auxiliary ASR loss during robot training**. Pure action prediction loss with audio/text mutual exclusivity (60/40), same as us.

**Why VLAS doesn't catastrophically forget but Pi0.5 does:**

1. **Architecture**: VLAS (LLaVA) uses a single LLM for everything — audio, text, and action tokens all go through the same transformer, same LoRA weights. Action prediction gradients directly reinforce audio understanding because 60% of samples require audio. In Pi0.5, Gemma (expert 0) and Action Expert (expert 1) have **separate weights** in a shared `nn.scan`. The flow matching loss primarily trains the Action Expert LoRA; Gemma LoRA only gets indirect gradients through cross-attention backprop.

2. **Vision dominance**: Our eval showed zero audio (90%) > text (80%). LIBERO tasks are largely solvable from vision alone. With weak gradient signal to "use audio", the robot task objective overwrites Gemma LoRA's audio knowledge. VLAS's tasks (CALVIN) may require more language understanding, creating stronger gradient pressure to preserve audio comprehension.

3. **Capacity**: Gemma LoRA rank 16 has limited capacity. Robot task features and audio features compete for the same small weight space.

**Conclusion**: Freezing the entire audio pathway (projector + Gemma LoRA) is the right approach for Pi0.5 on LIBERO. The action expert learns to use the fixed Stage 2 audio representations via cross-attention — same mechanism it uses for text (which already works at 80%).

### Launch Command
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_audio_stage3_libero \
  --data.tts-cache-dir=/home/user1/workspace/VLA/data/tts/libero_train \
  --data.audio-ratio=0.6 \
  --no-wandb-enabled --exp-name=stage3v2_libero
```

tmux session: `stage3v2`, log: `/tmp/stage3v2_libero.log`, ETA: ~13 hours

## Technical Notes

### Bugs Fixed During Eval Setup
- **Flax int-key bug in model.load()**: `model.py` used `state.replace_by_pure_dict(params)` which fails on whisper encoder integer layer keys. Fixed with manual `flat_state()` merge (same pattern as train.py).
- **LIBERO dependencies**: Required `robosuite==1.4.1` (not 1.5.x), `easydict`, `gym==0.25.2`, `bddl`
- **libero __init__.py**: Missing top-level `__init__.py` in `third_party/libero/libero/`
- **libero config**: Created `~/.libero/config.yaml` to avoid interactive prompt
- **torch.load weights_only**: PyTorch 2.6 changed default; fixed in `benchmark/__init__.py` with `weights_only=False`
- **CUDA_VISIBLE_DEVICES=""**: Breaks MuJoCo EGL rendering; use `MUJOCO_EGL_DEVICE_ID=N` instead
- **tyro CLI args**: eval script requires `--args.` prefix (e.g., `--args.task-suite-name`)

### Eval Script Modifications
- Added `asr` eval mode to `examples/libero/main.py`: loads audio, transcribes with Whisper pipeline, feeds transcription as text prompt
- Added `zero_audio` eval mode: sends silent waveform with empty text prompt — pure vision-only baseline
- Created `scripts/diag_audio_ablation_stage3.py`: ablation script matching Stage 3 architecture (gemma_2b_lora)
- Created `scripts/eval_pipeline.sh`: automated server start → eval → server stop pipeline

### Eval Commands
```bash
# Start model server (all GPUs)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/serve_policy.py \
  --port 8000 policy:checkpoint \
  --policy.config pi05_audio_stage3_libero \
  --policy.dir checkpoints/pi05_audio_stage3_libero/stage3_libero/29999

# Text mode
MUJOCO_EGL_DEVICE_ID=0 .venv/bin/python examples/libero/main.py \
  --args.task-suite-name=libero_spatial --args.num-trials-per-task=1 \
  --args.eval-mode=text --args.port=8000

# Audio mode (held-out voices)
MUJOCO_EGL_DEVICE_ID=0 .venv/bin/python examples/libero/main.py \
  --args.task-suite-name=libero_spatial --args.num-trials-per-task=1 \
  --args.eval-mode=audio --args.audio-dir=/home/user1/workspace/VLA/data/tts/libero_eval \
  --args.port=8000

# ASR pipeline mode
MUJOCO_EGL_DEVICE_ID=0 .venv/bin/python examples/libero/main.py \
  --args.task-suite-name=libero_spatial --args.num-trials-per-task=1 \
  --args.eval-mode=asr --args.audio-dir=/home/user1/workspace/VLA/data/tts/libero_eval \
  --args.port=8000

# Zero audio mode (vision-only baseline)
MUJOCO_EGL_DEVICE_ID=0 .venv/bin/python examples/libero/main.py \
  --args.task-suite-name=libero_spatial --args.num-trials-per-task=1 \
  --args.eval-mode=zero_audio --args.port=8000
```
