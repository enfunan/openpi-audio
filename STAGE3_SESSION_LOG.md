# Stage 3 Session Log

## What We Did This Session

### Starting Point
- Stage 2 (mixed ASR) complete at 15k steps, checkpoint at `mixed_asr/14999/params`
- Float32 LoRA fix in `lora.py` was written but not yet validated
- Stage 3 config `pi05_audio_stage3_libero` ready but untested

### Step 1: Validated Float32 LoRA Fix (text-only)

Ran 200-step test WITHOUT audio (`--data.tts-cache-dir` not set, `num_workers=2` default):
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_audio_stage3_libero \
  --num-train-steps=200 --no-wandb-enabled --exp-name=lora_f32_test --overwrite
```
**Result**: 200 steps clean, loss 0.085→0.082, grad_norm ~0.15. Float32 LoRA fix works for text-only.

### Step 2: Full Codebase Audit

Audited all modified files: `lora.py`, `pi0.py`, `whisper.py`, `config.py`, `train.py`, data pipeline, `gemma.py`, `pi0_config.py`. All passed.

### Step 3: Config Tuning (User-Directed)

Reviewed every change in `git diff`. User decisions:
- `warmup_steps`: Reverted 5000 → **1000** (NaN fix is in lora.py, not warmup)
- `Gemma LoRA LR`: Changed 0.3x → **0.5x** (user override: preserve Stage 2 audio knowledge)
- `log_interval`: Changed 10 → **50**
- Comments simplified

### Step 4: Eval Script Fix

Found train/eval mismatch in `examples/libero/main.py`: audio eval sent both audio AND text prompt, but training uses mutual exclusivity (prompt cleared when audio assigned).

**Fix** (line 198-200):
```python
if use_audio and audio_waveform is not None:
    element["audio"] = audio_waveform
    element["prompt"] = ""  # Match training mutual exclusivity
```

### Step 5: Pre-Flight Audit

Wrote `STAGE3_PREFLIGHT.md` covering 7 sections: loss formulation (CFM), data pipeline (60/40 audio/text, 40 tasks, 3.5 epochs), trainable components, warmup schedule, checkpoint loading, known risks, final config.

### Step 6: First Launch Attempt — NaN at Step 0

Launched with audio mixing. Immediate NaN.

**Root cause 1: `num_workers=2` data corruption**
- Default multi-worker loading with librosa/WhisperFeatureExtractor causes shared memory corruption
- Mel spectrograms had `max=3.675e+27` (deterministic corrupt value)
- **Fix**: `num_workers=0`

### Step 7: Second Attempt — NaN at Step 3

With `num_workers=0` + audio mixing: NaN at step 3.
```
Step 0: grad_norm=1.0032, loss=0.2176  ← clean (grad norms 10x higher than text-only)
Step 1: grad_norm=1.6441, loss=0.2820  ← clean
Step 2: grad_norm=1.4268, loss=0.2542  ← clean
Step 3: grad_norm=nan, loss=nan        ← dead
```

### Step 8: Diagnosing Forward vs Backward NaN

Added debug prints to `embed_prefix` in `pi0.py`. Found:
- Step 3: `img base_0_rgb: nan=True` (SigLIP output NaN)
- Step 3: `raw_audio: nan=False`, `text_tokens: nan=False`
- Step 2: `param_norm=2115.1724` (clean params entering step 3)

**Conclusion**: Forward pass NaN. Loss goes NaN before any backward pass at step 3.

### Step 9: Ruling Out Audio as the Cause

Key experiment: text-only with `num_workers=0` (same batch ordering as audio run):
```
Step 3: grad_norm=nan, loss=nan, param_norm=nan  ← SAME NaN!
```
**Audio is irrelevant.** The `num_workers=0` batch ordering puts a problematic batch at step 3 that triggers bfloat16 overflow in the forward pass. The earlier 200-step text-only test passed only because `num_workers=2` gave a different batch ordering.

### Step 10: Exhausted SigLIP Fixes

All previously tested and FAILED (from earlier sessions):
1. Float32 SigLIP (`dtype_mm="float32"`) — still NaN
2. Disabled remat (`remat_policy="everything_saveable"`) — still NaN
3. Stop gradient on SigLIP output — still NaN
4. Float32 softmax in SigLIP attention — still NaN

The bfloat16 forward pass overflow cannot be eliminated without changing the entire model to float32.

### Step 11: NaN Gradient Guard (The Fix)

Added to `train.py`:
```python
grad_norm = optax.global_norm(grads)
grad_finite = jnp.isfinite(grad_norm)

# ... compute updates normally ...

# Skip update if gradients are NaN/Inf
new_params = jax.tree.map(
    lambda new, old: jnp.where(grad_finite, new, old), new_params, params
)
new_opt_state = jax.tree.map(
    lambda new, old: jnp.where(grad_finite, new, old), new_opt_state, state.opt_state
)
```

Also added `nan_skipped` metric to training info for monitoring.

### Step 12: Full 30k Launch — SUCCESS

```bash
# Running in tmux session "stage3"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_audio_stage3_libero \
  --data.tts-cache-dir=/home/user1/workspace/VLA/data/tts/libero_train \
  --data.audio-ratio=0.6 \
  --no-wandb-enabled --exp-name=stage3_libero --overwrite
```

Training progress:
```
Step     0: loss=nan (skipped), param_norm=2115.1724   ← guard saved it
Step   100: loss=0.1394, nan_skipped=0.0
Step   500: loss=0.0658, nan_skipped=0.0
Step  1000: loss=~0.05,  nan_skipped=0.0
Step  5000: loss=~0.035, nan_skipped=0.0
Step 10000: loss=~0.03,  nan_skipped=0.0
Step 15000: loss=~0.027, nan_skipped=0.0
Step 20000: loss=~0.025, nan_skipped=0.0
```

- Only 1 NaN trigger (step 0) in 20k+ steps
- Loss converging: 0.14 → 0.025
- ~1.5s/step, ~13.5 hours total
- Checkpoints every 1,000 steps at `checkpoints/pi05_audio_stage3_libero/stage3_libero/`

## Files Modified (All Committed and Pushed)

### `src/openpi/models/lora.py`
Float32 LoRA computation in Einsum (attention) and FeedForward (MLP). Prevents bfloat16 gradient overflow through 18 Gemma transformer layers.

### `scripts/train.py`
- NaN gradient guard: skip param + optimizer update when `grad_norm` is NaN/Inf
- `nan_skipped` metric for monitoring
- Gradient scaling for `lr_scale_overrides` (pre-existing)

### `src/openpi/training/config.py`
Stage 3 config `pi05_audio_stage3_libero` (~line 978):
- `num_workers=0` (prevents librosa shared memory corruption)
- `warmup_steps=1000`, `peak_lr=2e-5`, `batch_size=32`, `num_train_steps=30000`
- Gemma LoRA at 0.5x LR, audio projector at 0.1x LR
- `log_interval=50`, `save_interval=1000`

### `examples/libero/main.py`
Clear text prompt when audio injected in eval (`element["prompt"] = ""`), matching training mutual exclusivity.

### Documentation
- `STAGE3_NAN_FIX.md` — Both root causes, both fixes, validation
- `STAGE3_PREFLIGHT.md` — 7-section pre-flight audit

## Git Commits
```
bbb3475 — NaN gradient guard + num_workers=0 + updated STAGE3_NAN_FIX.md
302924f — STAGE3_PREFLIGHT.md
916eda5 — Config tuning (Gemma LoRA 0.5x, log_interval=50) + eval fix
6cbafe1 — Float32 LoRA fix + original STAGE3_NAN_FIX.md
```
All pushed to `mine` remote (`git@github.com:enfunan/openpi-audio.git`).

## What To Do Next

### 1. Check if Training Finished
```bash
# Check tmux session
tmux attach -t stage3

# Or check log
grep "Step [0-9]" /tmp/stage3_libero.log | tail -5

# Check final checkpoint
ls checkpoints/pi05_audio_stage3_libero/stage3_libero/
```
Expected: step 29999 checkpoint, total ~13.5 hours.

### 2. Run Evaluation
```bash
# Start model server with final checkpoint
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/serve_policy.py pi05_audio_stage3_libero \
  --checkpoint-path=checkpoints/pi05_audio_stage3_libero/stage3_libero/29999/params

# In another terminal — run eval (text + audio modes)
.venv/bin/python examples/libero/main.py \
  --task-suite-name=libero_spatial \
  --num-trials-per-task=50 \
  --eval-mode=both \
  --audio-dir=/home/user1/workspace/VLA/data/tts/libero_eval
```
Primary metrics: libero_spatial + libero_object success rates.

### 3. Audio Ablation Diagnostic
Run `scripts/diag_audio_ignored.py` at checkpoints 5k, 15k, 30k.
Stage 2 baseline: +5.70 mean delta. If delta drops to ~0 → audio conditioning lost.

### 4. Risk: Overfitting
3.5 epochs over 273k frames. Loss flattened at ~0.025 since step 15k.
If eval performance doesn't improve past 15k checkpoint → consider using 15k checkpoint instead of 30k.

### 5. Risk: Audio Conditioning Degradation
Gemma LoRA at 0.5x LR will drift toward robot objectives.
Compare audio vs text eval success rates — they should be similar if audio conditioning is preserved.
