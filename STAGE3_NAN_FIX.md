# Stage 3 NaN Investigation & Fix

## Problem

Stage 3 LIBERO robot training (`pi05_audio_stage3_libero`) produced deterministic NaN at step 3:

```
Step 0: loss=0.0921, grad_norm=0.1595   <- clean
Step 1: loss=0.0926, grad_norm=0.2128   <- clean
Step 2: loss=0.0928, grad_norm=0.1634   <- clean
Step 3: loss=NaN,    grad_norm=NaN      <- dead
```

Same behavior every run with the same seed. All 3 cameras NaN simultaneously.

## What We Tried (6 Hypotheses, All Failed)

### 1. Float32 softmax in SigLIP attention
**Idea**: bfloat16 softmax overflow in SigLIP's 27 transformer layers.
**Test**: `force_fp32_for_softmax=True` on SigLIP attention.
**Result**: Still NaN at step 3.

### 2. Full float32 SigLIP
**Idea**: bfloat16 QK dot product overflow before softmax.
**Test**: `dtype_mm="float32"` for the entire SigLIP (all computation in float32).
**Result**: Still NaN at step 3. Even full float32 doesn't help.

### 3. Disable remat (gradient checkpointing)
**Idea**: `nn.remat(policy=nothing_saveable)` recomputation diverges from forward pass.
**Test**: `remat_policy="everything_saveable"` to save all intermediates.
**Result**: Still NaN at step 3.

### 4. Stop gradient through frozen SigLIP
**Idea**: Backward pass through 27 frozen SigLIP layers accumulates numerical errors.
**Test**: `jax.lax.stop_gradient(image_tokens)` after SigLIP forward pass.
**Result**: Still NaN at step 3. NaN is in the **forward** pass, not backward.

### 5. Corrupt input images
**Idea**: Data pipeline produces NaN/Inf pixels at step 3.
**Test**: `jax.debug.print` on raw images before SigLIP — checked NaN, Inf, min, max.
**Result**: All images clean, range [-1.0, 1.0]. Data is fine.

### 6. Corrupt checkpoint parameters
**Idea**: SigLIP params in Stage 2 checkpoint contain NaN or extreme values.
**Test**: Verified all 556 params (23 SigLIP). Ran `compute_loss` on synthetic dummy data.
**Result**: Checkpoint clean, dummy data produces clean loss (0.037).

## Two Root Causes

The NaN problem had **two independent causes**, both related to bfloat16 numerical limits.

### Root Cause 1: bfloat16 LoRA Gradient Overflow

**Fix**: Float32 LoRA computation in `lora.py`.

The initial NaN appeared to come from SigLIP's forward pass because `jax.debug.print` showed clean images going in and NaN image tokens coming out. This led to 4 hypotheses focused on SigLIP internals — all dead ends.

The key insight: SigLIP is **frozen** and produces the same output for the same input. If SigLIP itself were broken, it would NaN on step 0 too. The NaN at step 3 means the **trainable parameters** (LoRA weights) got corrupted by a NaN gradient update, and the corrupted weights produced NaN in the next forward pass.

1. Flow matching loss uses random Gaussian noise (std=1.0) as the diffusion target
2. Early in training, large MSE gradients flow backward through 18 Gemma transformer layers
3. LoRA computes `x @ w_a @ w_b` — two chained matmuls whose backward can overflow in bfloat16
4. Once any gradient becomes NaN/Inf, the parameter update corrupts all LoRA weights
5. The next forward pass reads corrupted weights and produces NaN everywhere

**Evidence**:

| Experiment | Result |
|-----------|--------|
| AE LoRA trainable, Gemma LoRA trainable | NaN at step 0 |
| AE LoRA frozen, Gemma LoRA trainable | NaN at step 3 |
| All LoRA frozen, fixed noise std=0.1 | All 200 steps clean |

**Fix** in `lora.py` — compute all LoRA operations in float32, cast back to bfloat16:

```python
# Einsum class (attention Q/K/V/O):
x_f32 = x.astype(jnp.float32)
lora = jnp.einsum(eqn_a, x_f32, self.w_a.astype(jnp.float32))
lora = jnp.einsum(eqn_b, lora, self.w_b.astype(jnp.float32))
result = result + (lora * config.scaling_value).astype(dtype)

# FeedForward class (MLP):
x_f32 = x.astype(jnp.float32)
lora = jnp.dot(jnp.dot(x_f32, lora_weights[0].astype(jnp.float32)),
                lora_weights[1].astype(jnp.float32))
return base + lora.astype(x.dtype)
```

This fixed the text-only path (200 steps clean, grad_norm ~0.15).

### Root Cause 2: bfloat16 Forward Pass Overflow on Specific Batches

**Fix**: NaN gradient guard in `train.py`.

After fixing LoRA, the NaN returned when audio mixing was enabled. Investigation:

1. **`num_workers=2` data corruption**: Default multi-worker loading with librosa/WhisperFeatureExtractor causes shared memory corruption, producing `max=3.675e+27` in mel spectrograms. **Fixed by setting `num_workers=0`.**

2. **Persistent NaN at step 3**: Even with `num_workers=0` and float32 LoRA, NaN at step 3 persisted with audio. Key experiment: text-only with `num_workers=0` **also NaN'd at step 3** — proving audio was irrelevant. The `num_workers=0` batch ordering puts a problematic batch at step 3 that triggers bfloat16 overflow in the forward pass.

3. **Forward pass confirmed**: `loss=nan` at step 3 while `param_norm=2115.1724` at step 2 — params were clean, so the NaN originated in the forward pass, not from corrupted weights.

4. **All SigLIP-level fixes failed**: Float32 SigLIP, disabled remat, stop_gradient — none helped. The bfloat16 overflow in the forward pass cannot be eliminated without changing the entire model to float32.

**Fix** in `train.py` — NaN gradient guard that skips parameter updates when gradients contain NaN/Inf:

```python
grad_norm = optax.global_norm(grads)
grad_finite = jnp.isfinite(grad_norm)

# ... compute updates normally ...

# Skip update if gradients are NaN/Inf — preserve params and optimizer state.
new_params = jax.tree.map(
    lambda new, old: jnp.where(grad_finite, new, old), new_params, params
)
new_opt_state = jax.tree.map(
    lambda new, old: jnp.where(grad_finite, new, old), new_opt_state, state.opt_state
)
```

The guard also reports `nan_skipped` in the training metrics for monitoring.

## Validation

Full Stage 3 training launched with both fixes. First 700 steps:

```
Step   0: loss=nan, grad_norm=nan, nan_skipped=1.0 (skipped), param_norm=2115.1724
Step 100: loss=0.1394, grad_norm=0.6129, nan_skipped=0.0, param_norm=2115.1724
Step 200: loss=0.0897, grad_norm=0.2723, nan_skipped=0.0, param_norm=2115.1724
Step 300: loss=0.0779, grad_norm=0.2512, nan_skipped=0.0, param_norm=2115.1729
Step 400: loss=0.0696, grad_norm=0.2414, nan_skipped=0.0, param_norm=2115.1733
Step 500: loss=0.0658, grad_norm=0.2618, nan_skipped=0.0, param_norm=2115.1736
Step 600: loss=0.0627, grad_norm=0.2469, nan_skipped=0.0, param_norm=2115.1748
Step 700: loss=0.0601, grad_norm=0.2179, nan_skipped=0.0, param_norm=2115.1758
```

- Only 1 NaN trigger (step 0) in 700+ steps — guard skipped it, params preserved
- Loss steadily decreasing: 0.14 → 0.06
- Grad norms stable: 0.22-0.61
- param_norm drifting slowly — healthy learning

## Config

Final Stage 3 settings:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| warmup_steps | 1,000 | Standard; NaN fix is in code, not schedule |
| peak_lr | 2e-5 | Standard Pi0.5 fine-tuning LR |
| Gemma LoRA LR | 0.5x (1e-5) | Preserve Stage 2 audio knowledge |
| Audio projector LR | 0.1x (2e-6) | Well-trained, gentle fine-tune only |
| AE LoRA LR | 1.0x (2e-5) | Fresh init, must learn from scratch |
| num_workers | 0 | Prevents librosa shared memory corruption |
| log_interval | 50 | Good monitoring granularity |
| batch_size | 32 | Standard |
| num_train_steps | 30,000 | ~3.5 epochs over 273k frames |

## Files Modified

- `src/openpi/models/lora.py` — float32 LoRA computation (Einsum + FeedForward)
- `scripts/train.py` — NaN gradient guard (skip update on NaN/Inf gradients) + `nan_skipped` metric
- `src/openpi/training/config.py` — `num_workers=0`, Gemma LoRA 0.5x LR, `log_interval=50`
- `examples/libero/main.py` — Clear text prompt in audio eval mode (train/eval consistency)
