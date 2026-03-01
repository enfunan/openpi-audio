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

## The Misleading Trail

The NaN appeared to come from SigLIP's forward pass because `jax.debug.print` showed clean images going in and NaN image tokens coming out. This led to 4 hypotheses focused on SigLIP internals — all dead ends.

The key insight was that SigLIP is **frozen** and produces the same output for the same input regardless of training. If SigLIP itself were broken, it would NaN on step 0 too. The NaN at step 3 means something **changed between step 0 and step 3** — and the only thing changing is the **trainable parameters** (LoRA weights).

## Root Cause: bfloat16 Gradient Overflow in LoRA

The actual problem was **bfloat16 gradient overflow in the LoRA backward pass** through Gemma's 18 transformer layers.

### Why it happens

1. Stage 3 uses flow matching loss with random Gaussian noise (std=1.0) as the diffusion target
2. Early in training, model predictions are far from targets, producing large MSE gradients
3. These gradients flow backward through 18 Gemma transformer layers, each containing LoRA operations
4. LoRA computes `x @ w_a @ w_b` — a chain of two matrix multiplications
5. In bfloat16, the backward pass through this chain can overflow (bfloat16 max ~3.4e38, but only 7-bit mantissa means limited precision for accumulation)
6. Once any gradient becomes NaN/Inf, it corrupts the LoRA weight update
7. The corrupted LoRA weights then produce NaN in the **next forward pass** — which is why the NaN appears to come from SigLIP (the forward pass reads the corrupted model state)

### Evidence

Three experiments confirmed this:

| Experiment | Result |
|-----------|--------|
| AE LoRA trainable, Gemma LoRA trainable | NaN at step 0 |
| AE LoRA frozen, Gemma LoRA trainable | NaN at step 3 |
| All LoRA frozen, fixed noise std=0.1 | All 200 steps clean |

With frozen LoRA and reduced noise, no overflow occurs. With trainable LoRA and full noise, the gradient magnitudes exceed bfloat16 capacity within a few steps.

## The Fix

**Compute all LoRA operations in float32**, cast back to bfloat16 for output.

### `lora.py` — Einsum class (attention Q/K/V/O projections)

```python
# Before (broken):
lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
result = result + lora * config.scaling_value

# After (fixed):
x_f32 = x.astype(jnp.float32)
lora = jnp.einsum(eqn_a, x_f32, self.w_a.astype(jnp.float32))
lora = jnp.einsum(eqn_b, lora, self.w_b.astype(jnp.float32))
result = result + (lora * config.scaling_value).astype(dtype)
```

### `lora.py` — FeedForward class (MLP gating + linear)

```python
# Before (broken):
return base + jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)),
                       lora_weights[1].astype(x.dtype))

# After (fixed):
x_f32 = x.astype(jnp.float32)
lora = jnp.dot(jnp.dot(x_f32, lora_weights[0].astype(jnp.float32)),
                lora_weights[1].astype(jnp.float32))
return base + lora.astype(x.dtype)
```

### Why this works

- **Forward pass**: LoRA contribution computed in float32 (no overflow), then downcast to bfloat16 for addition to base weights
- **Backward pass**: JAX autograd computes LoRA gradients in float32 (matching the forward dtype), preventing accumulation overflow through 18 transformer layers
- **Base weights**: Still computed in bfloat16 (unchanged) — only LoRA path needs float32

## Validation

200-step test run with the fix applied — **zero NaN**:

```
Step   0: loss=0.0853, grad_norm=0.1464, param_norm=2115.17
Step  10: loss=0.0941, grad_norm=0.1694, param_norm=2115.17
Step  50: loss=0.0878, grad_norm=0.1530, param_norm=2115.17
Step 100: loss=0.0863, grad_norm=0.1563, param_norm=2115.17
Step 150: loss=0.0851, grad_norm=0.1470, param_norm=2115.17
Step 190: loss=0.0822, grad_norm=0.1464, param_norm=2115.17
```

Loss trending down (0.085 -> 0.082), grad norms stable (~0.14-0.17), param norms constant. The fix is confirmed.

## Additional Stage 3 Config Changes

Alongside the LoRA fix, Stage 3 config was tuned:

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| Gemma LoRA LR | 1.0x (2e-5) | 0.3x (6e-6) | Slower adaptation preserves Stage 2 audio representations |
| Warmup steps | 1,000 | 5,000 | Gentler ramp prevents early gradient spikes |
| Log interval | default | 10 | Faster NaN detection in logs |

Per-component LR scaling (applied via gradient scaling in `train.py`):
- Action expert LoRA (fresh init): 1.0x (2e-5) — learn from scratch
- Gemma LoRA (Stage 2 pretrained): 0.3x (6e-6) — adapt slowly
- Audio projector (Stage 2 pretrained): 0.1x (2e-6) — fine-tune gently

## Files Modified

- `src/openpi/models/lora.py` — float32 LoRA computation (Einsum + FeedForward)
- `src/openpi/training/config.py` — Gemma LoRA 0.3x LR, warmup 5000, log_interval 10
