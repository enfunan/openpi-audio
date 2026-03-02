# Stage 3v2: Float32 Base Matmul Fix

## Problem

Stage 3 robot training needs to preserve Stage 2 audio conditioning (ablation delta +5.70). The solution is freezing both Gemma LoRA and audio projector at LR=0.0 via `lr_scale_overrides`. However, this produced persistent NaN in ~96% of batches — training was impossible.

### Why NaN Occurs at LR=0.0

The root cause is bfloat16 overflow in the Gemma forward pass. In `lora.py`, the base weight matmuls in both `Einsum` and `FeedForward._dot` operated entirely in bfloat16:

```python
# Einsum (line 57) — base matmul in bfloat16
result = jnp.einsum(eqn, x, self.w.astype(dtype))

# FeedForward._dot (line 148) — base matmul in bfloat16
base = jnp.dot(x, w.astype(x.dtype))
```

The MLP's 2048→16384→2048 expansion with GELU gating creates large intermediate values that lose precision in bfloat16, propagating to NaN through 18 transformer layers.

When any prefix-pathway parameter is trainable (e.g., projector at 0.1x LR), its updates shift activations out of the NaN-prone region. With everything frozen at LR=0.0, no parameter can change the prefix activations, trapping the model in the NaN region indefinitely.

### Previous Workarounds (Insufficient)

| Approach | Result |
|----------|--------|
| Projector at 0.1x LR | NaN eliminated, but audio delta dropped +5.70 → +1.83 in 2k steps (drift) |
| Projector at 0.01x LR | Still 96% NaN — too small to shift activations |
| `freeze_filter` for LoRA | Casts to bfloat16 + removes from backward pass → persistent NaN |

## Fix: Float32 Base Computation in `lora.py`

### Einsum.__call__

Compute base einsum in float32, matching the existing LoRA float32 pattern:

```python
def __call__(self, eqn: str, x):
    dtype = x.dtype
    # Compute base AND LoRA in float32 to prevent bfloat16 overflow
    # in the Gemma MLP/attention forward pass (18 layers of accumulation).
    x_f32 = x.astype(jnp.float32)
    result = jnp.einsum(eqn, x_f32, self.w.astype(jnp.float32))

    if config := self.lora_config:
        eqn_a, eqn_b = self._make_lora_eqns(eqn)
        lora = jnp.einsum(eqn_a, x_f32, self.w_a.astype(jnp.float32))
        lora = jnp.einsum(eqn_b, lora, self.w_b.astype(jnp.float32))
        result = result + lora * config.scaling_value

    return result.astype(dtype)
```

### FeedForward._dot

Compute base dot product in float32. Return float32 so intermediate ops (GELU, gate×ff1) stay in full precision:

```python
def _dot(self, x, w, lora_weights):
    x_f32 = x.astype(jnp.float32)
    base = jnp.dot(x_f32, w.astype(jnp.float32))
    if lora_weights is None:
        return base  # float32 — caller handles final dtype cast
    lora = jnp.dot(jnp.dot(x_f32, lora_weights[0].astype(jnp.float32)), lora_weights[1].astype(jnp.float32))
    return base + lora  # float32 — caller handles final dtype cast
```

### FeedForward.__call__

All intermediate operations now happen in float32. Cast back to bfloat16 only at the final output:

```python
def __call__(self, x):
    dtype = x.dtype
    # All _dot calls return float32; intermediate ops (GELU, gate*ff1) stay in float32.
    ff_gate = self._dot(x, self.w_gating[0], ...)   # float32
    gate_value = nn.gelu(ff_gate)                     # float32
    ff1 = self._dot(x, self.w_gating[1], ...)         # float32
    activations = gate_value * ff1                     # float32
    outputs = self._dot(activations, self.w_linear, self.w_linear_lora)  # float32
    return outputs.astype(dtype)  # back to bfloat16
```

### Config Change (`config.py`)

Audio projector LR changed from 0.01 to 0.0 (fully frozen):

```python
lr_scale_overrides={
    nnx_utils.PathRegex(".*audio_projector.*"): 0.0,        # was 0.01
    nnx_utils.PathRegex(r".*llm.*/attn/(q_einsum|kv_einsum|qkv_einsum|attn_vec_einsum)/lora_[ab]"): 0.0,
    nnx_utils.PathRegex(r".*llm.*/mlp/(gating_einsum|linear)_lora_[ab]"): 0.0,
},
```

## Verification

Training launched in tmux session `stage3v2` on 2026-03-02.

| Step | grad_norm | loss | nan_skipped | param_norm |
|------|-----------|------|-------------|------------|
| 0 | nan | nan | 1.0 | 2115.17 |
| 50 | nan | nan | 0.32 | 2115.17 |
| 100 | 1.04 | 0.2025 | **0.0** | 2115.17 |
| 150 | 0.99 | 0.1891 | **0.0** | 2115.17 |
| 200 | 0.73 | 0.1536 | **0.0** | 2115.17 |
| 250 | 0.49 | 0.1335 | **0.0** | 2115.17 |
| 300 | 0.34 | 0.1143 | **0.0** | 2115.17 |
| 350 | 0.28 | 0.1041 | **0.0** | 2115.17 |
| 400 | 0.27 | 0.0940 | **0.0** | 2115.17 |

- **NaN eliminated** from step 100 onward (0% rate vs 96% before)
- **Loss decreasing** steadily, tracking original Stage 3 curve
- **param_norm constant** at 2115.17 — confirms Gemma LoRA + audio projector truly frozen
- **Rate**: ~2.0s/it (~17h for 30k steps, ~20% slower than bfloat16 baseline of 13.5h)

## What's Already Float32 (Context)

These components already used float32 before this fix:
- Attention QK dot product (`gemma.py:217`)
- RMSNorm (`gemma.py:117`)
- LoRA operations (`lora.py:63-65` Einsum, `lora.py:152-153` FeedForward)

This fix extends float32 to the **base weight matmuls** — the only remaining bfloat16 compute in the Gemma forward pass.

## Files Modified

| File | Change |
|------|--------|
| `src/openpi/models/lora.py` | Float32 base computation in `Einsum.__call__` + `FeedForward._dot`/`__call__` |
| `src/openpi/training/config.py` | Audio projector LR: 0.01 → 0.0 |

## Next Steps

1. Monitor training through 30k steps (~22:00 UTC March 2)
2. Run audio ablation at step 2k — expect delta near +5.70 (Stage 2 level)
3. Full LIBERO eval at step 30k with text/audio/zero_audio modes
