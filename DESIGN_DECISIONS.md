# Design Decisions: Audio-Conditioned Pi0.5

This document records the reasoning, failed experiments, and architectural choices behind our audio-conditioned Pi0.5 pipeline. It serves as a technical journal for anyone continuing this work.

## 1. Starting Point: Why Add Audio to Pi0.5?

Pi0.5 is a powerful vision-language-action (VLA) model that takes text + images and produces robot actions via flow matching. But text input requires typing or a separate ASR system. We wanted **end-to-end speech-to-action**: speak a command, the robot executes it.

[VLAS (CVPR)](https://github.com/robopen/VLAS) showed this is possible — they added a Whisper encoder to LLaVA/LLaMA and trained a 3-stage pipeline (embedding alignment → ASR fine-tune → robot task). We aimed to replicate this approach on Pi0.5.

## 2. The VLAS Recipe and Why It Didn't Transfer Directly

### VLAS Architecture (reference)
```
Audio → Whisper → Projector → LLaMA → Robot Actions
Images → CLIP → LLaMA
```

VLAS uses **pure LLaMA** as the backbone — a text-only model that has never seen multimodal input. When VLAS feeds audio projector tokens into LLaMA's prefix positions, LLaMA has no prior expectation about what prefix embeddings should look like. Any reasonable distribution can work because LLaMA's attention layers will learn to extract information from whatever is there.

### Pi0.5 Architecture (our base)
```
Audio → Whisper → Projector → Gemma (PaliGemma) → shared attn → Action Expert → Actions
Images → SigLIP → Gemma (PaliGemma)
```

Pi0.5 uses **PaliGemma's Gemma** — a Gemma model that has been specifically fine-tuned to process SigLIP image tokens in prefix positions. This is the critical difference. PaliGemma's Gemma has learned specific attention patterns for processing visual tokens with particular statistical properties (norms, distribution shape). It has strong priors about what prefix tokens should look like.

### The Consequence

When we feed randomly-initialized audio projector tokens into PaliGemma's Gemma, the tokens have completely wrong statistics compared to what Gemma expects. Gemma's learned attention patterns route around these "alien" tokens — it effectively ignores them and relies on text context alone.

This is the fundamental challenge that shaped our entire multi-stage approach.

## 3. Stage 1 v1: ASR Cross-Entropy (Failed)

### What We Tried

Following VLAS exactly: feed audio tokens through frozen Gemma, predict transcription text via cross-entropy loss. Only the audio projector is trainable.

```
Audio → Whisper (frozen) → Projector (train) → Gemma (frozen) → predict text (CE loss)
```

### Training Results
- 5000 steps, batch 64, LR 1e-3 → 1e-5
- Loss: 62.77 → 5.89 (converged)
- Looked good on paper.

### Diagnostic Findings (the collapse)

We wrote diagnostic scripts to understand what was actually happening:

1. **Audio IS statistically used**: Substituting zero or random audio for real audio changes the loss (real=5.86, zero=10.59, random=82.2). So the projector learned *something*.

2. **But the projector collapsed to near-constant output**: Token norms were ~1371 (vs ~45 for text embeddings), with negligible variation across different audio samples. The projector found a single "good" embedding that slightly helps all samples.

3. **Greedy decoding is identical for all samples**: Every audio sample produces the same text output ("and the other two men who were in the house..."). The projector output shifts probability mass slightly but never changes the argmax prediction.

### Root Cause Analysis

The frozen Gemma is the bottleneck. Here's what happens at each position:

- **Position 299 (last audio → BOS)**: The BOS token after audio is trivially correct (p=1.0). No gradient signal flows to the audio projector from this position.
- **Position 300 (BOS → first word)**: The top-1 prediction is always "and" (p~0.13) regardless of audio content. The ground truth word has p~0.00-0.03. Audio provides a tiny probability shift but never enough to change the argmax.
- **Later positions**: Decent language modeling from text context alone, audio barely contributes.

The frozen Gemma simply cannot extract meaningful information from tokens that don't match its training distribution. The projector gets gradient signal that says "shift probabilities slightly" but never "change the prediction" — so it converges to a single point that marginally helps all samples.

### Key Insight

VLAS works because LLaMA has no multimodal fine-tuning bias. PaliGemma's Gemma has a strong bias toward SigLIP-like tokens. **We need to match Gemma's expected input distribution before we can train through it.**

## 4. Stage 1 v2: Direct Embedding Alignment via MSE (Current Approach)

### The Idea

Skip Gemma entirely. Instead, directly align audio projector output to Gemma's text embeddings using MSE loss. Since text embeddings (`embed_table[tokens] * sqrt(2048)`) are what Gemma already understands, making audio tokens match this distribution means they'll be processable as prefix tokens.

```
Audio → Whisper (frozen) → Projector (train) ─┐
                                                ├── MSE loss
Text → Gemma embed table (frozen) ─────────────┘
```

### Why MSE Instead of Other Losses?

- **Cosine similarity loss**: Would match direction but not scale. Audio tokens at norm ~1371 pointing in the right direction would still confuse Gemma's attention (which expects norm ~45).
- **Contrastive loss (InfoNCE)**: Would learn to distinguish samples but not necessarily match the exact distribution Gemma expects. Also more complex to tune.
- **MSE on mean-pooled representations**: Matches both direction AND scale. Forces the projector to output tokens in exactly the range Gemma expects. Simple, direct gradient signal.

### Why Mean-Pool?

Audio has 300 tokens, text has variable length (5-50 tokens typically). We can't do token-level alignment because there's no correspondence between audio frame 47 and text token 3. Mean-pooling gives us a single "summary" vector for each modality that captures the overall content, and sequence length mismatch doesn't matter.

### Training Results (Stage 1 v2)
- Loss drops fast: 28.15 → 0.40 in 100 steps, plateaus around 0.20-0.22
- This plateau is expected — the MSE can't go to zero because audio and text are fundamentally different modalities encoding the same content differently

### Diagnostic Results: Partial Success, But Collapse

After training completed (5000 steps, final loss ~0.21), diagnostics revealed:

**Good**: Cosine similarity between audio and text mean-pools was 0.95-0.97. The projector learned the right *direction*.

**Bad**: All audio samples produced **identical** embeddings:
- Audio-audio pairwise cosine similarity: **1.000** for all sample pairs
- Greedy decode: identical garbage for every sample
- Audio token norms: ~100 (vs text ~240) — 2.4x gap

**Root cause**: Mean-pooled MSE has a trivial solution. The projector found a single "centroid" direction that minimizes average MSE across all training samples. Since Gemma text embeddings are extremely similar to each other (pairwise cosine > 0.995 — see below), a single point achieves low MSE against all targets.

### Attempted Fix: InfoNCE Contrastive Loss

To force per-sample discrimination, we added InfoNCE (contrastive) loss alongside MSE:
```
L = MSE(audio_mean, text_mean) + InfoNCE(audio_mean, text_mean, temperature=0.07)
```

**Result**: Complete failure. InfoNCE loss stuck at log(64) = 4.16 (random chance for batch_size=64) for the entire training run.

**Investigation**: We measured pairwise cosine similarity between Gemma's text embeddings across a batch of 20 LibriSpeech transcriptions:
- Mean pairwise cosine: **0.9957**
- 166 of 190 pairs had cosine > 0.99
- Max cosine: 0.9990

**Verdict**: Gemma's text embedding space is so concentrated that all transcriptions are nearly identical when mean-pooled. Contrastive learning fundamentally cannot work when positives and negatives are indistinguishable (cosine > 0.995). This is not a hyperparameter issue — it's a property of the embedding space.

This makes sense: Gemma's embedding table maps tokens to a high-dimensional space where the information is in the *sequence* of tokens, not the *average* of their embeddings. Mean-pooling collapses sequence-level information into a single point.

### Resolution: Learnable Output Scale + Proceed to Stage 2

Since Stage 1 v2 successfully learned the right *direction* but wrong *scale*, and per-sample discrimination requires sequence-level processing (which only Gemma can do), we:

1. **Added a learnable `output_scale` parameter** to `DownsampleAudioProjector` (initialized to 2.4) to close the norm gap (audio ~100 → ~240 to match text ~240)
2. **Proceeded to Stage 2** where ASR cross-entropy through Gemma+LoRA provides the rich, per-token gradient signal needed for per-sample discrimination

The InfoNCE code was reverted. The loss function remains pure MSE.

### Why This Partial Success Is Enough

Stage 1 v2 achieved its real goal: **get the projector output into a distribution Gemma can process**. Even though all samples produce similar embeddings, those embeddings are in the right neighborhood of Gemma's embedding space. This is dramatically better than Stage 1 v1 where norms were ~1371 and completely outside Gemma's expected range.

Stage 2's ASR CE loss provides per-token gradient through Gemma+LoRA, which will teach both:
- The projector to produce *different* embeddings for different audio (per-sample discrimination)
- Gemma LoRA to extract and decode the audio information into text

## 5. Stage 2: ASR Fine-Tune with LoRA

### Why Not Full Fine-Tune?

VLAS fully unfreezes LLaMA in their Stage 2. We considered this but chose LoRA instead for several reasons:

1. **Catastrophic forgetting risk**: PaliGemma's Gemma was carefully fine-tuned for image understanding. Full fine-tuning on ASR-only data (no images) would likely degrade its image processing capabilities. We need both modalities to work in Stage 3.

2. **Data imbalance**: We have 97k LibriSpeech utterances but no paired image data for Stage 2. Full fine-tuning with this imbalanced data would shift Gemma's weights away from visual understanding.

3. **Parameter efficiency**: LoRA (rank 16) adds only ~6.5M trainable parameters vs modifying the full ~2B. This constrains the adaptation to a low-rank subspace, preserving most of Gemma's original capabilities.

4. **VLAS used pure LLaMA**: VLAS could afford full fine-tuning because LLaMA had no multimodal capabilities to preserve. We can't.

### LoRA Configuration
- **Rank 16, applied to attention + feedforward** in all Gemma layers
- **Both projector + LoRA are trainable**: The projector continues adapting (it was pre-aligned in Stage 1 but can be refined), and LoRA teaches Gemma's attention to extract semantic content from audio tokens
- **No EMA**: EMA would slow down LoRA parameter updates. Since LoRA params start near-zero (they modulate the base weights), EMA would keep them closer to zero for too long

### Positional Encoding: RoPE Has No Extrapolation Issue

A concern was raised: audio tokens occupy positions 0-299 (300 tokens), but PaliGemma's Gemma was trained with SigLIP image tokens at positions 0-255. Are positions 256-299 "extrapolating"?

**Answer: No.** Gemma uses Rotary Position Embeddings (RoPE), not learned positional embeddings:
- RoPE is a mathematical transformation (`cos(pos/freq)`, `sin(pos/freq)`) applied at inference time
- There is no "trained range" — RoPE works for any position value
- RoPE primarily encodes **relative** positions between tokens. The relative distance between audio token 0 and audio token 5 is the same regardless of where in the absolute sequence they appear
- Position 300 is mathematically no different from position 200 to RoPE

### Why Keep Audio Projector Trainable in Stage 2?

Stage 1 learned a coarse distribution match. Stage 2's ASR cross-entropy loss provides much richer gradient signal — it tells the projector not just "match this distribution" but "produce tokens that Gemma can decode into THIS specific text." Allowing the projector to refine its representations under ASR supervision improves the quality of audio encoding.

### Training Configuration

- **Data**: LibriSpeech train-clean-360 (97,243 utterances, 859 speakers, ~360 hours). 3.4x more data and 3.4x more speaker diversity than Stage 1's train-clean-100.
- **Learning rate**: 2e-5 → 2e-6, cosine decay, warmup 500 steps. Deliberately low because the projector is already pre-aligned.
- **Batch size**: 32 (4 per GPU × 8 GPUs). Smaller than Stage 1 (64) due to LoRA memory overhead.
- **Steps**: 10,000 (~3.3 epochs over 97k samples with batch 32).
- **Checkpoints**: Every 500 steps (for early stopping if needed).
- **Freeze filter**: `Any(PaliGemma/img, All(llm, Not(lora)), whisper_encoder)` — freezes SigLIP image encoder, Gemma base weights, Whisper encoder. Trains LoRA adapters + audio projector.

### Training Results — FAILED

- Step 0: loss=9.59, grad_norm=238.6 (below random chance ~12.5, confirming Stage 1 provides useful initialization)
- Step 100: loss=7.86, grad_norm=29.1 (rapid improvement during warmup)
- Step 5000: loss plateaued at ~5.6, grad_norm stable at ~2.3
- Step 10000: loss still ~5.6 (no improvement in second half of training)

### Post-Training Diagnostics

Ran `scripts/diag_decode.py` on the 10k-step checkpoint:

- **Greedy decode identical for all samples**: Every audio sample produces "Sub" as top-1 prediction (p=0.28-0.37)
- **Target word probabilities near-zero**: The model never learned to decode audio content into correct text
- **Same collapse pattern as Stage 1 v1**: Despite LoRA, Gemma still can't extract meaningful per-sample information from audio tokens

### Why It Failed

1. **Starting from collapsed projector**: Stage 1 v2's projector learned the right direction but collapsed to a single centroid. Starting from this gave LoRA only a uniform signal to work with.
2. **LibriSpeech-only data**: No robot-domain vocabulary. The projector never saw the kind of text it would need for downstream robot tasks.
3. **LoRA rank-16 may be too constrained**: Small rank limits how much Gemma can adapt its attention patterns to process audio tokens.

### Decision: Restart with Fresh Init + Mixed Data

Rather than trying to salvage Stage 2 v1, we pivoted to a joint training approach:
- **Fresh random projector + fresh LoRA** (not loaded from any failed checkpoint)
- **Mixed data**: 25% LibriSpeech + 75% DROID TTS robot instructions
- This is conceptually Stage 2 v2, configured as `pi05_audio_mixed_asr`

## 6. Critical Bug: Audio/Text Mutual Exclusivity

### The Bug

Our `AudioTextMixingTransform` (used in Stage 3 for 60/40 audio/text mixing) had a critical bug: when assigning TTS audio to a training sample, it added `audio_path` to the data dict but **did not remove the text prompt**. This meant:

- Audio samples had BOTH `audio_path` AND the full text prompt
- The model could simply ignore audio and use text for task understanding
- Audio would never be learned because it was always redundant

### How We Discovered It

Analysis of VLAS (Section 3.2) revealed their design: "randomly replaced half of the training samples with the synthesized speech instructions." In VLAS code, when audio is assigned, the text instruction is literally replaced with a single `<audio>` token — the model receives **either** audio **or** text, never both.

### The Fix

In `src/openpi/transforms.py`, when audio is assigned:
```python
data["audio_path"] = rng.choice(audio_files)
# Remove text prompt so model must use audio for task info.
data["prompt"] = np.asarray("")
```

Empty string (not `None`) avoids downstream `ValueError("Prompt is required")` in `TokenizePrompt`. The tokenizer produces BOS + padding tokens, giving the model zero text information to fall back on.

**Verified**: `scripts/check_audio_text_mixing.py` runs 100 samples and confirms:
- All audio samples: `prompt=""`, `audio_path` present
- All text samples: original prompt preserved, no `audio_path`

### Why This Is Critical

Without this fix, Stage 3 training would appear to succeed (loss decreases, actions look reasonable) but the model would be using **text** for all conditioning. At inference time with audio-only input, the model would have never learned to extract task information from audio tokens — it would produce random or default actions.

## 7. Training Data Strategy: LibriSpeech + DROID TTS

### The Problem

Stage 2 trains on LibriSpeech (generic speech), but Stage 3 uses robot manipulation commands. There's a vocabulary domain gap: LibriSpeech contains "the old man walked down the street" while robot tasks say "pick up the red mug and place it on the plate." If the projector only sees general speech, it may not transfer well to robot-specific vocabulary.

### DROID Dataset Analysis

Pi0.5 was trained on DROID (among other datasets). We downloaded the DROID annotations file (`/home/user1/workspace/VLA/data/droid_annotations.json`, 50,092 episodes):
- 3 language annotation slots per episode
- 31,420 unique instructions in slot 1
- 64,608 unique instructions across all 3 slots
- 2,694 unique vocabulary words
- Top verbs: put (13.9k), pick (10.4k), move (9.4k), remove (6.6k), take (4.3k)
- 92% vocabulary overlap with LIBERO — ideal training data for domain adaptation

### Mixing Strategy: 25% LibriSpeech / 75% DROID

We decided on a mixed dataset for Stage 2:
- **25% LibriSpeech train-clean-360**: 859 real speakers provide acoustic diversity (accents, speaking styles, noise). Avg 34.5 words/utterance, ~12.5s duration, ~45 tokens/sample
- **75% DROID TTS**: 31,420 unique robot manipulation instructions × 10 TTS voices = 314,000 samples. Avg 10.8 words, ~3.7s duration, ~14 tokens/sample

**Why 25/75 by sample count?** LibriSpeech utterances are ~3.4x longer than DROID commands. At 25/75 sample ratio, the **token exposure** is roughly balanced: 25% × 45 tokens ≈ 75% × 14 tokens. This ensures the model spends equal learning time on acoustic diversity and domain-relevant vocabulary.

### TTS Voice Plan — IMPLEMENTED

We evaluated 5 TTS engines (Piper, Kokoro, XTTS v2, StyleTTS2, edge-tts) and chose a three-way accent split:

| Dataset | Files | Engine | Voices | Accent | Synthesis Time |
|---------|-------|--------|--------|--------|----------------|
| DROID train | 30,000 .mp3 | edge-tts | 3 American (Aria, Guy, Jenny) | US English | ~13 min (concurrency=32) |
| LIBERO train | 2,240 .wav | Piper VCTK | 20 speakers | Multi-accent (UK, Indian, etc.) | ~14 min |
| LIBERO eval | 1,120 .mp3 | edge-tts | 10 voices | Held-out (GB/AU/IN/IE/ZA) | ~5 min |

**Why three-way accent split?**
- Train and eval use **strictly different voices** — no speaker leakage
- DROID train uses American English (majority accent in LibriTTS-R, consistent with LibriSpeech)
- LIBERO train uses multi-accent VCTK (109 speakers, 13 accent categories)
- LIBERO eval uses held-out accent-region voices (GB, AU, IN, IE, ZA) to test generalization

**Initial plan was 10 voices × 31.4k DROID instructions = 314k files** but:
- Piper (libritts_r, 904 speakers) was too slow: 1.7 files/s vs advertised ~80 files/s. Would take ~16 hours.
- Vocabulary analysis showed diminishing returns after 5k instructions (93% LIBERO coverage at 1k)
- Reduced to 10k instructions × 3 edge-tts voices = 30k files, sufficient for 75% of mixed dataset

**edge-tts concurrency optimization**: Benchmarked at 8/16/24/32/48 concurrent connections. Network-latency bound (~1.1s per request). Optimal at concurrency=32 (24 files/s), 3.4x faster than default 8. Plateaus at 48 due to connection resets.

All TTS data stored under `/home/user1/workspace/VLA/data/tts/` with `manifest.json` per dataset.

## 8. Stage 3: Robot Task Training — The Hardest Design Decision

### Pi0.5's Two-Expert Architecture

This is the critical context for Stage 3 design. Pi0.5 has two transformer experts that share attention:

```
Gemma Expert:     processes prefix tokens (images + audio + text)
Action Expert:    processes suffix tokens (action tokens for flow matching)

Shared attention: Action expert's queries attend to Gemma's keys/values
                  This is how action predictions are conditioned on the input
```

In the shared attention layers, Q/K/V matrices from both experts are concatenated. The action expert reads Gemma's representations through cross-attention in this shared space.

### Evolution of Our Freezing Strategy

We went through three iterations of the Stage 3 freezing strategy:

**Option A: Only train action side** (rejected)
- Train: action expert LoRA + action head projections
- Freeze: everything else including Gemma LoRA from Stage 2
- Problem: Gemma LoRA optimized for ASR can't adapt K/V for robot conditioning

**Option B: Gemma LoRA + action side, freeze audio projector** (initially chosen)
- Train: Gemma LoRA + action expert LoRA + action head projections
- Freeze: audio projector, all base weights, SigLIP, Whisper
- Rationale: projector produces correct distribution, flow matching gradients would corrupt it

**Option C: Everything trainable, projector at lower LR** (FINAL — approved)
- Train: Gemma LoRA + action expert LoRA + action head + audio projector (at 10x lower LR)
- Freeze: Gemma base weights + Whisper encoder + SigLIP
- Rationale: ASR-optimized representations may not be optimal for action prediction. Allowing the projector to adapt slowly preserves Stage 2 quality while enabling robot-specific tuning.

### Why We Changed from Option B to Option C

The argument for freezing the projector (Option B) assumed that the ASR-optimized embedding distribution is exactly what Stage 3 needs. But upon reflection:

1. **ASR ≠ action conditioning**: The projector learned to encode speech content for text prediction. Robot action prediction may benefit from slightly different emphasis (e.g., emphasizing object names vs verbs differently).
2. **Gemma LoRA can't compensate for everything**: If the projector's representations are suboptimal for robot conditioning, Gemma LoRA would need to learn a complex remapping. It's more efficient to let the projector adapt slightly.
3. **10x lower LR is safe**: At 2e-6 effective LR (vs 2e-5 for LoRA), the projector moves ~100x slower than LoRA in weight-space. Over 30k steps, it fine-tunes rather than retrains.

### Per-Parameter LR Scaling: Implementation

We implemented per-parameter LR groups via **gradient scaling** — the simplest possible approach:

```python
# In train_step (scripts/train.py):
for filt, scale in config.lr_scale_overrides.items():
    grads = nnx_utils.state_map(grads, filt, lambda p: p.replace(p.value * scale))
```

Scaling gradients by 0.1 before the optimizer is mathematically equivalent to using 10x lower learning rate for those parameters (since `update = -lr * grad`, scaling grad by 0.1 gives `update = -lr * 0.1 * grad = -(0.1*lr) * grad`).

This approach was chosen over:
- `optax.multi_transform`: Requires building parameter label pytrees that are incompatible with NNX State's `VariableState` leaves
- `optax.masked`: Doesn't support different scales per group, only on/off
- Per-parameter optimizer chains: Excessive complexity for a single scale factor

The `lr_scale_overrides` field on `TrainConfig` maps filters to scale factors:
```python
lr_scale_overrides={
    nnx_utils.PathRegex(".*audio_projector.*"): 0.1,  # 10x lower LR
}
```

### LIBERO Visual Ambiguity: Why Audio Can't Be Ignored

A key concern was whether the model would learn to ignore audio in Stage 3 because images alone provide enough information. Analysis of LIBERO's 130 tasks across 5 suites showed:

| Suite | Tasks | Scenes | Tasks sharing a scene | Ambiguity |
|-------|-------|--------|----------------------|-----------|
| libero_spatial | 10 | 1 | 10/10 (all same scene) | VERY HIGH — same objects, different target locations |
| libero_object | 10 | 1 | 10/10 (all same scene) | VERY HIGH — same scene, different target objects |
| libero_90 | 90 | 20 | 84/90 (18 scenes have 3+ tasks) | HIGH — most scenes shared |
| libero_10 | 10 | 9 | 2/10 | MODERATE |
| libero_goal | 10 | 10 | 0/10 (unique scenes) | LOW — each task has unique scene |

**Result: 120 out of 130 tasks (92%) share their scene with at least one other task.** For these tasks, the visual input alone is ambiguous — the model MUST use the instruction (audio or text) to know which task to perform. This makes the F6 failure mode (model ignores audio) much less likely when training on all 130 LIBERO tasks.

### 60/40 Audio/Text Mixing

Stage 3 uses 60% TTS audio and 40% text-only instruction mixing (following VLAS's `random.random() > 0.4`):

- **60% audio**: Enough exposure that the model learns to map audio → robot actions
- **40% text**: Preserves original text instruction capability and provides regularization
- **Mutual exclusivity**: When audio is assigned, text prompt is removed (see Section 6). The model receives either audio+images or text+images, never both.
- **Why not 100% audio?**: Would lose text instruction capability, and TTS has limited acoustic diversity

### TTS Pre-Synthesis

We pre-synthesize TTS audio for all robot task prompts rather than generating on-the-fly because:
- Deterministic training (same audio per prompt per epoch)
- No runtime dependency on TTS service
- Can use diverse voices for speaker variation (20 voices for LIBERO)
- Cached in a manifest (prompt_text → list of audio file paths) for fast lookup

## 9. What Could Go Wrong (Known Risks and Failure Mode Plan)

### F1: Stage 2 loss doesn't decrease (projector fundamentally broken)
- **Detection**: Loss stuck above 8.0 after 2000 steps
- **Recovery**: Try joint ASR (skip Stage 1, train projector + LoRA from scratch). Config `pi05_audio_joint_asr` is already prepared.

### F2: Stage 2 greedy decode shows no variation (projector still collapsed)
- **Detection**: `scripts/diag_decode.py` shows identical output for all samples
- **Recovery**: Increase LoRA rank to 32 or 64. If still collapsed, switch to joint ASR approach.

### F3: Stage 3 loss doesn't decrease (audio/action bridging fails)
- **Detection**: Loss flat after 5k steps compared to text-only baseline
- **Recovery**: Try higher audio_ratio (80/20), or larger action expert LoRA rank

### F4: Stage 3 audio performance poor but text fine (audio not learned)
- **Detection**: Eval with `--eval-mode audio` significantly worse than `--eval-mode text`
- **Recovery**: Check if AudioTextMixingTransform bug has resurfaced. Try 80/20 mixing. Verify TTS audio quality.

### F5: Stage 3 text performance degrades (catastrophic forgetting)
- **Detection**: Eval with `--eval-mode text` worse than pre-Stage-3 baseline
- **Recovery**: Lower audio_ratio to 40/60. Increase LoRA rank for more capacity.

### F6: Model ignores audio entirely (uses images only)
- **Detection**: Identical actions for same scene with different audio instructions
- **Likelihood**: LOW — 92% of LIBERO tasks share scenes, making instruction essential
- **Recovery**: Verify mutual exclusivity in transforms. Add audio-perturbation test (swap audio between samples, actions should change).

### F7: TTS vs real speech domain gap at inference
- **Detection**: Good performance with TTS audio, poor with real speech recordings
- **Recovery**: Add audio augmentation (noise injection, room reverb). Fine-tune on small set of real recordings.

### F8: Projector destabilized by Stage 3 gradients (despite low LR)
- **Detection**: Audio token norms diverge from text embedding norms during Stage 3
- **Recovery**: Freeze projector entirely (revert to Option B). Or reduce scale from 0.1 to 0.01.

### Stage 1 v2 plateau and collapse
Stage 1 v2 MSE plateaued at ~0.21 and the projector collapsed to a single direction. This was mitigated by adding a learnable output scale (2.4x) and proceeding to Stage 2 where ASR CE provides per-sample discrimination. **Status: mitigated, Stage 2 in progress.**

## 10. Comparison with VLAS

| Aspect | VLAS | Ours |
|--------|------|------|
| Base model | LLaVA + LLaMA | Pi0.5 (PaliGemma Gemma + Action Expert) |
| Audio encoder | Whisper-Large-V2 | Whisper-Large-V2 (same) |
| Projector | 2-layer MLP, 5x downsample | 2-layer MLP, 5x downsample (same) |
| Stage 1 | ASR CE through frozen LLaMA | MSE embedding alignment (no Gemma in loop) |
| Stage 2 | ASR CE, full unfreeze LLaMA | ASR CE, LoRA on Gemma (preserve image ability) |
| Stage 3 | Robot task, full unfreeze | Robot task, dual LoRA, projector at low LR |
| Audio/text mixing | Replace text with `<audio>` token | Remove text prompt (empty string) |
| Key challenge | None major (LLaMA is unbiased) | PaliGemma has strong multimodal bias |
| Action generation | Direct token prediction | Flow matching (continuous actions) |
| Training data | BridgeData V2 | LIBERO (130 tasks) |
| ASR training data | LibriSpeech only | 25% LibriSpeech + 75% DROID TTS (robot domain) |
| TTS voices | Not specified | 3 American (DROID), 20 multi-accent (LIBERO train), 10 held-out (LIBERO eval) |
| TTS engine | Not specified | edge-tts (DROID, eval) + Piper VCTK (LIBERO train) |

The core architectural difference — PaliGemma's multimodal fine-tuning vs LLaMA's text-only pretraining — forced us to develop a fundamentally different Stage 1 approach and use LoRA instead of full fine-tuning throughout.

## 11. Complete Training Plan

### Phase 1: TTS Synthesis — COMPLETE
| Dataset | Status | Files | Time |
|---------|--------|-------|------|
| DROID train (edge-tts) | DONE | 30,000 .mp3, 0 failed | 13 min |
| LIBERO train (Piper VCTK) | DONE | 2,240 .wav, 0 failed | 14 min |
| LIBERO eval (edge-tts) | DONE | 1,120 .mp3, 0 failed | 5 min |

### Phase 2: Stage 2 — Mixed ASR Training — READY TO LAUNCH

**Previous attempt (FAILED)**: `pi05_audio_stage2_asr_finetune` on LibriSpeech-only, loss plateaued at 5.6, greedy decode collapsed.

**Current plan**: `pi05_audio_mixed_asr`
- Data: 25% LibriSpeech train-clean-360 (97k) + 75% DROID TTS (30k) = 127,243 total
- Fresh random projector + fresh LoRA (verified: 0 projector/LoRA keys in base checkpoint)
- 7,500 steps, batch 32, LR cosine 2e-5→2e-6, warmup 500
- MixedASRDataset: runtime per-sample Bernoulli sampling (verified 23.7%/76.3% ratio)
- Checkpoint: `checkpoints/pi05_audio_mixed_asr/mixed_asr_7500/`

**Sanity checks completed**:
1. Config loads correctly with all fields
2. Dataset ratio verified (23.7% LibriSpeech / 76.3% DROID over 1000 samples)
3. Weight init: 0 audio_projector + 0 LoRA keys in base Pi0.5 checkpoint → fresh random init

**Launch command**:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_audio_mixed_asr \
  --data.data-dir=/home/user1/workspace/VLA/data/librispeech/LibriSpeech/train-clean-360 \
  --data.droid-tts-dir=/home/user1/workspace/VLA/data/tts/droid_train \
  --no-wandb-enabled --exp-name=mixed_asr_7500
```

**Go/no-go criteria before Stage 3**:
- Loss below 4.0 (well below random chance ~12.5)
- `scripts/diag_decode.py` shows varied greedy decode output across different audio samples
- Audio token norms in Gemma's expected range

### Phase 3: Stage 3 — LIBERO Robot Training
- Config: `pi05_audio_stage3_libero`
- Data: LIBERO (all 130 tasks), 60/40 audio/text mixing with `--data.tts-cache-dir` and `--data.audio-ratio=0.6`
- Trainable: Gemma LoRA + action expert LoRA + action head + audio projector (10x lower LR)
- 30k steps, batch 32, LR 2e-5→2e-6, gradient clipping 1.0
- **TODO**: Update `weight_loader` path to Stage 2 mixed ASR output checkpoint

**Launch command** (after updating weight_loader path):
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_audio_stage3_libero \
  --data.tts-cache-dir=/home/user1/workspace/VLA/data/tts/libero_train \
  --data.audio-ratio=0.6 \
  --no-wandb-enabled --exp-name=stage3_libero
```

**Validation checkpoints**: Eval every 5k steps with `examples/libero/main.py --eval-mode both`

### Phase 4: Evaluation
- `examples/libero/main.py --eval-mode text` (text-only baseline)
- `examples/libero/main.py --eval-mode audio --audio-dir /home/user1/workspace/VLA/data/tts/libero_eval` (audio-only, held-out accents)
- `examples/libero/main.py --eval-mode both` (mixed)
- Compare success rates across all 130 tasks

## 12. Implementation Notes

### Flax Int-Key Bug

Flax NNX's `replace_by_pure_dict` converts string digit keys (e.g., `'0'`, `'1'`) to integers internally, but the NNX state from Linen-bridged modules (like HuggingFace's FlaxWhisperEncoder) keeps them as strings. This causes a `KeyError` when loading checkpoints that contain Whisper encoder layers.

**Fix**: Replaced `state.replace_by_pure_dict(partial_params)` in `scripts/train.py` with a manual `flat_state()` merge that tries both string and integer key variants:
```python
flat_state = state.flat_state()
for kp, v in traverse_util.flatten_dict(partial_params).items():
    if kp in flat_state:
        flat_state[kp] = flat_state[kp].replace(v) ...
    else:
        alt_kp = tuple(int(k) if isinstance(k, str) and k.isdigit() else k for k in kp)
        if alt_kp in flat_state:
            flat_state[alt_kp] = flat_state[alt_kp].replace(v) ...
```

### Learnable Output Scale

Added to `DownsampleAudioProjector` in `whisper.py` to bridge the norm gap between audio projector output (~100) and Gemma text embeddings (~240). Initialized to 2.4. This is a single scalar parameter that gets multiplied with the projector output. Since `_merge_params` in `weight_loaders.py` uses `missing_regex=".*audio_projector.*"`, this parameter correctly falls back to its init value (2.4) when loading Stage 1 checkpoints that predate its addition.

### Per-Parameter LR via Gradient Scaling

Added `lr_scale_overrides` field to `TrainConfig` (dict mapping filter → float). In `train_step`, gradients for matched parameters are scaled before the optimizer:

```python
for filt, scale in config.lr_scale_overrides.items():
    grads = nnx_utils.state_map(grads, filt, lambda p: p.replace(p.value * scale))
```

This is mathematically equivalent to per-parameter LR groups but requires only 3 lines of code. No changes to the optimizer, no label pytrees, no NNX State compatibility issues.

## 13. Timeline and Iterations

1. **Initial implementation**: Added Whisper encoder + audio projector to Pi0.5 model
2. **Stage 1 v1 (ASR CE)**: Trained 5000 steps, loss converged but projector collapsed
3. **Diagnostic investigation**: Built scripts to analyze audio token norms, greedy decoding, attention patterns. Discovered the distribution mismatch problem.
4. **Literature review**: Re-examined VLAS approach, identified the LLaMA vs PaliGemma difference as root cause
5. **Stage 1 v2 design**: Proposed direct MSE embedding alignment to bypass Gemma
6. **Stage 1 v2 training**: Loss drops fast, plateaus at ~0.21
7. **Stage 1 v2 diagnostics**: Discovered projector collapse (single centroid direction). Attempted InfoNCE fix — failed because Gemma text embeddings are too similar (cosine > 0.995). Added learnable output scale (2.4x) to close norm gap.
8. **Stage 2 + 3 config design**: Deep analysis of Pi0.5 attention sharing, chose LoRA + freeze strategy
9. **Stage 2 v1 training (FAILED)**: ASR fine-tune on LibriSpeech-only. Loss plateaued at 5.6 after 10k steps, greedy decode identical for all samples. LoRA rank-16 on LibriSpeech-only insufficient.
10. **Audio/text mutual exclusivity bug**: Discovered and fixed — AudioTextMixingTransform now removes text when audio is assigned, following VLAS design.
11. **Training data strategy**: Analyzed DROID annotations (31.4k unique instructions), decided on 25% LibriSpeech + 75% DROID TTS mixing for improved domain coverage.
12. **LIBERO visual ambiguity analysis**: 92% of tasks share scenes → instruction is essential → audio ignorability risk is low.
13. **Stage 3 config finalized**: Projector unfrozen at 10x lower LR via gradient scaling. Per-parameter LR implemented as `lr_scale_overrides` on TrainConfig.
14. **TTS synthesis — COMPLETE**: Evaluated 5 engines (Piper, Kokoro, XTTS v2, StyleTTS2, edge-tts). Implemented three-way accent split: DROID train (edge-tts, 30k files), LIBERO train (Piper VCTK, 2,240 files), LIBERO eval (edge-tts held-out accents, 1,120 files). Optimized edge-tts from 7.5 files/s (concurrency 8) to 24 files/s (concurrency 32).
15. **Mixed ASR dataset — IMPLEMENTED**: `MixedASRDataset` class with per-sample Bernoulli sampling. `MixedASRDataConfig` + `pi05_audio_mixed_asr` training config. Data loader routing for `repo_id="mixed_asr"`. Sanity checked: ratio 23.7%/76.3%, fresh projector+LoRA init confirmed.
16. **Stage 2 v2 (mixed ASR) — READY TO LAUNCH**: Awaiting go-ahead. Fresh projector + LoRA, 25% LibriSpeech + 75% DROID TTS, 7,500 steps.
