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

### What We Expect From Diagnostics
- Audio token norms should match text norms (~40-60 range)
- Cosine similarity between audio and text mean-pools should be >0.5
- Different audio samples should produce different embeddings (no collapse)
- Greedy decode through Gemma should show varied output (unlike Stage 1 v1)

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

### Why Keep Audio Projector Trainable in Stage 2?

Stage 1 learned a coarse distribution match. Stage 2's ASR cross-entropy loss provides much richer gradient signal — it tells the projector not just "match this distribution" but "produce tokens that Gemma can decode into THIS specific text." Allowing the projector to refine its representations under ASR supervision improves the quality of audio encoding.

## 6. Stage 3: Robot Task Training — The Hardest Design Decision

### Pi0.5's Two-Expert Architecture

This is the critical context for Stage 3 design. Pi0.5 has two transformer experts that share attention:

```
Gemma Expert:     processes prefix tokens (images + audio + text)
Action Expert:    processes suffix tokens (action tokens for flow matching)

Shared attention: Action expert's queries attend to Gemma's keys/values
                  This is how action predictions are conditioned on the input
```

In the shared attention layers, Q/K/V matrices from both experts are concatenated. The action expert reads Gemma's representations through cross-attention in this shared space.

### The Three Options We Considered

**Option A: Only train action side**
- Train: action expert LoRA + action head projections
- Freeze: everything else including Gemma LoRA from Stage 2

Problem: The Gemma LoRA from Stage 2 was optimized for ASR (predict text from audio). Robot conditioning needs different K/V representations in the shared attention — specifically, representations that help the action expert predict motor commands, not transcriptions. Frozen Gemma LoRA can't adapt.

**Option B: Gemma LoRA + action side, freeze audio projector** (CHOSEN)
- Train: Gemma LoRA + action expert LoRA + action head projections
- Freeze: audio projector, all base weights, SigLIP, Whisper

This lets Gemma LoRA adapt its K/V representations from "good for ASR" to "good for robot conditioning." The audio projector stays frozen because it's already producing the right distribution — flow matching gradients would corrupt it.

**Option C: Everything trainable**
- Train: all LoRA + projector + action head

Problem: Flow matching loss optimizes for action prediction, not audio understanding. The audio projector would drift from the carefully learned text embedding distribution, potentially breaking the audio-to-embedding mapping entirely.

### Why Freeze the Audio Projector in Stage 3?

This deserves its own section because it's counterintuitive. You might think: "more trainable parameters = better, let the projector adapt to robot tasks."

The audio projector was specifically trained (Stage 1 + Stage 2) to output tokens in Gemma's expected embedding distribution. It learned:
- The right norm range (~40-60, matching text embeddings)
- Directions that encode speech content in Gemma's embedding space
- A mapping that Gemma LoRA (from Stage 2) knows how to decode

Flow matching loss in Stage 3 has a completely different objective: minimize the difference between predicted and ground-truth robot actions. The gradient to the audio projector from this loss says "change your output so the action expert can better predict motor commands" — this is a very different pressure than "match text embedding distribution." Over thousands of steps, this would push the projector to output tokens optimized for flow matching but no longer interpretable by Gemma's attention layers.

Instead, we let Gemma LoRA adapt. It can learn to transform audio projector outputs into K/V representations useful for robot conditioning, without disrupting the projector's learned audio encoding.

### 60/40 Audio/Text Mixing

Stage 3 uses 60% TTS audio and 40% text-only instruction mixing. Why?

- **60% audio**: Enough exposure to audio-conditioned training that the model learns to map audio → robot actions
- **40% text**: Preserves the original text instruction capability and provides a regularization signal — text instructions are "clean" conditioning (no TTS artifacts, no speaker variation), which helps stabilize training
- **Why not 100% audio?**: The model would lose the ability to follow text instructions, and TTS audio is synthesized (not real speech), so it has limited acoustic diversity

### TTS Pre-Synthesis

We pre-synthesize TTS audio for all robot task prompts rather than generating on-the-fly because:
- Deterministic training (same audio per prompt per epoch)
- No runtime dependency on TTS service
- Can use 50+ diverse voices for speaker variation
- Cached in a manifest (prompt_text → list of audio file paths) for fast lookup

## 7. What Could Go Wrong (Known Risks)

### Stage 1 v2 might plateau too high
If the MSE loss plateaus at 0.22 and doesn't go lower, the audio tokens might not be close enough to text embeddings for Gemma to process them well. Possible mitigation: add per-token alignment (not just mean-pool) or use a combined MSE + cosine loss.

### Stage 2 LoRA rank might be too low
Rank 16 might not have enough capacity for Gemma to learn audio → text decoding. If Stage 2 ASR quality is poor, try rank 32 or 64. The tradeoff is more parameters to train and higher risk of overfitting.

### Stage 3 audio/text mixing ratio
60/40 is our starting point. If audio performance is poor, try 80/20. If text performance degrades, try 50/50. This is an empirical hyperparameter.

### TTS vs real speech gap
Stage 3 uses synthesized TTS audio but deployment uses real human speech. There's likely a domain gap. Possible mitigations: data augmentation (noise, room simulation), or fine-tuning on a small set of real recordings.

## 8. Comparison with VLAS

| Aspect | VLAS | Ours |
|--------|------|------|
| Base model | LLaVA + LLaMA | Pi0.5 (PaliGemma Gemma + Action Expert) |
| Audio encoder | Whisper-Large-V2 | Whisper-Large-V2 (same) |
| Projector | 2-layer MLP, 5x downsample | 2-layer MLP, 5x downsample (same) |
| Stage 1 | ASR CE through frozen LLaMA | MSE embedding alignment (no Gemma in loop) |
| Stage 2 | ASR CE, full unfreeze LLaMA | ASR CE, LoRA on Gemma (preserve image ability) |
| Stage 3 | Robot task, full unfreeze | Robot task, LoRA on both experts, freeze projector |
| Key challenge | None major (LLaMA is unbiased) | PaliGemma has strong multimodal bias |
| Action generation | Direct token prediction | Flow matching (continuous actions) |

The core architectural difference — PaliGemma's multimodal fine-tuning vs LLaMA's text-only pretraining — forced us to develop a fundamentally different Stage 1 approach and use LoRA instead of full fine-tuning throughout.

## 9. Timeline and Iterations

1. **Initial implementation**: Added Whisper encoder + audio projector to Pi0.5 model
2. **Stage 1 v1 (ASR CE)**: Trained 5000 steps, loss converged but projector collapsed
3. **Diagnostic investigation**: Built scripts to analyze audio token norms, greedy decoding, attention patterns. Discovered the distribution mismatch problem.
4. **Literature review**: Re-examined VLAS approach, identified the LLaMA vs PaliGemma difference as root cause
5. **Stage 1 v2 design**: Proposed direct MSE embedding alignment to bypass Gemma
6. **Stage 1 v2 implementation + training**: Loss drops fast, plateaus at ~0.21 (currently finishing)
7. **Stage 2 + 3 config design**: Deep analysis of Pi0.5 attention sharing, chose LoRA + freeze strategy
8. **Next**: Run diagnostics on Stage 1 v2, then proceed through Stages 2 and 3
