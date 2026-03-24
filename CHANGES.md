# Audio-Conditioned Pi0.5 — Changes from Base OpenPI

## Overview

This fork extends Pi0.5 (Physical Intelligence's vision-language-action model) with **speech/audio conditioning** via a split LoRA architecture. The goal is to enable robot control from spoken instructions in addition to text, while preserving the original text-conditioned performance.

Additionally, we added a **language instruction robustness evaluation suite** to measure how sensitive the model is to instruction wording variations.

---

## 1. Audio Pipeline (JAX)

### `src/openpi/models/whisper.py` (NEW)
- **WhisperEncoder**: Wraps OpenAI's Whisper (large-v3, 128 mel bins) as a frozen feature extractor.
- Produces per-frame hidden states (1500 frames × 1280 dims for 30s audio).
- Why: Whisper provides strong multilingual speech representations. We use hidden states (not transcriptions) to preserve prosodic and acoustic information.

### `src/openpi/models/audio_projector.py` (NEW)
- **AudioProjector**: Linear projection from Whisper hidden dim → Gemma embedding dim.
- **AttentionPooling**: Learned query-based pooling that compresses 1500 Whisper frames into a fixed number of audio tokens (default 32). Uses cross-attention with learnable query vectors.
- Why: Raw Whisper output is too long (1500 tokens). Attention pooling compresses it to 32 tokens while letting the model learn which audio frames matter.

### `src/openpi/models/pi0.py` (MODIFIED)
- Added `embed_prefix` audio branch: Whisper encode → project → attention pool → inject as prefix tokens alongside image/text tokens.
- Added `compute_alignment_loss`: MSE loss between audio token representations and corresponding text token representations (teacher forcing alignment).
- Added `gradient_mode` parameter for split LoRA routing — controls whether audio_lora or task_lora gradients flow.
- Added `audio_position_mask` output from `embed_prefix` to identify which prefix positions are audio tokens.
- Why: Audio tokens need to be embedded in the same space as text/image tokens so the Gemma backbone can attend to them during action prediction.

### `src/openpi/models/lora.py` (MODIFIED)
- **Dual LoRA**: Added `audio_lora_a/b` alongside existing `task_lora_a/b` in both `Einsum` and `FeedForward` layers.
- `gradient_mode="audio"` routes through audio LoRA (with stop_gradient on task LoRA), and vice versa.
- Why: Split LoRA lets us train audio understanding (audio_lora) and robot action (task_lora) with separate learning rates and training stages, preventing catastrophic forgetting.

### `src/openpi/models/gemma.py` (MODIFIED)
- Threading `gradient_mode` through all Gemma transformer layers so each attention/FFN block routes to the correct LoRA.
- Why: The split LoRA routing decision needs to propagate through every layer of the backbone.

### `src/openpi/models/pi0_config.py` (MODIFIED)
- Added audio config fields: `audio_enabled`, `whisper_variant`, `audio_num_tokens`, `training_stage`, `rehearsal_lambda`, `alignment_loss_weight`, `asr_loss_weight`, etc.
- Why: Centralizes all audio hyperparameters in the model config.

---

## 2. Audio Pipeline (PyTorch Port)

### `src/openpi/models_pytorch/lora_pytorch.py` (NEW)
- **LoRAConfig / LoRALinear / inject_lora**: PyTorch implementation of split LoRA.
- `inject_lora()` replaces `nn.Linear` layers with `LoRALinear` that has separate `audio_A/B` and `task_A/B` weight matrices.
- LoRA B initialized to zeros (standard), params kept in float32 to avoid bfloat16 gradient overflow.
- Why: Needed for PyTorch DDP training on H100s. The JAX implementation can't easily use PyTorch's DDP.

### `src/openpi/models_pytorch/pi0_pytorch.py` (MODIFIED)
- **AudioProjectorPT / AttentionPoolingPT**: PyTorch equivalents of the JAX audio components.
- **compute_alignment_loss**: MSE alignment between audio and text representations.
- ASR rehearsal loss: Cross-entropy loss predicting text tokens from audio tokens to maintain speech understanding.
- Dual gradient computation: Separate backward passes for audio_lora and task_lora with `require_backward_grad_sync=False` on the first backward (DDP optimization).
- Why: Full PyTorch port enables training on NVIDIA hardware with standard DDP.

### `src/openpi/models_pytorch/gemma_pytorch.py` (MODIFIED)
- Added LoRA kwargs threading through HF Gemma forward pass.
- `decode_to_vocab()` for ASR loss computation.
- Float32 parameter list for LoRA params.
- Why: HuggingFace Gemma doesn't natively support LoRA kwargs, so we thread them via `**kwargs`.

### `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py` (MODIFIED)
- MLP and Attention layers accept and pass through `**kwargs` to support LoRA routing.
- Why: Required for split LoRA gradient routing through HF's Gemma implementation.

### `src/openpi/models_pytorch/preprocessing_pytorch.py` (MODIFIED)
- Added audio fields (`audio`, `audio_whisper_hidden`, `has_audio`) to `SimpleProcessedObservation`.
- Why: The preprocessing pipeline needs to handle audio data alongside images and state.

---

## 3. Training

### `scripts/train.py` (MODIFIED)
- JAX training loop with dual gradient computation for split LoRA.
- Separate optimizer states for audio_lora vs task_lora parameters.
- Why: Stage 1 trains audio components at 50× LR while keeping task_lora frozen (lr_scale=0.0).

### `scripts/train_pytorch.py` (MODIFIED)
- PyTorch DDP training with LoRA injection, freeze management, and dual backward passes.
- NaN gradient guard (skip step if grad_norm is NaN).
- Automatic LR scaling: audio projector/pooling at 50× in Stage 1, 10× in Stage 2.
- Why: Complete PyTorch training pipeline for audio-conditioned Pi0.5.

### `src/openpi/training/config.py` (MODIFIED)
- Added training configs:
  - `pi05_audio_stage1`: Audio alignment training (ASR + alignment loss, audio components at 50× LR)
  - `pi05_audio_stage1_pytorch`: PyTorch port of Stage 1
  - `pi05_audio_stage2_pytorch_h100`: Robot fine-tuning with audio LoRA + ASR rehearsal
  - Various ablation configs (64-tok, distillation, etc.)
- Audio-specific config fields: `tts_cache_dir`, `whisper_cache_dir`, `audio_ratio`, `clear_prompt_for_audio`, `auxiliary_tts_dir`, `auxiliary_ratio`, `rehearsal_lambda`, etc.
- Why: Each training stage has different hyperparameters and data mixing ratios.

### `src/openpi/training/weight_loaders.py` (MODIFIED)
- Support for loading split LoRA checkpoints and mapping audio component weights.
- Why: Stage 2 loads from Stage 1 checkpoint, needs to correctly map audio_lora and projector weights.

### `src/openpi/transforms.py` (MODIFIED)
- TTS audio loading, Whisper feature caching, audio augmentation transforms.
- DROID auxiliary data loading for ASR rehearsal.
- Audio-text pairing logic with configurable `audio_ratio` and `clear_prompt_for_audio`.
- Why: Training data pipeline needs to load/cache TTS audio, pair it with robot demonstrations, and mix in auxiliary ASR data.

---

## 4. Serving / Inference

### `src/openpi/serving/websocket_policy_server.py` (MODIFIED)
- Pass audio waveform through websocket to model server.
- Why: Inference-time audio needs to flow from the eval client to the policy server.

### `packages/openpi-client/src/openpi_client/websocket_client_policy.py` (MODIFIED)
- Increased `ping_timeout` to 300s for large model loading.
- Why: Model loading can take >60s on first inference, causing websocket timeouts.

---

## 5. Language Instruction Robustness Evaluation

### `examples/libero/main.py` (MODIFIED)
- Added `paraphrase` eval mode and `--paraphrase-map` CLI flag.
- When in paraphrase mode, the original task instruction is replaced with a modified version from a JSON mapping file before being sent to the model. The environment still uses the original instruction (correct sim setup and video naming).
- Added `from __future__ import annotations` for Python 3.8 compatibility.
- Why: To test whether the model's performance depends on exact instruction wording vs. true language understanding.

### Paraphrase / Robustness Map Files (NEW)
All in `data/`:
- `libero_paraphrased_instructions.json` — Easy paraphrase: simple synonym swaps ("pick up" → "grab", "put" → "place")
- `libero_paraphrased_hard.json` — Hard paraphrase: passive voice, reordered clauses, verbose rephrasing
- `libero_typos.json` — 1-3 realistic keyboard-proximity typos per instruction
- `libero_minimal.json` — Telegraphic style, stripped to bare minimum words ("black bowl between plate ramekin to plate")
- `libero_verbose.json` — Polite/verbose filler ("I would really appreciate if you could carefully...")
- `libero_synonym_objects.json` — Object name synonyms ("black bowl" → "dark bowl", "plate" → "flat dish", "stove" → "cooktop")
- `libero_negation.json` — Negated instructions ("don't touch the black bowl") to test if model ignores action semantics

### Eval Scripts (NEW)
- `scripts/eval_paraphrase.sh` — Single-GPU sequential text vs. paraphrase comparison
- `scripts/eval_paraphrase_parallel.sh` — 8-GPU parallel text vs. paraphrase (one server per GPU)
- `scripts/eval_robustness_parallel.sh` — 8-GPU parallel evaluation of all 5+ variants across 4 LIBERO suites

### Results (20 trials/task, 4 suites)

| Variant | Avg Accuracy | Δ vs Text | Interpretation |
|---------|-------------|-----------|----------------|
| Text (baseline) | 96.9% | — | — |
| Easy Paraphrase | 96.4% | -0.5pp | Robust |
| Hard Paraphrase | 95.4% | -1.5pp | Robust |
| Verbose/Polite | 96.1% | -0.8pp | Robust |
| Minimal/Telegraphic | 95.2% | -1.6pp | Robust |
| Typos | 92.0% | -4.9pp | Minor sensitivity |
| **Synonym Objects** | **86.9%** | **-10.0pp** | **Significant drop** |

**Key finding**: The model is robust to structural language changes (paraphrase, verbosity, minimalism) but **brittle to object name changes** (-10pp with synonyms). This suggests the model relies on exact object name token matching from training data rather than visual grounding. Combined with the minimal-instruction result (95.2% with just object names, no verbs), this indicates the model **overfits to object→action mappings** and largely ignores action verbs.

---

## 6. Other Files

### Diagnostic / Utility Scripts (NEW)
- `scripts/smoke_test_audio_pytorch.py` — Standalone smoke test for PyTorch audio pipeline (no data/checkpoints needed)
- `scripts/precompute_whisper_cache.py` — Pre-extract Whisper features for TTS audio files
- `scripts/diagnose_ckpt.py`, `diagnose_loss.py`, `diagnose_loss_v2.py` — Checkpoint and loss debugging tools
- `scripts/plot_training.py` — Training curve visualization
- `scripts/benchmark_latency.py`, `benchmark_whisper_latency.py` — Inference latency benchmarks
- `scripts/ablation_audio_jax.py` — Audio ablation experiments

### Paper
- `paper/speech_pi05_lbr.tex` — DAC 2026 LBR draft (Speech-Conditioned Pi0.5)
