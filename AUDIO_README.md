# Audio-Conditioned Pi0.5: Speech-to-Robot-Action Pipeline

This fork extends [OpenPI (Pi0/Pi0.5)](https://github.com/Physical-Intelligence/openpi) with **raw audio input** — enabling a robot to follow spoken language commands instead of (or in addition to) text prompts.

Inspired by [VLAS (CVPR)](https://github.com/robopen/VLAS), we add a Whisper-based audio encoder and a multi-stage training pipeline that progressively teaches Pi0.5 to understand speech and map it to robot actions.

## Architecture

```
                          ┌─────────────┐
  Audio ──► Whisper ──► Projector ──►  │              │
                          │   Gemma     │
  Images ──► SigLIP ──────────────►  │  (PaliGemma)  │──► shared ──► Action Expert ──► Robot Actions
                          │              │   attention
  Text ──► Embedder ──────────────►  │              │
                          └─────────────┘
```

- **Whisper-Large-V2** (frozen): Extracts 1500 audio hidden states from mel spectrograms
- **DownsampleAudioProjector** (trainable): 5x downsample + 2-layer MLP → 300 audio tokens at Gemma's embedding dimension
- **Gemma** (LoRA): Processes audio + image + text tokens with shared bidirectional attention
- **Action Expert** (LoRA): Cross-attends to Gemma's prefix representations to produce continuous robot actions via flow matching

## Three-Stage Training Pipeline

### Stage 1: Embedding Alignment (projector only)

Teaches the audio projector to output embeddings in the same distribution as Gemma's text embeddings. This is done via **direct MSE loss** between mean-pooled audio projector output and mean-pooled frozen text embeddings — Gemma is not in the loop, so gradients flow directly to the projector.

| | Details |
|---|---|
| **Config** | `pi05_audio_stage1_embed_align` |
| **Data** | LibriSpeech train-clean-100 (28.5k samples) |
| **Trainable** | Audio projector only (~15M params) |
| **Frozen** | Everything else |
| **Loss** | MSE: mean-pool(audio_tokens) vs mean-pool(text_embeddings) |
| **Hyperparams** | 5000 steps, batch 64, LR 1e-3 → 1e-5 (cosine) |

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_audio_stage1_embed_align \
  --data.data_dir=/path/to/LibriSpeech/train-clean-100 \
  --no-wandb-enabled --exp-name=stage1_embed_align_5k
```

**Why not ASR cross-entropy?** Our initial approach (Stage 1 v1) fed audio tokens through frozen Gemma with ASR cross-entropy loss. This failed because PaliGemma's Gemma was fine-tuned for SigLIP image tokens — randomly initialized audio tokens have the wrong distribution (~1371 norm vs ~45 for text), so Gemma routes around them and the projector collapses. Direct MSE bypasses Gemma entirely, giving the projector clean gradient signal to learn the correct distribution.

**Verification:** Run `scripts/diag_embed_align.py` after training. Check that audio token norms match text norms (~40-60), cosine similarity > 0.5, and embeddings vary across samples (no collapse).

### Stage 2: ASR Fine-tuning with LoRA (Gemma learns audio)

Now that the projector outputs are in the right distribution, we fine-tune Gemma via LoRA to actually **decode audio tokens into text**. This teaches Gemma's attention layers to extract semantic information from audio prefix tokens.

| | Details |
|---|---|
| **Config** | `pi05_audio_stage2_asr_finetune` |
| **Data** | LibriSpeech train-clean-360 (97k samples, 859 speakers) |
| **Trainable** | Gemma LoRA (rank 16, ~6.5M) + audio projector (~15M) |
| **Frozen** | Gemma base weights, action expert, SigLIP, Whisper |
| **Loss** | ASR cross-entropy (audio prefix → predict transcription) |
| **Hyperparams** | 10k steps, batch 32, LR 2e-5 → 2e-6 (cosine), no EMA |

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_audio_stage2_asr_finetune \
  --data.data_dir=/path/to/LibriSpeech/train-clean-360 \
  --no-wandb-enabled --exp-name=stage2_asr_finetune
```

**Verification:** Run `scripts/diag_decode.py` (update checkpoint path). Greedy decoding must produce **different text for different audio samples**. If all samples decode to the same text, the model hasn't learned to use audio.

### Stage 3: Robot Task Training with Audio/Text Mixing

Fine-tunes the full system for robot control with 60% TTS audio / 40% text-only instruction mixing. LoRA on both Gemma and action expert; audio projector is **frozen** to preserve its learned embedding distribution.

| | Details |
|---|---|
| **Config** | `pi05_audio_stage3_libero` / `pi05_audio_stage3_aloha` |
| **Data** | LeRobot (LIBERO or ALOHA) + pre-synthesized TTS audio |
| **Trainable** | Gemma LoRA + action expert LoRA + action head projections |
| **Frozen** | Audio projector, Gemma/action expert base weights, SigLIP, Whisper |
| **Loss** | Flow matching (standard Pi0.5 action prediction) |
| **Hyperparams** | 30k steps, batch 32, LR 2e-5 → 2e-6 (cosine), no EMA |

**Before training**, synthesize TTS audio for robot task prompts:
```bash
.venv/bin/python scripts/synthesize_tts.py \
  --repo_id physical-intelligence/libero \
  --output_dir ./tts_cache/libero \
  --num_speakers 50
```

Then train:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_audio_stage3_libero \
  --data.data_dir=/path/to/libero/data \
  --data.tts_cache_dir=./tts_cache/libero \
  --data.audio_ratio=0.6 \
  --no-wandb-enabled --exp-name=stage3_libero_audio
```

**Why freeze the audio projector in Stage 3?** The projector was specifically trained to match text embedding distribution. Flow matching gradients (robot action loss) would push it in a completely different direction, corrupting the carefully learned audio representations. Gemma LoRA adapts its K/V attention to bridge audio understanding with robot conditioning, while the 60/40 mixing ratio maintains audio comprehension.

## Evaluation

```bash
.venv/bin/python examples/libero/main.py \
  --checkpoint ./checkpoints/pi05_audio_stage3_libero/stage3_libero_audio/latest/params \
  --audio-dir ./tts_cache/libero \
  --eval-mode both
```

Evaluates in three modes:
- `text` — standard text prompt (baseline)
- `audio` — audio-only prompt (the new capability)
- `both` — audio + text together

## What's Trainable at Each Stage

| Component | Params | Stage 1 | Stage 2 | Stage 3 |
|-----------|--------|---------|---------|---------|
| Audio projector | ~15M | Train | Train | **Frozen** |
| Gemma LoRA (rank 16) | ~6.5M | — | Train | Train |
| Action expert LoRA (rank 32) | ~6.5M | — | — | Train |
| Action head (in/out proj, time MLP) | ~0.2M | — | — | Train |
| Gemma base weights | ~2B | Frozen | Frozen | Frozen |
| Action expert base | ~300M | Frozen | Frozen | Frozen |
| SigLIP | ~400M | Frozen | Frozen | Frozen |
| Whisper | ~600M | Frozen | Frozen | Frozen |

## Data Requirements

| Dataset | Size | Samples | Speakers | Used in |
|---------|------|---------|----------|---------|
| LibriSpeech train-clean-100 | 6.3 GB | 28,539 | 251 | Stage 1 |
| LibriSpeech train-clean-360 | 21 GB | 97,243 | 859 | Stage 2 |
| LeRobot LIBERO/ALOHA | varies | varies | — | Stage 3 |
| TTS cache (generated) | ~1 GB | varies | 50+ voices | Stage 3 |

## Key Files

| File | Purpose |
|------|---------|
| `src/openpi/models/pi0.py` | Core model: `compute_alignment_loss` (ASR CE), `compute_embedding_alignment_loss` (MSE) |
| `src/openpi/models/whisper.py` | WhisperEncoder + DownsampleAudioProjector |
| `src/openpi/models/lora.py` | LoRA implementation for Gemma Einsum + FeedForward |
| `src/openpi/training/config.py` | All training configs (Stage 1/2/3) |
| `src/openpi/training/librispeech_dataset.py` | LibriSpeech map-style dataset loader |
| `src/openpi/transforms.py` | AudioTextMixingTransform for Stage 3 |
| `scripts/synthesize_tts.py` | TTS pre-synthesis with edge-tts (60+ voices) |
| `scripts/diag_embed_align.py` | Diagnostic: embedding distribution match, cosine similarity, collapse check |
| `scripts/diag_decode.py` | Diagnostic: teacher-forced + greedy decode quality |
| `examples/libero/main.py` | LIBERO evaluation with `--eval-mode {text,audio,both}` |

## Current Progress

- [x] Audio encoder integration (Whisper + projector)
- [x] Stage 1 v1: ASR cross-entropy alignment (completed, but projector collapsed)
- [x] Stage 1 v2: Direct embedding alignment via MSE (training, loss plateau ~0.22)
- [ ] Stage 1 v2: Verify with diagnostic script
- [ ] Stage 2: ASR fine-tuning with Gemma LoRA (config ready)
- [ ] Stage 2: Verify greedy decode produces varied output per audio sample
- [ ] TTS synthesis for LIBERO task prompts
- [ ] Stage 3: LIBERO robot training with audio/text mixing
- [ ] Evaluation: text vs audio vs both success rates

## References

- [OpenPI / Pi0](https://github.com/Physical-Intelligence/openpi) — base VLA framework
- [VLAS (CVPR)](https://github.com/robopen/VLAS) — speech-conditioned VLA reference (LLaVA + Whisper + LLaMA)
- [Pi0 paper](https://arxiv.org/abs/2410.24164) — flow matching for robot action generation
