# Full Training Plan Audit — Audio-Conditioned Pi0.5

**Date**: 2026-02-28
**Status**: Stage 2 v2 ready to launch. DROID TTS expansion in progress.

---

## 1. End-to-End Consistency Check

### Architecture Flow
```
Stage 1 (COMPLETE):  Audio → Whisper (frozen) → Projector (train) → MSE vs Text Embeddings (frozen)
Stage 2 (READY):     Audio → Whisper (frozen) → Projector (train) → Gemma+LoRA (train) → CE text prediction
Stage 3 (CONFIG):    Audio → Whisper (frozen) → Projector (slow train) → Gemma+LoRA (train) → shared attn → Action Expert+LoRA (train) → flow matching
```

### Stage Transition Consistency

| Transition | What loads | What's fresh | Verified? |
|-----------|-----------|-------------|-----------|
| Base → Stage 2 | Pi0.5 base + Whisper | Projector + Gemma LoRA | YES — 0 projector/LoRA keys in base checkpoint |
| Stage 2 → Stage 3 | Stage 2 checkpoint (all weights) | Action expert LoRA | YES — path fixed to `mixed_asr/7500/params` |

### Trainable Parameter Audit

**Stage 2 (`pi05_audio_mixed_asr`)**:
| Component | Params | Trainable? | Notes |
|-----------|--------|-----------|-------|
| Gemma LoRA (rank 16) | ~27.9M (attn) + ~17.3M (FFN) = 45.2M | YES | Fresh zero-B init, random-A |
| Audio projector | ~17.3M | YES | Fresh random init |
| Gemma base weights | ~2B | FROZEN | `All(llm, Not(lora))` |
| SigLIP image encoder | ~400M | FROZEN | `PaliGemma/img` |
| Whisper encoder | ~600M | FROZEN | `whisper_encoder` |
| AlignmentPooler | ~4.2M | YES but harmless | Under `llm`, caught by `Not(lora)` bypass — BUT pooler is under `alignment_pooler` which is at model root, NOT under `llm`. So it IS trainable but has no gradient path in `asr_alignment` stage |
| Action head/time MLP | ~2.3M | YES but harmless | Same — trainable but no gradient path in ASR stage |

**Total trainable Stage 2**: ~51.7M, of which ~45.2M effective (LoRA + projector)

**Stage 3 (`pi05_audio_stage3_libero`)**:
| Component | Params | Trainable? | Notes |
|-----------|--------|-----------|-------|
| Gemma LoRA | ~45.2M | YES | Pre-trained from Stage 2 |
| Action expert LoRA | ~11.3M | YES | Fresh zero-B init |
| Audio projector | ~17.3M | YES at 0.1x LR | `lr_scale_overrides`: 2e-6 effective |
| Action head + time MLP | ~2.3M | YES | Full LR |
| Gemma base | ~2B | FROZEN | |
| Action expert base | ~300M | FROZEN | Under `llm` (suffix `_1` layers) |
| SigLIP | ~400M | FROZEN | |
| Whisper | ~600M | FROZEN | |

**Total trainable Stage 3**: ~73.8M

### Action Expert Freeze Verification

Critical question: Is the action expert base truly frozen in Stage 3?

**Answer: YES.** Pi0.5's action expert lives INSIDE `PaliGemma.llm.layers` with suffix `_1` (e.g., `layers_1/0/...`). The freeze filter `All(.*llm.*, Not(.*lora.*))` catches both:
- Gemma layers: `llm/layers/0/attn/...` (base frozen, LoRA trainable)
- Action expert layers: `llm/layers_1/0/attn/...` (base frozen, LoRA trainable)

Both experts' LoRA adapters are trainable; both experts' base weights are frozen.

---

## 2. Data Pipeline Audit

### Stage 2 Data: Mixed ASR

| Source | Samples | Audio Length | Tokens/sample | % by count | % by tokens |
|--------|---------|-------------|--------------|-----------|------------|
| LibriSpeech train-clean-360 | 97,243 | ~12.5s avg | ~45 | 25% (target) | ~51% |
| DROID TTS (10k instructions × 10 voices) | 100,000 | ~3.7s avg | ~14 | 75% (target) | ~49% |
| **Total** | **197,243** | — | — | 100% | ~50/50 |

**Ratio implementation**: Per-sample Bernoulli draw (`random.random() < 0.25`). Verified: 23.7%/76.3% over 1000 samples. Not exact per-batch but converges over training.

**Dataset class**: `MixedASRDataset` in `src/openpi/training/mixed_asr_dataset.py`
- LibriSpeech: reads `.flac` files + `.trans.txt` transcriptions
- DROID TTS: reads `manifest.json` → list of `.mp3` audio paths per instruction
- Both produce identical dict format: `{audio_path, prompt, state, actions, image, image_mask}`
- Dummy images/state/actions for ASR training (no robot data needed)

**Data loader routing**: `data_loader.py` routes `repo_id="mixed_asr"` to `MixedASRDataset`. Norm stats skipped for ASR datasets.

### Stage 3 Data: LIBERO + TTS

| Component | Details |
|-----------|---------|
| Robot data | LeRobot `physical-intelligence/libero`: **40 tasks**, 1,693 episodes, 273,465 frames |
| TTS train | 2,240 `.wav` files (112 instructions × 20 Piper VCTK speakers) |
| TTS eval | 1,120 `.mp3` files (112 instructions × 10 edge-tts held-out accent voices) |
| Audio/text mixing | 60% audio / 40% text, mutually exclusive (VLAS design) |

**LIBERO task count correction**: The original LIBERO paper describes 130 tasks across 5 suites. However, the LeRobot dataset `physical-intelligence/libero` contains only **40 tasks** (from `libero_spatial` + `libero_object` suites). All 40 LeRobot tasks are covered by our TTS synthesis (0 missing instructions).

**TTS coverage verification**:
```
LeRobot LIBERO tasks: 40
Unique instructions: 40 (all unique)
TTS instructions: 112
Instructions in TTS but not LeRobot: 72 (from other LIBERO suites — harmless extra)
Instructions in LeRobot but not TTS: 0 (complete coverage)
```

### TTS Data Summary

| Dataset | Files | Engine | Voices | Accent | Status |
|---------|-------|--------|--------|--------|--------|
| DROID train | 100,000 .mp3 | edge-tts | 10 American voices | US English | EXPANDING (was 30k with 3 voices) |
| LIBERO train | 2,240 .wav | Piper VCTK | 20 speakers | Multi-accent (UK, Indian, etc.) | COMPLETE |
| LIBERO eval | 1,120 .mp3 | edge-tts | 10 voices | Held-out (GB/AU/IN/IE/ZA) | COMPLETE |

**DROID voices (10 total)**:
- Original 3: Aria, Guy, Jenny
- Expansion 7: Ava, Andrew, Emma, Brian, Christopher, Michelle, Roger
- All `en-US-*Neural` (American English, non-Multilingual variants)

---

## 3. Step Count & Epoch Audit

### Stage 2: Mixed ASR (7,500 steps, batch 32)

| Source | Samples | Steps at 75%/25% | Presentations | Epochs |
|--------|---------|-------------------|--------------|--------|
| DROID TTS | 100,000 | 5,625 (75%) | 180,000 | **1.8** |
| LibriSpeech | 97,243 | 1,875 (25%) | 60,000 | **0.62** |

**Verdict**: Both under 2-epoch safety limit. Previously DROID was at 6.0 epochs (30k files) — the 10-voice expansion (Option A) fixes this.

### Stage 3: LIBERO (30,000 steps, batch 32)

| Component | Samples | Presentations | Epochs |
|-----------|---------|--------------|--------|
| LIBERO robot data | 273,465 frames | 960,000 | **3.5** |
| TTS audio (60%) | 2,240 files | ~576,000 | ~257 |
| Text (40%) | 40 instructions | ~384,000 | — |

**TTS epoch count is high** (257 epochs over 2,240 files), but this is expected for Stage 3:
- TTS is an augmentation, not the primary learning signal
- Each audio file pairs with a different image/state context → unique training signal
- VLAS used similar TTS sample reuse without issues
- The primary data (robot frames at 3.5 epochs) is the bottleneck

### Why 7,500 Steps for Stage 2?

| Factor | Consideration |
|--------|--------------|
| Data size | 197k samples ÷ 32 batch = 6,164 steps/epoch → 7,500 = 1.2 epochs |
| Comparison | Stage 2 v1 (failed) used 10k steps on 97k samples = 3.3 epochs. Overfit risk. |
| LoRA convergence | LoRA typically converges fast (1-2 epochs). 7,500 steps = 1.2 epochs is conservative. |
| Checkpoint safety | save_interval=500 → 15 checkpoints for early stopping |

---

## 4. Time Audit

### Training Time Estimates

| Stage | Steps | Time/step | Total | Basis |
|-------|-------|-----------|-------|-------|
| Stage 2 | 7,500 | 0.59 s/step | **1.2 hours** | Measured from Stage 2 v1 (identical architecture) |
| Stage 3 | 30,000 | 0.8–1.0 s/step | **6.7–8.3 hours** | Estimated: dual LoRA + LIBERO images + flow matching is heavier |

**Stage 2 speed basis**: Stage 2 v1 (`pi05_audio_stage2_asr_finetune`) ran 10k steps in ~98 minutes = 0.59 s/step. Stage 2 v2 (`pi05_audio_mixed_asr`) uses identical architecture: same model (`gemma_2b_lora`), same batch size (32), same training stage (`asr_alignment`), same hardware (8x L40S). The only difference is data composition, which doesn't affect step time.

**Stage 3 speed estimate**: More expensive than Stage 2 because:
- Dual LoRA (Gemma + action expert) vs single LoRA
- Real LIBERO images (224×224×3, 3 cameras) vs dummy images
- Flow matching loss vs CE loss
- Action expert forward pass in addition to Gemma

### Synthesis Time

| Task | Time | Status |
|------|------|--------|
| DROID TTS original (30k, 3 voices) | 13 min | COMPLETE |
| DROID TTS expansion (70k, 7 voices) | ~5 min | IN PROGRESS (parallel 7-process) |
| LIBERO train TTS | 14 min | COMPLETE |
| LIBERO eval TTS | 5 min | COMPLETE |

### Total Pipeline Time

| Phase | Time |
|-------|------|
| DROID TTS expansion | ~5 min |
| Stage 2 training | ~1.2 hours |
| Stage 2 diagnostics | ~10 min |
| Stage 3 training | ~7–8 hours |
| Stage 3 evaluation | ~1–2 hours |
| **Total** | **~10–12 hours** |

---

## 5. DROID Data Quantity Problem & Resolution

### The Problem

With 30k DROID TTS files (3 voices × 10k instructions):
- Stage 2: 7,500 steps × 32 batch × 75% DROID = 180,000 DROID presentations
- 180,000 ÷ 30,000 files = **6.0 epochs** — exceeds 2-epoch safety limit
- Risk: 3-voice overfitting (model memorizes speaker characteristics of Aria/Guy/Jenny)
- TTS-generated speech already has limited acoustic diversity; 6 epochs compounds this

### Options Evaluated

| Option | Description | DROID Epochs | Risk | Time Cost |
|--------|-------------|-------------|------|-----------|
| **A: More voices** | 10 voices × 10k = 100k files | **1.8** | Low | ~5 min synthesis |
| B: Fewer steps | 2,500 steps (÷3) | **2.0** | Training from random init with only 2,500 steps may be insufficient | 0 |
| C: Accept 6 epochs | Keep 30k files | **6.0** | 3-voice overfitting, limited generalization | 0 |

### Decision: Option A (approved)

Generate 7 additional American English voices via edge-tts:
- Ava, Andrew, Emma, Brian, Christopher, Michelle, Roger
- Same prompt set (10k instructions), same directory structure
- Parallel synthesis: 7 Python processes × 32 concurrent connections = ~229 files/s
- Manifest updated to include all 10 voices

**Result**: 100k DROID TTS files → 1.8 epochs in Stage 2.

---

## 6. Bugs Found & Fixed

### BUG 1: Stage 3 Weight Loader — FIXED

**Both Stage 3 configs** pointed to wrong/stale checkpoint paths:

| Config | Old Path (WRONG) | New Path (CORRECT) |
|--------|-----------------|-------------------|
| `pi05_audio_stage3_aloha` | `./checkpoints/pi05_audio_stage2_asr_finetune/stage2_asr_finetune/latest/params` | `./checkpoints/pi05_audio_mixed_asr/mixed_asr/7500/params` |
| `pi05_audio_stage3_libero` | `./checkpoints/pi05_audio_joint_asr/joint_asr/latest/params` | `./checkpoints/pi05_audio_mixed_asr/mixed_asr/7500/params` |

**Root cause**: Config names evolved (joint_asr → stage2_asr_finetune → mixed_asr) but weight_loader paths weren't updated.

**Override at launch**: `--weight-loader.params-path=./checkpoints/.../params`

### BUG 2: Audio/Text Mutual Exclusivity — FIXED (earlier)

When `AudioTextMixingTransform` assigned TTS audio, it kept the text prompt. Model could cheat by using text instead of learning audio. Fixed: `data["prompt"] = np.asarray("")` when audio is assigned.

### Non-Bug: Harmless Trainable Params in Stage 2

`AlignmentPooler` and `action_head/time_mlp` are technically trainable in Stage 2 (not matched by freeze filter) but have **no gradient path** in `asr_alignment` training stage. CE loss only flows through Gemma LoRA → projector. These params will remain at their initialized values. Not worth adding to the freeze filter as it would add complexity for zero benefit.

---

## 7. Go/No-Go Criteria

### Stage 2 → Stage 3 Transition

| Criterion | Threshold | How to check |
|-----------|----------|--------------|
| Loss | Below 4.0 (random chance ~12.5) | Training log |
| Greedy decode | Varied output per sample | `scripts/diag_decode.py` |
| Audio token norms | In Gemma range (~40-240) | `scripts/diag_embed_align.py` |
| Per-sample discrimination | Different audio → different text | `scripts/diag_decode.py` |

### Stage 3 → Evaluation

| Criterion | Threshold | How to check |
|-----------|----------|--------------|
| Training loss | Decreasing, below text-only baseline | Training log |
| Text eval | No degradation vs pre-Stage-3 | `examples/libero/main.py --eval-mode text` |
| Audio eval | Positive success rate | `examples/libero/main.py --eval-mode audio` |
| Audio vs text gap | Audio success ≥ 50% of text success | Both eval modes |

---

## 8. Known Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Stage 2 loss plateau (like v1) | LOW | HIGH | Fresh init + mixed data addresses root cause. Monitor loss every 500 steps. |
| DROID TTS voice homogeneity | LOW | MEDIUM | 10 voices (was 3). All American EN by design (matches LibriSpeech). |
| Stage 3 audio ignored | LOW | HIGH | 92% LIBERO tasks share scenes → instruction essential. Mutual exclusivity enforced. |
| Projector destabilized in Stage 3 | MEDIUM | MEDIUM | 10x lower LR (2e-6). Can freeze entirely if norms diverge. |
| TTS vs real speech gap | MEDIUM | MEDIUM | Held-out accent eval voices. Future: add noise augmentation or real recordings. |
| LoRA rank too small | MEDIUM | MEDIUM | Rank 16 may be insufficient. Can increase to 32/64 if collapse recurs. |

---

## 9. File Inventory

### Training Configs (`src/openpi/training/config.py`)
| Config | Stage | Status |
|--------|-------|--------|
| `pi05_audio_stage1_embed_align` | Stage 1 (MSE) | COMPLETE |
| `pi05_audio_stage2_asr_finetune` | Stage 2 v1 (LibriSpeech-only) | FAILED |
| `pi05_audio_mixed_asr` | Stage 2 v2 (mixed ASR) | READY |
| `pi05_audio_stage3_aloha` | Stage 3 (ALOHA) | CONFIG READY |
| `pi05_audio_stage3_libero` | Stage 3 (LIBERO) | CONFIG READY |

### Scripts
| Script | Purpose |
|--------|---------|
| `scripts/train.py` | Training entry point |
| `scripts/diag_decode.py` | Teacher-forced + greedy decode diagnostics |
| `scripts/diag_embed_align.py` | Embedding alignment diagnostics |
| `scripts/sanity_check_mixed_asr.py` | Mixed dataset ratio + weight init verification |
| `scripts/synthesize_droid_edge.py` | DROID TTS (3 voices, edge-tts) |
| `scripts/synthesize_droid_edge_expand.py` | DROID TTS expansion (7 voices, parallel) |
| `scripts/synthesize_piper.py` | LIBERO train TTS (Piper VCTK) |
| `scripts/synthesize_edge_tts.py` | LIBERO eval TTS (edge-tts held-out accents) |
| `scripts/bench_edge_tts.py` | edge-tts concurrency benchmark |

### Data Paths
| Data | Path |
|------|------|
| LibriSpeech train-clean-100 | `/home/user1/workspace/VLA/data/librispeech/LibriSpeech/train-clean-100` |
| LibriSpeech train-clean-360 | `/home/user1/workspace/VLA/data/librispeech/LibriSpeech/train-clean-360` |
| DROID TTS (train) | `/home/user1/workspace/VLA/data/tts/droid_train` |
| DROID instructions | `/home/user1/workspace/VLA/data/tts/droid_instructions_10k.txt` |
| LIBERO TTS (train) | `/home/user1/workspace/VLA/data/tts/libero_train` |
| LIBERO TTS (eval) | `/home/user1/workspace/VLA/data/tts/libero_eval` |
| LIBERO instructions | `/home/user1/workspace/VLA/data/tts/libero_instructions.txt` |

### Checkpoints
| Checkpoint | Path | Status |
|------------|------|--------|
| Pi0.5 base | `gs://openpi-assets/checkpoints/pi05_base/params` | Downloaded |
| Stage 1 v1 (failed) | `checkpoints/pi05_audio_stage1_asr/stage1_asr_ce_5k/4999/params` | Archived |
| Stage 1 v2 | `checkpoints/pi05_audio_stage1_embed_align/stage1_embed_align_5k/` | COMPLETE |
| Stage 2 v1 (failed) | `checkpoints/pi05_audio_stage2_asr_finetune/...` | Archived |
| Stage 2 v2 | `checkpoints/pi05_audio_mixed_asr/mixed_asr/7500/params` | PENDING |
| Stage 3 | `checkpoints/pi05_audio_stage3_libero/stage3_libero/` | PENDING |

---

## 10. Launch Sequence

### Step 1: Confirm DROID TTS Expansion
```bash
python3 -c "import json; m=json.load(open('/home/user1/workspace/VLA/data/tts/droid_train/manifest.json')); print(f'{sum(len(v) for v in m.values())} files, {len(m)} prompts')"
# Expected: 100000 files, 10000 prompts
```

### Step 2: Launch Stage 2 Training
```bash
tmux new-session -d -s mixed_asr "
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_audio_mixed_asr \
  --data.data-dir=/home/user1/workspace/VLA/data/librispeech/LibriSpeech/train-clean-360 \
  --data.droid-tts-dir=/home/user1/workspace/VLA/data/tts/droid_train \
  --no-wandb-enabled --exp-name=mixed_asr \
  2>&1 | tee logs/stage2_mixed_asr.log; exec bash"
```

### Step 3: Monitor Stage 2
```bash
# Check loss
grep 'Step ' logs/stage2_mixed_asr.log | tail -5

# Go/no-go at step 2000: loss should be below 6.0
# Go/no-go at step 5000: loss should be below 4.0
```

### Step 4: Run Stage 2 Diagnostics
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/diag_decode.py \
  --checkpoint checkpoints/pi05_audio_mixed_asr/mixed_asr/7500/params \
  --config pi05_audio_mixed_asr \
  --data-dir /home/user1/workspace/VLA/data/librispeech/LibriSpeech/train-clean-360
# Must see: varied greedy decode output per sample
```

### Step 5: Launch Stage 3 Training
```bash
tmux new-session -d -s stage3 "
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_audio_stage3_libero \
  --data.tts-cache-dir=/home/user1/workspace/VLA/data/tts/libero_train \
  --data.audio-ratio=0.6 \
  --no-wandb-enabled --exp-name=stage3_libero \
  2>&1 | tee logs/stage3_libero.log; exec bash"
```

### Step 6: Evaluate
```bash
# Text baseline
.venv/bin/python examples/libero/main.py --eval-mode text

# Audio only (held-out accents)
.venv/bin/python examples/libero/main.py --eval-mode audio \
  --audio-dir /home/user1/workspace/VLA/data/tts/libero_eval

# Both
.venv/bin/python examples/libero/main.py --eval-mode both \
  --audio-dir /home/user1/workspace/VLA/data/tts/libero_eval
```
