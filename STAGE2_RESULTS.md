# Stage 2 v2 Results — Mixed ASR Training

**Date**: 2026-02-28
**Config**: `pi05_audio_mixed_asr`
**Data**: 25% LibriSpeech train-clean-360 (97k) + 75% DROID TTS (100k) = 197k total
**Trainable**: Gemma LoRA (rank 16, fresh) + audio projector (fresh random init)
**Frozen**: Gemma base, action expert, SigLIP, Whisper
**Checkpoint**: `checkpoints/pi05_audio_mixed_asr/mixed_asr/7499/params`

---

## 1. Training Curve

| Step | Loss | Grad Norm | Notes |
|------|------|-----------|-------|
| 0 | 75.32 | 6869.9 | Fresh random init |
| 50 | 29.06 | 2014.8 | Rapid drop during warmup |
| 100 | 7.28 | 56.0 | Below random chance (~12.5) |
| 500 | 5.41 | 35.8 | End of warmup, already below v1's plateau |
| 1000 | ~4.7 | ~12 | |
| 2500 | 4.01 | 8.2 | Below go/no-go threshold (4.0) |
| 4000 | 3.87 | 7.0 | |
| 5000 | 3.78 | 6.3 | keep_period checkpoint |
| 7000 | 3.65 | 6.5 | |
| 7450 | 3.65 | 6.4 | Final logged step |

**Speed**: 0.59 s/step, total training time ~74 minutes.

**Loss never plateaued** — still decreasing at the final step. This is a critical improvement over Stage 2 v1 which plateaued at 5.6 from step 5000 onward.

### Comparison with Stage 2 v1 (FAILED)

| Metric | Stage 2 v1 | Stage 2 v2 |
|--------|-----------|-----------|
| Data | LibriSpeech-only (97k) | Mixed: 25% LibriSpeech + 75% DROID TTS (197k) |
| Init | Stage 1 v2 checkpoint (collapsed projector) | Fresh random (base Pi0.5 + Whisper) |
| Steps | 10,000 | 7,500 |
| Final loss | **5.6 (plateaued at step 5000)** | **3.65 (still decreasing)** |
| Loss trajectory | Plateau for 5000 steps | Monotonic decrease |

---

## 2. Diagnostic Results

### Step 500 Diagnostics

**Greedy decode**:
| Sample | GT | Decode |
|--------|-----|--------|
| 0 | "tom the piper's son" | "Put the blue bottle" |
| 100 | "when i heard him preach..." | "turn the blue bottle and put it to the right" |
| 500 | "the rusty bolt was shot back..." | "and the object of the table and put it to" |
| 1000 | "the implication of howard's..." | "i a the blue cup and put it on the" |
| 2000 | "the live light strikes..." | "turn the blue cup and put it on the table" |

- First tokens vary across samples (Put/turn/and/i/turn) — **not collapsed**
- All converge to DROID-style robot vocabulary — expected at 75% DROID data, step 500

**Audio ablation (step 500)**:
| Sample | Real audio loss | Zero audio loss | Delta |
|--------|----------------|----------------|-------|
| 0 | 5.80 | 16.09 | +10.29 |
| 100 | 8.18 | 12.02 | +3.84 |
| 500 | 8.46 | 13.79 | +5.34 |
| 1000 | 7.18 | 11.84 | +4.66 |
| 2000 | 9.29 | 14.18 | +4.89 |
| **Avg** | **7.78** | **13.58** | **+5.80** |

**Position 0 ablation (step 500)**:
| Sample | Real | Zero | Delta |
|--------|------|------|-------|
| 0 | 6.49 | 21.10 | +14.61 |
| 100 | 9.33 | 24.38 | +15.05 |
| 500 | 4.05 | 24.09 | +20.03 |
| 1000 | 4.13 | 24.09 | +19.96 |
| 2000 | 4.40 | 24.09 | +19.69 |
| **Avg** | **5.68** | **23.53** | **+17.85** |

### Step 7499 (Final) Diagnostics

**Greedy decode**:
| Sample | GT | Decode |
|--------|-----|--------|
| 0 | "tom the piper's son" | "and the first of the other" |
| 100 | "when i heard him preach..." | "and the other man had a long and long hair" |
| 500 | "the rusty bolt was shot back..." | "the man had a long and long hair and he" |
| 1000 | "the implication of howard's..." | "and the other thing was the most beautiful and most" |
| 2000 | "the live light strikes..." | "and the other thing was the most beautiful and most" |

- Shifted from DROID vocabulary to LibriSpeech-style prose
- Some variation (samples 0, 100, 500 differ) but samples 1000/2000 are identical
- Per-sample discrimination improved but not complete

**Audio ablation (step 7499)**:
| Sample | Real audio loss | Zero audio loss | Delta |
|--------|----------------|----------------|-------|
| 0 | 7.35 | 14.54 | +7.18 |
| 100 | 6.22 | 10.84 | +4.62 |
| 500 | 6.80 | 12.72 | +5.92 |
| 1000 | 6.29 | 10.40 | +4.11 |
| 2000 | 7.70 | 14.82 | +7.12 |
| **Avg** | **6.87** | **12.66** | **+5.79** |

**Position 0 ablation (step 7499)**:
| Sample | Real | Zero | Delta |
|--------|------|------|-------|
| 0 | 11.96 | 20.95 | +8.99 |
| 100 | 5.16 | 24.23 | +19.07 |
| 500 | 2.42 | 23.93 | +21.51 |
| 1000 | 2.71 | 23.93 | +21.23 |
| 2000 | 2.58 | 23.93 | +21.35 |
| **Avg** | **4.97** | **23.40** | **+18.43** |

### Audio Token Statistics

| Metric | Step 500 | Step 7499 |
|--------|---------|----------|
| Mean | 0.010 | 0.040 |
| Std | 1.85 | 3.36 |
| Norm | ~83.5 | **~152.1** |

Norms nearly doubled (83 → 152), moving toward text embedding range (~240). The projector is learning to scale its output to better match what Gemma expects.

---

## 3. Step 500 → Step 7499 Comparison

| Metric | Step 500 | Step 7499 | Trend |
|--------|---------|----------|-------|
| Loss | 5.41 | 3.65 | Improved |
| Mean ablation delta | +5.80 | +5.79 | Stable (audio consistently used) |
| Mean pos 0 delta | +17.85 | +18.43 | Slightly improved |
| Audio token norms | 83.5 | 152.1 | Growing toward text range (~240) |
| Greedy decode style | DROID robot vocabulary | LibriSpeech prose | Learning both domains |
| Per-sample variation | 5/5 different first tokens | 3/5 different outputs | Partial discrimination |
| Collapse? | No | No | Stable |

---

## 4. Go/No-Go Assessment for Stage 3

| Criterion | Threshold | Result | Status |
|-----------|----------|--------|--------|
| Final loss | Below 4.0 | **3.65** | **PASS** |
| Loss trajectory | No plateau | **Monotonic decrease** | **PASS** |
| Audio being used | Ablation delta > 0 | **+5.79 mean** | **PASS** |
| Position 0 audio-dependent | Large delta | **+18.43 mean** | **PASS** |
| No single-output collapse | Varied decode | **3/5 different** | **PASS** |
| Audio token norms | Moving toward text range | **152 (was 83)** | **PASS** |

### What Stage 2 Achieved

1. **Audio is strongly conditioned** — removing audio increases loss by +5.8 nats on average
2. **Position 0 is almost entirely audio-driven** — zero audio gives ~24 nats (random), real audio gives ~5 nats
3. **No collapse** — unlike Stage 2 v1's "Sub" for all samples, outputs vary
4. **Projector learned to scale** — norms grew from 83 to 152, approaching text embedding range
5. **Both domains learned** — model produces both DROID-style and LibriSpeech-style text

### What Stage 2 Did NOT Achieve

1. **Full per-sample ASR** — greedy decode doesn't produce the correct transcription for each sample
2. **Complete per-sample discrimination** — some sample pairs produce identical output (1000/2000)
3. **Exact norm matching** — norms at 152 vs text embeddings at ~240 (still a gap)

### Decision: Extend to 15,000 Steps

Despite passing all go/no-go criteria, the per-sample discrimination was deemed insufficient for Stage 3:
- Samples 1000/2000 produced identical greedy decode output
- Audio token norms at 152, still far from text range (~240)
- Loss was still decreasing (not saturated) — more training likely helps
- Risk: Stage 3 takes ~7-8 hours. If audio representations aren't discriminative enough, that time is wasted.

**Extended training**: 7,500 → 15,000 steps (resumed from step 7499 with `--resume`)
- Additional cost: ~73 minutes (7500 × 0.59s)
- DROID epochs at 15k: 3.6 (acceptable with 10 voices)
- LibriSpeech epochs at 15k: 1.2 (still low)
- LR schedule updated: cosine decay over 15k steps

---

## 5. Extended Training Results (Steps 7500–15000)

### Training Curve (extended)

| Step | Loss | Grad Norm |
|------|------|-----------|
| 7500 | 4.67 | 6.7 (first step after resume) |
| 8000 | 3.82 | 7.9 |
| 9450 | 3.45 | 7.4 |
| 14850 | 3.32 | 8.3 |
| 14950 | 3.43 | 8.0 |

Loss continued decreasing: 3.65 (step 7499) → 3.43 (step 14999). Total training time: ~148 min.

### Step 14999 Diagnostics

**Greedy decode**:
| Sample | GT | Decode |
|--------|-----|--------|
| 0 | "tom the piper's son" | "and the other thing" |
| 100 | "when i heard him preach..." | "and the other man had been a man of the" |
| 500 | "the rusty bolt was shot back..." | "the man had been a man of the same nature" |
| 1000 | "the implication of howard's..." | "and the other of the other the other of the" |
| 2000 | "the live light strikes..." | "and the other of the other the other of the" |

Samples 1000/2000 still identical. 3/5 unique outputs (same as step 7499).

**Audio ablation (step 14999)**:
| Sample | Real audio loss | Zero audio loss | Delta |
|--------|----------------|----------------|-------|
| 0 | 7.01 | 13.65 | +6.63 |
| 100 | 6.18 | 10.51 | +4.33 |
| 500 | 6.65 | 12.29 | +5.64 |
| 1000 | 6.18 | 11.06 | +4.88 |
| 2000 | 7.60 | 14.59 | +6.99 |
| **Avg** | **6.72** | **12.42** | **+5.70** |

**Position 0 ablation (step 14999)**:
| Sample | Real | Zero | Delta |
|--------|------|------|-------|
| 0 | 12.26 | 20.71 | +8.44 |
| 100 | 4.90 | 24.03 | +19.13 |
| 500 | 2.33 | 23.71 | +21.38 |
| 1000 | 3.04 | 23.71 | +20.67 |
| 2000 | 2.55 | 23.71 | +21.16 |
| **Avg** | **5.02** | **23.17** | **+18.16** |

**Audio token norms**: ~162 (up from 152 at step 7499)

### Full Comparison: Step 500 → 7499 → 14999

| Metric | Step 500 | Step 7499 | Step 14999 | Trend |
|--------|---------|----------|-----------|-------|
| Loss | 5.41 | 3.65 | **3.43** | Improving |
| Mean ablation delta | +5.80 | +5.79 | **+5.70** | Flat |
| Mean pos 0 delta | +17.85 | +18.43 | **+18.16** | Flat |
| Audio token norm | 83.5 | 152.1 | **162.0** | Slow growth |
| Unique greedy outputs | 5/5 | 3/5 | **3/5** | Flat |

### Conclusion

**Extending to 15k did NOT meaningfully improve per-sample discrimination:**
- Ablation deltas unchanged (~5.7 since step 500)
- Unique outputs unchanged (3/5 since step 7499)
- Norms grew marginally (152→162, still far from text ~240)
- Loss improved (3.65→3.43) but reflects better language modeling, not better audio discrimination

**The model has converged on its audio representation quality.** Further Stage 2 training shows diminishing returns. The audio processing infrastructure is functional — audio is strongly used for conditioning (+5.7 mean delta, +18.2 pos 0 delta) — but per-sample ASR discrimination is limited by the architecture/data.

**Decision: Proceed to Stage 3.** Stage 3's flow matching loss provides a fundamentally different learning signal (action prediction conditioned on audio) which may drive further audio-to-task discrimination that ASR CE loss alone cannot achieve.

---

## 6. Stage 3 Readiness

- **Checkpoint**: `checkpoints/pi05_audio_mixed_asr/mixed_asr/14999/params`
- **Stage 3 weight_loader**: Points to `./checkpoints/pi05_audio_mixed_asr/mixed_asr/14999/params`
- **Config**: `pi05_audio_stage3_libero` — ready
- **TTS data**: LIBERO train (2,240 files), LIBERO eval (1,120 files) — ready
- **Estimated time**: ~7-8 hours (30k steps × ~0.8-1.0 s/step)
