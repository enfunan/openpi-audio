# Installation Guide: OpenPi-Audio

Setup instructions for the Pi0.5 + Audio VLA on a fresh server.

## Prerequisites

- Ubuntu 22.04
- NVIDIA GPU(s) with CUDA drivers (tested: L40S 46GB, driver 535.x)
- At least 50GB disk for code + venv, plus ~300GB for checkpoints
- SSH access to the server

## 1. Push to GitHub (from nnmc60)

```bash
cd ~/workspace/VLA/openpi

# Stage all modified and new files
git add -A

# Commit
git commit -m "Stage 3v2: float32 base matmuls + frozen audio pathway

- Float32 base computation in lora.py (Einsum + FeedForward) to prevent
  bfloat16 overflow when audio pathway is frozen at LR=0.0
- Audio projector LR 0.01 -> 0.0 (fully frozen)
- Auto ablation watcher, eval pipeline scripts
- STAGE3V2_FLOAT32_FIX.md documenting the fix and results

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"

# Push to your fork
git push mine main
```

## 2. Clone on a New Server

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone the repo
mkdir -p ~/workspace/VLA
cd ~/workspace/VLA
git clone git@github.com:enfunan/openpi-audio.git openpi
cd openpi

# Initialize submodules (includes third_party/libero)
git submodule update --init --recursive
```

## 3. Create Python Environment

```bash
cd ~/workspace/VLA/openpi

# Create venv with Python 3.11 and install all dependencies
uv sync

# Verify
.venv/bin/python -c "import jax; print('JAX devices:', jax.devices())"
.venv/bin/python -c "import torch; print('Torch:', torch.__version__)"
.venv/bin/python -c "import openpi; print('OpenPi OK')"
```

## 4. Install LIBERO Dependencies (for eval only)

```bash
# LIBERO requires specific versions
uv pip install robosuite==1.4.1 gym==0.25.2 easydict bddl

# IMPORTANT: Fix PyTorch 2.6+ breaking change in libero
# In third_party/libero/libero/libero/benchmark/__init__.py, change:
#   torch.load(...)  ->  torch.load(..., weights_only=False)

# Create LIBERO config (avoids interactive prompt)
mkdir -p ~/.libero
cat > ~/.libero/config.yaml << EOF
assets: $(pwd)/third_party/libero/libero/libero/assets
bddl_files: $(pwd)/third_party/libero/libero/libero/bddl_files
benchmark_root: $(pwd)/third_party/libero/libero/libero
datasets: $(pwd)/third_party/libero/libero/libero/../datasets
init_states: $(pwd)/third_party/libero/libero/libero/init_files
EOF
```

## 5. Download Checkpoints

Checkpoints are not in git (too large). Copy from nnmc60 or download:

```bash
# Option A: Copy from nnmc60
mkdir -p checkpoints/pi05_audio_mixed_asr/mixed_asr/14999
mkdir -p checkpoints/pi05_audio_stage3_libero/stage3_libero/29999

rsync -az nnmc60:~/workspace/VLA/openpi/checkpoints/pi05_audio_mixed_asr/mixed_asr/14999/ \
  checkpoints/pi05_audio_mixed_asr/mixed_asr/14999/

rsync -az nnmc60:~/workspace/VLA/openpi/checkpoints/pi05_audio_stage3_libero/stage3_libero/29999/ \
  checkpoints/pi05_audio_stage3_libero/stage3_libero/29999/

# Option B: Download base model (if starting fresh)
# The base Pi0.5 weights are downloaded automatically on first run from GCS.
```

## 6. Download/Generate TTS Data

```bash
# Option A: Copy pre-generated TTS from nnmc60
mkdir -p data/tts
rsync -az nnmc60:~/workspace/VLA/data/tts/ ../data/tts/

# Option B: Generate fresh TTS (see scripts in data/tts/)
# DROID: edge-tts, 10 American voices, 100k files
# LIBERO train: Piper VCTK, 20 speakers, 2240 files
# LIBERO eval: edge-tts, 10 held-out accents, 1120 files
```

## 7. Verify Setup

```bash
cd ~/workspace/VLA/openpi

# Quick import test
.venv/bin/python -c "
from openpi.models import pi0_config
from openpi.training import config
print('Config loaded:', [c.name for c in config.get_all_configs() if 'audio' in c.name])
"

# Run tests
.venv/bin/python -m pytest tests/ -x -q
```

## Common Commands

### Training

```bash
# Stage 2: Mixed ASR (15k steps)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_audio_mixed_asr \
  --no-wandb-enabled --exp-name=mixed_asr

# Stage 3: LIBERO robot training (30k steps)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_audio_stage3_libero \
  --data.tts-cache-dir=/path/to/data/tts/libero_train \
  --data.audio-ratio=0.6 \
  --no-wandb-enabled --exp-name=stage3v2_libero
```

### Serving + Eval

```bash
# Start model server (on GPU)
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  .venv/bin/python scripts/serve_policy.py \
  --port 8001 \
  policy:checkpoint \
  --policy.config pi05_audio_stage3_libero \
  --policy.dir checkpoints/pi05_audio_stage3_libero/stage3v2_libero/29999

# Run LIBERO eval (in another terminal)
MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES="" \
  .venv/bin/python examples/libero/main.py \
  --args.task-suite-name=libero_spatial \
  --args.num-trials-per-task=3 \
  --args.eval-mode=audio \
  --args.audio-dir=/path/to/data/tts/libero_eval \
  --args.port=8001

# Audio ablation diagnostic
.venv/bin/python scripts/diag_audio_ablation_stage3.py \
  checkpoints/pi05_audio_stage3_libero/stage3v2_libero/5000/params
```

## Known Issues

- **num_workers**: Must be 0 for audio configs (librosa shared memory corruption with >0)
- **LIBERO deps**: robosuite must be 1.4.1 (not 1.5.x), gym must be 0.25.2
- **MuJoCo rendering**: Use `MUJOCO_EGL_DEVICE_ID=N` (not `CUDA_VISIBLE_DEVICES=""`)
- **torch.load**: PyTorch 2.6+ needs `weights_only=False` in libero benchmark/__init__.py
- **Boolean flags**: Use `--no-wandb-enabled` (not `--wandb-enabled=False`)
- **Whisper OOM**: Use batch_size=8 in eval scripts

## Directory Structure

```
openpi/
├── src/openpi/          # Main source code
│   ├── models/          # Pi0, Gemma, Whisper, LoRA
│   ├── training/        # Training configs, data loaders
│   └── policies/        # Serving policies
├── scripts/             # Training, serving, diagnostic scripts
├── examples/libero/     # LIBERO eval script
├── third_party/libero/  # LIBERO benchmark (submodule)
├── packages/            # openpi-client package
├── checkpoints/         # Model checkpoints (not in git)
├── data/                # Training/eval data (not in git)
├── DESIGN_DECISIONS.md  # Technical journal
├── STAGE3V2_FLOAT32_FIX.md  # Float32 fix documentation
└── STAGE3_EVAL_RESULTS.md   # Eval results
```
