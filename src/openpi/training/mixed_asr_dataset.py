"""Mixed ASR dataset: LibriSpeech + DROID TTS at configurable ratio.

Combines real speech (LibriSpeech) with synthesized robot instructions (DROID TTS)
for joint ASR training. Sampling ratio is enforced at runtime per-sample, not
by pre-mixing the datasets.

Usage in config:
    MixedASRDataConfig(
        librispeech_dir="/path/to/train-clean-360",
        droid_tts_dir="/path/to/droid_train",
        librispeech_ratio=0.25,  # 25% LibriSpeech, 75% DROID TTS
    )
"""

import json
import logging
import pathlib
import random
from typing import SupportsIndex

import numpy as np

logger = logging.getLogger(__name__)

_IMAGE_H, _IMAGE_W = 224, 224
_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


class DroidTTSDataset:
    """Dataset over pre-synthesized DROID TTS audio files.

    Reads a manifest.json mapping instruction text -> list of audio file paths.
    Each (instruction, audio_file) pair is one sample.
    """

    def __init__(self, tts_dir: str, action_dim: int = 32, action_horizon: int = 50):
        self._action_dim = action_dim
        self._action_horizon = action_horizon
        self._samples: list[tuple[str, str]] = []  # (audio_path, instruction)

        manifest_path = pathlib.Path(tts_dir) / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest.json found in {tts_dir}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        for instruction, audio_paths in manifest.items():
            for audio_path in audio_paths:
                if pathlib.Path(audio_path).exists():
                    self._samples.append((audio_path, instruction))

        logger.info(f"DroidTTSDataset: loaded {len(self._samples)} samples "
                    f"({len(manifest)} instructions) from {tts_dir}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()
        audio_path, instruction = self._samples[idx]

        dummy_image = np.zeros((_IMAGE_H, _IMAGE_W, 3), dtype=np.uint8)
        dummy_mask = np.bool_(False)

        return {
            "audio_path": audio_path,
            "prompt": instruction,
            "state": np.zeros((self._action_dim,), dtype=np.float32),
            "actions": np.zeros((self._action_horizon, self._action_dim), dtype=np.float32),
            "image": {k: dummy_image for k in _IMAGE_KEYS},
            "image_mask": {k: dummy_mask for k in _IMAGE_KEYS},
        }


class MixedASRDataset:
    """Combines LibriSpeech and DROID TTS datasets with runtime ratio sampling.

    Each __getitem__ call draws from LibriSpeech with probability `librispeech_ratio`
    and from DROID TTS with probability `1 - librispeech_ratio`. The index is mapped
    to the appropriate sub-dataset using modular arithmetic to ensure full coverage.

    The reported __len__ is the sum of both datasets so the dataloader sees all samples
    over one epoch. The ratio is enforced probabilistically per sample.
    """

    def __init__(
        self,
        librispeech_dir: str,
        droid_tts_dir: str,
        librispeech_ratio: float = 0.25,
        action_dim: int = 32,
        action_horizon: int = 50,
        seed: int = 42,
    ):
        from openpi.training.librispeech_dataset import LibriSpeechDataset

        self._librispeech = LibriSpeechDataset(
            data_dir=librispeech_dir,
            action_dim=action_dim,
            action_horizon=action_horizon,
        )
        self._droid = DroidTTSDataset(
            tts_dir=droid_tts_dir,
            action_dim=action_dim,
            action_horizon=action_horizon,
        )
        self._ratio = librispeech_ratio
        self._rng = random.Random(seed)

        total = len(self._librispeech) + len(self._droid)
        logger.info(
            f"MixedASRDataset: {len(self._librispeech)} LibriSpeech + "
            f"{len(self._droid)} DROID TTS = {total} total, "
            f"ratio={librispeech_ratio:.0%}/{1-librispeech_ratio:.0%}"
        )

    def __len__(self) -> int:
        return len(self._librispeech) + len(self._droid)

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()
        # Stochastic per-sample draw: ~25% LibriSpeech, ~75% DROID TTS.
        # Not exact per-batch, but converges over training.
        use_librispeech = self._rng.random() < self._ratio

        if use_librispeech:
            sub_idx = idx % len(self._librispeech)
            return self._librispeech[sub_idx]
        else:
            sub_idx = idx % len(self._droid)
            return self._droid[sub_idx]
