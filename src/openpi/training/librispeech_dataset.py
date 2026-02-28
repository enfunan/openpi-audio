"""LibriSpeech dataset wrapper for ASR alignment training (Stage 1).

Returns dicts compatible with the openpi data pipeline, providing:
- audio_path: path to the FLAC audio file
- prompt: transcription text (used as the alignment target)
- Dummy state, actions, image, and image_mask fields for pipeline compatibility.
"""

import logging
import pathlib
from typing import SupportsIndex

import numpy as np

logger = logging.getLogger(__name__)

# Default image resolution expected by the model pipeline.
_IMAGE_H, _IMAGE_W = 224, 224
# Default image keys expected by the model.
_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


class LibriSpeechDataset:
    """A map-style dataset over LibriSpeech data for ASR alignment.

    Expects LibriSpeech directory structure:
        data_dir/
          <speaker-id>/
            <chapter-id>/
              <speaker-id>-<chapter-id>.trans.txt
              <speaker-id>-<chapter-id>-<utterance-id>.flac
              ...

    Each sample is a dict with:
        - "audio_path": str, path to .flac file
        - "prompt": str, lowercase transcription
        - "state": np.zeros (dummy)
        - "actions": np.zeros (dummy)
        - "image": dict of dummy images
        - "image_mask": dict of dummy masks
    """

    def __init__(self, data_dir: str, action_dim: int = 32, action_horizon: int = 50):
        self._data_dir = pathlib.Path(data_dir)
        self._action_dim = action_dim
        self._action_horizon = action_horizon
        self._samples: list[tuple[str, str]] = []  # (audio_path, transcription)
        self._load_samples()

    def _load_samples(self):
        """Scan the LibriSpeech directory for transcript files and build the sample list."""
        trans_files = sorted(self._data_dir.rglob("*.trans.txt"))
        if not trans_files:
            raise FileNotFoundError(
                f"No .trans.txt files found under {self._data_dir}. "
                "Make sure data_dir points to a LibriSpeech split (e.g., train-clean-100)."
            )

        for trans_file in trans_files:
            chapter_dir = trans_file.parent
            with open(trans_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) != 2:
                        continue
                    utterance_id, transcription = parts
                    audio_path = chapter_dir / f"{utterance_id}.flac"
                    if audio_path.exists():
                        self._samples.append((str(audio_path), transcription.lower()))

        logger.info(f"LibriSpeechDataset: loaded {len(self._samples)} samples from {self._data_dir}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()
        audio_path, transcription = self._samples[idx]

        # Dummy image data — the ASR alignment stage doesn't use images,
        # but the pipeline requires these fields.
        dummy_image = np.zeros((_IMAGE_H, _IMAGE_W, 3), dtype=np.uint8)
        dummy_mask = np.bool_(False)

        return {
            "audio_path": audio_path,
            "prompt": transcription,
            "state": np.zeros((self._action_dim,), dtype=np.float32),
            "actions": np.zeros((self._action_horizon, self._action_dim), dtype=np.float32),
            "image": {k: dummy_image for k in _IMAGE_KEYS},
            "image_mask": {k: dummy_mask for k in _IMAGE_KEYS},
        }
