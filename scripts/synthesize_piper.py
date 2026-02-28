#!/usr/bin/env python3
"""Synthesize TTS audio using Piper (local, CPU-based, fast).

Supports two modes:
  - libritts_r (904 speakers): for DROID training data
  - vctk (109 speakers): for LIBERO training data

Usage:
    python scripts/synthesize_piper.py \
        --model libritts_r \
        --prompts_file /path/to/instructions.txt \
        --output_dir /path/to/output \
        --voices_per_prompt 10 \
        --workers 16
"""

import argparse
import hashlib
import json
import logging
import pathlib
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = pathlib.Path("/home/user1/workspace/VLA/data/tts/piper_models")

MODELS = {
    "libritts_r": {
        "onnx": MODEL_DIR / "en_US-libritts_r-medium.onnx",
        "config": MODEL_DIR / "en_US-libritts_r-medium.onnx.json",
    },
    "vctk": {
        "onnx": MODEL_DIR / "en_GB-vctk-medium.onnx",
        "config": MODEL_DIR / "en_GB-vctk-medium.onnx.json",
    },
}


def get_speaker_ids(config_path: pathlib.Path) -> list[int]:
    """Extract available speaker IDs from the model config."""
    with open(config_path) as f:
        config = json.load(f)
    speaker_id_map = config.get("speaker_id_map", {})
    if not speaker_id_map:
        return [0]
    return sorted(speaker_id_map.values())


def synthesize_one(args_tuple):
    """Synthesize a single utterance. Called from process pool."""
    model_path, text, speaker_id, output_path = args_tuple
    output_path = pathlib.Path(output_path)
    if output_path.exists() and output_path.stat().st_size > 100:
        return True  # already cached

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "piper",
                "--model", str(model_path),
                "--speaker", str(speaker_id),
                "--output_file", str(output_path),
            ],
            input=text,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"Failed: {output_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Synthesize TTS with Piper")
    parser.add_argument("--model", choices=["libritts_r", "vctk"], required=True)
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--voices_per_prompt", type=int, default=10)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_info = MODELS[args.model]
    model_path = model_info["onnx"]
    speaker_ids = get_speaker_ids(model_info["config"])
    logger.info(f"Model: {args.model}, {len(speaker_ids)} available speakers")

    with open(args.prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(prompts)} prompts")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    manifest = {}
    tasks = []

    for prompt in prompts:
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
        prompt_dir = output_dir / prompt_hash
        prompt_dir.mkdir(parents=True, exist_ok=True)
        manifest[prompt] = []

        # Pick random speakers for this prompt
        selected_speakers = rng.sample(speaker_ids, min(args.voices_per_prompt, len(speaker_ids)))

        for i, spk_id in enumerate(selected_speakers):
            audio_path = prompt_dir / f"speaker_{spk_id:04d}.wav"
            manifest[prompt].append(str(audio_path))
            tasks.append((str(model_path), prompt, spk_id, str(audio_path)))

    # Check how many are already cached
    cached = sum(1 for _, _, _, p in tasks if pathlib.Path(p).exists() and pathlib.Path(p).stat().st_size > 100)
    to_synthesize = len(tasks) - cached
    logger.info(f"Total: {len(tasks)} files, {cached} cached, {to_synthesize} to synthesize")

    if to_synthesize == 0:
        logger.info("All files already cached!")
    else:
        start_time = time.time()
        completed = 0
        failed = 0

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(synthesize_one, t): t for t in tasks}
            for future in as_completed(futures):
                success = future.result()
                completed += 1
                if not success:
                    failed += 1
                if completed % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (len(tasks) - completed) / rate if rate > 0 else 0
                    logger.info(f"Progress: {completed}/{len(tasks)} ({failed} failed), "
                                f"{rate:.1f} files/s, ETA {eta/60:.1f} min")

        elapsed = time.time() - start_time
        logger.info(f"Done! {completed} files in {elapsed:.1f}s ({completed/elapsed:.1f} files/s), {failed} failed")

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_files = sum(len(v) for v in manifest.values())
    logger.info(f"Manifest: {manifest_path} ({len(manifest)} prompts, {total_files} files)")


if __name__ == "__main__":
    main()
