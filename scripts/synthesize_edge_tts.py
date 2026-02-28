#!/usr/bin/env python3
"""Synthesize TTS audio using edge-tts for held-out evaluation voices.

Uses explicit accent-diverse Azure voices that are NEVER used during training.

Usage:
    python scripts/synthesize_edge_tts.py \
        --prompts_file /path/to/libero_instructions.txt \
        --output_dir /path/to/libero_eval
"""

import argparse
import asyncio
import hashlib
import json
import logging
import pathlib
import time

import edge_tts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Held-out evaluation voices: explicit non-American accents, NEVER seen during training.
EVAL_VOICES = [
    ("en-GB-SoniaNeural", "british_f"),
    ("en-GB-RyanNeural", "british_m"),
    ("en-AU-NatashaNeural", "australian_f"),
    ("en-AU-WilliamNeural", "australian_m"),
    ("en-IN-NeerjaNeural", "indian_f"),
    ("en-IN-PrabhatNeural", "indian_m"),
    ("en-IE-ConnorNeural", "irish_m"),
    ("en-IE-EmilyNeural", "irish_f"),
    ("en-ZA-LeahNeural", "south_african_f"),
    ("en-ZA-LukeNeural", "south_african_m"),
]


async def synthesize_one(voice: str, text: str, output_path: pathlib.Path) -> bool:
    if output_path.exists() and output_path.stat().st_size > 100:
        return True
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
        return True
    except Exception as e:
        logger.warning(f"Failed {voice}: {e}")
        return False


async def main_async(prompts: list[str], output_dir: pathlib.Path):
    manifest = {}
    semaphore = asyncio.Semaphore(5)  # conservative to avoid rate limiting
    total = len(prompts) * len(EVAL_VOICES)
    completed = 0
    failed = 0

    async def _synth(voice_id, voice_label, text, path):
        nonlocal completed, failed
        async with semaphore:
            ok = await synthesize_one(voice_id, text, path)
            completed += 1
            if not ok:
                failed += 1
            if completed % 100 == 0:
                logger.info(f"Progress: {completed}/{total} ({failed} failed)")

    tasks = []
    for prompt in prompts:
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
        prompt_dir = output_dir / prompt_hash
        prompt_dir.mkdir(parents=True, exist_ok=True)
        manifest[prompt] = []

        for voice_id, voice_label in EVAL_VOICES:
            audio_path = prompt_dir / f"{voice_label}.mp3"
            manifest[prompt].append(str(audio_path))
            tasks.append(_synth(voice_id, voice_label, prompt, audio_path))

    start = time.time()
    await asyncio.gather(*tasks)
    elapsed = time.time() - start
    logger.info(f"Done! {completed} files in {elapsed:.1f}s, {failed} failed")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest: {manifest_path} ({len(manifest)} prompts, {sum(len(v) for v in manifest.values())} files)")


def main():
    parser = argparse.ArgumentParser(description="Synthesize held-out eval TTS with edge-tts")
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.prompts_file) as f:
        prompts = list(set(line.strip() for line in f if line.strip()))
    prompts.sort()
    logger.info(f"Loaded {len(prompts)} unique prompts")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(main_async(prompts, output_dir))


if __name__ == "__main__":
    main()
