#!/usr/bin/env python3
"""Synthesize DROID TTS audio using edge-tts with American English voices.

Generates 3 voice variants per instruction using only American English voices
(held-out accent voices are reserved for evaluation).

Usage:
    python scripts/synthesize_droid_edge.py \
        --prompts_file /path/to/droid_instructions_10k.txt \
        --output_dir /path/to/droid_train \
        --max_concurrent 8
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

# American English voices ONLY — accent-diverse voices are held out for eval.
AMERICAN_VOICES = [
    ("en-US-AriaNeural", "us_aria"),
    ("en-US-GuyNeural", "us_guy"),
    ("en-US-JennyNeural", "us_jenny"),
]


async def synthesize_one(voice: str, text: str, output_path: pathlib.Path, retries: int = 3) -> bool:
    if output_path.exists() and output_path.stat().st_size > 100:
        return True
    for attempt in range(retries):
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(output_path))
            return True
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))
            else:
                logger.warning(f"Failed after {retries} attempts: {voice} '{text[:40]}...': {e}")
                return False
    return False


async def main_async(prompts: list[str], output_dir: pathlib.Path, max_concurrent: int):
    manifest = {}
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(prompts) * len(AMERICAN_VOICES)
    completed = [0]
    failed = [0]
    start = time.time()

    async def _synth(voice_id, voice_label, text, path):
        async with semaphore:
            ok = await synthesize_one(voice_id, text, path)
            completed[0] += 1
            if not ok:
                failed[0] += 1
            if completed[0] % 500 == 0:
                elapsed = time.time() - start
                rate = completed[0] / elapsed
                eta = (total - completed[0]) / rate / 60 if rate > 0 else 0
                logger.info(f"Progress: {completed[0]}/{total} ({failed[0]} failed), "
                            f"{rate:.1f} files/s, ETA {eta:.1f} min")

    tasks = []
    for prompt in prompts:
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
        prompt_dir = output_dir / prompt_hash
        prompt_dir.mkdir(parents=True, exist_ok=True)
        manifest[prompt] = []

        for voice_id, voice_label in AMERICAN_VOICES:
            audio_path = prompt_dir / f"{voice_label}.mp3"
            manifest[prompt].append(str(audio_path))
            tasks.append(_synth(voice_id, voice_label, prompt, audio_path))

    logger.info(f"Synthesizing {total} files with {max_concurrent} concurrent connections...")

    # Process in batches to avoid overwhelming the event loop
    batch_size = 200
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        await asyncio.gather(*batch)

    elapsed = time.time() - start
    logger.info(f"Done! {completed[0]} files in {elapsed:.0f}s ({completed[0]/elapsed:.1f} files/s), "
                f"{failed[0]} failed")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    logger.info(f"Manifest: {manifest_path} ({len(manifest)} prompts, "
                f"{sum(len(v) for v in manifest.values())} files)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_concurrent", type=int, default=8)
    args = parser.parse_args()

    with open(args.prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(prompts)} prompts, {len(AMERICAN_VOICES)} voices each = "
                f"{len(prompts) * len(AMERICAN_VOICES)} total files")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(main_async(prompts, output_dir, args.max_concurrent))


if __name__ == "__main__":
    main()
