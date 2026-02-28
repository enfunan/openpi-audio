#!/usr/bin/env python3
"""Expand DROID TTS with 7 additional voices, one process per voice in parallel.

Reuses the existing directory structure and updates manifest.json.
Each voice runs as an independent subprocess for maximum throughput.

Usage:
    python scripts/synthesize_droid_edge_expand.py \
        --prompts_file /path/to/droid_instructions_10k.txt \
        --output_dir /path/to/droid_train \
        --max_concurrent 32
"""

import argparse
import asyncio
import hashlib
import json
import logging
import multiprocessing
import pathlib
import time

import edge_tts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 7 new American English voices (non-Multilingual variants)
NEW_VOICES = [
    ("en-US-AvaNeural", "us_ava"),
    ("en-US-AndrewNeural", "us_andrew"),
    ("en-US-EmmaNeural", "us_emma"),
    ("en-US-BrianNeural", "us_brian"),
    ("en-US-ChristopherNeural", "us_christopher"),
    ("en-US-MichelleNeural", "us_michelle"),
    ("en-US-RogerNeural", "us_roger"),
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
                logger.warning(f"Failed: {voice} '{text[:40]}...': {e}")
                return False
    return False


async def synthesize_voice(voice_id: str, voice_label: str, prompts: list[str],
                           output_dir: pathlib.Path, max_concurrent: int):
    """Synthesize all prompts for a single voice."""
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(prompts)
    completed = [0]
    failed = [0]
    start = time.time()

    async def _synth(text, path):
        async with semaphore:
            ok = await synthesize_one(voice_id, text, path)
            completed[0] += 1
            if not ok:
                failed[0] += 1
            if completed[0] % 1000 == 0:
                elapsed = time.time() - start
                rate = completed[0] / elapsed
                eta = (total - completed[0]) / rate / 60 if rate > 0 else 0
                logger.info(f"[{voice_label}] {completed[0]}/{total} "
                            f"({failed[0]} failed), {rate:.1f}/s, ETA {eta:.1f}m")

    tasks = []
    for prompt in prompts:
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
        prompt_dir = output_dir / prompt_hash
        prompt_dir.mkdir(parents=True, exist_ok=True)
        audio_path = prompt_dir / f"{voice_label}.mp3"
        tasks.append(_synth(prompt, audio_path))

    # Process in batches
    batch_size = 200
    for i in range(0, len(tasks), batch_size):
        await asyncio.gather(*tasks[i:i + batch_size])

    elapsed = time.time() - start
    logger.info(f"[{voice_label}] DONE: {completed[0]} files in {elapsed:.0f}s "
                f"({completed[0]/elapsed:.1f}/s), {failed[0]} failed")
    return failed[0]


def run_voice(args_tuple):
    """Entry point for each subprocess."""
    voice_id, voice_label, prompts, output_dir, max_concurrent = args_tuple
    logger.info(f"[{voice_label}] Starting: {len(prompts)} files, concurrency={max_concurrent}")
    failed = asyncio.run(synthesize_voice(
        voice_id, voice_label, prompts, pathlib.Path(output_dir), max_concurrent
    ))
    return voice_label, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_concurrent", type=int, default=32)
    args = parser.parse_args()

    with open(args.prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]

    output_dir = pathlib.Path(args.output_dir)
    logger.info(f"Expanding DROID TTS: {len(prompts)} prompts × {len(NEW_VOICES)} voices "
                f"= {len(prompts) * len(NEW_VOICES)} new files")

    start = time.time()

    # Launch all 7 voices in parallel processes
    voice_args = [
        (vid, vlabel, prompts, str(output_dir), args.max_concurrent)
        for vid, vlabel in NEW_VOICES
    ]

    with multiprocessing.Pool(len(NEW_VOICES)) as pool:
        results = pool.map(run_voice, voice_args)

    elapsed = time.time() - start
    total_failed = sum(f for _, f in results)
    total_files = len(prompts) * len(NEW_VOICES)
    logger.info(f"All voices done: {total_files} files in {elapsed:.0f}s "
                f"({total_files/elapsed:.1f}/s), {total_failed} failed")

    # Update manifest with all 10 voices
    logger.info("Updating manifest.json with all 10 voices...")
    all_voices = [
        ("en-US-AriaNeural", "us_aria"),
        ("en-US-GuyNeural", "us_guy"),
        ("en-US-JennyNeural", "us_jenny"),
    ] + NEW_VOICES

    manifest = {}
    for prompt in prompts:
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
        prompt_dir = output_dir / prompt_hash
        paths = []
        for _, voice_label in all_voices:
            audio_path = prompt_dir / f"{voice_label}.mp3"
            if audio_path.exists() and audio_path.stat().st_size > 100:
                paths.append(str(audio_path))
        manifest[prompt] = paths

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    total_in_manifest = sum(len(v) for v in manifest.values())
    logger.info(f"Manifest: {manifest_path} ({len(manifest)} prompts, {total_in_manifest} files)")


if __name__ == "__main__":
    main()
