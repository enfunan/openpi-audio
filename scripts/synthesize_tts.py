#!/usr/bin/env python3
"""Pre-synthesize TTS audio for robot task prompts.

Uses edge-tts to generate speech audio with multiple speaker voices for each
unique task prompt in a LeRobot dataset. Outputs a manifest.json mapping each
prompt to a list of audio file paths.

Usage:
    python scripts/synthesize_tts.py \
        --repo_id your-lerobot-dataset \
        --output_dir ./tts_cache \
        --num_speakers 50

Requirements:
    pip install edge-tts lerobot
"""

import argparse
import asyncio
import hashlib
import json
import logging
import pathlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# A curated set of edge-tts voices with diverse accents and genders.
# These are a subset of the 300+ voices available in edge-tts.
VOICE_POOL = [
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-DavisNeural",
    "en-US-AmberNeural",
    "en-US-AnaNeural",
    "en-US-AndrewNeural",
    "en-US-BrandonNeural",
    "en-US-ChristopherNeural",
    "en-US-CoraNeural",
    "en-US-ElizabethNeural",
    "en-US-EricNeural",
    "en-US-JacobNeural",
    "en-US-MichelleNeural",
    "en-US-MonicaNeural",
    "en-US-RogerNeural",
    "en-US-SteffanNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-GB-LibbyNeural",
    "en-GB-AbbiNeural",
    "en-GB-AlfieNeural",
    "en-GB-BellaNeural",
    "en-GB-ElliotNeural",
    "en-GB-EthanNeural",
    "en-GB-HollieNeural",
    "en-GB-MaisieNeural",
    "en-GB-NoahNeural",
    "en-GB-OliverNeural",
    "en-GB-OliviaNeural",
    "en-GB-ThomasNeural",
    "en-AU-NatashaNeural",
    "en-AU-WilliamNeural",
    "en-AU-AnnetteNeural",
    "en-AU-CarlyNeural",
    "en-AU-DarrenNeural",
    "en-AU-DuncanNeural",
    "en-AU-ElsieNeural",
    "en-AU-FreyaNeural",
    "en-AU-JoanneNeural",
    "en-AU-KenNeural",
    "en-AU-KimNeural",
    "en-AU-NeilNeural",
    "en-AU-TimNeural",
    "en-IN-NeerjaNeural",
    "en-IN-PrabhatNeural",
    "en-CA-ClaraNeural",
    "en-CA-LiamNeural",
    "en-IE-ConnorNeural",
    "en-IE-EmilyNeural",
    "en-NZ-MitchellNeural",
    "en-NZ-MollyNeural",
    "en-ZA-LeahNeural",
    "en-ZA-LukeNeural",
    "en-SG-LunaNeural",
    "en-SG-WayneNeural",
    "en-PH-JamesNeural",
    "en-PH-RosaNeural",
    "en-KE-AsiliaNeural",
    "en-KE-ChilembaNeural",
    "en-HK-SamNeural",
    "en-HK-YanNeural",
]


def get_prompts_from_lerobot(repo_id: str) -> list[str]:
    """Extract unique task prompts from a LeRobot dataset."""
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    prompts = list(set(dataset_meta.tasks.values()))
    logger.info(f"Found {len(prompts)} unique prompts in {repo_id}")
    return sorted(prompts)


def get_prompts_from_file(prompts_file: str) -> list[str]:
    """Read prompts from a text file (one per line)."""
    with open(prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(prompts)} prompts from {prompts_file}")
    return prompts


async def synthesize_one(voice: str, text: str, output_path: pathlib.Path) -> bool:
    """Synthesize a single TTS audio file."""
    import edge_tts

    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
        return True
    except Exception as e:
        logger.warning(f"Failed to synthesize with voice {voice}: {e}")
        return False


async def synthesize_all(
    prompts: list[str],
    output_dir: pathlib.Path,
    num_speakers: int,
    max_concurrent: int = 10,
):
    """Synthesize TTS audio for all prompts with multiple speakers."""
    voices = VOICE_POOL[:num_speakers]
    if len(voices) < num_speakers:
        logger.warning(
            f"Requested {num_speakers} speakers but only {len(voices)} voices available. "
            f"Using {len(voices)} voices."
        )

    manifest: dict[str, list[str]] = {}
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(prompts) * len(voices)
    completed = 0

    async def _synth_with_sem(voice: str, text: str, path: pathlib.Path):
        nonlocal completed
        async with semaphore:
            result = await synthesize_one(voice, text, path)
            completed += 1
            if completed % 50 == 0:
                logger.info(f"Progress: {completed}/{total}")
            return result

    tasks = []
    for prompt in prompts:
        # Use a hash of the prompt for the directory name to avoid filesystem issues.
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
        prompt_dir = output_dir / prompt_hash
        prompt_dir.mkdir(parents=True, exist_ok=True)
        manifest[prompt] = []

        for i, voice in enumerate(voices):
            audio_path = prompt_dir / f"speaker_{i:03d}_{voice}.mp3"
            manifest[prompt].append(str(audio_path))

            if audio_path.exists():
                completed += 1
                continue

            tasks.append(_synth_with_sem(voice, prompt, audio_path))

    if tasks:
        logger.info(f"Synthesizing {len(tasks)} new audio files ({completed} already cached)...")
        await asyncio.gather(*tasks)
    else:
        logger.info("All audio files already cached.")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Pre-synthesize TTS audio for robot task prompts.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="LeRobot dataset repo ID to extract prompts from.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to a text file with one prompt per line (alternative to --repo_id).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for synthesized audio and manifest.",
    )
    parser.add_argument(
        "--num_speakers",
        type=int,
        default=50,
        help="Number of speaker voices to synthesize per prompt.",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=10,
        help="Maximum number of concurrent TTS requests.",
    )
    args = parser.parse_args()

    if args.repo_id is None and args.prompts_file is None:
        parser.error("Either --repo_id or --prompts_file must be provided.")

    if args.repo_id is not None:
        prompts = get_prompts_from_lerobot(args.repo_id)
    else:
        prompts = get_prompts_from_file(args.prompts_file)

    if not prompts:
        logger.error("No prompts found. Exiting.")
        return

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = asyncio.run(
        synthesize_all(prompts, output_dir, args.num_speakers, args.max_concurrent)
    )

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_files = sum(len(v) for v in manifest.values())
    logger.info(f"Done! Manifest written to {manifest_path}")
    logger.info(f"  {len(manifest)} prompts x {args.num_speakers} speakers = {total_files} audio files")


if __name__ == "__main__":
    main()
