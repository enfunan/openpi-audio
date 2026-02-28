"""Sanity check: verify AudioTextMixingTransform removes text when audio is assigned.

Creates a temporary TTS manifest with fake audio paths, then runs the transform
on synthetic samples to verify:
1. Audio samples: audio_path present, prompt is empty string
2. Text samples: no audio_path, prompt preserved
"""

import json
import tempfile
import pathlib
import numpy as np

from openpi.transforms import AudioTextMixingTransform


def main():
    # Create a fake manifest with 3 task instructions.
    instructions = [
        "pick up the red mug and place it on the plate",
        "open the middle drawer of the cabinet",
        "turn on the stove and put the moka pot on it",
    ]
    manifest = {inst: [f"/fake/audio/{i}_speaker_{j}.wav" for j in range(5)] for i, inst in enumerate(instructions)}

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = pathlib.Path(tmpdir) / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        # Create transform with 60% audio ratio.
        transform = AudioTextMixingTransform(audio_ratio=0.6, tts_cache_dir=tmpdir)

        # Run 100 samples and collect results.
        audio_samples = []
        text_samples = []

        for i in range(100):
            inst = instructions[i % len(instructions)]
            data = {"prompt": np.asarray(inst), "image": f"img_{i}.jpg", "state": np.zeros(7)}
            result = transform(data)

            if "audio_path" in result:
                audio_samples.append(result)
            else:
                text_samples.append(result)

        # Report results.
        print(f"Total samples: 100")
        print(f"Audio samples: {len(audio_samples)} ({len(audio_samples)}%)")
        print(f"Text samples:  {len(text_samples)} ({len(text_samples)}%)")
        print()

        # Check audio samples: prompt must be empty, audio_path must exist.
        print("=" * 60)
        print("AUDIO SAMPLES (first 5):")
        print("=" * 60)
        all_audio_ok = True
        for i, s in enumerate(audio_samples[:5]):
            prompt_val = s["prompt"]
            if isinstance(prompt_val, np.ndarray):
                prompt_val = prompt_val.item()
            audio_path = s["audio_path"]
            prompt_empty = prompt_val == ""
            has_audio = bool(audio_path)

            status = "OK" if (prompt_empty and has_audio) else "FAIL"
            if status == "FAIL":
                all_audio_ok = False

            print(f"  Sample {i}: [{status}]")
            print(f"    prompt = {repr(prompt_val)}")
            print(f"    audio_path = {audio_path}")
            print()

        # Verify ALL audio samples have empty prompt.
        for s in audio_samples:
            p = s["prompt"]
            if isinstance(p, np.ndarray):
                p = p.item()
            if p != "":
                all_audio_ok = False

        print(f"All audio samples have empty prompt: {all_audio_ok}")
        print()

        # Check text samples: prompt must be preserved, no audio_path.
        print("=" * 60)
        print("TEXT SAMPLES (first 5):")
        print("=" * 60)
        all_text_ok = True
        for i, s in enumerate(text_samples[:5]):
            prompt_val = s["prompt"]
            if isinstance(prompt_val, np.ndarray):
                prompt_val = prompt_val.item()
            has_audio = "audio_path" in s
            prompt_nonempty = len(prompt_val) > 0
            no_audio = not has_audio

            status = "OK" if (prompt_nonempty and no_audio) else "FAIL"
            if status == "FAIL":
                all_text_ok = False

            print(f"  Sample {i}: [{status}]")
            print(f"    prompt = {repr(prompt_val)}")
            print(f"    audio_path = {s.get('audio_path', 'NOT PRESENT')}")
            print()

        # Verify ALL text samples have non-empty prompt.
        for s in text_samples:
            p = s["prompt"]
            if isinstance(p, np.ndarray):
                p = p.item()
            if p == "" or "audio_path" in s:
                all_text_ok = False

        print(f"All text samples have preserved prompt: {all_text_ok}")
        print()

        # Final verdict.
        print("=" * 60)
        if all_audio_ok and all_text_ok:
            print("PASS: Audio and text are mutually exclusive.")
        else:
            print("FAIL: Audio/text mixing is broken.")
            if not all_audio_ok:
                print("  - Some audio samples still have text prompt!")
            if not all_text_ok:
                print("  - Some text samples are missing prompt or have audio!")
        print("=" * 60)


if __name__ == "__main__":
    main()
