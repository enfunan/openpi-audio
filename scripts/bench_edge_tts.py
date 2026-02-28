#!/usr/bin/env python3
"""Benchmark edge-tts at different concurrency levels to find optimal setting."""

import asyncio
import tempfile
import time
import pathlib

import edge_tts

# 50 test prompts — enough to saturate concurrency
TEST_PROMPTS = [
    f"Pick up the {obj} and place it on the {loc}"
    for obj in ["red block", "blue cup", "green bottle", "yellow ball", "white plate"]
    for loc in ["table", "shelf", "counter", "tray", "drawer", "bin", "rack", "stand", "box", "mat"]
]

VOICE = "en-US-AriaNeural"


async def synthesize_one(voice, text, path, semaphore):
    async with semaphore:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(path))


async def bench(concurrency: int, num_files: int = 50):
    semaphore = asyncio.Semaphore(concurrency)
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks = []
        for i in range(num_files):
            path = pathlib.Path(tmpdir) / f"test_{i}.mp3"
            text = TEST_PROMPTS[i % len(TEST_PROMPTS)]
            tasks.append(synthesize_one(VOICE, text, path, semaphore))

        start = time.time()
        await asyncio.gather(*tasks)
        elapsed = time.time() - start

        rate = num_files / elapsed
        avg_latency = elapsed / num_files * concurrency  # avg per-request latency
        print(f"  concurrency={concurrency:3d}  |  {num_files} files in {elapsed:5.1f}s  |  "
              f"{rate:5.1f} files/s  |  avg_latency_per_req={avg_latency:.2f}s")
        return concurrency, rate


async def main():
    print("Benchmarking edge-tts concurrency (50 files each):\n")

    # Warmup
    print("  Warmup...")
    await bench(4, num_files=8)
    print()

    results = []
    for c in [8, 16, 24, 32, 48]:
        _, rate = await bench(c, num_files=50)
        results.append((c, rate))

    print(f"\nSummary:")
    best_c, best_rate = max(results, key=lambda x: x[1])
    for c, rate in results:
        marker = " <-- best" if c == best_c else ""
        print(f"  concurrency={c:3d}  ->  {rate:.1f} files/s{marker}")

    # Check if bottleneck is CPU or network
    print(f"\nBottleneck analysis:")
    rate_8 = next(r for c, r in results if c == 8)
    rate_32 = next(r for c, r in results if c == 32)
    speedup = rate_32 / rate_8
    print(f"  8->32 concurrency speedup: {speedup:.2f}x")
    if speedup > 2.5:
        print(f"  -> Network latency bound (more concurrency helps)")
    elif speedup > 1.5:
        print(f"  -> Mixed (some network, some CPU)")
    else:
        print(f"  -> CPU bound (more concurrency won't help much)")


if __name__ == "__main__":
    asyncio.run(main())
