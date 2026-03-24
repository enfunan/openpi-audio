"""Precompute Whisper encoder embeddings for all TTS audio files.

Runs Whisper v3 once on every TTS audio file and saves the encoder hidden states
as .npy files. During training, these cached embeddings replace the live Whisper
forward pass, giving ~2x speedup and freeing ~3-5GB GPU memory.

Usage:
    # On GPU (fast, ~10 min):
    python scripts/precompute_whisper_cache.py --tts-dir data/tts/libero_train --output-dir data/whisper_cache/libero_train

    # On CPU (slow, ~2h):
    JAX_PLATFORMS=cpu python scripts/precompute_whisper_cache.py --tts-dir data/tts/libero_train --output-dir data/whisper_cache/libero_train

Output structure mirrors input:
    data/whisper_cache/libero_train/
        <episode_id>/
            speaker_0003.npy   # shape (1500, 1280), float32
            speaker_0008.npy
            ...
"""

import argparse
import logging
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import librosa
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def compute_mel_spectrogram(waveform: np.ndarray, n_mels: int = 128, n_fft: int = 400, hop_length: int = 160, sample_rate: int = 16000) -> np.ndarray:
    """Compute log-mel spectrogram matching Whisper v3 preprocessing."""
    max_samples = 30 * sample_rate  # 30s
    if len(waveform) < max_samples:
        waveform = np.pad(waveform, (0, max_samples - len(waveform)))
    else:
        waveform = waveform[:max_samples]

    # STFT
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    stft_frames = []
    for i in range(0, len(waveform) - n_fft + 1, hop_length):
        frame = waveform[i:i + n_fft] * window
        spectrum = np.fft.rfft(frame)
        stft_frames.append(spectrum)

    if not stft_frames:
        return np.zeros((n_mels, 3000), dtype=np.float32)

    magnitudes = np.abs(np.array(stft_frames).T) ** 2

    # Mel filterbank
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    low_freq_mel = 0.0
    high_freq_mel = hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        for j in range(bin_points[i], bin_points[i + 1]):
            if bin_points[i + 1] != bin_points[i]:
                filters[i, j] = (j - bin_points[i]) / (bin_points[i + 1] - bin_points[i])
        for j in range(bin_points[i + 1], bin_points[i + 2]):
            if bin_points[i + 2] != bin_points[i + 1]:
                filters[i, j] = (bin_points[i + 2] - j) / (bin_points[i + 2] - bin_points[i + 1])

    mel_spec = filters @ magnitudes
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    # Pad/trim to exactly 3000 frames
    if log_spec.shape[1] < 3000:
        log_spec = np.pad(log_spec, ((0, 0), (0, 3000 - log_spec.shape[1])))
    else:
        log_spec = log_spec[:, :3000]

    return log_spec.astype(np.float32)


def process_shard(gpu_id: int, file_paths: list[Path], tts_dir: Path, output_dir: Path, whisper_variant: str, batch_size: int):
    """Process a shard of files on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Force JAX to reinitialize with this GPU only
    import jax
    jax.config.update("jax_platforms", "cuda")

    import jax.numpy as jnp

    logger.info(f"[GPU {gpu_id}] Processing {len(file_paths)} files, batch_size={batch_size}")

    # Load Whisper model
    from transformers import FlaxWhisperForConditionalGeneration, WhisperConfig
    from transformers.models.whisper.modeling_flax_whisper import FlaxWhisperEncoder

    model = FlaxWhisperForConditionalGeneration.from_pretrained(whisper_variant, from_pt=True)
    encoder_params = {"params": model.params["model"]["encoder"]}
    del model  # free decoder memory

    config = WhisperConfig.from_pretrained(whisper_variant)
    encoder = FlaxWhisperEncoder(config, dtype=jnp.float32)

    @jax.jit
    def encode_batch(mel_batch):
        outputs = encoder.apply(encoder_params, input_features=mel_batch, deterministic=True)
        return outputs[0]

    total_batches = (len(file_paths) + batch_size - 1) // batch_size
    start_time = time.time()

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(file_paths))
        batch_paths = file_paths[batch_start:batch_end]

        mel_batch = []
        for audio_path in batch_paths:
            waveform, _ = librosa.load(str(audio_path), sr=16000)
            mel = compute_mel_spectrogram(waveform.astype(np.float32))
            mel_batch.append(mel)

        while len(mel_batch) < batch_size:
            mel_batch.append(np.zeros_like(mel_batch[0]))

        mel_array = jnp.array(np.stack(mel_batch))
        hidden_states = encode_batch(mel_array)
        hidden_np = np.asarray(hidden_states)

        for i, audio_path in enumerate(batch_paths):
            rel = audio_path.relative_to(tts_dir)
            cache_path = output_dir / rel.with_suffix(".npy")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, hidden_np[i])

        if (batch_idx + 1) % 5 == 0 or batch_idx == 0 or batch_idx == total_batches - 1:
            elapsed = time.time() - start_time
            files_done = batch_end
            rate = files_done / elapsed if elapsed > 0 else 0
            eta = (len(file_paths) - files_done) / rate if rate > 0 else 0
            logger.info(
                f"[GPU {gpu_id}] Batch {batch_idx + 1}/{total_batches} | "
                f"{files_done}/{len(file_paths)} files | "
                f"{rate:.1f} files/s | ETA: {eta:.0f}s"
            )

    elapsed = time.time() - start_time
    logger.info(f"[GPU {gpu_id}] Done! {len(file_paths)} files in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Precompute Whisper encoder embeddings")
    parser.add_argument("--tts-dir", type=str, required=True, help="Input TTS directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output cache directory")
    parser.add_argument("--whisper-variant", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs to use")
    args = parser.parse_args()

    tts_dir = Path(args.tts_dir)
    output_dir = Path(args.output_dir)

    # Collect all audio files
    audio_files = []
    for ext in ("*.wav", "*.mp3", "*.flac"):
        audio_files.extend(sorted(tts_dir.rglob(ext)))

    logger.info(f"Found {len(audio_files)} audio files in {tts_dir}")
    if not audio_files:
        return

    # Check how many are already cached
    to_process = []
    already_cached = 0
    for audio_path in audio_files:
        rel = audio_path.relative_to(tts_dir)
        cache_path = output_dir / rel.with_suffix(".npy")
        if cache_path.exists():
            already_cached += 1
        else:
            to_process.append(audio_path)

    logger.info(f"Already cached: {already_cached}, to process: {len(to_process)}")
    if not to_process:
        logger.info("All files already cached. Done.")
        return

    num_gpus = min(args.num_gpus, len(to_process))
    logger.info(f"Using {num_gpus} GPUs, batch_size={args.batch_size} per GPU")

    # Split files across GPUs
    import multiprocessing as mp
    shards = [[] for _ in range(num_gpus)]
    for i, f in enumerate(to_process):
        shards[i % num_gpus].append(f)

    start_time = time.time()

    # Launch one process per GPU
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=process_shard,
            args=(gpu_id, shards[gpu_id], tts_dir, output_dir, args.whisper_variant, args.batch_size),
        )
        p.start()
        processes.append(p)
        logger.info(f"Launched GPU {gpu_id} with {len(shards[gpu_id])} files")

    # Wait for all
    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    cached_files = list(output_dir.rglob("*.npy"))
    cache_size_gb = sum(f.stat().st_size for f in cached_files) / 1e9
    logger.info(f"All done! {len(cached_files)} files cached in {elapsed:.1f}s. Cache size: {cache_size_gb:.1f} GB")


if __name__ == "__main__":
    main()
