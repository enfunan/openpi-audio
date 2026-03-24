from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class AudioPreprocess(DataTransformFn):
    """Preprocesses audio into Whisper-compatible format for the model.

    Supports two modes:
    1. **Cached** (fast): If "audio_whisper_hidden" is already in data (precomputed),
       passes it through directly. No mel computation needed.
    2. **Live** (slow): If "audio_waveform" is in data, computes (128, 3000) mel spectrogram
       for live Whisper encoding in the model.

    If neither is present, produces zeros with audio_mask=False.
    """

    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 400
    hop_length: int = 160
    max_duration: float = 30.0  # Whisper expects 30s

    def __call__(self, data: DataDict) -> DataDict:
        target_length = int(self.max_duration * self.sample_rate)  # 480000
        n_frames = 3000  # Whisper mel frames for 30s

        if "audio_whisper_hidden" in data and data["audio_whisper_hidden"] is not None:
            # Precomputed Whisper embeddings — no mel needed
            data["audio_mask"] = np.bool_(True)
            # Still need a dummy audio field for Observation dataclass (won't be used)
            if "audio" not in data:
                data["audio"] = np.zeros((self.n_mels, n_frames), dtype=np.float32)
            data.pop("audio_waveform", None)
        elif "audio_waveform" in data and data["audio_waveform"] is not None:
            waveform = np.asarray(data.pop("audio_waveform"), dtype=np.float32)

            # Pad or trim to 30s
            if len(waveform) < target_length:
                waveform = np.pad(waveform, (0, target_length - len(waveform)))
            else:
                waveform = waveform[:target_length]

            # Compute log-mel spectrogram using numpy
            mel = self._compute_mel_spectrogram(waveform)
            data["audio"] = mel  # (128, 3000)
            data["audio_mask"] = np.bool_(True)
        else:
            # No audio - provide zeros with mask=False
            data["audio"] = np.zeros((self.n_mels, n_frames), dtype=np.float32)
            data["audio_mask"] = np.bool_(False)
            data.pop("audio_waveform", None)

        return data

    def _compute_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """Compute log-mel spectrogram matching Whisper preprocessing."""
        # STFT
        window = np.hanning(self.n_fft + 1)[:-1].astype(np.float32)
        stft_frames = []
        for i in range(0, len(waveform) - self.n_fft + 1, self.hop_length):
            frame = waveform[i:i + self.n_fft] * window
            spectrum = np.fft.rfft(frame)
            stft_frames.append(np.abs(spectrum) ** 2)

        if not stft_frames:
            return np.zeros((self.n_mels, 3000), dtype=np.float32)

        magnitudes = np.stack(stft_frames, axis=0).T  # (n_fft//2+1, num_frames)

        # Mel filterbank
        mel_filters = self._mel_filterbank(self.n_mels, self.n_fft, self.sample_rate)
        mel_spec = mel_filters @ magnitudes  # (n_mels, num_frames)

        # Log scale (matching Whisper)
        log_spec = np.log10(np.maximum(mel_spec, 1e-10))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        # Pad or trim to 3000 frames
        target_frames = 3000
        if log_spec.shape[1] < target_frames:
            log_spec = np.pad(log_spec, ((0, 0), (0, target_frames - log_spec.shape[1])))
        else:
            log_spec = log_spec[:, :target_frames]

        return log_spec.astype(np.float32)

    @staticmethod
    def _mel_filterbank(n_mels: int, n_fft: int, sample_rate: int) -> np.ndarray:
        """Create mel filterbank matrix."""

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
                filters[i, j] = (j - bin_points[i]) / max(bin_points[i + 1] - bin_points[i], 1)
            for j in range(bin_points[i + 1], bin_points[i + 2]):
                filters[i, j] = (bin_points[i + 2] - j) / max(bin_points[i + 2] - bin_points[i + 1], 1)

        return filters


@dataclasses.dataclass(frozen=True)
class AudioTextMixingTransform(DataTransformFn):
    """Injects TTS audio into training samples.

    With probability `audio_ratio`, looks up a pre-synthesized TTS audio file
    for the current prompt and injects it as `audio_waveform`.

    If `whisper_cache_dir` is set, also loads the precomputed Whisper encoder
    embedding (.npy file) as `audio_whisper_hidden`, avoiding live Whisper
    computation during training (~2x speedup).

    If `clear_prompt` is True (Stage 2 style), clears the text prompt when
    audio is injected so the model must rely on audio alone.
    If False (Stage 1 style), keeps text alongside audio for teacher forcing.

    If `auxiliary_tts_dir` is set, with probability `auxiliary_ratio` a random
    prompt+audio pair is drawn from the auxiliary manifest instead of matching
    the episode's prompt. This allows mixing in out-of-domain TTS data (e.g.
    DROID instructions during LIBERO training) for acoustic diversity. The
    episode's text prompt is replaced with the auxiliary prompt for ASR targets.
    """

    audio_ratio: float = 1.0
    tts_cache_dir: str = ""
    whisper_cache_dir: str = ""
    clear_prompt: bool = False
    auxiliary_tts_dir: str = ""
    auxiliary_whisper_cache_dir: str = ""
    auxiliary_ratio: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "_manifest", None)
        object.__setattr__(self, "_aux_manifest", None)
        object.__setattr__(self, "_aux_prompts", None)
        object.__setattr__(self, "_rng", None)

    def _load_manifest(self, tts_dir: str) -> dict:
        import json
        import pathlib

        manifest_path = pathlib.Path(tts_dir) / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"TTS manifest not found at {manifest_path}. "
                "Run scripts/synthesize_tts.py first to generate the TTS cache."
            )
        with open(manifest_path) as f:
            return json.load(f)

    def _get_manifest(self) -> dict:
        if self._manifest is None:
            object.__setattr__(self, "_manifest", self._load_manifest(self.tts_cache_dir))
        return self._manifest

    def _get_aux_manifest(self) -> dict:
        if self._aux_manifest is None:
            manifest = self._load_manifest(self.auxiliary_tts_dir)
            object.__setattr__(self, "_aux_manifest", manifest)
            object.__setattr__(self, "_aux_prompts", list(manifest.keys()))
        return self._aux_manifest

    def _get_rng(self):
        import random
        if self._rng is None:
            object.__setattr__(self, "_rng", random.Random())
        return self._rng

    def _load_audio(self, audio_path: str, tts_dir: str, whisper_cache_dir: str, data: DataDict):
        """Load audio as precomputed Whisper embedding or raw waveform."""
        import pathlib

        if whisper_cache_dir:
            tts_base = pathlib.Path(tts_dir)
            cache_base = pathlib.Path(whisper_cache_dir)
            audio_p = pathlib.Path(audio_path)
            try:
                rel_path = audio_p.relative_to(tts_base)
            except ValueError:
                try:
                    rel_path = audio_p.relative_to(tts_base.resolve())
                except ValueError:
                    # audio_path may be absolute from a different machine;
                    # extract relative part after the tts dir name.
                    tts_name = tts_base.name  # e.g. "libero_train"
                    parts = audio_p.parts
                    for i, part in enumerate(parts):
                        if part == tts_name:
                            rel_path = pathlib.Path(*parts[i + 1:])
                            break
                    else:
                        # Last resort: use just the filename
                        rel_path = pathlib.Path(audio_p.name)
            cache_path = cache_base / rel_path.with_suffix(".npy")
            if cache_path.exists():
                data["audio_whisper_hidden"] = np.load(cache_path)  # (1500, 1280)
                return
        # Fallback to live Whisper — load waveform
        import librosa
        waveform, _ = librosa.load(audio_path, sr=16000)
        data["audio_waveform"] = waveform.astype(np.float32)

    def __call__(self, data: DataDict) -> DataDict:
        if not self.tts_cache_dir:
            return data

        rng = self._get_rng()

        prompt = data.get("prompt", "")
        if isinstance(prompt, np.ndarray):
            prompt = str(prompt.item()) if prompt.ndim == 0 else str(prompt)

        if rng.random() < self.audio_ratio and prompt:
            # Decide whether to use auxiliary TTS (out-of-domain, e.g. DROID)
            use_auxiliary = (
                self.auxiliary_tts_dir
                and self.auxiliary_ratio > 0
                and rng.random() < self.auxiliary_ratio
            )

            if use_auxiliary:
                aux_manifest = self._get_aux_manifest()
                aux_prompt = rng.choice(self._aux_prompts)
                audio_files = aux_manifest[aux_prompt]
                audio_path = rng.choice(audio_files)
                self._load_audio(audio_path, self.auxiliary_tts_dir, self.auxiliary_whisper_cache_dir, data)
                # Replace prompt so ASR targets match the audio content
                data["prompt"] = np.asarray(aux_prompt)
            else:
                manifest = self._get_manifest()
                audio_files = manifest.get(prompt)
                if audio_files:
                    audio_path = rng.choice(audio_files)
                    self._load_audio(audio_path, self.tts_cache_dir, self.whisper_cache_dir, data)

            if self.clear_prompt:
                # Save the original prompt before clearing so that ASR rehearsal
                # in Stage 2 can still produce meaningful target tokens.
                current_prompt = data.get("prompt", "")
                if isinstance(current_prompt, np.ndarray):
                    current_prompt = str(current_prompt.item()) if current_prompt.ndim == 0 else str(current_prompt)
                if current_prompt:
                    data["original_prompt"] = current_prompt
                data["prompt"] = np.asarray("")

        return data


@dataclasses.dataclass(frozen=True)
class GenerateASRTargets(DataTransformFn):
    """Generates ASR target tokens for ASR loss.

    Must be applied AFTER TokenizePrompt. In Stage 1 (text not cleared),
    simply copies tokenized_prompt into asr_target_tokens. In Stage 2
    (clear_prompt=True), if ``original_prompt`` exists, tokenizes it to
    produce ASR targets with the original instruction text, since
    tokenized_prompt is empty after clearing.

    When ``original_prompt`` is tokenized, the result is also stored in
    ``original_tokenized_prompt`` / ``original_tokenized_prompt_mask`` so
    the model can use the full instruction text as prefix input during the
    ASR rehearsal forward pass.

    Args:
        tokenizer: Optional PaligemmaTokenizer instance. Required when
            ``original_prompt`` may be present (Stage 2). If not provided,
            falls back to copying tokenized_prompt (Stage 1 behavior).
        max_token_len: Maximum token length for tokenization.
        discrete_state_input: Whether to include state in tokenization.
    """

    tokenizer: _tokenizer.PaligemmaTokenizer | None = None

    def __call__(self, data: DataDict) -> DataDict:
        original_prompt = data.pop("original_prompt", None)

        if "tokenized_prompt" not in data or "tokenized_prompt_mask" not in data:
            return data

        if original_prompt is not None and self.tokenizer is not None:
            # Stage 2 path: original_prompt was saved before clearing.
            # Tokenize the original instruction text for ASR targets and
            # as the prefix input for the ASR rehearsal forward pass.
            if not isinstance(original_prompt, str):
                original_prompt = str(original_prompt)

            orig_tokens, orig_mask = self.tokenizer.tokenize(original_prompt)

            data["asr_target_tokens"] = orig_tokens
            data["asr_target_mask"] = orig_mask
            data["original_tokenized_prompt"] = orig_tokens.copy()
            data["original_tokenized_prompt_mask"] = orig_mask.copy()
        else:
            # Stage 1 path (or no clearing): copy tokenized_prompt as-is.
            data["asr_target_tokens"] = data["tokenized_prompt"].copy()
            data["asr_target_mask"] = data["tokenized_prompt_mask"].copy()

            # When tokenizer is provided (audio-enabled model), always produce
            # original_tokenized_prompt for shape consistency across batches.
            # If text wasn't cleared, the original IS the current prompt.
            if self.tokenizer is not None:
                data["original_tokenized_prompt"] = data["tokenized_prompt"].copy()
                data["original_tokenized_prompt_mask"] = data["tokenized_prompt_mask"].copy()

        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
