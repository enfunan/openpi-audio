from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import pathlib
from typing import Any

import imageio
import librosa
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    # Task range (for parallel eval — split tasks across GPUs)
    task_start: int = 0  # First task index (inclusive)
    task_end: int = -1  # Last task index (exclusive), -1 = all tasks

    #################################################################################################################
    # Audio evaluation parameters
    #################################################################################################################
    audio_dir: str = ""  # TTS cache directory (empty = text-only eval)
    eval_mode: str = "text"  # "text", "audio", "asr", "zero_audio", "zero_text", "paraphrase", or "both"
    whisper_model: str = "openai/whisper-large-v2"  # Whisper model for ASR pipeline mode
    paraphrase_map: str = ""  # JSON file mapping original instructions to paraphrased versions


def _load_tts_manifest(audio_dir: str) -> dict[str, list[str]]:
    """Load TTS manifest mapping task descriptions to lists of audio file paths."""
    manifest_path = pathlib.Path(audio_dir) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    logging.info(f"Loaded TTS manifest with {len(manifest)} tasks from {manifest_path}")
    return manifest


def _load_random_audio(manifest: dict[str, list[str]], task_description: str) -> np.ndarray | None:
    """Load a random TTS audio waveform for the given task description."""
    if task_description not in manifest:
        logging.warning(f"No TTS audio found for task: {task_description}")
        return None
    audio_files = manifest[task_description]
    audio_path = audio_files[np.random.randint(len(audio_files))]
    waveform, _ = librosa.load(audio_path, sr=16000)
    return waveform


def _load_paraphrase_map(path: str) -> dict[str, str]:
    """Load JSON mapping of original instructions to paraphrased versions."""
    with open(path) as f:
        mapping = json.load(f)
    logging.info(f"Loaded paraphrase map with {len(mapping)} entries from {path}")
    return mapping


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Determine which evaluation modes to run
    if args.eval_mode == "both":
        modes = ["text", "audio"]
    else:
        modes = [args.eval_mode]

    # Load paraphrase map if paraphrase mode is requested
    paraphrase_map = None
    if args.eval_mode == "paraphrase":
        if not args.paraphrase_map:
            raise ValueError("--paraphrase-map is required for paraphrase eval mode")
        paraphrase_map = _load_paraphrase_map(args.paraphrase_map)

    # Load TTS manifest if audio mode is requested
    manifest = None
    if args.audio_dir and args.eval_mode in ("audio", "asr", "both"):
        manifest = _load_tts_manifest(args.audio_dir)

    # Load Whisper model for ASR pipeline mode
    whisper_pipeline = None
    if args.eval_mode == "asr":
        import transformers
        whisper_pipeline = transformers.pipeline(
            "automatic-speech-recognition",
            model=args.whisper_model,
            device="cpu",
        )
        logging.info(f"Loaded Whisper ASR pipeline: {args.whisper_model}")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Run evaluation for each mode
    results_by_mode: dict[str, dict[str, Any]] = {}
    for mode in modes:
        use_audio = mode == "audio" and manifest is not None
        use_zero_audio = mode == "zero_audio"
        use_zero_text = mode == "zero_text"
        use_random_audio = mode == "random_audio"
        use_asr = mode == "asr" and manifest is not None and whisper_pipeline is not None
        logging.info(f"\n{'='*60}")
        logging.info(f"Running evaluation in {mode.upper()} mode")
        logging.info(f"{'='*60}")

        # Start evaluation
        total_episodes, total_successes = 0, 0
        task_end = args.task_end if args.task_end > 0 else num_tasks_in_suite
        task_range = range(max(0, args.task_start), min(task_end, num_tasks_in_suite))
        for task_id in tqdm.tqdm(task_range, desc=f"{mode} tasks"):
            # Get task
            task = task_suite.get_task(task_id)

            # Get default LIBERO initial states
            initial_states = task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

            # Paraphrase mode: substitute the task description with the paraphrased version
            prompt_description = task_description
            if mode == "paraphrase" and paraphrase_map is not None:
                if task_description in paraphrase_map:
                    prompt_description = paraphrase_map[task_description]
                    logging.info(f"Paraphrase: '{task_description}' -> '{prompt_description}'")
                else:
                    logging.warning(f"No paraphrase found for: '{task_description}', using original")

            # Start episodes
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"{mode} episodes"):
                logging.info(f"\nTask: {task_description} (mode={mode}, prompt='{prompt_description}')")

                # Reset environment
                env.reset()
                action_plan = collections.deque()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Pre-load audio for this episode (same waveform reused across replan steps)
                audio_waveform = None
                asr_transcription = None
                if use_audio:
                    audio_waveform = _load_random_audio(manifest, task_description)
                if use_zero_audio:
                    # Send 3 seconds of silence — tests if model ignores audio entirely
                    audio_waveform = np.zeros(16000 * 3, dtype=np.float32)
                if use_random_audio:
                    # Send 3 seconds of random noise — tests if model uses audio content
                    audio_waveform = np.random.randn(16000 * 3).astype(np.float32)
                if use_asr:
                    audio_waveform = _load_random_audio(manifest, task_description)
                    if audio_waveform is not None:
                        result = whisper_pipeline({"raw": audio_waveform, "sampling_rate": 16000})
                        asr_transcription = result["text"].strip().lower()
                        logging.info(f"ASR transcription: '{asr_transcription}' (ground truth: '{task_description}')")

                # Setup
                t = 0
                replay_images = []

                logging.info(f"Starting episode {task_episodes+1}...")
                while t < max_steps + args.num_steps_wait:
                    try:
                        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                        # and we need to wait for them to fall
                        if t < args.num_steps_wait:
                            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        # Get preprocessed image
                        # IMPORTANT: rotate 180 degrees to match train preprocessing
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                        )
                        wrist_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                        )

                        # Save preprocessed image for replay video
                        replay_images.append(img)

                        if not action_plan:
                            # Finished executing previous action chunk -- compute new chunk
                            # Prepare observations dict
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": np.concatenate(
                                    (
                                        obs["robot0_eef_pos"],
                                        _quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"],
                                    )
                                ),
                                "prompt": str(prompt_description),
                            }

                            # Inject audio waveform when in audio mode.
                            # Clear text prompt to match training (VLAS mutual exclusivity).
                            if use_audio and audio_waveform is not None:
                                element["audio"] = audio_waveform
                                element["prompt"] = ""

                            # Zero audio mode: silent waveform, no text prompt.
                            if use_zero_audio:
                                element["audio"] = audio_waveform
                                element["prompt"] = ""

                            # Random audio mode: random noise waveform, no text prompt.
                            if use_random_audio:
                                element["audio"] = audio_waveform
                                element["prompt"] = ""

                            # Zero text mode: empty text, no audio. Pure vision baseline.
                            if use_zero_text:
                                element["prompt"] = ""

                            # ASR pipeline: use Whisper transcription as text prompt.
                            if use_asr and asr_transcription is not None:
                                element["prompt"] = asr_transcription

                            # Query model to get action
                            action_chunk = client.infer(element)["actions"]
                            assert (
                                len(action_chunk) >= args.replan_steps
                            ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                            action_plan.extend(action_chunk[: args.replan_steps])

                        action = action_plan.popleft()

                        # Execute action in environment
                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        logging.error(f"Caught exception: {e}")
                        break

                task_episodes += 1
                total_episodes += 1

                # Save a replay video of the episode
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{mode}_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

                # Log current results
                logging.info(f"Success: {done}")
                logging.info(f"[{mode}] # episodes completed so far: {total_episodes}")
                logging.info(f"[{mode}] # successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            # Log final results
            logging.info(f"[{mode}] Current task success rate: {float(task_successes) / float(task_episodes)}")
            logging.info(f"[{mode}] Current total success rate: {float(total_successes) / float(total_episodes)}")

        results_by_mode[mode] = {
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "success_rate": float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0,
        }
        logging.info(f"[{mode}] Total success rate: {results_by_mode[mode]['success_rate']:.4f}")
        logging.info(f"[{mode}] Total episodes: {total_episodes}")

    # Print summary when running both modes
    if len(modes) > 1:
        logging.info(f"\n{'='*60}")
        logging.info("COMPARISON SUMMARY")
        logging.info(f"{'='*60}")
        for mode, result in results_by_mode.items():
            logging.info(
                f"  {mode.upper():>5}: {result['success_rate']*100:.1f}% "
                f"({result['total_successes']}/{result['total_episodes']})"
            )


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
