#!/usr/bin/env bash
# Run audio vs zero_audio eval on v1 checkpoint, with per-task results.
# Groups results by high-ambiguity vs low-ambiguity tasks.
#
# Usage: bash scripts/eval_ambiguity.sh [gpu_id] [port]
#   gpu_id: GPU for model server (default: 0)
#   port: server port (default: 8001)
#
# Requires: GPUs to be free (don't run during training).

set -euo pipefail
cd /home/user1/workspace/VLA/openpi

GPU_ID=${1:-0}
PORT=${2:-8001}
CKPT="checkpoints/pi05_audio_stage3_libero/stage3_libero/29999"
AUDIO_DIR="/home/user1/workspace/VLA/data/tts/libero_eval"
LOG_DIR="/tmp/eval_ambiguity"
mkdir -p "$LOG_DIR"

echo "$(date '+%H:%M:%S') === Ambiguity Eval: v1 checkpoint 29999 ==="
echo "$(date '+%H:%M:%S') GPU: ${GPU_ID}, Port: ${PORT}"
echo "$(date '+%H:%M:%S') Checkpoint: ${CKPT}"
echo ""

# Start model server
echo "$(date '+%H:%M:%S') Starting model server..."
CUDA_VISIBLE_DEVICES=${GPU_ID} XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  .venv/bin/python scripts/serve_policy.py \
  --port ${PORT} \
  policy:checkpoint \
  --policy.config pi05_audio_stage3_libero \
  --policy.dir "${CKPT}" \
  > "${LOG_DIR}/server.log" 2>&1 &
SERVER_PID=$!

echo "$(date '+%H:%M:%S') Waiting for server (PID ${SERVER_PID})..."
for i in $(seq 1 180); do
  if ss -tlnp | grep -q ":${PORT} "; then
    echo "$(date '+%H:%M:%S') Server ready after ${i}s"
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "$(date '+%H:%M:%S') Server died! Check ${LOG_DIR}/server.log"
    tail -20 "${LOG_DIR}/server.log"
    exit 1
  fi
  sleep 1
done

if ! ss -tlnp | grep -q ":${PORT} "; then
  echo "$(date '+%H:%M:%S') Server failed to start within 180s"
  kill $SERVER_PID 2>/dev/null
  exit 1
fi

# Run evals: 4 runs (2 suites × 2 modes)
for SUITE in libero_spatial libero_object; do
  for MODE in audio zero_audio; do
    LOG="${LOG_DIR}/${SUITE}_${MODE}.log"
    echo ""
    echo "$(date '+%H:%M:%S') === ${SUITE} / ${MODE} (3 trials) ==="

    MUJOCO_EGL_DEVICE_ID=${GPU_ID} CUDA_VISIBLE_DEVICES="" \
      .venv/bin/python examples/libero/main.py \
      --args.task-suite-name="${SUITE}" \
      --args.num-trials-per-task=3 \
      --args.eval-mode="${MODE}" \
      --args.audio-dir="${AUDIO_DIR}" \
      --args.port=${PORT} \
      --args.video-out-path="${LOG_DIR}/videos/${SUITE}_${MODE}" \
      2>&1 | tee "${LOG}"

    echo "$(date '+%H:%M:%S') === ${SUITE} / ${MODE} done ==="
  done
done

# Stop server
echo ""
echo "$(date '+%H:%M:%S') Stopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true

# ==================== Parse and report results ====================
echo ""
echo "============================================================"
echo "  AMBIGUITY ANALYSIS — v1 checkpoint 29999, 3 trials each"
echo "============================================================"

python3 - "${LOG_DIR}" <<'PYEOF'
import re, sys, os

log_dir = sys.argv[1]

# Task ambiguity classification
# libero_spatial: task_id -> ambiguity
spatial_high = {0, 1, 6, 8}  # between/next_to tasks
spatial_low = {2, 3, 4, 5, 7, 9}  # unique location tasks

# libero_object: task_id -> ambiguity
object_high = {1, 3, 4, 5, 6}  # visually similar items (cream cheese, bbq, ketchup, tomato, butter)
object_low = {0, 2, 7, 8, 9}  # distinctive items (alphabet soup, salad dressing, milk, pudding, OJ)

classifications = {
    "libero_spatial": (spatial_high, spatial_low),
    "libero_object": (object_high, object_low),
}

spatial_names = [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
]
object_names = [
    "pick up the alphabet soup and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
    "pick up the orange juice and place it in the basket",
]
task_names = {"libero_spatial": spatial_names, "libero_object": object_names}

def parse_per_task(log_path):
    """Parse per-task success from eval log. Returns dict of task_id -> (successes, trials)."""
    results = {}
    task_id = -1
    successes = 0
    trials = 0
    with open(log_path) as f:
        for line in f:
            # Detect new task starting (tqdm or task log)
            if "Current task success rate" in line:
                m = re.search(r"success rate: ([\d.]+)", line)
                if m and trials > 0:
                    results[task_id] = (successes, trials)
                task_id += 1
                successes = 0
                trials = 0
            elif "Success: True" in line:
                successes += 1
                trials += 1
            elif "Success: False" in line:
                trials += 1
    # Capture last task
    if trials > 0:
        results[task_id] = (successes, trials)
    return results

for suite in ["libero_spatial", "libero_object"]:
    high_ids, low_ids = classifications[suite]
    names = task_names[suite]

    print(f"\n{'='*70}")
    print(f"  {suite.upper()}")
    print(f"{'='*70}")

    for mode in ["audio", "zero_audio"]:
        log_path = os.path.join(log_dir, f"{suite}_{mode}.log")
        if not os.path.exists(log_path):
            print(f"  {mode}: log not found")
            continue
        results = parse_per_task(log_path)
        print(f"\n  --- {mode.upper()} mode ---")
        for tid in sorted(results.keys()):
            s, t = results[tid]
            amb = "HIGH" if tid in high_ids else "LOW "
            name_short = names[tid][:55] if tid < len(names) else "?"
            print(f"    Task {tid} [{amb}] {s}/{t}  {name_short}")

    # Aggregate by ambiguity
    print(f"\n  --- AGGREGATED ---")
    print(f"  {'Category':<20} {'Audio':>12} {'Zero Audio':>12} {'Gap':>8}")
    print(f"  {'-'*52}")

    for label, ids in [("HIGH ambiguity", high_ids), ("LOW ambiguity", low_ids), ("ALL", high_ids | low_ids)]:
        audio_s, audio_t = 0, 0
        zero_s, zero_t = 0, 0

        audio_log = os.path.join(log_dir, f"{suite}_audio.log")
        zero_log = os.path.join(log_dir, f"{suite}_zero_audio.log")

        if os.path.exists(audio_log):
            r = parse_per_task(audio_log)
            for tid in ids:
                if tid in r:
                    audio_s += r[tid][0]
                    audio_t += r[tid][1]

        if os.path.exists(zero_log):
            r = parse_per_task(zero_log)
            for tid in ids:
                if tid in r:
                    zero_s += r[tid][0]
                    zero_t += r[tid][1]

        audio_pct = f"{audio_s}/{audio_t} ({100*audio_s/audio_t:.0f}%)" if audio_t else "N/A"
        zero_pct = f"{zero_s}/{zero_t} ({100*zero_s/zero_t:.0f}%)" if zero_t else "N/A"
        gap = ""
        if audio_t and zero_t:
            gap_val = 100*audio_s/audio_t - 100*zero_s/zero_t
            gap = f"{gap_val:+.0f}%"
        print(f"  {label:<20} {audio_pct:>12} {zero_pct:>12} {gap:>8}")

print()
PYEOF

echo ""
echo "$(date '+%H:%M:%S') === Ambiguity eval complete ==="
