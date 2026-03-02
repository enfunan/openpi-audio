#!/usr/bin/env bash
# Self-contained ambiguity eval on a remote node.
# Everything stays inside /home/user1/workspace/eval_v1/
# Usage: bash scripts/eval_ambiguity_remote.sh <remote_host> <gpu_id>
#
# Cleanup after: ssh <host> 'rm -rf /home/user1/workspace/eval_v1'

set -euo pipefail

REMOTE=${1:-nnmc61}
GPU_ID=${2:-0}
PORT=8001
BASE="/home/user1/workspace/eval_v1"
OPENPI="${BASE}/openpi"
PYTHON="${OPENPI}/.venv/bin/python"
CKPT="${OPENPI}/checkpoints/pi05_audio_stage3_libero/stage3_libero/29999"
AUDIO_DIR="${BASE}/data/tts/libero_eval"
LOG_DIR="${BASE}/logs"
LIBERO_ROOT="${OPENPI}/third_party/libero/libero/libero"

echo "$(date '+%H:%M:%S') === Setting up eval on ${REMOTE} ==="

# Create libero config inside eval_v1
ssh ${REMOTE} "mkdir -p ${BASE}/libero_config ${LOG_DIR}/videos
cat > ${BASE}/libero_config/config.yaml << 'YAML'
assets: ${LIBERO_ROOT}/assets
bddl_files: ${LIBERO_ROOT}/bddl_files
benchmark_root: ${LIBERO_ROOT}
datasets: ${LIBERO_ROOT}/../datasets
init_states: ${LIBERO_ROOT}/init_files
YAML
"

# Write the actual config with expanded paths
ssh ${REMOTE} "cat > ${BASE}/libero_config/config.yaml << YAML
assets: ${LIBERO_ROOT}/assets
bddl_files: ${LIBERO_ROOT}/bddl_files
benchmark_root: ${LIBERO_ROOT}
datasets: ${LIBERO_ROOT}/../datasets
init_states: ${LIBERO_ROOT}/init_files
YAML
echo 'Config written'
cat ${BASE}/libero_config/config.yaml"

echo "$(date '+%H:%M:%S') Starting model server on ${REMOTE} GPU ${GPU_ID}..."

# Start server (all env vars keep things inside eval_v1)
ssh ${REMOTE} "
  cd ${OPENPI}
  CUDA_VISIBLE_DEVICES=${GPU_ID} \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  LIBERO_CONFIG_PATH=${BASE}/libero_config \
  nohup ${PYTHON} scripts/serve_policy.py \
    --port ${PORT} \
    policy:checkpoint \
    --policy.config pi05_audio_stage3_libero \
    --policy.dir ${CKPT} \
    > ${LOG_DIR}/server.log 2>&1 &
  echo \"Server PID: \$!\"
"

echo "$(date '+%H:%M:%S') Waiting for server..."
for i in $(seq 1 180); do
  if ssh ${REMOTE} "ss -tlnp 2>/dev/null | grep -q ':${PORT} '" 2>/dev/null; then
    echo "$(date '+%H:%M:%S') Server ready after ${i}s"
    break
  fi
  sleep 1
done

# Verify server is running
if ! ssh ${REMOTE} "ss -tlnp 2>/dev/null | grep -q ':${PORT} '"; then
  echo "$(date '+%H:%M:%S') Server failed to start! Last log:"
  ssh ${REMOTE} "tail -20 ${LOG_DIR}/server.log"
  exit 1
fi

# Run evals: 2 suites × 2 modes = 4 runs
for SUITE in libero_spatial libero_object; do
  for MODE in audio zero_audio; do
    LOG="${LOG_DIR}/${SUITE}_${MODE}.log"
    echo ""
    echo "$(date '+%H:%M:%S') === ${SUITE} / ${MODE} (3 trials) ==="

    ssh ${REMOTE} "
      cd ${OPENPI}
      MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
      CUDA_VISIBLE_DEVICES='' \
      LIBERO_CONFIG_PATH=${BASE}/libero_config \
      ${PYTHON} examples/libero/main.py \
        --args.task-suite-name=${SUITE} \
        --args.num-trials-per-task=3 \
        --args.eval-mode=${MODE} \
        --args.audio-dir=${AUDIO_DIR} \
        --args.port=${PORT} \
        --args.video-out-path=${LOG_DIR}/videos/${SUITE}_${MODE} \
        2>&1 | tee ${LOG}
    "

    echo "$(date '+%H:%M:%S') === ${SUITE} / ${MODE} done ==="
  done
done

# Stop server
echo ""
echo "$(date '+%H:%M:%S') Stopping server..."
ssh ${REMOTE} "pkill -f 'serve_policy.*${PORT}' 2>/dev/null || true"

# Copy logs back and parse results
echo "$(date '+%H:%M:%S') Copying logs..."
mkdir -p /tmp/eval_ambiguity
scp -r ${REMOTE}:${LOG_DIR}/*.log /tmp/eval_ambiguity/

# Parse and report
echo ""
echo "============================================================"
echo "  AMBIGUITY ANALYSIS — v1 checkpoint 29999, 3 trials each"
echo "============================================================"

python3 - /tmp/eval_ambiguity <<'PYEOF'
import re, sys, os

log_dir = sys.argv[1]

# Task ambiguity classification
spatial_high = {0, 1, 6, 8}  # between/next_to tasks
spatial_low = {2, 3, 4, 5, 7, 9}  # unique location tasks
object_high = {1, 3, 4, 5, 6}  # visually similar (cream cheese, bbq, ketchup, tomato, butter)
object_low = {0, 2, 7, 8, 9}  # distinctive (alphabet soup, salad dressing, milk, pudding, OJ)

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
    """Parse per-task success counts from eval log."""
    results = {}
    task_id = -1
    successes = 0
    trials = 0
    with open(log_path) as f:
        for line in f:
            if "Current task success rate" in line:
                if trials > 0:
                    results[task_id] = (successes, trials)
                task_id += 1
                successes = 0
                trials = 0
            elif "Success: True" in line:
                successes += 1
                trials += 1
            elif "Success: False" in line:
                trials += 1
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
echo "$(date '+%H:%M:%S') === Done ==="
echo "Cleanup: ssh ${REMOTE} 'rm -rf ${BASE}'"
