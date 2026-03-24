#!/usr/bin/env bash
# Paraphrase robustness experiment: compare text vs paraphrase eval on all LIBERO suites.
#
# Usage: bash scripts/eval_paraphrase.sh <checkpoint_dir> <gpu_id> <port> <log_prefix> [num_trials] [config_name]
#
# Example:
#   bash scripts/eval_paraphrase.sh gs://openpi-assets/checkpoints/pi05_libero 0 8000 paraphrase_test 20
#   bash scripts/eval_paraphrase.sh checkpoints/stage2_h100/15000 0 8000 audio_paraphrase 20 pi05_audio_stage2_pytorch_h100

set -euo pipefail
cd /home/user1/workspace/VLA/openpi2

GPU_ID=${1:-0}
PORT=${2:-8000}
LOG_PREFIX=${3:-paraphrase_v1}
NUM_TRIALS=${4:-20}
# Optional: provide checkpoint dir and config for custom models
CKPT_DIR=${5:-}
CONFIG_NAME=${6:-pi05_libero}

PARAPHRASE_MAP="data/libero_paraphrased_instructions.json"
LOG_DIR="/tmp/eval_paraphrase/${LOG_PREFIX}"
mkdir -p "$LOG_DIR"

echo "$(date '+%H:%M:%S') === Paraphrase Robustness Experiment: ${LOG_PREFIX} ==="
if [[ -n "$CKPT_DIR" ]]; then
  echo "$(date '+%H:%M:%S') Checkpoint: ${CKPT_DIR} (config: ${CONFIG_NAME})"
else
  echo "$(date '+%H:%M:%S') Using default pi05_libero checkpoint (auto-download from GCS)"
fi
echo "$(date '+%H:%M:%S') GPU: ${GPU_ID}, Port: ${PORT}, Trials: ${NUM_TRIALS}"
echo "$(date '+%H:%M:%S') Paraphrase map: ${PARAPHRASE_MAP}"
echo "$(date '+%H:%M:%S') Logs: ${LOG_DIR}/"
echo ""

# ==================== Start model server ====================
echo "$(date '+%H:%M:%S') Starting model server..."
if [[ -n "$CKPT_DIR" ]]; then
  OPENPI_NO_COMPILE=1 CUDA_VISIBLE_DEVICES=${GPU_ID} XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=src:${PYTHONPATH:-} \
    .venv/bin/python scripts/serve_policy.py \
    --port ${PORT} \
    policy:checkpoint \
    --policy.config ${CONFIG_NAME} \
    --policy.dir "${CKPT_DIR}" \
    > "${LOG_DIR}/server.log" 2>&1 &
else
  OPENPI_NO_COMPILE=1 CUDA_VISIBLE_DEVICES=${GPU_ID} XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=src:${PYTHONPATH:-} \
    .venv/bin/python scripts/serve_policy.py \
    --port ${PORT} \
    --env LIBERO \
    > "${LOG_DIR}/server.log" 2>&1 &
fi
SERVER_PID=$!

cleanup() {
    echo "$(date '+%H:%M:%S') Cleaning up server (PID ${SERVER_PID})..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT

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
  exit 1
fi

# ==================== Run text vs paraphrase eval ====================
RUNS=(
  "text|text"
  "paraphrase|paraphrase"
)

SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

RUN_COUNT=0
TOTAL_RUNS=$(( ${#RUNS[@]} * ${#SUITES[@]} ))

for SUITE in "${SUITES[@]}"; do
  for RUN_SPEC in "${RUNS[@]}"; do
    IFS='|' read -r LABEL MODE <<< "$RUN_SPEC"
    RUN_COUNT=$((RUN_COUNT + 1))
    LOG="${LOG_DIR}/${SUITE}_${LABEL}.log"

    echo ""
    echo "$(date '+%H:%M:%S') === [${RUN_COUNT}/${TOTAL_RUNS}] ${SUITE} / ${LABEL} (${NUM_TRIALS} trials) ==="

    EVAL_CMD=(
      examples/libero/.venv/bin/python examples/libero/main.py
      --args.task-suite-name="${SUITE}"
      --args.num-trials-per-task="${NUM_TRIALS}"
      --args.eval-mode="${MODE}"
      --args.port=${PORT}
      --args.video-out-path="${LOG_DIR}/videos/${SUITE}_${LABEL}"
    )

    if [[ "$MODE" == "paraphrase" ]]; then
      EVAL_CMD+=(--args.paraphrase-map="${PARAPHRASE_MAP}")
    fi

    MUJOCO_EGL_DEVICE_ID=${GPU_ID} MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="" \
      PYTHONPATH=src:third_party/libero:${PYTHONPATH:-} \
      "${EVAL_CMD[@]}" 2>&1 | tee "${LOG}"

    echo "$(date '+%H:%M:%S') === ${SUITE} / ${LABEL} done ==="
  done
done

# ==================== Parse and report results ====================
echo ""
echo "$(date '+%H:%M:%S') ============================================================"
echo "  PARAPHRASE ROBUSTNESS — ${LOG_PREFIX}, ${NUM_TRIALS} trials"
echo "============================================================"

python3 - "${LOG_DIR}" "${NUM_TRIALS}" <<'PYEOF'
import sys, os, re

log_dir = sys.argv[1]
num_trials = int(sys.argv[2])

def parse_log(log_path):
    """Parse per-task success counts from eval log."""
    task_results = []
    current_successes = 0
    current_trials = 0
    with open(log_path) as f:
        for line in f:
            if "Success: True" in line:
                current_successes += 1
                current_trials += 1
            elif "Success: False" in line:
                current_trials += 1
            if "Current task success rate" in line:
                task_results.append((current_successes, current_trials))
                current_successes = 0
                current_trials = 0
    total_s = sum(s for s, _ in task_results)
    total_t = sum(t for _, t in task_results)
    return task_results, total_s, total_t

suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
modes = ["text", "paraphrase"]

results = {}
for suite in suites:
    results[suite] = {}
    for mode in modes:
        log_path = os.path.join(log_dir, f"{suite}_{mode}.log")
        if os.path.exists(log_path):
            task_results, total_s, total_t = parse_log(log_path)
            rate = total_s / total_t * 100 if total_t > 0 else 0
            results[suite][mode] = {"successes": total_s, "trials": total_t, "rate": rate, "tasks": task_results}
        else:
            results[suite][mode] = None

# Print table
print(f"\n{'Suite':<20} {'Text':>12} {'Paraphrase':>12} {'Delta':>8}")
print("-" * 56)
text_total_s, text_total_t = 0, 0
para_total_s, para_total_t = 0, 0

for suite in suites:
    text_r = results[suite].get("text")
    para_r = results[suite].get("paraphrase")
    text_str = f"{text_r['rate']:.1f}% ({text_r['successes']}/{text_r['trials']})" if text_r else "N/A"
    para_str = f"{para_r['rate']:.1f}% ({para_r['successes']}/{para_r['trials']})" if para_r else "N/A"
    delta = ""
    if text_r and para_r:
        d = para_r['rate'] - text_r['rate']
        delta = f"{d:+.1f}pp"
        text_total_s += text_r['successes']
        text_total_t += text_r['trials']
        para_total_s += para_r['successes']
        para_total_t += para_r['trials']
    print(f"{suite:<20} {text_str:>12} {para_str:>12} {delta:>8}")

# Average
if text_total_t > 0 and para_total_t > 0:
    text_avg = text_total_s / text_total_t * 100
    para_avg = para_total_s / para_total_t * 100
    delta_avg = para_avg - text_avg
    print("-" * 56)
    print(f"{'AVERAGE':<20} {text_avg:>11.1f}% {para_avg:>11.1f}% {delta_avg:>+7.1f}pp")

    print(f"\nConclusion: ", end="")
    if abs(delta_avg) < 2.0:
        print(f"Model is ROBUST to paraphrasing (delta={delta_avg:+.1f}pp)")
    else:
        print(f"Paraphrasing {'HURTS' if delta_avg < 0 else 'HELPS'} by {abs(delta_avg):.1f}pp — model may be sensitive to instruction wording")

# Per-task breakdown for suites where paraphrase hurts
print("\n--- Per-task breakdown (suites where paraphrase drops > 5pp) ---")
for suite in suites:
    text_r = results[suite].get("text")
    para_r = results[suite].get("paraphrase")
    if text_r and para_r and (para_r['rate'] - text_r['rate']) < -5:
        print(f"\n{suite}:")
        for i, (t_task, p_task) in enumerate(zip(text_r['tasks'], para_r['tasks'])):
            t_rate = t_task[0] / t_task[1] * 100 if t_task[1] > 0 else 0
            p_rate = p_task[0] / p_task[1] * 100 if p_task[1] > 0 else 0
            delta = p_rate - t_rate
            if abs(delta) > 10:
                print(f"  Task {i}: text={t_rate:.0f}% para={p_rate:.0f}% ({delta:+.0f}pp)")

PYEOF

echo ""
echo "$(date '+%H:%M:%S') Done. Logs saved to ${LOG_DIR}/"
