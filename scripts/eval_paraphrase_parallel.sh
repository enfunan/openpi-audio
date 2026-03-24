#!/usr/bin/env bash
# Paraphrase robustness experiment — 8-GPU PARALLEL version.
# Runs 8 jobs (4 suites × 2 modes) concurrently, one server per GPU.
#
# Usage: bash scripts/eval_paraphrase_parallel.sh [num_trials] [log_prefix]
#
# Default: 20 trials, uses pi05_libero checkpoint (auto-download from GCS).

set -euo pipefail
cd /home/user1/workspace/VLA/openpi2

NUM_TRIALS=${1:-20}
LOG_PREFIX=${2:-paraphrase_v1}

PARAPHRASE_MAP="data/libero_paraphrased_instructions.json"
LOG_DIR="/tmp/eval_paraphrase/${LOG_PREFIX}"
mkdir -p "$LOG_DIR"

# 8 jobs: 4 suites × 2 modes
JOBS=(
  "0|8000|libero_spatial|text"
  "1|8001|libero_spatial|paraphrase"
  "2|8002|libero_object|text"
  "3|8003|libero_object|paraphrase"
  "4|8004|libero_goal|text"
  "5|8005|libero_goal|paraphrase"
  "6|8006|libero_10|text"
  "7|8007|libero_10|paraphrase"
)

SERVER_PIDS=()
CLIENT_PIDS=()

cleanup() {
    echo ""
    echo "$(date '+%H:%M:%S') Cleaning up all servers..."
    for pid in "${SERVER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${CLIENT_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "$(date '+%H:%M:%S') === Paraphrase Robustness (8-GPU Parallel) ==="
echo "$(date '+%H:%M:%S') Trials per task: ${NUM_TRIALS}"
echo "$(date '+%H:%M:%S') Logs: ${LOG_DIR}/"
echo "$(date '+%H:%M:%S') Jobs: ${#JOBS[@]} (4 suites × 2 modes)"
echo ""

# ==================== Ensure checkpoint is cached ====================
# Launch one server first to download the checkpoint if needed, then launch the rest.
echo "$(date '+%H:%M:%S') Launching first server (GPU 0) to ensure checkpoint is cached..."
IFS='|' read -r GPU0 PORT0 SUITE0 MODE0 <<< "${JOBS[0]}"
OPENPI_NO_COMPILE=1 CUDA_VISIBLE_DEVICES=${GPU0} XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  PYTHONPATH=src:${PYTHONPATH:-} \
  .venv/bin/python scripts/serve_policy.py \
    --port ${PORT0} \
    --env LIBERO \
  > "${LOG_DIR}/server_gpu${GPU0}.log" 2>&1 &
SERVER_PIDS+=($!)
echo "  GPU ${GPU0} -> port ${PORT0} (PID ${SERVER_PIDS[0]}) — ${SUITE0}/${MODE0}"

# Wait for first server to be ready (means checkpoint is fully cached)
echo "$(date '+%H:%M:%S') Waiting for first server (checkpoint download + model load)..."
for attempt in $(seq 1 600); do
    if ss -tlnp 2>/dev/null | grep -q ":${PORT0} "; then
        echo "$(date '+%H:%M:%S') First server ready after ${attempt}s"
        break
    fi
    if ! kill -0 "${SERVER_PIDS[0]}" 2>/dev/null; then
        echo "$(date '+%H:%M:%S') ERROR: First server died!"
        tail -20 "${LOG_DIR}/server_gpu${GPU0}.log"
        exit 1
    fi
    if (( attempt % 30 == 0 )); then
        echo "$(date '+%H:%M:%S')   waiting... (${attempt}s)"
    fi
    sleep 1
done

# ==================== Launch remaining 7 servers ====================
echo "$(date '+%H:%M:%S') Launching remaining 7 servers..."
for i in $(seq 1 $((${#JOBS[@]} - 1))); do
    IFS='|' read -r GPU PORT SUITE MODE <<< "${JOBS[$i]}"

    OPENPI_NO_COMPILE=1 CUDA_VISIBLE_DEVICES=${GPU} XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
      PYTHONPATH=src:${PYTHONPATH:-} \
      .venv/bin/python scripts/serve_policy.py \
        --port ${PORT} \
        --env LIBERO \
      > "${LOG_DIR}/server_gpu${GPU}.log" 2>&1 &
    SERVER_PIDS+=($!)
    echo "  GPU ${GPU} -> port ${PORT} (PID $!) — ${SUITE}/${MODE}"
    sleep 2  # small stagger to avoid filesystem contention
done

# Wait for all servers to be ready
echo ""
echo "$(date '+%H:%M:%S') Waiting for all servers..."
ALL_READY=false
for attempt in $(seq 1 300); do
    READY_COUNT=0
    for JOB in "${JOBS[@]}"; do
        IFS='|' read -r GPU PORT SUITE MODE <<< "$JOB"
        if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
            READY_COUNT=$((READY_COUNT + 1))
        fi
    done
    if [[ $READY_COUNT -eq ${#JOBS[@]} ]]; then
        ALL_READY=true
        echo "$(date '+%H:%M:%S') All ${#JOBS[@]} servers ready after ${attempt}s"
        break
    fi
    # Check for dead servers
    for i in "${!SERVER_PIDS[@]}"; do
        if ! kill -0 "${SERVER_PIDS[$i]}" 2>/dev/null; then
            IFS='|' read -r GPU PORT SUITE MODE <<< "${JOBS[$i]}"
            echo "$(date '+%H:%M:%S') WARNING: Server on GPU ${GPU} (port ${PORT}) died!"
            tail -5 "${LOG_DIR}/server_gpu${GPU}.log"
            # Don't exit — try to continue with remaining servers
        fi
    done
    if (( attempt % 30 == 0 )); then
        echo "$(date '+%H:%M:%S')   ${READY_COUNT}/${#JOBS[@]} servers ready (${attempt}s elapsed)..."
    fi
    sleep 1
done

if [[ "$ALL_READY" != "true" ]]; then
    echo "$(date '+%H:%M:%S') ERROR: Not all servers started within 300s"
    exit 1
fi

# ==================== Launch 8 eval clients in parallel ====================
echo ""
echo "$(date '+%H:%M:%S') Launching ${#JOBS[@]} eval clients..."
for JOB in "${JOBS[@]}"; do
    IFS='|' read -r GPU PORT SUITE MODE <<< "$JOB"
    LABEL="${SUITE}_${MODE}"
    LOG="${LOG_DIR}/${LABEL}.log"

    EVAL_CMD=(
      examples/libero/.venv/bin/python examples/libero/main.py
      --args.task-suite-name="${SUITE}"
      --args.num-trials-per-task="${NUM_TRIALS}"
      --args.eval-mode="${MODE}"
      --args.port=${PORT}
      --args.video-out-path="${LOG_DIR}/videos/${LABEL}"
    )

    if [[ "$MODE" == "paraphrase" ]]; then
      EVAL_CMD+=(--args.paraphrase-map="${PARAPHRASE_MAP}")
    fi

    MUJOCO_EGL_DEVICE_ID=${GPU} MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="" \
      PYTHONPATH=/home/user1/workspace/VLA/openpi_og/third_party/libero:/home/user1/workspace/VLA/openpi2/packages/openpi-client/src:${PYTHONPATH:-} \
      "${EVAL_CMD[@]}" > "${LOG}" 2>&1 &
    CLIENT_PIDS+=($!)
    echo "  [GPU ${GPU}] ${SUITE} / ${MODE} -> ${LOG} (PID $!)"
done

# ==================== Wait for all clients and report progress ====================
echo ""
echo "$(date '+%H:%M:%S') All clients launched. Monitoring progress..."

while true; do
    DONE_COUNT=0
    RUNNING=""
    for i in "${!CLIENT_PIDS[@]}"; do
        IFS='|' read -r GPU PORT SUITE MODE <<< "${JOBS[$i]}"
        if ! kill -0 "${CLIENT_PIDS[$i]}" 2>/dev/null; then
            DONE_COUNT=$((DONE_COUNT + 1))
        else
            # Count completed episodes
            LABEL="${SUITE}_${MODE}"
            EP_COUNT=$(grep -c "Success:" "${LOG_DIR}/${LABEL}.log" 2>/dev/null || echo "0")
            TOTAL=$((10 * NUM_TRIALS))
            RUNNING="${RUNNING}  ${SUITE}/${MODE}: ${EP_COUNT}/${TOTAL}\n"
        fi
    done
    echo -e "$(date '+%H:%M:%S') Completed: ${DONE_COUNT}/${#JOBS[@]} jobs"
    if [[ -n "$RUNNING" ]]; then
        echo -e "$RUNNING"
    fi
    if [[ $DONE_COUNT -eq ${#JOBS[@]} ]]; then
        break
    fi
    sleep 60
done

echo "$(date '+%H:%M:%S') All eval clients finished!"

# ==================== Parse and report results ====================
echo ""
echo "============================================================"
echo "  PARAPHRASE ROBUSTNESS — ${LOG_PREFIX}, ${NUM_TRIALS} trials"
echo "============================================================"

python3 - "${LOG_DIR}" "${NUM_TRIALS}" <<'PYEOF'
import sys, os, json

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
        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            task_results, total_s, total_t = parse_log(log_path)
            rate = total_s / total_t * 100 if total_t > 0 else 0
            results[suite][mode] = {"successes": total_s, "trials": total_t, "rate": rate, "tasks": task_results}
        else:
            results[suite][mode] = None

# Print table
print(f"\n{'Suite':<20} {'Text':>15} {'Paraphrase':>15} {'Delta':>8}")
print("-" * 62)
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
    print(f"{suite:<20} {text_str:>15} {para_str:>15} {delta:>8}")

if text_total_t > 0 and para_total_t > 0:
    text_avg = text_total_s / text_total_t * 100
    para_avg = para_total_s / para_total_t * 100
    delta_avg = para_avg - text_avg
    print("-" * 62)
    print(f"{'AVERAGE':<20} {text_avg:>14.1f}% {para_avg:>14.1f}% {delta_avg:>+7.1f}pp")

    print(f"\nConclusion: ", end="")
    if abs(delta_avg) < 2.0:
        print(f"Model is ROBUST to paraphrasing (delta={delta_avg:+.1f}pp)")
    else:
        print(f"Paraphrasing {'HURTS' if delta_avg < 0 else 'HELPS'} by {abs(delta_avg):.1f}pp — model may be sensitive to instruction wording")

# Per-task breakdown where paraphrase hurts
print("\n--- Per-task breakdown (suites where paraphrase drops > 5pp) ---")
for suite in suites:
    text_r = results[suite].get("text")
    para_r = results[suite].get("paraphrase")
    if text_r and para_r and (para_r['rate'] - text_r['rate']) < -5:
        print(f"\n{suite}:")
        for i, (t_task, p_task) in enumerate(zip(text_r['tasks'], para_r['tasks'])):
            t_rate = t_task[0] / t_task[1] * 100 if t_task[1] > 0 else 0
            p_rate = p_task[0] / p_task[1] * 100 if p_task[1] > 0 else 0
            d = p_rate - t_rate
            if abs(d) > 10:
                print(f"  Task {i}: text={t_rate:.0f}% para={p_rate:.0f}% ({d:+.0f}pp)")

# Save results as JSON
results_json = {}
for suite in suites:
    for mode in modes:
        r = results[suite].get(mode)
        if r:
            results_json[f"{suite}_{mode}"] = {"rate": r["rate"], "successes": r["successes"], "trials": r["trials"]}
with open(os.path.join(log_dir, "results.json"), "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\nResults saved to {os.path.join(log_dir, 'results.json')}")

PYEOF

echo ""
echo "$(date '+%H:%M:%S') Done. Logs saved to ${LOG_DIR}/"
