#!/usr/bin/env bash
# Language Robustness Experiment — runs multiple instruction variants across 8 GPUs.
# Each GPU runs one model server. Eval clients are distributed across GPUs.
# Text baseline is reused from a previous run if available.
#
# Usage: bash scripts/eval_robustness_parallel.sh [num_trials] [log_prefix]

set -euo pipefail
cd /home/user1/workspace/VLA/openpi2

NUM_TRIALS=${1:-20}
LOG_PREFIX=${2:-robustness_v1}
LOG_DIR="/tmp/eval_paraphrase/${LOG_PREFIX}"
mkdir -p "$LOG_DIR"

OG_LIBERO_PYTHONPATH="/home/user1/workspace/VLA/openpi_og/third_party/libero"
CLIENT_PYTHONPATH="${OG_LIBERO_PYTHONPATH}:/home/user1/workspace/VLA/openpi2/packages/openpi-client/src"
CLIENT_PYTHON="examples/libero/.venv/bin/python"

# Variant definitions: label|json_path
VARIANTS=(
  "hard|data/libero_paraphrased_hard.json"
  "typos|data/libero_typos.json"
  "minimal|data/libero_minimal.json"
  "verbose|data/libero_verbose.json"
  "synonym|data/libero_synonym_objects.json"
)

SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

# Build full job list: suite|variant_label|json_path
JOBS=()
for SUITE in "${SUITES[@]}"; do
  for VSPEC in "${VARIANTS[@]}"; do
    IFS='|' read -r VLABEL VJSON <<< "$VSPEC"
    JOBS+=("${SUITE}|${VLABEL}|${VJSON}")
  done
done

NUM_JOBS=${#JOBS[@]}
NUM_GPUS=8
echo "$(date '+%H:%M:%S') === Language Robustness (8-GPU Parallel) ==="
echo "$(date '+%H:%M:%S') Variants: ${#VARIANTS[@]} (hard, typos, minimal, verbose, synonym)"
echo "$(date '+%H:%M:%S') Suites: ${#SUITES[@]} (spatial, object, goal, 10)"
echo "$(date '+%H:%M:%S') Total jobs: ${NUM_JOBS}, Trials/task: ${NUM_TRIALS}"
echo "$(date '+%H:%M:%S') Logs: ${LOG_DIR}/"
echo ""

SERVER_PIDS=()

cleanup() {
    echo ""
    echo "$(date '+%H:%M:%S') Cleaning up all servers..."
    for pid in "${SERVER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT

# ==================== Launch 8 servers ====================
echo "$(date '+%H:%M:%S') Launching first server to ensure checkpoint is cached..."
OPENPI_NO_COMPILE=1 CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  PYTHONPATH=src:${PYTHONPATH:-} \
  .venv/bin/python scripts/serve_policy.py --port 8000 --env LIBERO \
  > "${LOG_DIR}/server_gpu0.log" 2>&1 &
SERVER_PIDS+=($!)

for attempt in $(seq 1 600); do
    if ss -tlnp 2>/dev/null | grep -q ":8000 "; then
        echo "$(date '+%H:%M:%S') First server ready after ${attempt}s"
        break
    fi
    if ! kill -0 "${SERVER_PIDS[0]}" 2>/dev/null; then
        echo "ERROR: First server died!"; tail -20 "${LOG_DIR}/server_gpu0.log"; exit 1
    fi
    (( attempt % 30 == 0 )) && echo "$(date '+%H:%M:%S')   waiting... (${attempt}s)"
    sleep 1
done

echo "$(date '+%H:%M:%S') Launching remaining 7 servers..."
for GPU in $(seq 1 7); do
    PORT=$((8000 + GPU))
    OPENPI_NO_COMPILE=1 CUDA_VISIBLE_DEVICES=${GPU} XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
      PYTHONPATH=src:${PYTHONPATH:-} \
      .venv/bin/python scripts/serve_policy.py --port ${PORT} --env LIBERO \
      > "${LOG_DIR}/server_gpu${GPU}.log" 2>&1 &
    SERVER_PIDS+=($!)
    sleep 2
done

echo "$(date '+%H:%M:%S') Waiting for all servers..."
for attempt in $(seq 1 300); do
    READY=0
    for GPU in $(seq 0 7); do
        PORT=$((8000 + GPU))
        ss -tlnp 2>/dev/null | grep -q ":${PORT} " && READY=$((READY + 1))
    done
    if [[ $READY -eq 8 ]]; then
        echo "$(date '+%H:%M:%S') All 8 servers ready after ${attempt}s"
        break
    fi
    (( attempt % 30 == 0 )) && echo "$(date '+%H:%M:%S')   ${READY}/8 servers ready (${attempt}s)..."
    sleep 1
done

# ==================== Distribute and run jobs ====================
# Assign jobs round-robin to GPUs, then run sequentially per GPU.
# Each GPU gets ~ceil(20/8) = 2-3 jobs.

# Create job queues per GPU
for GPU in $(seq 0 7); do
    eval "GPU_JOBS_${GPU}=()"
done

for i in "${!JOBS[@]}"; do
    GPU=$((i % NUM_GPUS))
    eval "GPU_JOBS_${GPU}+=(\"${JOBS[$i]}\")"
done

echo ""
echo "$(date '+%H:%M:%S') Job assignment:"
for GPU in $(seq 0 7); do
    eval "QUEUE=(\"\${GPU_JOBS_${GPU}[@]}\")"
    LABELS=""
    for JOB in "${QUEUE[@]}"; do
        IFS='|' read -r SUITE VLABEL VJSON <<< "$JOB"
        LABELS="${LABELS} ${SUITE}/${VLABEL}"
    done
    echo "  GPU ${GPU}:${LABELS}"
done

# Launch one background worker per GPU
echo ""
echo "$(date '+%H:%M:%S') Launching workers..."
WORKER_PIDS=()

for GPU in $(seq 0 7); do
    PORT=$((8000 + GPU))
    eval "QUEUE=(\"\${GPU_JOBS_${GPU}[@]}\")"

    (
        for JOB in "${QUEUE[@]}"; do
            IFS='|' read -r SUITE VLABEL VJSON <<< "$JOB"
            LABEL="${SUITE}_${VLABEL}"
            LOG="${LOG_DIR}/${LABEL}.log"

            echo "$(date '+%H:%M:%S') [GPU ${GPU}] Starting ${LABEL}..."

            MUJOCO_EGL_DEVICE_ID=${GPU} MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="" \
              PYTHONPATH=${CLIENT_PYTHONPATH}:${PYTHONPATH:-} \
              ${CLIENT_PYTHON} examples/libero/main.py \
              --args.task-suite-name="${SUITE}" \
              --args.num-trials-per-task="${NUM_TRIALS}" \
              --args.eval-mode="paraphrase" \
              --args.paraphrase-map="${VJSON}" \
              --args.port=${PORT} \
              --args.video-out-path="${LOG_DIR}/videos/${LABEL}" \
              > "${LOG}" 2>&1

            EP=$(grep -c "Success:" "${LOG}" 2>/dev/null || echo "0")
            SUCC=$(grep -c "Success: True" "${LOG}" 2>/dev/null || echo "0")
            echo "$(date '+%H:%M:%S') [GPU ${GPU}] Done ${LABEL}: ${SUCC}/${EP}"
        done
    ) &
    WORKER_PIDS+=($!)
done

# ==================== Monitor progress ====================
echo ""
echo "$(date '+%H:%M:%S') All workers launched. Monitoring..."

while true; do
    DONE=0
    for pid in "${WORKER_PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null || DONE=$((DONE + 1))
    done

    # Count completed episodes across all logs
    TOTAL_EP=0
    TOTAL_EXPECTED=$((NUM_JOBS * 10 * NUM_TRIALS))
    for JOB in "${JOBS[@]}"; do
        IFS='|' read -r SUITE VLABEL VJSON <<< "$JOB"
        LABEL="${SUITE}_${VLABEL}"
        EP="$(grep -c 'Success:' "${LOG_DIR}/${LABEL}.log" 2>/dev/null || true)"
        EP="${EP:-0}"
        TOTAL_EP=$((TOTAL_EP + EP))
    done
    if [[ $TOTAL_EXPECTED -gt 0 ]]; then
        PCT=$((TOTAL_EP * 100 / TOTAL_EXPECTED))
    else
        PCT=0
    fi

    echo "$(date '+%H:%M:%S') Progress: ${TOTAL_EP}/${TOTAL_EXPECTED} episodes (${PCT}%), workers done: ${DONE}/8"

    if [[ $DONE -eq 8 ]]; then
        break
    fi
    sleep 120
done

echo ""
echo "$(date '+%H:%M:%S') All workers finished!"

# ==================== Parse and report results ====================
echo ""
echo "============================================================"
echo "  LANGUAGE ROBUSTNESS — ${LOG_PREFIX}, ${NUM_TRIALS} trials"
echo "============================================================"

python3 - "${LOG_DIR}" "${NUM_TRIALS}" <<'PYEOF'
import sys, os, json

log_dir = sys.argv[1]
num_trials = int(sys.argv[2])

def parse_log(log_path):
    task_results = []
    cs, ct = 0, 0
    with open(log_path) as f:
        for line in f:
            if "Success: True" in line: cs += 1; ct += 1
            elif "Success: False" in line: ct += 1
            if "Current task success rate" in line:
                task_results.append((cs, ct)); cs, ct = 0, 0
    ts = sum(s for s, _ in task_results)
    tt = sum(t for _, t in task_results)
    return ts, tt

suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
variants = ["text", "hard", "typos", "minimal", "verbose", "synonym"]

# Try to load text baseline from v1
v1_dir = os.path.join(os.path.dirname(log_dir), "paraphrase_v1")

results = {}
for suite in suites:
    results[suite] = {}
    for var in variants:
        if var == "text":
            log_path = os.path.join(v1_dir, f"{suite}_text.log")
        else:
            log_path = os.path.join(log_dir, f"{suite}_{var}.log")
        if os.path.exists(log_path) and os.path.getsize(log_path) > 100:
            ts, tt = parse_log(log_path)
            results[suite][var] = (ts, tt, ts/tt*100 if tt > 0 else 0)
        else:
            results[suite][var] = None

# Header
hdr = f"{'Suite':<18}"
for v in variants:
    hdr += f" {v:>10}"
hdr += "     worst_drop"
print(f"\n{hdr}")
print("-" * (18 + 11 * len(variants) + 15))

totals = {v: [0, 0] for v in variants}
for suite in suites:
    row = f"{suite:<18}"
    text_rate = results[suite].get("text")
    text_r = text_rate[2] if text_rate else None
    worst_drop = 0
    for v in variants:
        r = results[suite].get(v)
        if r:
            row += f" {r[2]:>9.1f}%"
            totals[v][0] += r[0]; totals[v][1] += r[1]
            if text_r and v != "text":
                drop = r[2] - text_r
                if drop < worst_drop:
                    worst_drop = drop
        else:
            row += f" {'N/A':>10}"
    row += f"  {worst_drop:>+8.1f}pp" if text_r else ""
    print(row)

# Average row
print("-" * (18 + 11 * len(variants) + 15))
row = f"{'AVERAGE':<18}"
text_avg = None
for v in variants:
    s, t = totals[v]
    if t > 0:
        avg = s / t * 100
        row += f" {avg:>9.1f}%"
        if v == "text": text_avg = avg
    else:
        row += f" {'N/A':>10}"
print(row)

# Summary
print("\nSummary:")
if text_avg:
    for v in variants:
        if v == "text": continue
        s, t = totals[v]
        if t > 0:
            avg = s / t * 100
            delta = avg - text_avg
            severity = "ROBUST" if abs(delta) < 2 else ("MINOR" if abs(delta) < 5 else "SIGNIFICANT")
            print(f"  {v:>10}: {avg:.1f}% ({delta:+.1f}pp vs text) [{severity}]")

# Save JSON
out = {}
for suite in suites:
    for v in variants:
        r = results[suite].get(v)
        if r:
            out[f"{suite}_{v}"] = {"rate": round(r[2], 1), "successes": r[0], "trials": r[1]}
with open(os.path.join(log_dir, "results.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"\nResults saved to {os.path.join(log_dir, 'results.json')}")

PYEOF

echo ""
echo "$(date '+%H:%M:%S') Done. Logs at ${LOG_DIR}/"
