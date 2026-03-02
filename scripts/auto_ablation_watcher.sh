#!/usr/bin/env bash
# Auto-run audio ablation at key checkpoints during Stage 3v2 training.
# Usage: bash scripts/auto_ablation_watcher.sh
#
# Watches for checkpoints at steps: 1999, 4999, 9999, 14999, 19999, 24999, 29999
# Runs ablation on a single GPU (GPU 7) to avoid interfering with training (GPUs 0-7).

set -euo pipefail

CKPT_BASE="checkpoints/pi05_audio_stage3_libero/stage3v2_libero"
LOG_DIR="/tmp/ablation_logs"
mkdir -p "$LOG_DIR"

# Steps to run ablation on (checkpoint dirs use step number)
STEPS=(2000 5000 10000 15000 20000 25000 29999)

# Track which steps we've already processed
declare -A DONE

echo "$(date '+%H:%M:%S') Auto-ablation watcher started."
echo "$(date '+%H:%M:%S') Watching: ${CKPT_BASE}"
echo "$(date '+%H:%M:%S') Target steps: ${STEPS[*]}"
echo "$(date '+%H:%M:%S') Logs: ${LOG_DIR}/"
echo ""

while true; do
    all_done=true
    for step in "${STEPS[@]}"; do
        # Skip already-processed steps
        if [[ -n "${DONE[$step]:-}" ]]; then
            continue
        fi

        ckpt_path="${CKPT_BASE}/${step}/params"
        if [[ -d "${CKPT_BASE}/${step}" ]]; then
            # Wait a few seconds for checkpoint write to finish
            sleep 10

            log_file="${LOG_DIR}/ablation_step${step}.log"
            echo "$(date '+%H:%M:%S') === Running ablation for step ${step} ==="
            echo "$(date '+%H:%M:%S') Checkpoint: ${ckpt_path}"
            echo "$(date '+%H:%M:%S') Log: ${log_file}"

            # Run on CPU to avoid interfering with training (all 8 GPUs at ~42/46GB)
            JAX_PLATFORMS=cpu .venv/bin/python scripts/diag_audio_ablation_stage3.py \
                "${ckpt_path}" 2>&1 | tee "${log_file}"

            echo ""
            echo "$(date '+%H:%M:%S') === Ablation step ${step} complete ==="
            echo ""

            DONE[$step]=1
        else
            all_done=false
        fi
    done

    # Exit if all steps processed
    if $all_done; then
        echo "$(date '+%H:%M:%S') All ablation steps complete!"
        break
    fi

    # Poll every 60 seconds
    sleep 60
done

# Print summary
echo ""
echo "============================================================"
echo "ABLATION SUMMARY"
echo "============================================================"
for step in "${STEPS[@]}"; do
    log_file="${LOG_DIR}/ablation_step${step}.log"
    if [[ -f "${log_file}" ]]; then
        echo "--- Step ${step} ---"
        grep -E "(Delta|CONCLUSION)" "${log_file}" || echo "  (no results found)"
    fi
done
