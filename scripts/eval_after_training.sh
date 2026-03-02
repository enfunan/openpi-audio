#!/usr/bin/env bash
# Wait for Stage 3v2 training to finish, then run ambiguity eval on v1 checkpoint.
# Usage: bash scripts/eval_after_training.sh

set -euo pipefail
cd /home/user1/workspace/VLA/openpi

echo "$(date '+%H:%M:%S') Waiting for training to complete (watching tmux:stage3v2)..."

while true; do
    # Check if stage3v2 tmux session is still running a python process
    if ! tmux has-session -t stage3v2 2>/dev/null; then
        echo "$(date '+%H:%M:%S') tmux session stage3v2 gone — training finished"
        break
    fi

    # Check if python training process is still running in the session
    PANE_PID=$(tmux list-panes -t stage3v2 -F '#{pane_pid}' 2>/dev/null || echo "")
    if [[ -n "$PANE_PID" ]]; then
        PYTHON_RUNNING=$(pgrep -P "$PANE_PID" -f "python.*train.py" 2>/dev/null || echo "")
        if [[ -z "$PYTHON_RUNNING" ]]; then
            echo "$(date '+%H:%M:%S') Training process finished (tmux session still open)"
            break
        fi
    fi

    sleep 120
done

# Wait a bit for GPU memory to be freed
echo "$(date '+%H:%M:%S') Waiting 30s for GPU memory cleanup..."
sleep 30

# Check GPU memory is actually free
FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
echo "$(date '+%H:%M:%S') GPU 0 free memory: ${FREE_MEM} MiB"
if [[ "$FREE_MEM" -lt 30000 ]]; then
    echo "$(date '+%H:%M:%S') WARNING: GPU 0 still has low free memory. Waiting 60s more..."
    sleep 60
fi

echo "$(date '+%H:%M:%S') Launching ambiguity eval on v1 checkpoint..."
bash scripts/eval_ambiguity.sh 0 8001 2>&1 | tee /tmp/eval_ambiguity/full_run.log

echo "$(date '+%H:%M:%S') All done!"
