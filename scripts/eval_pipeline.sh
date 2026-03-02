#!/bin/bash
# Usage: eval_pipeline.sh <checkpoint_dir> <port> <gpu_ids> <log_prefix>
# Starts a model server, runs libero_spatial and libero_object evals, then stops the server.
set -e

CKPT_DIR=$1
PORT=$2
GPU_IDS=$3
LOG_PREFIX=$4

cd /home/user1/workspace/VLA/openpi

echo "[${LOG_PREFIX}] Starting server on port ${PORT} with GPUs ${GPU_IDS}..."
echo "[${LOG_PREFIX}] Checkpoint: ${CKPT_DIR}"

# Start model server in background
CUDA_VISIBLE_DEVICES=${GPU_IDS} XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  .venv/bin/python scripts/serve_policy.py \
  --port ${PORT} \
  policy:checkpoint \
  --policy.config pi05_audio_stage3_libero \
  --policy.dir "${CKPT_DIR}" \
  > /tmp/${LOG_PREFIX}_server.log 2>&1 &
SERVER_PID=$!

# Wait for server to be ready (check if port is listening)
echo "[${LOG_PREFIX}] Waiting for server (PID ${SERVER_PID}) to be ready..."
for i in $(seq 1 120); do
  if ss -tlnp | grep -q ":${PORT} "; then
    echo "[${LOG_PREFIX}] Server ready after ${i}s"
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[${LOG_PREFIX}] Server died! Check /tmp/${LOG_PREFIX}_server.log"
    cat /tmp/${LOG_PREFIX}_server.log
    exit 1
  fi
  sleep 1
done

if ! ss -tlnp | grep -q ":${PORT} "; then
  echo "[${LOG_PREFIX}] Server failed to start within 120s"
  kill $SERVER_PID 2>/dev/null
  exit 1
fi

# Run libero_spatial eval
echo "[${LOG_PREFIX}] Running libero_spatial eval..."
CUDA_VISIBLE_DEVICES="" .venv/bin/python examples/libero/main.py \
  --args.task-suite-name=libero_spatial \
  --args.num-trials-per-task=50 \
  --args.eval-mode=both \
  --args.audio-dir=/home/user1/workspace/VLA/data/tts/libero_eval \
  --args.port=${PORT} \
  --args.video-out-path=data/libero/videos/${LOG_PREFIX}_spatial \
  2>&1 | tee /tmp/${LOG_PREFIX}_spatial.log

# Run libero_object eval
echo "[${LOG_PREFIX}] Running libero_object eval..."
CUDA_VISIBLE_DEVICES="" .venv/bin/python examples/libero/main.py \
  --args.task-suite-name=libero_object \
  --args.num-trials-per-task=50 \
  --args.eval-mode=both \
  --args.audio-dir=/home/user1/workspace/VLA/data/tts/libero_eval \
  --args.port=${PORT} \
  --args.video-out-path=data/libero/videos/${LOG_PREFIX}_object \
  2>&1 | tee /tmp/${LOG_PREFIX}_object.log

# Stop server
echo "[${LOG_PREFIX}] Stopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "[${LOG_PREFIX}] Done!"
echo ""
echo "===== ${LOG_PREFIX} SUMMARY ====="
echo "--- libero_spatial ---"
grep -E "COMPARISON SUMMARY|Total success rate|TEXT:|AUDIO:" /tmp/${LOG_PREFIX}_spatial.log | tail -6
echo "--- libero_object ---"
grep -E "COMPARISON SUMMARY|Total success rate|TEXT:|AUDIO:" /tmp/${LOG_PREFIX}_object.log | tail -6
