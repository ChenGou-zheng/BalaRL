#!/usr/bin/env bash
# Launch BalaRL PPO training on server.
# Detects GPU, sets up logging, runs in background with nohup.
#
# Usage:
#   ./scripts/run_train.sh            # default 5M steps
#   ./scripts/run_train.sh 10000000   # 10M steps
#   ./scripts/run_train.sh 1000000 --cuda  # 1M steps on GPU

set -e

TIMESTEPS=${1:-5000000}
DEVICE_ARG=${2:-}

# Detect GPU
if [ -z "$DEVICE_ARG" ] && command -v nvidia-smi &> /dev/null; then
    DEVICE="cuda"
    echo "GPU detected, using CUDA"
elif [ "$DEVICE_ARG" = "--cuda" ]; then
    DEVICE="cuda"
else
    DEVICE="cpu"
    echo "Using CPU"
fi

# Create directories
mkdir -p models logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="ppo_${TIMESTAMP}"
LOG_FILE="logs/${RUN_NAME}.log"

echo "=== Starting BalaRL PPO Training ==="
echo "Run name:    $RUN_NAME"
echo "Timesteps:   $TIMESTEPS"
echo "Device:      $DEVICE"
echo "Log:         $LOG_FILE"
echo ""

# Launch training in background with nohup
nohup uv run python -u -m balarl.scripts.train \
    --timesteps "$TIMESTEPS" \
    --n-envs 8 \
    --device "$DEVICE" \
    --log-dir "logs/$RUN_NAME" \
    --model-dir "models/$RUN_NAME" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training launched! PID=$PID"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
echo ""
echo "Stop training:"
echo "  kill $PID"
echo ""
echo "PID saved to: logs/${RUN_NAME}.pid"
echo "$PID" > "logs/${RUN_NAME}.pid"
