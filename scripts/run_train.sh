#!/usr/bin/env bash
# Launch BalaRL PPO training on server with BC warm-start.
# Detects GPU, uses SubprocVecEnv on Linux, runs in background.
#
# Usage:
#   ./scripts/run_train.sh            # default 5M steps, with BC
#   ./scripts/run_train.sh 10000000   # 10M steps

set -e

TIMESTEPS=${1:-5000000}
DEVICE="cuda"
TRAJ_FILE="trajectories/expert_trajectories.pkl"

# Detect GPU
if ! command -v nvidia-smi &> /dev/null; then
    DEVICE="cpu"
    echo "No GPU detected, using CPU"
else
    echo "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi

# Create directories
mkdir -p models logs trajectories

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="ppo_${TIMESTAMP}"

# Step 1: Generate expert trajectories (if not already present)
if [ ! -f "$TRAJ_FILE" ]; then
    echo ""
    echo "=== Generating expert trajectories (one-time) ==="
    uv run python -u -m balarl.scripts.generate_trajectories \
        --n-episodes 500 \
        --min-ante 3 \
        --save "$TRAJ_FILE"
    echo "Trajectories saved to: $TRAJ_FILE"
fi

# Step 2: Launch training with BC warm-start
echo ""
echo "=== BalaRL PPO Training ==="
echo "Run:       $RUN_NAME"
echo "Steps:     $TIMESTEPS"
echo "Device:    $DEVICE"
echo "BC trajs:  $TRAJ_FILE"
echo ""

mkdir -p "logs/${RUN_NAME}"
mkdir -p "models/${RUN_NAME}"

nohup uv run python -u -m balarl.scripts.train \
    --server \
    --timesteps "$TIMESTEPS" \
    --device "$DEVICE" \
    --bc-trajectories "$TRAJ_FILE" \
    --bc-epochs 50 \
    --log-dir "logs/$RUN_NAME" \
    --model-dir "models/$RUN_NAME" \
    > "logs/${RUN_NAME}/stdout.log" 2>&1 &

PID=$!

sleep 2
if kill -0 "$PID" 2>/dev/null; then
    echo "Training started (PID=$PID)"
    echo ""
    echo "Monitor:  tail -f logs/${RUN_NAME}/stdout.log"
    echo "GPU:      nvidia-smi"
    echo "Stop:     kill $PID"
    echo "$PID" > "logs/${RUN_NAME}/pid.txt"
else
    echo "ERROR: Training failed to start. Check logs/${RUN_NAME}/stdout.log"
    exit 1
fi
