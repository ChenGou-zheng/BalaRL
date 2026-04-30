#!/usr/bin/env bash
# Sync training results FROM server TO local machine.
# Run this ON YOUR LOCAL MACHINE.
#
# Usage:
#   ./scripts/sync_results.sh user@server:/path/to/BalaRL

set -e

REMOTE=${1:-}
if [ -z "$REMOTE" ]; then
    echo "Usage: ./scripts/sync_results.sh user@server:/path/to/BalaRL"
    echo "Example: ./scripts/sync_results.sh root@192.168.1.100:/home/user/BalaRL"
    exit 1
fi

echo "Syncing from: $REMOTE"

# Pull models and logs (skip .venv, __pycache__, etc.)
rsync -avz --progress \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    "$REMOTE/models/" "./models/"
rsync -avz --progress \
    --exclude '.venv/' \
    "$REMOTE/logs/" "./logs/"

echo ""
echo "=== Sync complete! ==="
echo "Models: ./models/"
echo "Logs:   ./logs/"
