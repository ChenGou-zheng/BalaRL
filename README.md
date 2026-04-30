# BalaRL — Reinforcement Learning for Balatro

PPO reinforcement learning agent for the poker roguelike deckbuilder Balatro.

## Quick Start (Local)

```bash
# Install
uv venv --python 3.11 && uv pip install -e ".[training]"

# Run tests
uv run python tests/test_engine/test_smoke.py

# Quick training test (10k steps, ~1 min)
uv run python -u -m balarl.scripts.train --quick-test
```

## Server Training

### 1. Push to Git & Clone on Server

```bash
# Local
git remote add origin <your-repo-url>
git add -A && git commit -m "BalaRL: PPO training pipeline"
git push -u origin main

# Server
git clone <your-repo-url> && cd BalaRL
```

### 2. One-time Setup

```bash
chmod +x scripts/*.sh
./scripts/setup_server.sh
```

### 3. Launch Training

```bash
# 5M steps (default)
./scripts/run_train.sh

# Or specify steps + device
./scripts/run_train.sh 10000000 --cuda
```

### 4. Monitor

```bash
tail -f logs/ppo_*/ppo_*.log
nvidia-smi  # GPU usage
```

### 5. Sync Results Back

```bash
# On your local machine
./scripts/sync_results.sh user@server:/path/to/BalaRL
```

## Project Structure

```
BalaRL/
├── src/balarl/
│   ├── engine/          # Game engine (cards, scoring, jokers, shop, etc.)
│   ├── env/             # Gymnasium environment
│   ├── agents/          # Expert/heuristic agent
│   └── training/        # PPO pipeline (feature extractor, curriculum, callbacks)
├── scripts/             # Server deployment scripts
├── tests/               # Unit and smoke tests
└── configs/             # Training configs
```

## Phase Progress

- [x] Phase 0: Project infrastructure, game engine (150 jokers, 12 hand types, 29 boss blinds)
- [x] Phase 1: Expert agent baseline (avg ante 3.6/8, 2/20 wins)
- [x] Phase 2: PPO training pipeline (SB3, curriculum learning, checkpointing)
- [ ] Phase 3: Endless mode optimization (MCTS, Decision Transformer)
