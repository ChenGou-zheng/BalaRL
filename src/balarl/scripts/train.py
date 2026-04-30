#!/usr/bin/env python3
"""Entry point for training a PPO agent on Balatro.

Usage:
    python -m balarl.scripts.train --quick-test    # 10k step smoke test
    python -m balarl.scripts.train --timesteps 1e6 # Custom timesteps
    python -m balarl.scripts.train --full          # Full training run
"""

from __future__ import annotations

import argparse
import sys

from balarl.training.ppo_train import train_ppo
from balarl.training.config import TrainingConfig, QUICK_TEST_CONFIG, SERVER_TRAIN_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on Balatro")
    parser.add_argument("--quick-test", action="store_true", help="Run 10k step smoke test")
    parser.add_argument("--full", action="store_true", help="Run full training (10M steps)")
    parser.add_argument("--timesteps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=None, help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Model save directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.quick_test:
        config = QUICK_TEST_CONFIG
    elif args.full:
        config = SERVER_TRAIN_CONFIG
    else:
        config = TrainingConfig()

    # Override from command line
    if args.timesteps is not None:
        config.total_timesteps = args.timesteps
    if args.n_envs is not None:
        config.n_envs = args.n_envs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.no_curriculum:
        config.use_curriculum = False
    if args.device:
        config.device = args.device
    if args.log_dir:
        config.log_dir = args.log_dir
        config.tensorboard_log = f"{args.log_dir}/tensorboard"
    if args.model_dir:
        config.model_dir = args.model_dir
    if args.seed is not None:
        config.seed = args.seed

    print(f"Training mode: {'quick test' if args.quick_test else 'full' if args.full else 'custom'}")
    print(f"Config: {config.total_timesteps:,} steps, {config.n_envs} envs, lr={config.learning_rate}")

    model, path = train_ppo(config)
    print(f"\nDone! Model saved to: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
