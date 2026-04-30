#!/usr/bin/env python3
"""Entry point for training a PPO agent on Balatro.

Usage:
    python -u -m balarl.scripts.train --quick-test     # 5k step smoke test
    python -u -m balarl.scripts.train --server          # Server-optimized: SubprocVecEnv, 5M steps
    python -u -m balarl.scripts.train --timesteps 1e6   # Custom steps
"""

from __future__ import annotations

import argparse
import platform
import sys

from balarl.training.ppo_train import train_ppo
from balarl.training.config import TrainingConfig, QUICK_TEST_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on Balatro")
    parser.add_argument("--quick-test", action="store_true", help="Smoke test (5K steps)")
    parser.add_argument("--server", action="store_true", help="Server-optimized training")
    parser.add_argument("--timesteps", type=int, default=None, help="Total timesteps")
    parser.add_argument("--n-envs", type=int, default=None, help="Parallel environments")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Model save directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum")
    args = parser.parse_args()

    if args.quick_test:
        config = TrainingConfig(
            total_timesteps=5_000, n_envs=2, n_steps=256, batch_size=32, n_epochs=4,
            features_dim=128, net_arch_pi=[64, 64], net_arch_vf=[64, 64],
            checkpoint_freq=10_000, log_freq=500, save_freq=5_000,
            device=args.device or "cpu",
        )
    elif args.server:
        is_linux = platform.system() == "Linux"
        config = TrainingConfig(
            total_timesteps=5_000_000, n_envs=8 if is_linux else 4,
            n_steps=2048, batch_size=128, n_epochs=10,
            learning_rate=3e-4,
            features_dim=512, net_arch_pi=[256, 256], net_arch_vf=[256, 256],
            checkpoint_freq=50_000, log_freq=2_000, save_freq=100_000,
            use_curriculum=True,
            device=args.device or ("cuda" if is_linux else "cpu"),
        )
    else:
        config = TrainingConfig()

    # CLI overrides
    if args.timesteps: config.total_timesteps = args.timesteps
    if args.n_envs: config.n_envs = args.n_envs
    if args.lr: config.learning_rate = args.lr
    if args.device: config.device = args.device
    if args.log_dir: config.log_dir = args.log_dir
    if args.model_dir: config.model_dir = args.model_dir
    if args.seed is not None: config.seed = args.seed
    if args.no_curriculum: config.use_curriculum = False

    model, path = train_ppo(config)
    print(f"\nDone. Model: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
