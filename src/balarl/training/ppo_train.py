"""PPO training pipeline for Balatro - core training loop with SB3."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback, EvalCallback,
)

from balarl.env.balatro_env import BalatroEnv
from balarl.training.feature_extractor import BalatroFeatureExtractor
from balarl.training.config import TrainingConfig, QUICK_TEST_CONFIG
from balarl.training.curriculum import CurriculumScheduler
from balarl.training.callbacks import BalatroMetricsCallback, TrainingLogger


class CurriculumEnvWrapper(gym.Wrapper):
    """Wraps BalatroEnv to enforce curriculum max ante."""

    def __init__(self, env: BalatroEnv, max_ante: int = 8):
        super().__init__(env)
        self._max_ante = max_ante

    @property
    def max_ante(self) -> int:
        return self._max_ante

    def set_max_ante(self, ante: int):
        self._max_ante = ante

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Terminate if exceeding curriculum max
        if hasattr(self.env, 'state') and self.env.state.ante > self._max_ante:
            terminated = True
            info["curriculum_limit"] = True
        return obs, reward, terminated, truncated, info


def make_env(seed: int, max_ante: int = 8, rank: int = 0) -> callable:
    """Create a factory for BalatroEnv with curriculum wrapper."""
    def _init():
        env = BalatroEnv(seed=seed + rank)
        if max_ante < 100:
            env = CurriculumEnvWrapper(env, max_ante)
        env = Monitor(env)
        return env
    return _init


def create_vec_env(n_envs: int, seed: int, max_ante: int = 8, use_subproc: bool = False) -> gym.vector.VectorEnv:
    """Create vectorized Balatro environments."""
    env_fns = [make_env(seed, max_ante, i) for i in range(n_envs)]

    if use_subproc and n_envs > 1:
        try:
            return SubprocVecEnv(env_fns)
        except Exception:
            pass
    return DummyVecEnv(env_fns)


def train_ppo(config: TrainingConfig) -> tuple[PPO, Path]:
    """Run PPO training on Balatro.

    Args:
        config: Training configuration.

    Returns:
        Tuple of (trained_model, save_path).
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(config.model_dir) / f"ppo_{run_id}"
    save_path.mkdir(parents=True, exist_ok=True)

    log_path = Path(config.log_dir) / f"ppo_{run_id}"
    log_path.mkdir(parents=True, exist_ok=True)

    tb_path = Path(config.tensorboard_log) / f"ppo_{run_id}"
    tb_path.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(save_path / "config.json", "w") as f:
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
        json.dump(config_dict, f, indent=2)

    # Create environments
    max_ante = config.curriculum_max_ante if config.use_curriculum else 100
    env = create_vec_env(config.n_envs, config.seed, max_ante, use_subproc=False)

    # Eval environment
    # Eval environment (only used for final eval)
    eval_env = create_vec_env(1, config.seed + 1000, 100, use_subproc=False)

    # Build policy kwargs with feature extractor
    policy_kwargs = {
        "features_extractor_class": BalatroFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": config.features_dim},
        "net_arch": {
            "pi": config.net_arch_pi,
            "vf": config.net_arch_vf,
        },
    }

    # Create model (no tensorboard for now)
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        clip_range_vf=config.clip_range_vf,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        target_kl=config.target_kl,
        tensorboard_log=None,  # Disabled for speed
        device=config.device,
        verbose=config.verbose,
    )

    # Setup curriculum
    curriculum = CurriculumScheduler() if config.use_curriculum else None

    # Callbacks (lightweight)
    callbacks = []

    checkpoint_cb = CheckpointCallback(
        save_freq=max(config.checkpoint_freq, config.total_timesteps),
        save_path=str(save_path / "checkpoints"),
        name_prefix="ppo_balatro",
    )
    callbacks.append(checkpoint_cb)

    if config.verbose >= 1:
        metrics_cb = BalatroMetricsCallback(str(log_path), log_freq=config.log_freq)
        callbacks.append(metrics_cb)

    # Train
    print(f"\n{'='*60}")
    print(f"Starting PPO training: {run_id}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Environments: {config.n_envs}")
    print(f"Feature dim: {config.features_dim}")
    print(f"Device: {config.device}")
    print(f"Save path: {save_path}")
    print(f"{'='*60}\n")

    t_start = time.perf_counter()

    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList(callbacks),
            log_interval=10,
            progress_bar=config.progress_bar,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    elapsed = time.perf_counter() - t_start

    # Save final model
    final_model_path = save_path / "ppo_final"
    model.save(str(final_model_path))

    print(f"\nTraining complete: {elapsed:.0f}s ({config.total_timesteps:,} steps)")
    print(f"Model saved to: {final_model_path}")

    # Quick final eval
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"Final eval: {mean_reward:.1f} +/- {std_reward:.1f}")

    return model, save_path
