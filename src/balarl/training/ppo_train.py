"""PPO training pipeline for Balatro - core training loop with SB3."""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback,
)

from balarl.env.balatro_env import BalatroEnv
from balarl.training.feature_extractor import BalatroFeatureExtractor
from balarl.training.config import TrainingConfig


# ═══════════════════════════════════════════════════════════════
# Picklable environment factory (must be module-level for SubprocVecEnv)
# ═══════════════════════════════════════════════════════════════

IS_LINUX = platform.system() == "Linux"

class CurriculumEnvWrapper(gym.Wrapper):
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
        if hasattr(self.env, 'state') and self.env.state.ante > self._max_ante:
            terminated = True
            info["curriculum_limit"] = True
        return obs, reward, terminated, truncated, info


# Global state for env factory (set by train_ppo before creating vec env)
_ENV_SEED = 42
_ENV_MAX_ANTE = 8


def _env_factory(idx: int) -> gym.Env:
    """Module-level factory for SubprocVecEnv pickling."""
    env = BalatroEnv(seed=_ENV_SEED + idx)
    if _ENV_MAX_ANTE < 100:
        env = CurriculumEnvWrapper(env, _ENV_MAX_ANTE)
    env = Monitor(env)
    return env


def create_vec_env(n_envs: int, seed: int, max_ante: int = 8) -> gym.vector.VectorEnv:
    """Create vectorized environments, using SubprocVecEnv on Linux."""
    global _ENV_SEED, _ENV_MAX_ANTE
    _ENV_SEED = seed
    _ENV_MAX_ANTE = max_ante

    if IS_LINUX and n_envs > 1:
        # Use "fork" start method for speed on Linux
        try:
            import multiprocessing as mp
            ctx = mp.get_context("fork")
            return SubprocVecEnv(
                [_make_env_tuple(idx) for idx in range(n_envs)],
                context=ctx,
            )
        except Exception as e:
            print(f"  SubprocVecEnv failed ({e}), falling back to DummyVecEnv")

    # DummyVecEnv fallback
    return DummyVecEnv([_make_env_tuple(idx) for idx in range(n_envs)])


def _make_env_tuple(idx: int):
    """Returns (callable,) tuple for VecEnv construction."""
    return lambda: _env_factory(idx)


# ═══════════════════════════════════════════════════════════════
# Stdout progress callback (prints training progress periodically)
# ═══════════════════════════════════════════════════════════════

class ProgressCallback(BaseCallback):
    """Print training metrics to stdout every N steps."""

    def __init__(self, print_freq: int = 2000, n_envs: int = 1):
        super().__init__()
        self.print_freq = print_freq
        self._n_envs = n_envs
        self._last_print = 0
        # Per-env accumulators
        self._cumrew = np.zeros(n_envs, dtype=np.float64)
        self._eplen = np.zeros(n_envs, dtype=np.int32)
        self._cur_ante = np.ones(n_envs, dtype=np.int32)  # current ante per env
        # Episode-level history
        self._ep_rewards: List[float] = []
        self._ep_antes: List[int] = []
        self._ep_lengths: List[int] = []

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals["rewards"])
        dones = np.asarray(self.locals["dones"])
        new_obs = self.locals.get("new_obs", {})
        n = min(len(dones), self._n_envs)

        for i in range(n):
            self._cumrew[i] += float(rewards[i])
            self._eplen[i] += 1
            if dones[i]:
                # Capture episode data using cached ante (from previous step's new_obs)
                self._ep_rewards.append(float(self._cumrew[i]))
                self._ep_lengths.append(int(self._eplen[i]))
                self._ep_antes.append(int(self._cur_ante[i]))
                self._cumrew[i] = 0.0
                self._eplen[i] = 0

        # Update cached ante from latest observation (after processing dones,
        # so the done-step's reset ante doesn't overwrite the captured value)
        if isinstance(new_obs, dict) and "ante" in new_obs:
            a = new_obs["ante"]
            for i in range(n):
                self._cur_ante[i] = int(a[i]) if a.ndim >= 1 else int(a)

        # Print
        if self.num_timesteps - self._last_print >= self.print_freq:
            self._last_print = self.num_timesteps
            elapsed = time.perf_counter() - getattr(self.model, "_start_time", time.perf_counter())
            fps = int(self.num_timesteps / max(1, elapsed))
            n_recent = min(20, len(self._ep_rewards))
            avg_rew = np.mean(self._ep_rewards[-n_recent:]) if n_recent else 0.0
            avg_ante = np.mean(self._ep_antes[-n_recent:]) if n_recent else 0.0
            avg_len = np.mean(self._ep_lengths[-n_recent:]) if n_recent else 0.0
            print(
                f"  [{self.num_timesteps:>10,} steps]  "
                f"fps={fps:>6}  "
                f"eps={len(self._ep_rewards):>6}  "
                f"rew={avg_rew:>8.1f}  "
                f"ante={avg_ante:>5.1f}  "
                f"len={avg_len:>5.0f}",
                flush=True,
            )
        return True


# ═══════════════════════════════════════════════════════════════
# Main training function
# ═══════════════════════════════════════════════════════════════

def train_ppo(config: TrainingConfig) -> tuple[PPO, Path]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(config.model_dir) / f"ppo_{run_id}"
    save_path.mkdir(parents=True, exist_ok=True)

    log_path = Path(config.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(save_path / "config.json", "w") as f:
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
        json.dump(config_dict, f, indent=2)

    # Create environments
    max_ante = config.curriculum_max_ante if config.use_curriculum else 100
    print(f"  Creating {config.n_envs} envs (SubprocVecEnv={IS_LINUX})...", flush=True)
    env = create_vec_env(config.n_envs, config.seed, max_ante)
    print(f"  Env type: {type(env).__name__}", flush=True)

    # Policy
    policy_kwargs = {
        "features_extractor_class": BalatroFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": config.features_dim},
        "net_arch": {
            "pi": config.net_arch_pi,
            "vf": config.net_arch_vf,
        },
    }

    # Model
    print(f"  Creating PPO model (device={config.device})...", flush=True)
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
        tensorboard_log=None,
        device=config.device,
        verbose=0,  # We handle our own logging
    )

    # Callbacks
    callbacks = [
        ProgressCallback(print_freq=config.log_freq, n_envs=config.n_envs),
        CheckpointCallback(
            save_freq=config.checkpoint_freq,
            save_path=str(save_path / "checkpoints"),
            name_prefix="ppo_balatro",
        ),
    ]

    # Print header
    print(f"\n{'='*55}")
    print(f"  BalaRL PPO Training")
    print(f"  Run:      {run_id}")
    print(f"  Steps:    {config.total_timesteps:,}")
    print(f"  Envs:     {config.n_envs} ({type(env).__name__})")
    print(f"  n_steps:  {config.n_steps}")
    print(f"  Batch:    {config.batch_size} x {config.n_epochs} epochs")
    print(f"  Device:   {config.device}")
    print(f"  Dim:      {config.features_dim}")
    print(f"  Save:     {save_path}")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    if torch.cuda.is_available():
        print(f"  GPU:      {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory // 2**30}GB)")
    print(f"{'='*55}\n")

    t_start = time.perf_counter()
    model._start_time = t_start

    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList(callbacks),
            log_interval=200,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n  Training interrupted by user", flush=True)

    elapsed = time.perf_counter() - t_start

    # Save final model
    final_model_path = save_path / "ppo_final"
    model.save(str(final_model_path))

    print(f"\n  Done: {elapsed:.0f}s ({config.total_timesteps:,} steps)")
    print(f"  Model: {final_model_path}", flush=True)

    return model, save_path
