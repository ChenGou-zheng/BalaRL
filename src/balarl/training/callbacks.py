"""Training callbacks for Balatro PPO - metrics logging, checkpointing, eval."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class BalatroMetricsCallback(BaseCallback):
    """Logs Balatro-specific metrics during training."""

    def __init__(self, log_dir: str, log_freq: int = 1000):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_freq = log_freq
        self.episode_rewards: List[float] = []
        self.episode_antes: List[int] = []
        self.episode_scores: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_wins: List[bool] = []
        self._episode_reward = 0.0
        self._episode_start_step = 0

    def _on_step(self) -> bool:
        self._episode_reward += self.locals["rewards"][0] if isinstance(self.locals["rewards"], np.ndarray) else float(self.locals["rewards"])

        for i, done in enumerate(self.locals["dones"] if isinstance(self.locals["dones"], (list, np.ndarray)) else [self.locals["dones"]]):
            if done:
                infos = self.locals.get("infos", [])
                info = infos[i] if isinstance(infos, (list, np.ndarray)) and i < len(infos) else {}

                self.episode_rewards.append(float(self._episode_reward))
                self.episode_antes.append(int(info.get("ante", 1)))
                self.episode_scores.append(float(info.get("final_score", 0)))
                self.episode_lengths.append(self.num_timesteps - self._episode_start_step)
                self.episode_wins.append(bool(info.get("won", False)))
                self._episode_reward = 0.0
                self._episode_start_step = self.num_timesteps

        if self.num_timesteps > 0 and self.num_timesteps % self.log_freq == 0:
            self._log_metrics()

        return True

    def _log_metrics(self):
        if not self.episode_rewards:
            return
        n = max(1, len(self.episode_rewards))
        self.logger.record("balatro/mean_reward", np.mean(self.episode_rewards[-n:]))
        self.logger.record("balatro/mean_ante", np.mean(self.episode_antes[-n:]))
        self.logger.record("balatro/mean_ep_length", np.mean(self.episode_lengths[-n:]))
        if self.episode_wins:
            self.logger.record("balatro/win_rate", np.mean(self.episode_wins[-n:]))


class CurriculumCallback(BaseCallback):
    """Updates environment curriculum level based on performance."""

    def __init__(self, curriculum, env, eval_freq: int = 5000):
        super().__init__()
        self.curriculum = curriculum
        self.env = env
        self.eval_freq = eval_freq
        self._episode_results: List[int] = []

    def _on_step(self) -> bool:
        for done in (self.locals.get("dones", []) if isinstance(self.locals.get("dones"), (list, np.ndarray)) else [self.locals.get("dones")]):
            if done:
                info = self.locals.get("infos", [])
                ante = info[0].get("ante", 1) if isinstance(info, (list, np.ndarray)) and len(info) > 0 else 1
                self._episode_results.append(int(ante))

        if self.num_timesteps > 0 and self.num_timesteps % self.eval_freq == 0 and self._episode_results:
            avg_ante = np.mean(self._episode_results[-20:]) if len(self._episode_results) >= 20 else np.mean(self._episode_results)
            self.curriculum.record_episode(int(avg_ante), died=False)
            self._episode_results = []
            old_max = getattr(self.env, "max_ante", 0)
            new_max = self.curriculum.max_ante
            if hasattr(self.env, "set_max_ante") and new_max != old_max:
                self.env.set_max_ante(new_max)
                self.logger.record("balatro/curriculum_max_ante", new_max)

        return True


class TrainingLogger:
    """Logs training progress to JSON for later analysis."""

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: List[Dict] = []

    def log(self, timesteps: int, metrics: Dict):
        metrics["timesteps"] = timesteps
        self.entries.append(metrics)

    def save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.entries, f, indent=2)

    def flush(self):
        self.save()
        self.entries = []
