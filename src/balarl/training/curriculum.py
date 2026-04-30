"""Curriculum learning scheduler for Balatro PPO training.

Progressively increases difficulty by raising the max ante the agent
must survive, starting from low antes and gradually increasing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CurriculumStage:
    max_ante: int
    success_rate: float  # win rate needed to advance
    min_episodes: int     # minimum episodes at this stage


DEFAULT_CURRICULUM = [
    CurriculumStage(max_ante=2, success_rate=0.6, min_episodes=50),
    CurriculumStage(max_ante=3, success_rate=0.5, min_episodes=50),
    CurriculumStage(max_ante=4, success_rate=0.4, min_episodes=50),
    CurriculumStage(max_ante=5, success_rate=0.35, min_episodes=50),
    CurriculumStage(max_ante=6, success_rate=0.3, min_episodes=50),
    CurriculumStage(max_ante=7, success_rate=0.25, min_episodes=50),
    CurriculumStage(max_ante=8, success_rate=0.2, min_episodes=50),
]


class CurriculumScheduler:
    """Manages progressive difficulty in training."""

    def __init__(self, stages: List[CurriculumStage] | None = None):
        self.stages = stages or DEFAULT_CURRICULUM
        self.current_stage = 0
        self.episodes_at_stage = 0
        self.recent_results: List[bool] = []  # True = survived, False = died

    @property
    def max_ante(self) -> int:
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage].max_ante
        return 100  # Beyond curriculum = full game

    @property
    def current_stage_info(self) -> CurriculumStage | None:
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage]
        return None

    def record_episode(self, reached_ante: int, died: bool):
        """Record the result of an episode."""
        survived = reached_ante >= self.max_ante and not died
        self.recent_results.append(survived)
        self.episodes_at_stage += 1

        # Check stage advancement
        stage = self.current_stage_info
        if stage is None:
            return

        if self.episodes_at_stage < stage.min_episodes:
            return

        # Use last N results for rate calculation
        window = min(self.episodes_at_stage, 100)
        recent = self.recent_results[-window:]
        if not recent:
            return

        success_rate = sum(recent) / len(recent)
        if success_rate >= stage.success_rate:
            self.current_stage += 1
            self.episodes_at_stage = 0
            self.recent_results = []

    def to_dict(self) -> dict:
        return {
            "current_stage": self.current_stage,
            "max_ante": self.max_ante,
            "episodes_at_stage": self.episodes_at_stage,
        }
