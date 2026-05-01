"""Reward shaping for the Balatro environment.

Provides reward signals that guide RL learning:
  - Progress reward: advancing toward blind target
  - Milestone bonuses: crossing 25/50/75/100% thresholds
  - Hand quality reward: stronger poker hands get higher base reward
  - Efficiency reward: using fewer cards / conserving hands
  - Economy reward: earning interest, efficient spending
  - Win/loss terminal rewards
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


class RewardShaper:
    """Calculates shaped rewards for Balatro gameplay."""

    def __init__(self):
        self.episode_total = 0.0

    def reset(self):
        self.episode_total = 0.0

    def hand_reward(
        self,
        score: int,
        old_progress: float,
        new_progress: float,
        hand_type_name: str,
        cards_played: int,
        hands_left: int,
        ante: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Reward for playing a hand."""
        breakdown: Dict[str, float] = {}
        reward = 0.0

        # 1. Progress reward (main signal)
        progress_reward = 15.0 * new_progress
        breakdown["progress"] = progress_reward
        reward += progress_reward

        # 2. Milestone bonuses
        milestone = 0.0
        thresholds = [(0.25, 5.0), (0.5, 10.0), (0.75, 15.0), (1.0, 25.0)]
        for thresh, bonus in thresholds:
            if old_progress < thresh <= new_progress:
                milestone += bonus
        breakdown["milestone"] = milestone
        reward += milestone

        # 3. Score magnitude reward
        if ante <= 3:
            score_reward = min(10.0, score / 100.0)
        else:
            score_reward = min(10.0, 3.0 * np.log10(max(1, score)))
        breakdown["score"] = score_reward
        reward += score_reward

        # 4. Hand quality
        quality_values = {
            "High Card": 0.1, "One Pair": 0.5, "Two Pair": 1.0,
            "Three of a Kind": 2.0, "Straight": 2.5, "Flush": 2.5,
            "Full House": 3.5, "Four of a Kind": 5.0,
            "Straight Flush": 7.0, "Five of a Kind": 10.0,
            "Flush House": 12.0, "Flush Five": 15.0,
        }
        quality = quality_values.get(hand_type_name, 0.0)
        breakdown["hand_quality"] = quality
        reward += quality * 2.0

        # 5. Efficiency (fewer cards, saving hands)
        efficiency = 0.0
        if quality > 1.0 and cards_played <= 3:
            efficiency = 2.0
        elif hands_left <= 2 and cards_played <= 4:
            efficiency = 1.5
        breakdown["efficiency"] = efficiency
        reward += efficiency * 1.5

        # 6. Strategic play
        strategy = 0.0
        if new_progress > 0.7 and hands_left >= 3:
            strategy = 2.0
        elif new_progress < 0.3 and quality >= 2.5:
            strategy = 3.0
        breakdown["strategy"] = strategy
        reward += strategy * 2.0

        self.episode_total += reward
        return min(reward, 100.0), breakdown

    def discard_reward(self, discarded_count: int, old_progress: float, discards_left: int) -> float:
        reward = 0.1
        if old_progress < 0.5 and discards_left > 1:
            reward += 0.3
        elif old_progress > 0.8:
            reward -= 0.2
        self.episode_total += reward
        return reward

    def blind_clear_reward(self, ante: int) -> float:
        return min(50.0, 20.0 + 10.0 * ante)

    def blind_fail_penalty(self, progress: float) -> float:
        return -20.0 * (1.0 - progress)

    def ante_termination_bonus(self, final_ante: int) -> float:
        return 10.0 * final_ante

    def shop_buy_reward(self, item_type: str) -> float:
        rewards = {"JOKER": 2.0, "PACK": 1.0, "CARD": 0.5, "VOUCHER": 3.0}
        return rewards.get(item_type, 1.0)

    def invalid_action_penalty(self) -> float:
        return -1.0
