"""ScoreEngine - Balatro scoring with hand type levels and planet cards.

Hand types follow Balatro's 12-type system (including secret hands).
Each hand type has base chips/mult, leveled up by planet cards (+10 chips, +1 mult per level).
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Tuple, Callable, Optional


class HandType(IntEnum):
    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_KIND = 7
    STRAIGHT_FLUSH = 8
    FIVE_KIND = 9
    FLUSH_HOUSE = 10
    FLUSH_FIVE = 11

    @property
    def name(self) -> str:
        return {
            HandType.HIGH_CARD: "High Card",
            HandType.ONE_PAIR: "One Pair",
            HandType.TWO_PAIR: "Two Pair",
            HandType.THREE_KIND: "Three of a Kind",
            HandType.STRAIGHT: "Straight",
            HandType.FLUSH: "Flush",
            HandType.FULL_HOUSE: "Full House",
            HandType.FOUR_KIND: "Four of a Kind",
            HandType.STRAIGHT_FLUSH: "Straight Flush",
            HandType.FIVE_KIND: "Five of a Kind",
            HandType.FLUSH_HOUSE: "Flush House",
            HandType.FLUSH_FIVE: "Flush Five",
        }[self]


BASE_HAND_VALUES: Dict[HandType, Tuple[int, int]] = {
    HandType.HIGH_CARD:       (5, 1),
    HandType.ONE_PAIR:       (10, 2),
    HandType.TWO_PAIR:       (20, 2),
    HandType.THREE_KIND:     (30, 3),
    HandType.STRAIGHT:       (30, 4),
    HandType.FLUSH:          (35, 4),
    HandType.FULL_HOUSE:     (40, 4),
    HandType.FOUR_KIND:      (60, 7),
    HandType.STRAIGHT_FLUSH: (100, 8),
    HandType.FIVE_KIND:      (120, 12),
    HandType.FLUSH_HOUSE:    (140, 14),
    HandType.FLUSH_FIVE:     (160, 16),
}

PLANET_HAND_MAP: Dict[str, HandType] = {
    "Pluto":      HandType.HIGH_CARD,
    "Mercury":    HandType.ONE_PAIR,
    "Venus":      HandType.TWO_PAIR,
    "Earth":      HandType.THREE_KIND,
    "Mars":       HandType.STRAIGHT,
    "Jupiter":    HandType.FLUSH,
    "Saturn":     HandType.FULL_HOUSE,
    "Uranus":     HandType.FOUR_KIND,
    "Neptune":    HandType.STRAIGHT_FLUSH,
    "Planet X":   HandType.FIVE_KIND,
    "Ceres":      HandType.FLUSH_HOUSE,
    "Eris":       HandType.FLUSH_FIVE,
}

MAX_HAND_LEVEL = 15


class ScoreEngine:
    """Tracks hand type levels and computes base chips/mult for scoring."""

    def __init__(self):
        self.hand_levels: Dict[HandType, int] = {ht: 1 for ht in HandType}
        self.hand_play_counts: Dict[HandType, int] = {ht: 0 for ht in HandType}
        self.modifiers: List[Callable] = []

    def get_level(self, hand_type: HandType) -> int:
        return self.hand_levels.get(hand_type, 1)

    def set_level(self, hand_type: HandType, level: int):
        self.hand_levels[hand_type] = max(1, min(level, MAX_HAND_LEVEL))

    def apply_planet(self, hand_type: HandType):
        current = self.get_level(hand_type)
        self.hand_levels[hand_type] = min(current + 1, MAX_HAND_LEVEL)

    def apply_planet_by_name(self, planet_name: str) -> Optional[HandType]:
        hand_type = PLANET_HAND_MAP.get(planet_name)
        if hand_type is not None:
            self.apply_planet(hand_type)
        return hand_type

    def get_base_chips_mult(self, hand_type: HandType) -> Tuple[int, float]:
        base_chips, base_mult = BASE_HAND_VALUES.get(hand_type, (5, 1))
        level = self.get_level(hand_type)
        level_bonus = level - 1
        return base_chips + level_bonus * 10, base_mult + level_bonus

    def record_play(self, hand_type: HandType):
        self.hand_play_counts[hand_type] = self.hand_play_counts.get(hand_type, 0) + 1

    def get_play_count(self, hand_type: HandType) -> int:
        return self.hand_play_counts.get(hand_type, 0)

    def register_modifier(self, fn: Callable):
        self.modifiers.append(fn)

    def apply_modifiers(self, score: int, cards: List[int]) -> int:
        for fn in self.modifiers:
            score = fn(score, cards, self)
        return int(score)

    def reset(self):
        self.hand_levels = {ht: 1 for ht in HandType}
        self.hand_play_counts = {ht: 0 for ht in HandType}
        self.modifiers.clear()
