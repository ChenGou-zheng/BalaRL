"""Observation space builder for the Balatro environment.

Produces a Dict observation space with:
  - hand: card IDs for 8 hand slots
  - hand_one_hot: (8, 52) one-hot encoding
  - hand_ranks, hand_suits: per-card features
  - joker_ids: 10 joker slots with IDs
  - joker_count, joker_slots
  - money, ante, round, hands_left, discards_left, hand_size
  - chips_scored, chips_needed, progress_ratio
  - selected_cards: 8-bit mask
  - rank_counts: 13-rank frequency in hand
  - suit_counts: 4-suit frequency in hand
  - straight_potential, flush_potential: hand quality indicators
  - hand_levels: 12 hand type levels
  - phase: current game phase
  - action_mask: legal action mask
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from gymnasium import spaces

from balarl.engine.cards import Card, Suit, Rank
from balarl.env.action_space import (
    Phase, PLAY_ACTIONS, SHOP_ACTIONS, BLIND_ACTIONS,
    action_space_size as total_actions,
)


def create_observation_space() -> spaces.Dict:
    return spaces.Dict({
        # Hand cards
        "hand": spaces.Box(-1, 51, (8,), dtype=np.int32),
        "hand_one_hot": spaces.Box(0, 1, (8, 52), dtype=np.float32),
        "hand_ranks": spaces.Box(0, 14, (8,), dtype=np.int32),
        "hand_suits": spaces.Box(0, 3, (8,), dtype=np.int32),

        # Selection mask
        "selected_cards": spaces.MultiBinary(8),

        # Game state scalars
        "money": spaces.Box(-20, 9999, (), dtype=np.int32),
        "ante": spaces.Box(1, 1000, (), dtype=np.int32),
        "round": spaces.Box(1, 3, (), dtype=np.int32),
        "hands_left": spaces.Box(0, 12, (), dtype=np.int32),
        "discards_left": spaces.Box(0, 10, (), dtype=np.int32),
        "hand_size": spaces.Box(5, 12, (), dtype=np.int32),

        # Scoring
        "chips_scored": spaces.Box(0, 1_000_000_000, (), dtype=np.int64),
        "chips_needed": spaces.Box(0, 1_000_000_000, (), dtype=np.int64),
        "progress_ratio": spaces.Box(0.0, 2.0, (), dtype=np.float32),

        # Jokers
        "joker_ids": spaces.Box(0, 200, (10,), dtype=np.int32),
        "joker_count": spaces.Box(0, 10, (), dtype=np.int32),
        "joker_slots": spaces.Box(0, 10, (), dtype=np.int32),

        # Hand analysis
        "rank_counts": spaces.Box(0, 4, (13,), dtype=np.int32),
        "suit_counts": spaces.Box(0, 8, (4,), dtype=np.int32),
        "straight_potential": spaces.Box(0.0, 1.0, (), dtype=np.float32),
        "flush_potential": spaces.Box(0.0, 1.0, (), dtype=np.float32),

        # Hand levels
        "hand_levels": spaces.Box(0, 50, (12,), dtype=np.int32),

        # Consumables
        "consumable_count": spaces.Box(0, 5, (), dtype=np.int32),
        "consumables": spaces.Box(0, 100, (5,), dtype=np.int32),

        # Shop
        "shop_items": spaces.Box(0, 300, (10,), dtype=np.int32),
        "shop_costs": spaces.Box(0, 5000, (10,), dtype=np.int32),
        "shop_reroll_cost": spaces.Box(0, 9999, (), dtype=np.int32),

        # Boss blind
        "boss_blind_active": spaces.Box(0, 1, (), dtype=np.int32),
        "boss_blind_type": spaces.Box(0, 30, (), dtype=np.int32),
        "face_down_cards": spaces.MultiBinary(8),

        # Phase
        "phase": spaces.Box(0, 2, (), dtype=np.int32),
        "action_mask": spaces.MultiBinary(total_actions()),
    })


HAND_TYPE_ORDER = [
    "High Card", "One Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind",
    "Straight Flush", "Five of a Kind", "Flush House", "Flush Five",
]

CONSUMABLE_ID_MAP: Dict[str, int] = {}
CONSUMABLE_LIST: List[str] = []


def _init_consumable_ids():
    global CONSUMABLE_ID_MAP, CONSUMABLE_LIST
    if CONSUMABLE_ID_MAP:
        return
    from balarl.engine.consumables import TAROT_NAMES, PLANET_NAMES, SPECTRAL_NAMES
    idx = 1
    for name in TAROT_NAMES:
        CONSUMABLE_ID_MAP[name] = idx
        CONSUMABLE_LIST.append(name)
        idx += 1
    for name in PLANET_NAMES:
        CONSUMABLE_ID_MAP[name] = idx
        CONSUMABLE_LIST.append(name)
        idx += 1
    for name in SPECTRAL_NAMES:
        CONSUMABLE_ID_MAP[name] = idx
        CONSUMABLE_LIST.append(name)
        idx += 1


def consumable_to_id(name: str) -> int:
    _init_consumable_ids()
    return CONSUMABLE_ID_MAP.get(name, 0)


def id_to_consumable(idx: int) -> str:
    _init_consumable_ids()
    if 0 <= idx < len(CONSUMABLE_LIST):
        return CONSUMABLE_LIST[idx]
    return ""


def build_hand_features(hand: List[Card]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rank_counts = np.zeros(13, dtype=np.int32)
    suit_counts = np.zeros(4, dtype=np.int32)

    for card in hand:
        if card:
            rank_counts[card.rank.value - 2] += 1
            suit_counts[card.suit.value] += 1

    sorted_ranks = sorted(set(c.rank.value for c in hand if c))
    consecutive = 0
    max_consecutive = 0
    for i in range(1, len(sorted_ranks)):
        if sorted_ranks[i] - sorted_ranks[i - 1] <= 2:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    straight_pot = min(1.0, max_consecutive / 4.0) if len(hand) >= 5 else 0.0

    max_suit = suit_counts.max()
    flush_pot = min(1.0, max_suit / 5.0) if len(hand) >= 5 else 0.0

    return rank_counts, suit_counts, np.float32(straight_pot), np.float32(flush_pot)
