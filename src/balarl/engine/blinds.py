"""Blind chip scaling - target scores for each ante/round."""

from __future__ import annotations

from typing import Dict

BLIND_CHIPS: Dict[int, Dict[str, int]] = {
    1:  {"Small Blind": 300,  "Big Blind": 450,  "Boss Blind": 600},
    2:  {"Small Blind": 450,  "Big Blind": 675,  "Boss Blind": 900},
    3:  {"Small Blind": 600,  "Big Blind": 900,  "Boss Blind": 1200},
    4:  {"Small Blind": 900,  "Big Blind": 1350, "Boss Blind": 1800},
    5:  {"Small Blind": 1350, "Big Blind": 2025, "Boss Blind": 2700},
    6:  {"Small Blind": 2100, "Big Blind": 3150, "Boss Blind": 4200},
    7:  {"Small Blind": 3300, "Big Blind": 4950, "Boss Blind": 6600},
    8:  {"Small Blind": 5250, "Big Blind": 7875, "Boss Blind": 10500},
}

BLIND_REWARDS: Dict[str, int] = {
    "Small Blind": 2,
    "Big Blind": 4,
    "Boss Blind": 6,
}

def get_blind_chips(ante: int, blind_type: str) -> int:
    if ante <= 8:
        return BLIND_CHIPS.get(ante, {}).get(blind_type, 300)
    base = BLIND_CHIPS[8].get(blind_type, 5250)
    return int(base * (1.5 ** (ante - 8)))

def get_blind_reward(ant: int, blind_type: str) -> int:
    base = BLIND_REWARDS.get(blind_type, 3)
    return base + ant

def get_interest_cap() -> int:
    return 5
