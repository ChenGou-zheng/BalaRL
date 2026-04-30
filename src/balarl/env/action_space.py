"""Action space definitions for the Balatro Gymnasium environment.

Hierarchical action space:
  Phase 0 (PLAY):  Select cards (0-7) + Play Hand (8) + Discard (9) + Use Consumable (10-11)
  Phase 1 (SHOP):  Skip (0) + Reroll (1) + Buy items (2-11) + Sell Joker (12-16)
  Phase 2 (BLIND_SELECT):  Select Small (0) or Big (1) Blind (Boss (2) only when applicable)
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Tuple


class Phase(IntEnum):
    PLAY = 0
    SHOP = 1
    BLIND_SELECT = 2


# ─── PLAY phase actions ─────────────────────────────────────────────

NUM_CARD_SLOTS = 8
NUM_CONSUMABLE_SLOTS = 2

SELECT_CARD_BASE = 0
SELECT_CARD_COUNT = NUM_CARD_SLOTS

PLAY_HAND = SELECT_CARD_BASE + SELECT_CARD_COUNT  # 8
DISCARD = PLAY_HAND + 1                           # 9

USE_CONSUMABLE_BASE = DISCARD + 1                 # 10
USE_CONSUMABLE_COUNT = NUM_CONSUMABLE_SLOTS

PLAY_ACTIONS = SELECT_CARD_COUNT + 1 + 1 + USE_CONSUMABLE_COUNT  # 12

# ─── SHOP phase actions ─────────────────────────────────────────────

SHOP_SKIP = 0
SHOP_REROLL = 1
SHOP_BUY_BASE = 2
SHOP_BUY_COUNT = 10   # up to 10 shop items
SHOP_SELL_BASE = SHOP_BUY_BASE + SHOP_BUY_COUNT  # 12
SHOP_SELL_COUNT = 5   # up to 5 jokers to sell

SHOP_ACTIONS = 1 + 1 + SHOP_BUY_COUNT + SHOP_SELL_COUNT  # 17

# ─── BLIND SELECT actions ───────────────────────────────────────────

SELECT_SMALL_BLIND = 0
SELECT_BIG_BLIND = 1
SELECT_BOSS_BLIND = 2

BLIND_ACTIONS = 3

# ─── Action encoding helpers ────────────────────────────────────────


def encode_action(phase: Phase, action_dict: Dict[str, int]) -> int:
    """Encode a phase-specific action into a unified action ID."""
    if phase == Phase.PLAY:
        if action_dict["type"] == "select_card":
            return SELECT_CARD_BASE + action_dict["card_idx"]
        elif action_dict["type"] == "play":
            return PLAY_HAND
        elif action_dict["type"] == "discard":
            return DISCARD
        elif action_dict["type"] == "consumable":
            return USE_CONSUMABLE_BASE + action_dict["consumable_idx"]
    elif phase == Phase.SHOP:
        if action_dict["type"] == "skip":
            return SHOP_SKIP
        elif action_dict["type"] == "reroll":
            return SHOP_REROLL
        elif action_dict["type"] == "buy":
            return SHOP_BUY_BASE + action_dict["item_idx"]
        elif action_dict["type"] == "sell":
            return SHOP_SELL_BASE + action_dict["joker_idx"]
    elif phase == Phase.BLIND_SELECT:
        if action_dict["type"] == "small":
            return SELECT_SMALL_BLIND
        elif action_dict["type"] == "big":
            return SELECT_BIG_BLIND
        elif action_dict["type"] == "boss":
            return SELECT_BOSS_BLIND
    return 0


def decode_action(phase: Phase, action_id: int) -> Dict[str, int]:
    """Decode a unified action ID into a phase-specific action dict."""
    if phase == Phase.PLAY:
        if SELECT_CARD_BASE <= action_id < SELECT_CARD_BASE + SELECT_CARD_COUNT:
            return {"type": "select_card", "card_idx": action_id - SELECT_CARD_BASE}
        elif action_id == PLAY_HAND:
            return {"type": "play"}
        elif action_id == DISCARD:
            return {"type": "discard"}
        elif USE_CONSUMABLE_BASE <= action_id < USE_CONSUMABLE_BASE + USE_CONSUMABLE_COUNT:
            return {"type": "consumable", "consumable_idx": action_id - USE_CONSUMABLE_BASE}
    elif phase == Phase.SHOP:
        if action_id == SHOP_SKIP:
            return {"type": "skip"}
        elif action_id == SHOP_REROLL:
            return {"type": "reroll"}
        elif SHOP_BUY_BASE <= action_id < SHOP_BUY_BASE + SHOP_BUY_COUNT:
            return {"type": "buy", "item_idx": action_id - SHOP_BUY_BASE}
        elif SHOP_SELL_BASE <= action_id < SHOP_SELL_BASE + SHOP_SELL_COUNT:
            return {"type": "sell", "joker_idx": action_id - SHOP_SELL_BASE}
    elif phase == Phase.BLIND_SELECT:
        if action_id == SELECT_SMALL_BLIND:
            return {"type": "small"}
        elif action_id == SELECT_BIG_BLIND:
            return {"type": "big"}
        elif action_id == SELECT_BOSS_BLIND:
            return {"type": "boss"}
    return {"type": "unknown"}


def get_legal_actions(phase: Phase, game_state: dict) -> List[int]:
    """Get list of legal action IDs for the current phase."""
    legal = []

    if phase == Phase.PLAY:
        hand_size = game_state.get("hand_size", 8)
        for i in range(hand_size):
            legal.append(SELECT_CARD_BASE + i)
        if game_state.get("selected_cards", []):
            legal.append(PLAY_HAND)
        if game_state.get("discards_left", 0) > 0:
            legal.append(DISCARD)
        for i in range(min(2, len(game_state.get("consumables", [])))):
            legal.append(USE_CONSUMABLE_BASE + i)

    elif phase == Phase.SHOP:
        legal.append(SHOP_SKIP)
        legal.append(SHOP_REROLL)
        for i in range(min(10, len(game_state.get("shop_items", [])))):
            legal.append(SHOP_BUY_BASE + i)
        for i in range(len(game_state.get("joker_ids", []))):
            legal.append(SHOP_SELL_BASE + i)

    elif phase == Phase.BLIND_SELECT:
        legal.append(SELECT_SMALL_BLIND)
        legal.append(SELECT_BIG_BLIND)
        if game_state.get("round", 1) >= 3:
            legal.append(SELECT_BOSS_BLIND)

    return legal


def action_space_size() -> int:
    return max(PLAY_ACTIONS, SHOP_ACTIONS, BLIND_ACTIONS)
