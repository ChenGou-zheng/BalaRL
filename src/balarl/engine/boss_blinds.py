"""Boss Blinds - special blind effects that modify gameplay rules."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple


BOSS_BLIND_DB: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "The Hook",
        "effect": "Discard 2 random cards from your hand after playing a hand",
        "type": "discard_random",
    },
    2: {
        "name": "The Mouth",
        "effect": "Play only 1 hand type this round",
        "type": "restrict_hand_types",
    },
    3: {
        "name": "The Eye",
        "effect": "No repeat hand types. Each hand must be different.",
        "type": "no_repeat_hands",
    },
    4: {
        "name": "The Tooth",
        "effect": "-$1 per card played. Lose $1 for each card used in scoring.",
        "type": "money_per_card_played",
    },
    5: {
        "name": "The Fish",
        "effect": "Cards drawn face down until a flush is played",
        "type": "face_down_until_flush",
    },
    6: {
        "name": "The Plant",
        "effect": "All face cards are debuffed. Cannot score face cards.",
        "type": "debuff_face_cards",
    },
    7: {
        "name": "The Wall",
        "effect": "Required score is doubled. Extra large blind.",
        "type": "double_chips_needed",
    },
    8: {
        "name": "The Mark",
        "effect": "All Spade cards are debuffed.",
        "type": "debuff_suit_spades",
    },
    9: {
        "name": "The Wheel",
        "effect": "1 in 7 cards drawn face down",
        "type": "random_face_down",
    },
    10: {
        "name": "The Arm",
        "effect": "Level of played poker hand decreases by 1",
        "type": "hand_level_decrease",
    },
    11: {
        "name": "The Water",
        "effect": "Start with 0 discards",
        "type": "no_discards",
    },
    12: {
        "name": "The Flint",
        "effect": "Base Chips and Mult are halved. Chips and mult are halved.",
        "type": "half_base_chips_mult",
    },
    13: {
        "name": "The Club",
        "effect": "All Club cards are debuffed.",
        "type": "debuff_suit_clubs",
    },
    14: {
        "name": "The Ox",
        "effect": "Playing a most played hand sets money to $0",
        "type": "most_played_resets_money",
    },
    15: {
        "name": "The Pillar",
        "effect": "Cards played this ante score as 0 until a hand is won without them",
        "type": "debuff_played_this_ante",
    },
    16: {
        "name": "The Needle",
        "effect": "Play only 1 hand. You only get 1 hand to beat the blind.",
        "type": "one_hand",
    },
    17: {
        "name": "The Head",
        "effect": "All Heart cards are debuffed.",
        "type": "debuff_suit_hearts",
    },
    18: {
        "name": "The Window",
        "effect": "All Diamond cards are debuffed.",
        "type": "debuff_suit_diamonds",
    },
    19: {
        "name": "The Goad",
        "effect": "All Spade cards are debuffed (same as The Mark).",
        "type": "debuff_suit_spades",
    },
    20: {
        "name": "The Manacle",
        "effect": "-1 hand size",
        "type": "hand_size_reduction",
    },
    21: {
        "name": "The Psychic",
        "effect": "Must play exactly 5 cards. Cannot play fewer than 5.",
        "type": "must_play_five",
    },
    22: {
        "name": "The Serpent",
        "effect": "After playing a hand, always draw 3 cards",
        "type": "force_draw_three",
    },
    23: {
        "name": "The House",
        "effect": "First hand of each round is drawn face down",
        "type": "first_hand_face_down",
    },
    24: {
        "name": "The Amber Acorn",
        "effect": "All jokers are shuffled and flipped face down",
        "type": "shuffle_jokers",
    },
    25: {
        "name": "The Conductor",
        "effect": "Clubs and Spades are debuffed",
        "type": "debuff_suits_clubs_spades",
    },
    26: {
        "name": "The Chalice",
        "effect": "Red seals are debuffed. Red seal cards do not retrigger.",
        "type": "debuff_red_seals",
    },
    27: {
        "name": "Crimson Heart",
        "effect": "One random joker is debuffed each round",
        "type": "debuff_random_joker",
    },
    28: {
        "name": "Verdant Leaf",
        "effect": "All cards debuffed until 1 joker is sold",
        "type": "debuff_until_joker_sold",
    },
    29: {
        "name": "Amber Acorn",
        "effect": "Flips and shuffles all joker cards",
        "type": "shuffle_jokers",
    },
}

BOSS_NAMES = [v["name"] for v in BOSS_BLIND_DB.values()]
BOSS_BY_NAME = {v["name"]: k for k, v in BOSS_BLIND_DB.items()}


class BossBlindManager:
    """Manages boss blind selection and effects."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.seen: List[str] = []
        self.active_blind: Optional[str] = None

    def select_boss(self, ante: int) -> str:
        available = [v["name"] for v in BOSS_BLIND_DB.values() if v["name"] not in self.seen]
        if not available:
            available = BOSS_NAMES
            self.seen = []
        chosen = self.rng.choice(available)
        self.seen.append(chosen)
        self.active_blind = chosen
        return chosen

    def get_effect(self, name: str) -> str:
        return BOSS_BLIND_DB.get(BOSS_BY_NAME.get(name, 0), {}).get("effect", "")

    def get_type(self, name: str) -> str:
        return BOSS_BLIND_DB.get(BOSS_BY_NAME.get(name, 0), {}).get("type", "")

    def on_round_start(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.active_blind:
            return {}

        boss_type = self.get_type(self.active_blind)
        effects: Dict[str, Any] = {}

        if boss_type == "no_discards":
            game_state["discards_left"] = 0
            effects["discards_set_to_zero"] = True
        elif boss_type == "one_hand":
            game_state["hands_left"] = 1
            effects["hands_set_to_one"] = True
        elif boss_type == "hand_size_reduction":
            game_state["hand_size"] = max(5, game_state.get("hand_size", 8) - 1)
            effects["hand_size_reduced"] = True
        elif boss_type == "double_chips_needed":
            game_state["chips_needed"] = game_state.get("chips_needed", 300) * 2
            effects["chips_doubled"] = True

        return effects

    def on_hand_drawn(self, cards: List[Any], game_state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.active_blind:
            return {}

        boss_type = self.get_type(self.active_blind)
        effects: Dict[str, Any] = {}

        if boss_type == "first_hand_face_down" and game_state.get("hands_played_this_round", 0) == 0:
            effects["face_down_cards"] = list(range(len(cards)))
        elif boss_type == "random_face_down":
            face_down = [i for i in range(len(cards)) if self.rng.random() < 1 / 7]
            if face_down:
                effects["face_down_cards"] = face_down
        elif boss_type == "force_draw_three":
            effects["force_draw"] = 3

        return effects

    def on_hand_played(self, cards: List[Any], hand_type: str, game_state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.active_blind:
            return {}

        boss_type = self.get_type(self.active_blind)
        effects: Dict[str, Any] = {}

        if boss_type == "discard_random":
            n = min(2, len(game_state.get("hand_indexes", [])))
            if n > 0:
                indices = list(range(len(game_state.get("hand_indexes", []))))
                discarded = self.rng.sample(indices, k=n)
                effects["discarded_cards"] = sorted(discarded, reverse=True)
        elif boss_type == "money_per_card_played":
            cost = len(cards)
            game_state["money"] = max(0, game_state.get("money", 0) - cost)
            effects["money_lost"] = cost
        elif boss_type == "hand_level_decrease":
            effects["level_decrease"] = True

        return effects

    def modify_scoring(self, base_chips: float, base_mult: float, cards: List[Any], hand_type: str) -> Tuple[float, float]:
        if not self.active_blind:
            return base_chips, base_mult

        boss_type = self.get_type(self.active_blind)
        chips, mult = base_chips, base_mult

        if boss_type == "half_base_chips_mult":
            chips /= 2
            mult /= 2

        return chips, mult

    def is_card_debuffed(self, card: Any, game_state: Dict[str, Any]) -> bool:
        if not self.active_blind:
            return False

        boss_type = self.get_type(self.active_blind)

        if hasattr(card, 'suit'):
            suit_name = card.suit.name if hasattr(card.suit, 'name') else str(card.suit)
            suit_map = {
                "debuff_suit_spades": "Spades",
                "debuff_suit_hearts": "Hearts",
                "debuff_suit_clubs": "Clubs",
                "debuff_suit_diamonds": "Diamonds",
            }
            for btype, suit in suit_map.items():
                if boss_type == btype and suit_name == suit:
                    return True

        if boss_type == "debuff_face_cards" and hasattr(card, 'is_face'):
            return card.is_face

        return False

    def reset(self):
        self.active_blind = None
