"""Joker effects execution engine. Routes joker abilities by game phase."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Optional

from balarl.engine.cards import Card, Suit, Rank, Enhancement, Edition, Seal
from balarl.engine.scoring import HandType
from balarl.engine.jokers import JokerInfo, JOKER_ID_TO_INFO


class JokerState:
    """Mutable per-joker state for jokers that track state across rounds."""

    __slots__ = ("name", "mult", "x_mult", "chips", "money", "counter", "stored")

    def __init__(self, name: str):
        self.name = name
        self.mult = 0.0
        self.x_mult = 1.0
        self.chips = 0.0
        self.money = 0
        self.counter = 0
        self.stored: Dict[str, Any] = {}


class JokerEffects:
    """Applies joker effects during different game phases."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.states: Dict[str, JokerState] = {}

    def get_state(self, name: str) -> JokerState:
        if name not in self.states:
            self.states[name] = JokerState(name)
        return self.states[name]

    def apply(self, joker: JokerInfo, phase: str, context: dict, game_state: dict) -> Optional[dict]:
        state = self.get_state(joker.name)

        if phase == "individual_scoring":
            return self._individual_scoring(joker, context, game_state, state)
        elif phase == "scoring":
            return self._scoring(joker, context, game_state, state)
        elif phase == "before_scoring":
            return self._before_scoring(joker, context, game_state, state)
        elif phase == "discard":
            return self._discard(joker, context, game_state, state)
        elif phase == "end_round":
            return self._end_round(joker, context, game_state, state)

        return None

    # ─── individual scoring (per-card) ──────────────────────────────

    def _individual_scoring(self, joker: JokerInfo, ctx: dict, gs: dict, st: JokerState) -> Optional[dict]:
        card: Optional[Card] = ctx.get("card")
        if card is None:
            return None

        rank = card.rank.value
        suit = card.suit

        handlers: dict[str, Callable[[], Optional[dict]]] = {
            "Scary Face":  lambda: {"chips": 30} if 11 <= rank <= 13 else None,
            "Smiley Face": lambda: {"mult": 5} if 11 <= rank <= 13 else None,
            "Scholar":     lambda: {"chips": 20, "mult": 4} if rank == 14 else None,
            "Even Steven": lambda: {"mult": 4} if rank % 2 == 0 else None,
            "Odd Todd":    lambda: {"chips": 31} if rank % 2 == 1 else None,
            "Hanging Chad": lambda: {"retrigger": 2} if ctx.get("score_position") == 0 else None,
            "Walkie Talkie": lambda: {"chips": 10, "mult": 4} if rank in (4, 10) else None,
            "Wee Joker":   lambda: {"chips": 8} if rank == 2 else None,
            "Fibonacci":   lambda: {"mult": 8} if rank in (2, 3, 5, 8, 14) else None,
            "Sock & Buskin": lambda: {"retrigger": 1} if 11 <= rank <= 13 else None,
            "Hack":        lambda: {"retrigger": 1} if 2 <= rank <= 5 else None,
            "Arrowhead":   lambda: {"chips": 50} if suit == Suit.SPADES else None,
            "Onyx Agate":  lambda: {"mult": 7} if suit == Suit.CLUBS else None,
            "Rough Gem":   lambda: {"money": 1} if suit == Suit.DIAMONDS else None,
            "Bloodstone":  lambda: {"x_mult": 2} if suit == Suit.HEARTS and self.rng.random() < 0.5 else None,
            "Business Card": lambda: {"money": 2} if 11 <= rank <= 13 and self.rng.random() < 0.5 else None,
            "Photograph":  lambda: {"x_mult": 2} if 11 <= rank <= 13 and ctx.get("is_first_face") else None,
            "8 Ball":      lambda: None if rank != 8 or self.rng.random() >= 0.25 else {},
            "Triboulet":   lambda: {"x_mult": 2} if rank in (12, 13) else None,
        }

        fn = handlers.get(joker.name)
        if fn:
            return fn()

        suit_map = {
            2: ("Greedy Joker", Suit.DIAMONDS, {"mult": 3}),
            3: ("Lusty Joker", Suit.HEARTS, {"mult": 3}),
            4: ("Wrathful Joker", Suit.SPADES, {"mult": 3}),
            5: ("Gluttonous Joker", Suit.CLUBS, {"mult": 3}),
        }
        for jid, s, eff in suit_map.values():
            if joker.id == jid and suit == s:
                return eff

        return None

    # ─── scoring (hand-level) ───────────────────────────────────────

    def _scoring(self, joker: JokerInfo, ctx: dict, gs: dict, st: JokerState) -> Optional[dict]:
        hand_type_str: str = ctx.get("hand_type", "")
        cards: list = ctx.get("cards", [])
        scoring_cards: list = ctx.get("scoring_cards", [])
        n_scoring = len(scoring_cards)
        n_cards = len(cards)

        basic_effects: dict[str, Callable[[], Optional[dict]]] = {
            "Joker":           lambda: {"mult": 4},
            "Misprint":        lambda: {"mult": self.rng.randint(0, 23)},
            "Gros Michel":     lambda: {"mult": 15},
            "Cavendish":       lambda: {"x_mult": 3},
            "Half Joker":      lambda: {"mult": 20} if n_scoring <= 3 else None,
            "Banner":          lambda: {"chips": 30 * gs.get("discards_left", 0)},
            "Mystic Summit":   lambda: {"mult": 15} if gs.get("discards_left", 0) == 0 else None,
            "Blue Joker":      lambda: {"chips": 2 * len(gs.get("deck", []))},
            "Abstract Joker":  lambda: {"mult": 3 * len(gs.get("joker_ids", []))},
            "Acrobat":         lambda: {"x_mult": 3} if gs.get("hands_left", 1) == 1 else None,
            "Ice Cream":       lambda: {"chips": max(0, 100 - 5 * gs.get("hands_played", 0))} if gs.get("hands_played", 0) <= 20 else None,
            "Popcorn":         lambda: {"mult": max(0, 20 - 4 * gs.get("rounds_played", 0))},
            "Swashbuckler":    lambda: {"mult": sum(getattr(j, 'base_cost', 1) for j in gs.get("jokers", []))},
            "Bull":            lambda: {"chips": 2 * gs.get("money", 0)},
            "Bootstraps":      lambda: {"mult": 2 * (gs.get("money", 0) // 5)},
            "Supernova":       lambda: {"mult": gs.get("hand_play_counts", {}).get(hand_type_str.lower().replace(" ", "_"), 0)},
            "Green Joker":     lambda: {"mult": st.mult},
            "Ride the Bus":    lambda: {"mult": st.mult},
            "Fortune Teller":  lambda: {"mult": st.mult},
            "Red Card":        lambda: {"mult": st.mult},
            "Erosion":         lambda: {"mult": 4 * max(0, 52 - len(gs.get("deck", [])))},
            "Blackboard":      lambda: {"x_mult": 3} if all(_card_suit(c) in ("Spades", "Clubs") for c in cards) else None,
            "Flower Pot":      lambda: {"x_mult": 3} if len(set(_card_suit(c) for c in scoring_cards)) == 4 else None,
            "Seeing Double":   lambda: {"x_mult": 2} if _has_suit(scoring_cards, "Clubs") and len(set(_card_suit(c) for c in scoring_cards)) > 1 else None,
            "Baron":           lambda: {"x_mult": 1.5 ** _count_held_rank(cards, scoring_cards, 13)} if _count_held_rank(cards, scoring_cards, 13) > 0 else None,
            "Shoot the Moon":  lambda: {"mult": 13 * _count_held_rank(cards, scoring_cards, 12)} if _count_held_rank(cards, scoring_cards, 12) > 0 else None,
            "Raised Fist":     lambda: {"mult": 2 * _lowest_rank_in_hand(cards, scoring_cards)},
            "Ramen":           lambda: {"x_mult": max(0.01, st.x_mult)} if st.x_mult < 2.0 else None,
            "Stuntman":        lambda: {"chips": 250},
            "Joker Stencil":   lambda: {"x_mult": 1 + gs.get("joker_slots", 5) - len(gs.get("joker_ids", []))},
            "Driver's License": lambda: {"x_mult": 3} if gs.get("enhanced_count", 0) >= 16 else None,
            "Campfire":        lambda: {"x_mult": 1.0 + st.x_mult} if st.x_mult > 1.0 else None,
            "Constellation":   lambda: {"x_mult": 1.0 + st.x_mult} if st.x_mult > 1.0 else None,
            "Hologram":        lambda: {"x_mult": 1.0 + st.x_mult} if st.x_mult > 1.0 else None,
            "Vampire":         lambda: {"x_mult": 1.0 + st.x_mult} if st.x_mult > 1.0 else None,
            "Lucky Cat":       lambda: {"x_mult": 1.0 + st.x_mult} if st.x_mult > 1.0 else None,
            "Steel Joker":     lambda: {"x_mult": 1.0 + 0.2 * gs.get("steel_count", 0)} if gs.get("steel_count", 0) > 0 else None,
            "Glass Joker":     lambda: {"x_mult": 1.0 + 0.75 * st.x_mult} if st.x_mult > 0 else None,
            "Obelisk":         lambda: {"x_mult": 1.0 + 0.2 * st.x_mult} if st.x_mult > 0 else None,
            "Madness":         lambda: {"x_mult": 1.0 + 0.5 * st.x_mult} if st.x_mult > 0 else None,
            "Canio":           lambda: {"x_mult": 1.0 + st.x_mult} if st.x_mult > 0 else None,
            "Yorick":          lambda: {"x_mult": 1.0 + st.x_mult} if st.x_mult > 0 else None,
        }

        fn = basic_effects.get(joker.name)
        if fn:
            return fn()

        hand_type_effects: dict[str, tuple[str, dict]] = {
            "Jolly Joker":      ("One Pair", {"mult": 8}),
            "Zany Joker":       ("Three of a Kind", {"mult": 12}),
            "Mad Joker":        ("Two Pair", {"mult": 10}),
            "Crazy Joker":      ("Straight", {"mult": 12}),
            "Droll Joker":      ("Flush", {"mult": 10}),
            "Sly Joker":        ("One Pair", {"chips": 50}),
            "Wily Joker":       ("Three of a Kind", {"chips": 100}),
            "Clever Joker":     ("Two Pair", {"chips": 80}),
            "Devious Joker":    ("Straight", {"chips": 100}),
            "Crafty Joker":     ("Flush", {"chips": 80}),
            "The Duo":          ("One Pair", {"x_mult": 2}),
            "The Trio":         ("Three of a Kind", {"x_mult": 3}),
            "The Family":       ("Four of a Kind", {"x_mult": 4}),
            "The Order":        ("Straight", {"x_mult": 3}),
            "The Tribe":        ("Flush", {"x_mult": 2}),
            "Spare Trousers":   ("Two Pair", {}),
            "Runner":           ("Straight", {}),
            "Square Joker":     ("", {}),
        }

        if joker.name in hand_type_effects:
            target, effect = hand_type_effects[joker.name]
            if hand_type_str == target or (target == ""):
                return effect if effect else None

        return None

    # ─── before scoring ─────────────────────────────────────────────

    def _before_scoring(self, joker: JokerInfo, ctx: dict, gs: dict, st: JokerState) -> Optional[dict]:
        hand_type_str = ctx.get("hand_type", "")
        scoring_cards: list = ctx.get("scoring_cards", [])

        if joker.name == "Green Joker":
            st.mult += 1
            return None
        if joker.name == "Ride the Bus":
            has_face = any(11 <= _card_rank(c) <= 13 for c in scoring_cards)
            if has_face:
                st.mult = 0
            else:
                st.mult += 1
            return None
        if joker.name == "Square Joker":
            if len(scoring_cards) == 4:
                st.chips += 4
            return None
        if joker.name == "Spare Trousers":
            if hand_type_str == "Two Pair":
                st.mult += 2
            return None
        if joker.name == "Runner":
            if hand_type_str == "Straight":
                st.chips += 15
            return None
        if joker.name in ("Fortune Teller", "Red Card", "Flash Card"):
            return {"mult": st.mult} if st.mult > 0 else None

        return None

    # ─── discard phase ──────────────────────────────────────────────

    def _discard(self, joker: JokerInfo, ctx: dict, gs: dict, st: JokerState) -> Optional[dict]:
        discarded = ctx.get("discarded_cards", [])
        n_discard = len(discarded)
        is_first = ctx.get("is_first_discard", False)
        last_card = ctx.get("last_discarded_card")

        if joker.name == "Trading Card":
            if is_first and n_discard == 1:
                return {"money": 3, "destroy_card": True}
        if joker.name == "Faceless Joker":
            if n_discard >= 3 and all(11 <= _card_rank(c) <= 13 for c in discarded if hasattr(c, 'rank')):
                return {"money": 5}
        if joker.name == "Mail-In Rebate":
            if n_discard == 1 and last_card:
                return {"money": 5}
        if joker.name == "Green Joker":
            st.mult = max(0, st.mult - 1)
        if joker.name == "Hit the Road":
            jacks = sum(1 for c in discarded if hasattr(c, 'rank') and _card_rank(c) == 11)
            if jacks > 0:
                st.x_mult += 0.5 * jacks
        if joker.name == "Burnt Joker":
            if is_first and n_discard > 0:
                return {"upgrade_hand": True}

        return None

    # ─── end of round ───────────────────────────────────────────────

    def _end_round(self, joker: JokerInfo, ctx: dict, gs: dict, st: JokerState) -> Optional[dict]:
        if joker.name == "Golden Joker":
            return {"money": 4}
        if joker.name == "Rocket":
            st.money += 2
            if gs.get("round", 1) % 3 == 0:
                st.money += 2
            return {"money": st.money}
        if joker.name == "Egg":
            return {"sell_value_increase": 3}
        if joker.name == "Gift Card":
            return {"global_sell_value_increase": 1}
        if joker.name == "To the Moon":
            return {"extra_interest": 1}
        if joker.name == "Cloud 9":
            nine_count = sum(1 for c in gs.get("deck", []) if hasattr(c, 'rank') and _card_rank(c) == 9)
            return {"money": nine_count}
        if joker.name == "Satellite":
            return {"money": gs.get("planets_used", 0)}
        if joker.name == "Delayed Gratification":
            if gs.get("discards_left", 0) == gs.get("max_discards", 3):
                return {"money": 2 * gs.get("max_discards", 3)}
        if joker.name == "Gros Michel":
            if self.rng.random() < 1 / 6:
                return {"destroy": True}
        if joker.name == "Cavendish":
            if self.rng.random() < 1 / 1000:
                return {"destroy": True}
        if joker.name == "Turtle Bean":
            st.counter += 1
        if joker.name == "Popcorn":
            st.mult = max(0, st.mult - 4) if st.mult else 20 - 4 * (gs.get("rounds_played", 0) + 1)
        if joker.name == "Invisible Joker":
            st.counter += 1
            if st.counter >= 2:
                return {"ready_to_duplicate": True}
        if joker.name == "Ceremonial Dagger":
            return {"sacrifice_right": True}
        if joker.name == "Perkeo":
            return {"create_negative_copy": True}
        if joker.name == "Cartomancer":
            return {"create_tarot": True}

        return None


# ─── card helpers (work with card objects or dicts) ────────────────


def _card_suit(card) -> str:
    if hasattr(card, 'suit'):
        s = card.suit
        if hasattr(s, 'name'):
            return s.name
        return str(s)
    if isinstance(card, dict):
        s = card.get('suit', '')
        return s
    return ''


def _card_rank(card) -> int:
    if hasattr(card, 'rank'):
        r = card.rank
        if hasattr(r, 'value'):
            return r.value
        return int(r)
    if isinstance(card, dict):
        return card.get('rank', 0)
    return 0


def _has_suit(cards, suit_name: str) -> bool:
    return any(_card_suit(c) == suit_name for c in cards)


def _count_held_rank(all_cards, scoring_cards, rank_val: int) -> int:
    scored_ids = set(id(c) for c in scoring_cards)
    return sum(1 for c in all_cards if _card_rank(c) == rank_val and id(c) not in scored_ids)


def _lowest_rank_in_hand(all_cards, scoring_cards) -> int:
    scored_ids = set(id(c) for c in scoring_cards)
    held = [c for c in all_cards if id(c) not in scored_ids]
    if not held:
        return 1
    return min(_card_rank(c) for c in held)
