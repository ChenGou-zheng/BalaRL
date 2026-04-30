"""Consumable items - Tarot, Planet, and Spectral cards with their effects."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Union

from balarl.engine.cards import Card, Suit, Rank
from balarl.engine.scoring import HandType, PLANET_HAND_MAP


class ConsumableManager:
    """Manages consumable usage and effects."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def use_consumable(self, name: str, game_state: Dict[str, Any], target_cards: List[Any] = None) -> Dict[str, Any]:
        if target_cards is None:
            target_cards = []

        if name.startswith("The ") or name in [
            "Strength", "Death", "Temperance", "Justice", "Judgement"
        ]:
            return self._use_tarot(name, game_state, target_cards)
        elif name in PLANET_HAND_MAP:
            return self._use_planet(name, game_state)
        elif name in ["Familiar", "Grim", "Incantation", "Talisman", "Aura", "Wraith",
                       "Black Hole", "Deja Vu", "Ankh", "Immolate", "Medium", "Ectoplasm",
                       "Cryptid", "The Soul"]:
            return self._use_spectral(name, game_state, target_cards)
        else:
            return {"success": False, "error": f"Unknown consumable: {name}"}

    def _use_tarot(self, name: str, gs: Dict[str, Any], targets: List[Any]) -> Dict[str, Any]:
        result = {"success": True, "tarot_used": name}

        if name == "The Fool":
            result["effect"] = "Create a random planet card"
        elif name == "The Magician":
            for card in targets[:2]:
                result["enhancement"] = "lucky"
            result["cards_affected"] = len(targets[:2])
        elif name == "The High Priestess":
            result["effect"] = "Create 2 random planet cards"
        elif name == "The Empress":
            for card in targets[:2]:
                result["enhancement"] = "mult"
            result["cards_affected"] = len(targets[:2])
        elif name == "The Emperor":
            result["effect"] = "Create 2 random tarot cards"
        elif name == "The Hierophant":
            for card in targets[:2]:
                result["enhancement"] = "bonus"
            result["cards_affected"] = len(targets[:2])
        elif name == "The Lovers":
            if targets:
                result["enhancement"] = "wild"
                result["cards_affected"] = 1
        elif name == "The Chariot":
            if targets:
                result["enhancement"] = "steel"
                result["cards_affected"] = 1
        elif name == "Strength":
            for card in targets[:2]:
                result["rank_increase"] = 1
            result["cards_affected"] = len(targets[:2])
        elif name == "The Hermit":
            hand = gs.get("hand", [])
            money_gained = len(hand)
            gs["money"] = gs.get("money", 0) + money_gained
            result["money_gained"] = money_gained
        elif name == "Wheel of Fortune":
            if targets and self.rng.random() < 0.25:
                result["edition"] = self.rng.choice(["foil", "holographic", "polychrome"])
            else:
                result["effect"] = "Nope!"
        elif name == "Justice":
            if targets:
                result["enhancement"] = "glass"
                result["cards_affected"] = 1
        elif name == "The Hanged Man":
            result["cards_destroyed"] = min(2, len(targets))
            result["cards_affected"] = min(2, len(targets))
        elif name == "Death":
            if len(targets) >= 2:
                result["effect"] = "Convert left card to right card"
                result["cards_affected"] = 2
        elif name == "Temperance":
            sell_total = sum(gs.get("joker_sell_values", {}).values())
            gs["money"] = gs.get("money", 0) + sell_total
            result["money_gained"] = sell_total
        elif name == "The Devil":
            if targets:
                result["enhancement"] = "gold"
                result["cards_affected"] = 1
        elif name == "The Tower":
            if targets:
                result["enhancement"] = "stone"
                result["cards_affected"] = 1
        elif name == "The Star":
            for card in targets[:3]:
                result["suit_change"] = "Diamonds"
            result["cards_affected"] = len(targets[:3])
        elif name == "The Moon":
            for card in targets[:3]:
                result["suit_change"] = "Clubs"
            result["cards_affected"] = len(targets[:3])
        elif name == "The Sun":
            for card in targets[:3]:
                result["suit_change"] = "Hearts"
            result["cards_affected"] = len(targets[:3])
        elif name == "The World":
            for card in targets[:3]:
                result["suit_change"] = "Spades"
            result["cards_affected"] = len(targets[:3])
        elif name == "Judgement":
            result["effect"] = "Create a random joker card"

        return result

    def _use_planet(self, name: str, gs: Dict[str, Any]) -> Dict[str, Any]:
        hand_type = PLANET_HAND_MAP.get(name)
        if hand_type:
            return {
                "success": True,
                "planet_used": name,
                "hand_type": hand_type,
                "level_increase": 1,
            }
        return {"success": False, "error": f"Unknown planet: {name}"}

    def _use_spectral(self, name: str, gs: Dict[str, Any], targets: List[Any]) -> Dict[str, Any]:
        result = {"success": True, "spectral_used": name}

        if name == "Familiar":
            result["effect"] = "Destroy 1 card, add 3 enhanced face cards"
            result["cards_affected"] = 1
        elif name == "Grim":
            result["effect"] = "Destroy 1 card, add 2 enhanced aces"
            result["cards_affected"] = 1
        elif name == "Incantation":
            result["effect"] = "Destroy 1 card, add 4 numbered cards"
            result["cards_affected"] = 1
        elif name == "Talisman":
            if targets:
                result["enhancement"] = "gold"
                result["cards_affected"] = 1
        elif name == "Aura":
            if targets:
                result["edition"] = self.rng.choice(["foil", "holographic", "polychrome"])
                result["cards_affected"] = 1
        elif name == "Wraith":
            result["effect"] = "Destroy 1 card, add joker"
            result["cards_affected"] = 1
        elif name == "Black Hole":
            result["effect"] = "Upgrade every poker hand by 1 level"
            result["mass_level_up"] = True
        elif name == "Deja Vu":
            if targets:
                result["seal"] = "red"
                result["cards_affected"] = 1
        elif name == "Ankh":
            result["effect"] = "Create a copy of a random joker, destroy others"
        elif name == "Immolate":
            result["effect"] = "Destroy 5 cards, gain $20"
            gs["money"] = gs.get("money", 0) + 20
            result["money_gained"] = 20
        elif name == "Medium":
            if targets:
                result["seal"] = "purple"
                result["cards_affected"] = 1
        elif name == "Ectoplasm":
            result["effect"] = "Add Negative to a random joker, -1 hand size"
        elif name == "Cryptid":
            result["effect"] = "Create 2 copies of 1 selected card"
            result["cards_affected"] = 1
        elif name == "The Soul":
            result["effect"] = "Create a Legendary joker"

        return result


TAROT_NAMES = [
    "The Fool", "The Magician", "The High Priestess", "The Empress",
    "The Emperor", "The Hierophant", "The Lovers", "The Chariot",
    "Strength", "The Hermit", "Wheel of Fortune", "Justice",
    "The Hanged Man", "Death", "Temperance", "The Devil",
    "The Tower", "The Star", "The Moon", "The Sun", "Judgement", "The World",
]

PLANET_NAMES = list(PLANET_HAND_MAP.keys())

SPECTRAL_NAMES = [
    "Familiar", "Grim", "Incantation", "Talisman", "Aura", "Wraith",
    "Black Hole", "Deja Vu", "Ankh", "Immolate", "Medium", "Ectoplasm",
    "Cryptid", "The Soul",
]

ALL_CONSUMABLES = TAROT_NAMES + PLANET_NAMES + SPECTRAL_NAMES
