"""Balatro shop system - inventory generation, purchasing, rerolls, pack opening."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from balarl.engine.cards import Card
from balarl.engine.jokers import JokerInfo, JOKER_LIBRARY, JOKER_ID_TO_INFO


class ItemType(IntEnum):
    PACK = 0
    CARD = 1
    JOKER = 2
    VOUCHER = 3


class ShopAction(IntEnum):
    SKIP = 0
    REROLL = 1
    BUY_PACK = 2
    BUY_JOKER = 3
    BUY_CARD = 4
    BUY_VOUCHER = 5


@dataclass
class ShopItem:
    item_type: ItemType
    name: str
    cost: int
    payload: Dict
    sold: bool = False


@dataclass
class PlayerState:
    money: int
    jokers: List[int] = field(default_factory=list)
    joker_slots: int = 5
    consumables: List[str] = field(default_factory=list)
    consumable_slots: int = 2
    vouchers: List[str] = field(default_factory=list)
    deck: List[int] = field(default_factory=list)
    enhanced_count: int = 0
    steel_count: int = 0

    @property
    def chips(self) -> int:
        return self.money


PACK_NAMES = ["Standard Pack", "Joker Pack", "Tarot Pack", "Planet Pack", "Spectral Pack"]

COST_TABLE: Dict[str, int] = {
    "Standard Pack": 4,
    "Joker Pack": 5,
    "Tarot Pack": 4,
    "Planet Pack": 6,
    "Spectral Pack": 8,
    "Voucher: Magic Trick": 10,
    "Voucher: Minimalist": 10,
}

ANTE_COST_MULT = 1.15
REROLL_BASE_COST = 5
MAX_JOKERS_DEFAULT = 5


class Shop:
    """Generates shop inventory and processes purchases."""

    def __init__(self, ante: int, player: PlayerState, seed: Optional[int] = None):
        self.ante = ante
        self.player = player
        self.rng = random.Random(seed)
        self.inventory: List[ShopItem] = []
        self.reroll_cost = REROLL_BASE_COST
        self._generate_inventory()

    def _cost_mult(self) -> float:
        return ANTE_COST_MULT ** (self.ante - 1)

    def _generate_inventory(self):
        self.inventory.clear()
        mult = self._cost_mult()

        packs = ["Standard Pack", "Joker Pack", self.rng.choice(["Tarot Pack", "Planet Pack", "Spectral Pack"])]
        for pname in packs:
            self.inventory.append(ShopItem(ItemType.PACK, pname, int(COST_TABLE[pname] * mult), {"pack_type": pname}))

        available = [j for j in JOKER_LIBRARY if j.base_cost > 0 and j.id not in self.player.jokers]
        for joker in self.rng.sample(available, k=min(3, len(available))):
            self.inventory.append(ShopItem(ItemType.JOKER, joker.name, int(joker.base_cost * mult), {"joker_id": joker.id}))

        vname = self.rng.choice(["Voucher: Magic Trick", "Voucher: Minimalist"])
        self.inventory.append(ShopItem(ItemType.VOUCHER, vname, int(COST_TABLE[vname] * mult), {"voucher": vname.split(": ")[1]}))

        for _ in range(2):
            c = self.rng.randint(0, 51)
            self.inventory.append(ShopItem(ItemType.CARD, f"Card {c}", 3, {"card": c}))

    def get_observation(self) -> Dict:
        return {
            "shop_item_type": [int(i.item_type) for i in self.inventory],
            "shop_name": [i.name for i in self.inventory],
            "shop_cost": [i.cost for i in self.inventory],
            "shop_payload": [i.payload for i in self.inventory],
            "shop_sold": [i.sold for i in self.inventory],
        }

    def _open_pack(self, pack_type: str) -> List[int]:
        new_cards = []
        if pack_type == "Standard Pack":
            count = 5
        elif pack_type in ("Tarot Pack", "Planet Pack", "Spectral Pack"):
            count = 2
        else:
            count = 1

        for _ in range(count):
            card = self.rng.randint(0, 51)
            self.player.deck.append(card)
            new_cards.append(card)
        return new_cards

    def step(self, action: ShopAction, item_idx: int = -1) -> Tuple[float, bool, Dict]:
        info: Dict = {}
        reward = 0.0

        if action == ShopAction.SKIP:
            return 0.0, True, info

        if action == ShopAction.REROLL:
            cost = int(self.reroll_cost * self._cost_mult())
            if self.player.money < cost:
                return -1.0, False, {"error": "Not enough money for reroll"}
            self.player.money -= cost
            self.reroll_cost = int(self.reroll_cost * 1.5)
            self._generate_inventory()
            return 0.0, False, info

        if item_idx < 0 or item_idx >= len(self.inventory):
            return -1.0, False, {"error": f"Invalid item index: {item_idx}"}

        item = self.inventory[item_idx]
        if item.sold:
            return -1.0, False, {"error": "Item already sold"}

        if self.player.money < item.cost:
            return -1.0, False, {"error": "Not enough money"}

        self.player.money -= item.cost
        item.sold = True
        info["purchased"] = item.name

        if action == ShopAction.BUY_PACK:
            pack_type = item.payload.get("pack_type", item.name)
            info["new_cards"] = self._open_pack(pack_type)
            reward = 1.0
        elif action == ShopAction.BUY_CARD:
            self.player.deck.append(item.payload["card"])
            reward = 0.5
        elif action == ShopAction.BUY_JOKER:
            if len(self.player.jokers) >= self.player.joker_slots:
                return -1.0, False, {"error": "Joker slots full"}
            self.player.jokers.append(item.payload["joker_id"])
            reward = 2.0
        elif action == ShopAction.BUY_VOUCHER:
            self.player.vouchers.append(item.payload["voucher"])
            reward = 3.0

        return reward, False, info

    def sell_joker(self, joker_idx: int, sell_value: int = 1) -> Optional[int]:
        if 0 <= joker_idx < len(self.player.jokers):
            joker_id = self.player.jokers.pop(joker_idx)
            self.player.money += sell_value
            return joker_id
        return None
