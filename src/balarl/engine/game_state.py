"""UnifiedGameState - single source of truth for all Balatro game state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from balarl.engine.cards import Card
from balarl.engine.jokers import JokerInfo, JOKER_ID_TO_INFO

BLIND_TYPES = ["Small Blind", "Big Blind", "Boss Blind"]


@dataclass
class UnifiedGameState:
    """Centralized game state used by all subsystems."""

    ante: int = 1
    round: int = 1
    chips_needed: int = 300
    chips_scored: int = 0
    round_chips_scored: int = 0
    money: int = 4

    deck: List[Card] = field(default_factory=list)
    hand_indexes: List[int] = field(default_factory=list)
    selected_cards: List[int] = field(default_factory=list)
    hands_left: int = 4
    discards_left: int = 3
    hand_size: int = 8
    max_discards: int = 3
    hands_played: int = 0
    hands_played_total: int = 0
    hands_played_this_round: int = 0

    joker_ids: List[int] = field(default_factory=list)
    consumables: List[str] = field(default_factory=list)
    consumable_slots: int = 2
    vouchers: List[str] = field(default_factory=list)
    joker_slots: int = 5

    joker_sell_values: Dict[int, int] = field(default_factory=dict)
    enhancements: Dict[int, str] = field(default_factory=dict)
    permanent_chip_bonuses: Dict[int, int] = field(default_factory=dict)

    shop_reroll_cost: int = 5

    hand_levels: Dict[str, int] = field(default_factory=dict)
    hand_play_counts: Dict[str, int] = field(default_factory=dict)
    most_played_hand: str = ""
    last_played_hand: str = ""

    active_boss_blind: str | None = None
    boss_blind_active: bool = False
    face_down_cards: List[int] = field(default_factory=list)

    planets_used: int = 0
    tarots_used: int = 0
    packs_skipped: int = 0
    cards_added: int = 0
    cards_destroyed: int = 0
    cards_discarded_total: int = 0
    enhanced_count: int = 0
    steel_count: int = 0
    glass_destroyed: int = 0
    rounds_played: int = 0

    game_over: bool = False
    won: bool = False

    @property
    def hand(self) -> List[Card]:
        return [self.deck[i] for i in self.hand_indexes if i < len(self.deck)]

    @property
    def jokers(self) -> List[JokerInfo]:
        return [JOKER_ID_TO_INFO[jid] for jid in self.joker_ids if jid in JOKER_ID_TO_INFO]

    @property
    def joker_names(self) -> List[str]:
        return [JOKER_ID_TO_INFO[jid].name for jid in self.joker_ids if jid in JOKER_ID_TO_INFO]

    @property
    def deck_size(self) -> int:
        return len(self.deck)

    @property
    def blind_type(self) -> str:
        return BLIND_TYPES[self.round - 1] if 1 <= self.round <= 3 else "Unknown"

    def has_joker(self, name: str) -> bool:
        return name in self.joker_names

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deck": self.deck,
            "hand": self.hand,
            "joker_ids": self.joker_ids,
            "jokers": self.jokers,
            "joker_slots": self.joker_slots,
            "consumables": self.consumables,
            "vouchers": self.vouchers,
            "money": self.money,
            "ante": self.ante,
            "round": self.round,
            "blind_type": self.blind_type,
            "hands_left": self.hands_left,
            "discards_left": self.discards_left,
            "hand_size": self.hand_size,
            "hands_played": self.hands_played,
            "hands_played_this_round": self.hands_played_this_round,
            "rounds_played": self.rounds_played,
            "chips_needed": self.chips_needed,
            "chips_scored": self.chips_scored,
            "round_chips_scored": self.round_chips_scored,
            "hand_levels": self.hand_levels,
            "hand_play_counts": self.hand_play_counts,
            "most_played_hand": self.most_played_hand,
            "last_played_hand": self.last_played_hand,
            "active_boss_blind": self.active_boss_blind,
            "boss_blind_active": self.boss_blind_active,
            "planets_used": self.planets_used,
            "tarots_used": self.tarots_used,
            "packs_skipped": self.packs_skipped,
            "cards_added": self.cards_added,
            "cards_destroyed": self.cards_destroyed,
            "cards_discarded_total": self.cards_discarded_total,
            "enhanced_count": self.enhanced_count,
            "steel_count": self.steel_count,
            "glass_destroyed": self.glass_destroyed,
            "joker_sell_values": self.joker_sell_values,
            "enhancements": self.enhancements,
            "permanent_chip_bonuses": self.permanent_chip_bonuses,
        }

    def copy(self) -> UnifiedGameState:
        return UnifiedGameState(
            ante=self.ante,
            round=self.round,
            chips_needed=self.chips_needed,
            chips_scored=self.chips_scored,
            round_chips_scored=self.round_chips_scored,
            money=self.money,
            deck=list(self.deck),
            hand_indexes=list(self.hand_indexes),
            selected_cards=list(self.selected_cards),
            hands_left=self.hands_left,
            discards_left=self.discards_left,
            hand_size=self.hand_size,
            max_discards=self.max_discards,
            hands_played=self.hands_played,
            hands_played_total=self.hands_played_total,
            hands_played_this_round=self.hands_played_this_round,
            joker_ids=list(self.joker_ids),
            consumables=list(self.consumables),
            consumable_slots=self.consumable_slots,
            vouchers=list(self.vouchers),
            joker_slots=self.joker_slots,
            joker_sell_values=dict(self.joker_sell_values),
            enhancements=dict(self.enhancements),
            permanent_chip_bonuses=dict(self.permanent_chip_bonuses),
            shop_reroll_cost=self.shop_reroll_cost,
            hand_levels=dict(self.hand_levels),
            hand_play_counts=dict(self.hand_play_counts),
            most_played_hand=self.most_played_hand,
            last_played_hand=self.last_played_hand,
            active_boss_blind=self.active_boss_blind,
            boss_blind_active=self.boss_blind_active,
            face_down_cards=list(self.face_down_cards),
            planets_used=self.planets_used,
            tarots_used=self.tarots_used,
            packs_skipped=self.packs_skipped,
            cards_added=self.cards_added,
            cards_destroyed=self.cards_destroyed,
            cards_discarded_total=self.cards_discarded_total,
            enhanced_count=self.enhanced_count,
            steel_count=self.steel_count,
            glass_destroyed=self.glass_destroyed,
            rounds_played=self.rounds_played,
            game_over=self.game_over,
            won=self.won,
        )
