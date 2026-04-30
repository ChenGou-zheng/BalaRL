"""Card primitives for Balatro - suits, ranks, enhancements, editions, seals."""

from __future__ import annotations

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional


class Suit(IntEnum):
    SPADES = 0
    HEARTS = 1
    DIAMONDS = 2
    CLUBS = 3

    @property
    def symbol(self) -> str:
        return ["\u2660", "\u2665", "\u2666", "\u2663"][self.value]

    @property
    def name(self) -> str:
        return ["Spades", "Hearts", "Diamonds", "Clubs"][self.value]


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def is_face(self) -> bool:
        return self.value in (11, 12, 13)

    @property
    def base_chips(self) -> int:
        if self.value <= 10:
            return self.value
        elif self == Rank.ACE:
            return 11
        else:
            return 10

    @property
    def short_name(self) -> str:
        if self.value <= 10:
            return str(self.value)
        return {11: "J", 12: "Q", 13: "K", 14: "A"}[self.value]


class Enhancement(IntEnum):
    NONE = 0
    BONUS = 1          # +30 chips
    MULT = 2           # +4 mult
    WILD = 3           # acts as any suit
    GLASS = 4          # x2 mult, 25% destroy
    STEEL = 5          # x1.5 mult while held in hand
    STONE = 6          # +50 chips, no rank/suit
    GOLD = 7           # $3 when played
    LUCKY = 8          # 20% chance +20 mult or +$20


class Edition(IntEnum):
    NONE = 0
    FOIL = 1           # +50 chips
    HOLOGRAPHIC = 2    # +10 mult
    POLYCHROME = 3     # x1.5 mult
    NEGATIVE = 4       # +1 joker slot


class Seal(IntEnum):
    NONE = 0
    RED = 1            # retrigger this card once
    BLUE = 2           # create planet card if held at end of round
    GOLD = 3           # $3 when played
    PURPLE = 4         # create tarot when discarded


ENHANCEMENT_EFFECTS = {
    Enhancement.BONUS:     {"chips": 30},
    Enhancement.MULT:      {"mult": 4},
    Enhancement.WILD:      {"acts_as_any_suit": True},
    Enhancement.GLASS:     {"x_mult": 2, "destroy_chance": 0.25},
    Enhancement.STEEL:     {"x_mult": 1.5},
    Enhancement.STONE:     {"chips": 50, "no_suit_rank": True},
    Enhancement.GOLD:      {"money": 3},
    Enhancement.LUCKY:     {"money_chance": 0.2, "mult_chance": 0.2, "money_amount": 20, "mult_amount": 20},
}

EDITION_EFFECTS = {
    Edition.FOIL:         {"chips": 50},
    Edition.HOLOGRAPHIC:  {"mult": 10},
    Edition.POLYCHROME:   {"x_mult": 1.5},
    Edition.NEGATIVE:     {"joker_slot": 1},
}

SEAL_EFFECTS = {
    Seal.RED:    {"retrigger": 1},
    Seal.BLUE:   {"create_planet_if_held": True},
    Seal.GOLD:   {"money": 3},
    Seal.PURPLE: {"create_tarot_when_discarded": True},
}


@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit

    @property
    def base_chips(self) -> int:
        return self.rank.base_chips

    @property
    def is_face(self) -> bool:
        return self.rank.is_face

    @property
    def card_id(self) -> int:
        return self.suit.value * 13 + (self.rank.value - 2)

    @staticmethod
    def from_id(card_id: int) -> Card:
        suit = Suit(card_id // 13)
        rank = Rank((card_id % 13) + 2)
        return Card(rank=rank, suit=suit)

    def __int__(self) -> int:
        return self.card_id

    def __str__(self) -> str:
        return f"{self.rank.short_name}{self.suit.symbol}"


CARDS_PER_SUIT = 13
TOTAL_CARDS = 52


def create_standard_deck() -> list[Card]:
    return [Card(rank=rank, suit=suit) for suit in Suit for rank in Rank]
