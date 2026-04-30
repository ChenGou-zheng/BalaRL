"""Poker hand evaluation - classifies 5-8 card hands into Balatro's 12 hand types.

Handles utility joker effects (Four Fingers, Shortcut, Smeared Joker) that modify
hand construction rules.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from balarl.engine.cards import Card, Suit, Rank
from balarl.engine.scoring import HandType


def _rank_counts(cards: list[Card]) -> dict[int, list[Card]]:
    counts: dict[int, list[Card]] = defaultdict(list)
    for c in cards:
        counts[c.rank.value].append(c)
    return dict(counts)


def _suit_counts(cards: list[Card]) -> dict[int, list[Card]]:
    counts: dict[int, list[Card]] = defaultdict(list)
    for c in cards:
        counts[c.suit.value].append(c)
    return dict(counts)


def _has_straight(ranks: set[int], length: int, shortcut: bool) -> tuple[bool, list[int]]:
    ordered = sorted(ranks, reverse=True)
    straights: list[int] = []

    for i in range(len(ordered)):
        seq = [ordered[i]]
        gaps = 0
        for j in range(i + 1, len(ordered)):
            diff = ordered[j - 1] - ordered[j]
            if diff == 1:
                seq.append(ordered[j])
            elif diff == 2 and shortcut and gaps == 0:
                gaps += 1
                seq.append(ordered[j])
            else:
                break
        if len(seq) >= length:
            straights = seq
            break

    if 14 in ranks and not straights:
        wheel = [14, 2, 3, 4, 5]
        present = [r for r in wheel if r in ranks]
        if len(present) >= length:
            straights = present

    if straights:
        result_ranks = straights[:length]
        return True, result_ranks
    return False, []


def _collect_straight_cards(cards: list[Card], straight_ranks: list[int]) -> list[Card]:
    result = []
    seen = set()
    for rank_val in straight_ranks:
        for c in cards:
            if c.rank.value == rank_val and c.card_id not in seen:
                result.append(c)
                seen.add(c.card_id)
                break
    return result


def classify_hand(
    cards: list[Card],
    four_fingers: bool = False,
    shortcut: bool = False,
    smeared: bool = False,
) -> tuple[HandType, list[Card]]:
    """Classify the best poker hand from a list of 1-8 cards.

    Args:
        cards: The cards to classify.
        four_fingers: Four Fingers joker - flushes/straights need 4 cards instead of 5.
        shortcut: Shortcut joker - straights can skip one rank gap.
        smeared: Smeared Joker - hearts/diamonds considered same, spades/clubs same.

    Returns:
        (hand_type, best_cards) tuple. best_cards are the cards forming the hand.
    """
    n = len(cards)
    if n == 0:
        return HandType.HIGH_CARD, []

    rc = _rank_counts(cards)

    suit_for = {}
    for c in cards:
        if smeared:
            if c.suit in (Suit.HEARTS, Suit.DIAMONDS):
                suit_for[c.card_id] = 0  # red
            else:
                suit_for[c.card_id] = 1  # black
        else:
            suit_for[c.card_id] = c.suit.value

    sc: dict[int, list[Card]] = defaultdict(list)
    for c in cards:
        sc[suit_for[c.card_id]].append(c)

    flush_needed = 4 if four_fingers else 5
    straight_needed = 4 if four_fingers else 5

    flush_suit: Optional[int] = None
    flush_cards: list[Card] = []
    for suit_val, suit_cards in sc.items():
        if len(suit_cards) >= flush_needed:
            flush_suit = suit_val
            flush_cards = suit_cards[:flush_needed] if len(suit_cards) <= 5 else suit_cards
            break

    all_ranks = set(c.rank.value for c in cards)
    has_straight, straight_ranks = _has_straight(all_ranks, straight_needed, shortcut)
    straight_cards = _collect_straight_cards(cards, straight_ranks) if has_straight else []

    rank_items = sorted(rc.items(), key=lambda x: (-len(x[1]), -x[0]))
    max_same = len(rank_items[0][1]) if rank_items else 0
    second_same = len(rank_items[1][1]) if len(rank_items) > 1 else 0

    # 12. Flush Five: five of a kind + flush
    if max_same >= 5 and flush_suit is not None:
        flush_ranks = set(c.rank.value for c in flush_cards)
        if any(len(rc[r]) >= 5 for r in flush_ranks):
            best_rank = max(r for r in flush_ranks if len(rc[r]) >= 5)
            return HandType.FLUSH_FIVE, rc[best_rank][:5]

    # 11. Flush House: full house + flush
    if max_same >= 3 and second_same >= 2 and flush_suit is not None:
        three_cards = rank_items[0][1][:3]
        two_cards = rank_items[1][1][:2]
        return HandType.FLUSH_HOUSE, three_cards + two_cards

    # 10. Five of a Kind
    if max_same >= 5:
        return HandType.FIVE_KIND, rank_items[0][1][:5]

    # 9. Straight Flush
    if has_straight and flush_suit is not None:
        flush_rank_set = set(c.rank.value for c in flush_cards)
        sf_has, sf_ranks = _has_straight(flush_rank_set, straight_needed, shortcut)
        if sf_has:
            return HandType.STRAIGHT_FLUSH, _collect_straight_cards(flush_cards, sf_ranks)

    # 8. Four of a Kind
    if max_same >= 4:
        four = rank_items[0][1][:4]
        kicker = [c for c in cards if c.rank.value != rank_items[0][0]][:1]
        return HandType.FOUR_KIND, four + kicker

    # 7. Full House
    if max_same >= 3 and second_same >= 2:
        three = rank_items[0][1][:3]
        two = rank_items[1][1][:2]
        return HandType.FULL_HOUSE, three + two

    # 6. Flush
    if flush_suit is not None:
        return HandType.FLUSH, flush_cards[:5] if len(flush_cards) <= 5 else flush_cards

    # 5. Straight
    if has_straight:
        return HandType.STRAIGHT, straight_cards[:straight_needed]

    # 4. Three of a Kind
    if max_same >= 3:
        three = rank_items[0][1][:3]
        kickers = [c for c in cards if c.rank.value != rank_items[0][0]][:2]
        return HandType.THREE_KIND, three + kickers

    # 3. Two Pair
    if max_same >= 2 and second_same >= 2:
        high = rank_items[0][1][:2]
        low = rank_items[1][1][:2]
        kicker = [c for c in cards if c.rank.value not in (rank_items[0][0], rank_items[1][0])][:1]
        return HandType.TWO_PAIR, high + low + kicker

    # 2. One Pair
    if max_same >= 2:
        pair = rank_items[0][1][:2]
        kickers = [c for c in cards if c.rank.value != rank_items[0][0]][:3]
        return HandType.ONE_PAIR, pair + kickers

    # 1. High Card
    highest = sorted(cards, key=lambda c: c.rank.value, reverse=True)[:1]
    return HandType.HIGH_CARD, highest


def evaluate_all_hands(
    cards: list[Card],
    four_fingers: bool = False,
    shortcut: bool = False,
    smeared: bool = False,
) -> dict[HandType, list[list[Card]]]:
    """Find all valid hands of each type in the given cards."""
    results: dict[HandType, list[list[Card]]] = {ht: [] for ht in HandType}
    hand_type, best_cards = classify_hand(cards, four_fingers, shortcut, smeared)
    results[hand_type].append(best_cards)
    return results
