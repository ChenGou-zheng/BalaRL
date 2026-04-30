"""Expert/Heuristic agent for Balatro - used for behavioral cloning data generation.

Strategy:
  PLAY phase: Find the best 5-card hand from the 8 cards, select all 5, then play.
              If no good hand possible, discard weak cards.
  SHOP phase: Buy one joker if affordable, then skip.
  BLIND_SELECT: Always pick Big Blind for more money.
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np

from balarl.engine.cards import Card, Suit, Rank
from balarl.engine.hand_eval import classify_hand
from balarl.env.action_space import (
    SELECT_CARD_BASE, PLAY_HAND, DISCARD,
    SHOP_SKIP, SHOP_REROLL, SHOP_BUY_BASE,
    SELECT_SMALL_BLIND, SELECT_BIG_BLIND,
)
from balarl.env.observation import HAND_TYPE_ORDER

HAND_TYPE_INDEX = {name: i for i, name in enumerate(HAND_TYPE_ORDER)}


class ExpertAgent:
    """Heuristic agent that builds 5-card hands and plays them."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self._target_cards: Optional[List[int]] = None
        self._select_idx = 0
        self._shop_bought = False

    def act(self, obs: Dict[str, np.ndarray]) -> int:
        phase_val = int(obs["phase"])
        legal = _legal_from_mask(obs.get("action_mask"))
        if not legal:
            return 0

        if phase_val == 0:
            return self._play_action(obs, legal)
        elif phase_val == 1:
            return self._shop_action(obs, legal)
        else:
            return self._blind_action(obs, legal)

    def _play_action(self, obs: Dict, legal: List[int]) -> int:
        self._shop_bought = False
        selected = set(int(i) for i in np.where(obs["selected_cards"] > 0)[0])
        discards_left = int(obs["discards_left"])
        progress = float(obs["progress_ratio"])
        hand_size = int(sum(obs["hand"] >= 0))

        # If we have 5 cards selected, play them
        if len(selected) == 5 and PLAY_HAND in legal:
            self._target_cards = None
            self._select_idx = 0
            return PLAY_HAND

        # Continue selecting from current target
        if self._target_cards is not None and self._select_idx < len(self._target_cards) and len(selected) < 5:
            next_card = self._target_cards[self._select_idx]
            if next_card in selected:
                self._select_idx += 1
                return self._play_action(obs, legal)
            action = SELECT_CARD_BASE + next_card
            if action in legal:
                self._select_idx += 1
                return action

        # Need a new target: find the best 5-card hand from available cards
        hand_cards = _parse_hand(obs)
        if len(hand_cards) >= 5:
            self._target_cards = _find_best_5_card_combo(hand_cards, hand_size)
            self._select_idx = 0

            # Deselect wrong cards
            if len(selected) > 0 and self._target_cards:
                wrong = selected - set(self._target_cards)
                if wrong:
                    idx = sorted(wrong)[0]
                    action = SELECT_CARD_BASE + idx
                    if action in legal:
                        return action

            if self._target_cards and self._select_idx < len(self._target_cards):
                next_card = self._target_cards[self._select_idx]
                action = SELECT_CARD_BASE + next_card
                if action in legal:
                    self._select_idx += 1
                    return action

        # Discard weak cards if behind
        if discards_left > 0 and DISCARD in legal and progress < 0.5:
            weak = _find_weak_cards(hand_cards, hand_size)
            if weak:
                weak_set = set(weak)
                for s in list(selected):
                    if s not in weak_set:
                        action = SELECT_CARD_BASE + s
                        if action in legal:
                            return action
                for w in weak:
                    if w not in selected:
                        action = SELECT_CARD_BASE + w
                        if action in legal:
                            return action
                if selected == weak_set:
                    self._target_cards = None
                    self._select_idx = 0
                    return DISCARD

        # Play what we have as last resort
        if len(selected) > 0 and PLAY_HAND in legal:
            self._target_cards = None
            self._select_idx = 0
            return PLAY_HAND

        # Fallback
        card_acts = [a for a in legal if SELECT_CARD_BASE <= a < SELECT_CARD_BASE + hand_size]
        if card_acts:
            return card_acts[0]
        return legal[0]

    def _shop_action(self, obs: Dict, legal: List[int]) -> int:
        money = int(obs["money"])
        joker_count = int(obs["joker_count"])

        if self._shop_bought and SHOP_SKIP in legal:
            return SHOP_SKIP

        if joker_count < 5 and money >= 5:
            buy_acts = [a for a in legal if SHOP_BUY_BASE <= a < SHOP_BUY_BASE + 10]
            # Jokers are typically at shop positions 3-5. Try to buy them preferentially.
            joker_acts = [a for a in buy_acts if a >= SHOP_BUY_BASE + 3]
            if joker_acts:
                self._shop_bought = True
                return joker_acts[0]
            if buy_acts:
                self._shop_bought = True
                return buy_acts[0]

        if SHOP_SKIP in legal:
            return SHOP_SKIP
        return legal[0]

    def _blind_action(self, obs: Dict, legal: List[int]) -> int:
        if SELECT_BIG_BLIND in legal:
            return SELECT_BIG_BLIND
        return SELECT_SMALL_BLIND if SELECT_SMALL_BLIND in legal else legal[0]


def _parse_hand(obs: Dict) -> List[Card]:
    hand_ids = obs.get("hand", [])
    hand_ranks = obs.get("hand_ranks", [])
    hand_suits = obs.get("hand_suits", [])
    cards = []
    for i in range(min(len(hand_ids), len(hand_ranks), len(hand_suits))):
        cid = int(hand_ids[i])
        if cid >= 0:
            rank = Rank(int(hand_ranks[i])) if int(hand_ranks[i]) > 0 else Rank.TWO
            suit = Suit(int(hand_suits[i]))
            cards.append(Card(rank=rank, suit=suit))
    return cards


def _find_best_5_card_combo(hand: List[Card], hand_size: int) -> List[int]:
    n = min(hand_size, len(hand))
    if n < 5:
        return list(range(n))
    best_indices = list(range(5))
    best_rank = -1
    for combo in combinations(range(n), 5):
        cards = [hand[i] for i in combo]
        ht, _ = classify_hand(cards)
        rank = HAND_TYPE_INDEX.get(ht.name, -1)
        if rank > best_rank:
            best_rank = rank
            best_indices = list(combo)
    return best_indices


def _find_weak_cards(hand: List[Card], hand_size: int) -> List[int]:
    n = min(hand_size, len(hand))
    if n < 5:
        return list(range(n))
    rank_counts: Dict[int, List[int]] = {}
    for i in range(n):
        rv = hand[i].rank.value
        if rv not in rank_counts:
            rank_counts[rv] = []
        rank_counts[rv].append(i)
    weak = []
    for rv, indices in sorted(rank_counts.items()):
        if len(indices) == 1:
            weak.append(indices[0])
    return weak


def _legal_from_mask(mask) -> List[int]:
    if mask is None:
        return list(range(20))
    arr = np.array(mask).flatten()
    return [int(i) for i, v in enumerate(arr) if v > 0]


def run_expert_episode(env, agent, max_steps: int = 1000):
    obs, info = env.reset()
    agent._target_cards = None
    agent._select_idx = 0
    agent._shop_bought = False
    done = False
    total_reward = 0.0
    steps = 0
    final_ante = 1
    while not done and steps < max_steps:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        final_ante = int(obs.get("ante", final_ante))
    return total_reward, final_ante, steps
