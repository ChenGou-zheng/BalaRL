"""Balatro Gymnasium Environment - Main RL environment for Balatro.

Phase flow: BLIND_SELECT -> PLAY (repeat until blind beaten/failed) -> SHOP -> repeat
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from balarl.engine.cards import Card, Suit, Rank, Enhancement, Edition, Seal, create_standard_deck
from balarl.engine.scoring import ScoreEngine, HandType
from balarl.engine.hand_eval import classify_hand, evaluate_all_hands
from balarl.engine.jokers import JokerInfo, JOKER_LIBRARY, JOKER_ID_TO_INFO, JOKER_NAME_TO_ID
from balarl.engine.joker_effects import JokerEffects
from balarl.engine.shop import Shop, ShopItem, ItemType, ShopAction as ShopAct, PlayerState
from balarl.engine.game_state import UnifiedGameState
from balarl.engine.blinds import get_blind_chips, get_blind_reward
from balarl.engine.boss_blinds import BossBlindManager
from balarl.engine.consumables import ConsumableManager, TAROT_NAMES, PLANET_NAMES, SPECTRAL_NAMES

from balarl.env.action_space import (
    Phase, SELECT_CARD_BASE, SELECT_CARD_COUNT, PLAY_HAND, DISCARD,
    USE_CONSUMABLE_BASE, USE_CONSUMABLE_COUNT,
    SHOP_SKIP, SHOP_REROLL, SHOP_BUY_BASE, SHOP_BUY_COUNT,
    SHOP_SELL_BASE, SHOP_SELL_COUNT,
    SELECT_SMALL_BLIND, SELECT_BIG_BLIND, SELECT_BOSS_BLIND,
    get_legal_actions, decode_action, action_space_size,
)
from balarl.env.observation import create_observation_space, build_hand_features, HAND_TYPE_ORDER, consumable_to_id
from balarl.env.reward import RewardShaper


class BalatroEnv(gym.Env):
    """Balatro reinforcement learning environment.

    Three phases:
      - BLIND_SELECT: Choose which blind type to play
      - PLAY: Select cards, play hand, discard, use consumables
      - SHOP: Buy/sell jokers, cards, packs, vouchers

    Antes 1-8 are the normal game; antes 9+ are endless mode.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode: str | None = None, seed: int | None = None):
        super().__init__()
        self.render_mode = render_mode
        self._seed = seed

        self.action_space = spaces.Discrete(action_space_size())
        self.observation_space = create_observation_space()

        self.rng = random.Random(seed)
        self.state = UnifiedGameState()
        self.scorer = ScoreEngine()
        self.joker_effects = JokerEffects(seed)
        self.boss_manager = BossBlindManager(seed)
        self.consumable_manager = ConsumableManager(seed)
        self.reward_shaper = RewardShaper()

        self.shop: Optional[Shop] = None
        self.selected_blind: Optional[str] = None
        self.played_hand_types_this_round: List[str] = []
        self._current_hand_type: str = ""
        self._total_steps = 0

    # ═══════════════════════════════════════════════════════════════
    # Gymnasium Interface
    # ═══════════════════════════════════════════════════════════════

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[Dict, Dict]:
        if seed is not None:
            self._seed = seed
            self.rng = random.Random(seed)
            self.joker_effects = JokerEffects(seed)
            self.boss_manager = BossBlindManager(seed)
            self.consumable_manager = ConsumableManager(seed)

        self.state = UnifiedGameState()
        self.scorer.reset()
        self.reward_shaper.reset()

        deck = create_standard_deck()
        self.rng.shuffle(deck)
        self.state.deck = deck

        self.state.ante = 1
        self.state.round = 1
        self.state.money = 4
        self.state.hand_size = 8
        self.state.hands_left = 4
        self.state.discards_left = 3
        self.state.max_discards = 3
        self.state.joker_slots = 5
        self.state.consumable_slots = 2

        for ht in HandType:
            self.state.hand_levels[ht.name] = 1
            self.state.hand_play_counts[ht.name] = 0

        self.state.chips_needed = get_blind_chips(self.state.ante, "Small Blind")
        self.state.round_chips_scored = 0
        self.state.chips_scored = 0

        self.shop = None
        self.selected_blind = None
        self.played_hand_types_this_round = []
        self._total_steps = 0
        self._current_hand_type = ""

        self._deal_hand()
        self.state.phase = "play"

        return self._build_obs(), self._build_info()

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        self._total_steps += 1
        terminated = False
        truncated = False
        reward = 0.0
        info: Dict[str, Any] = {}

        if not self._is_legal(action):
            reward = self.reward_shaper.invalid_action_penalty()
            info["error"] = "Illegal action"
            return self._build_obs(), reward, terminated, truncated, info

        if self.state.phase == "play":
            reward, terminated, info = self._step_play(action)
        elif self.state.phase == "shop":
            reward, terminated, info = self._step_shop(action)
        elif self.state.phase == "blinds":
            reward, terminated, info = self._step_blind_select(action)

        # Check for truncated (max steps)
        if self._total_steps > 5000:
            truncated = True

        return self._build_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            print(self._render_text())
        elif self.render_mode == "human":
            print(self._render_text())

    def close(self):
        pass

    # ═══════════════════════════════════════════════════════════════
    # Phase: BLIND_SELECT
    # ═══════════════════════════════════════════════════════════════

    def _step_blind_select(self, action: int) -> Tuple[float, bool, Dict]:
        blind_types = ["Small Blind", "Big Blind"]
        if self.state.round >= 3:
            blind_types.append("Boss Blind")

        decoded = decode_action(Phase.BLIND_SELECT, action)
        idx = {"small": 0, "big": 1, "boss": 2}.get(decoded.get("type", ""), 0)

        if idx >= len(blind_types):
            return -1.0, False, {"error": "Invalid blind selection"}

        self.selected_blind = blind_types[idx]

        # Boss blinds
        if self.selected_blind == "Boss Blind":
            boss_name = self.boss_manager.select_boss(self.state.ante)
            self.state.active_boss_blind = boss_name
            self.state.boss_blind_active = True

        self.state.chips_needed = get_blind_chips(self.state.ante, self.selected_blind)
        self.state.round_chips_scored = 0

        # Reset round state
        self.state.hands_left = 4
        self.state.discards_left = self.state.max_discards
        self.state.hands_played_this_round = 0
        self.played_hand_types_this_round = []

        self._deal_hand()
        self.state.phase = "play"

        return 0.1, False, {"blind_selected": self.selected_blind}

    # ═══════════════════════════════════════════════════════════════
    # Phase: PLAY
    # ═══════════════════════════════════════════════════════════════

    def _step_play(self, action: int) -> Tuple[float, bool, Dict]:
        decoded = decode_action(Phase.PLAY, action)
        atype = decoded["type"]

        if atype == "select_card":
            return self._play_select_card(decoded["card_idx"])
        elif atype == "play":
            return self._play_hand()
        elif atype == "discard":
            return self._play_discard()
        elif atype == "consumable":
            return self._play_consumable(decoded["consumable_idx"])

        return -1.0, False, {"error": f"Unknown play action: {atype}"}

    def _play_select_card(self, card_idx: int) -> Tuple[float, bool, Dict]:
        if card_idx >= len(self.state.hand_indexes):
            return -1.0, False, {"error": f"Invalid card index {card_idx}"}

        if card_idx in self.state.selected_cards:
            self.state.selected_cards.remove(card_idx)
        else:
            self.state.selected_cards.append(card_idx)

        return 0.0, False, {"selected": card_idx, "total_selected": len(self.state.selected_cards)}

    def _play_hand(self) -> Tuple[float, bool, Dict]:
        if len(self.state.selected_cards) == 0:
            return -1.0, False, {"error": "No cards selected"}
        if len(self.state.selected_cards) > 5:
            return -1.0, False, {"error": "Too many cards selected"}

        # Get card objects for selected cards
        selected_cards = []
        for idx in self.state.selected_cards:
            if idx < len(self.state.hand_indexes):
                deck_idx = self.state.hand_indexes[idx]
                if deck_idx < len(self.state.deck):
                    selected_cards.append(self.state.deck[deck_idx])

        # Classify hand type
        four_fingers = self.state.has_joker("Four Fingers")
        shortcut = self.state.has_joker("Shortcut")
        smeared = self.state.has_joker("Smeared Joker")

        hand_type, best_cards = classify_hand(selected_cards, four_fingers, shortcut, smeared)
        hand_type_name = hand_type.name

        # Check boss blind restrictions - debuffed cards just don't score
        debuffed_mask = [False] * len(selected_cards)
        if self.state.boss_blind_active and self.boss_manager.active_blind:
            for i, card in enumerate(selected_cards):
                if self.boss_manager.is_card_debuffed(card, self.state.to_dict()):
                    debuffed_mask[i] = True

        # Calculate base chips and mult from hand level
        base_chips, base_mult = self.scorer.get_base_chips_mult(hand_type)

        # Apply joker effects
        game_dict = self.state.to_dict()
        game_dict["hands_left"] = self.state.hands_left
        game_dict["discards_left"] = self.state.discards_left
        game_dict["hand_play_counts"] = self.state.hand_play_counts
        game_dict["hands_played_this_round"] = self.state.hands_played_this_round
        game_dict["rounds_played"] = self.state.rounds_played
        game_dict["max_discards"] = self.state.max_discards
        game_dict["steel_count"] = self.state.steel_count
        game_dict["enhanced_count"] = self.state.enhanced_count

        # Sum card base chips (skip debuffed)
        total_chips = base_chips
        for i, card in enumerate(selected_cards):
            if debuffed_mask[i]:
                continue
            bonus = self.state.permanent_chip_bonuses.get(card.card_id, 0)
            total_chips += card.base_chips + bonus

        total_add_mult = 0.0
        total_x_mult = 1.0
        extra_money = 0
        steel_mult = 1.0

        # Steel cards in hand (not played)
        played_ids = [self.state.hand_indexes[i] for i in self.state.selected_cards if i < len(self.state.hand_indexes)]
        for i, idx in enumerate(self.state.hand_indexes):
            if idx not in played_ids:
                card = self.state.deck[idx]
                enh = self.state.enhancements.get(idx, "")
                if enh == "steel":
                    steel_mult *= 1.5

        # Apply joker effects: before_scoring
        for jid in self.state.joker_ids:
            if jid not in JOKER_ID_TO_INFO:
                continue
            joker = JOKER_ID_TO_INFO[jid]
            ctx = {
                "phase": "before_scoring",
                "cards": selected_cards,
                "scoring_cards": selected_cards,
                "hand_type": hand_type_name,
            }
            effect = self.joker_effects.apply(joker, "before_scoring", ctx, game_dict)
            if effect:
                total_add_mult += effect.get("mult", 0)
                total_chips += effect.get("chips", 0)

        # Apply joker effects: individual_scoring (per card, skip debuffed)
        for pos, card in enumerate(selected_cards):
            if debuffed_mask[pos]:
                continue
            is_face = card.is_face
            for jid in self.state.joker_ids:
                if jid not in JOKER_ID_TO_INFO:
                    continue
                joker = JOKER_ID_TO_INFO[jid]
                ctx = {
                    "phase": "individual_scoring",
                    "card": card,
                    "cards": selected_cards,
                    "scoring_cards": selected_cards,
                    "hand_type": hand_type_name,
                    "score_position": pos,
                    "is_first_face": is_face and pos == 0,
                }
                effect = self.joker_effects.apply(joker, "individual_scoring", ctx, game_dict)
                if effect:
                    total_add_mult += effect.get("mult", 0)
                    total_chips += effect.get("chips", 0)
                    if "x_mult" in effect:
                        total_x_mult *= effect["x_mult"]
                    extra_money += effect.get("money", 0)

        # Apply joker effects: scoring (hand-level)
        for jid in self.state.joker_ids:
            if jid not in JOKER_ID_TO_INFO:
                continue
            joker = JOKER_ID_TO_INFO[jid]
            ctx = {
                "phase": "scoring",
                "cards": selected_cards,
                "scoring_cards": selected_cards,
                "hand_type": hand_type_name,
            }
            effect = self.joker_effects.apply(joker, "scoring", ctx, game_dict)
            if effect:
                total_add_mult += effect.get("mult", 0)
                total_chips += effect.get("chips", 0)
                if "x_mult" in effect:
                    total_x_mult *= effect["x_mult"]

        # Calculate final score
        final_mult = (base_mult + total_add_mult) * total_x_mult * steel_mult
        final_score = int(total_chips * final_mult)
        final_score = max(1, final_score)

        # Boss blind modifications
        if self.state.boss_blind_active and self.boss_manager.active_blind:
            chips_ch, mult_ch = self.boss_manager.modify_scoring(total_chips, final_mult, selected_cards, hand_type_name)
            final_score = int(chips_ch * mult_ch)

        # Track hand type
        self._current_hand_type = hand_type_name
        self.played_hand_types_this_round.append(hand_type_name)

        # Record stats
        self.state.hand_play_counts[hand_type_name] = self.state.hand_play_counts.get(hand_type_name, 0) + 1
        self.scorer.record_play(hand_type)
        self.state.hands_played_total += 1
        self.state.hands_played_this_round += 1

        # Calculate reward
        old_progress = min(1.0, self.state.round_chips_scored / max(1, self.state.chips_needed))
        self.state.round_chips_scored += final_score
        self.state.chips_scored += final_score
        self.state.money += extra_money
        new_progress = min(1.0, self.state.round_chips_scored / max(1, self.state.chips_needed))

        reward, reward_breakdown = self.reward_shaper.hand_reward(
            final_score, old_progress, new_progress, hand_type_name,
            len(selected_cards), self.state.hands_left, self.state.ante,
        )

        info = {
            "hand_type": hand_type_name,
            "score": final_score,
            "reward_breakdown": reward_breakdown,
        }

        # Boss blind post-hand effects
        if self.state.boss_blind_active and self.boss_manager.active_blind:
            boss_effects = self.boss_manager.on_hand_played(
                selected_cards, hand_type_name, self.state.to_dict()
            )
            if "money_lost" in boss_effects:
                self.state.money = max(0, self.state.money - boss_effects["money_lost"])

        # Clear selection
        self.state.selected_cards = []

        # Check round completion
        if self.state.round_chips_scored >= self.state.chips_needed:
            return self._on_blind_cleared(reward, info)
        elif self.state.hands_left <= 0:
            return self._on_blind_failed(reward, info)
        else:
            self.state.hands_left -= 1
            self._deal_hand()
            return reward, False, info

    def _play_discard(self) -> Tuple[float, bool, Dict]:
        if self.state.discards_left <= 0:
            return -1.0, False, {"error": "No discards left"}
        if len(self.state.selected_cards) == 0:
            return -1.0, False, {"error": "No cards selected to discard"}

        discarded_cards = []
        discard_indices = sorted(self.state.selected_cards, reverse=True)

        for idx in discard_indices:
            if idx < len(self.state.hand_indexes):
                deck_idx = self.state.hand_indexes.pop(idx)
                if deck_idx < len(self.state.deck):
                    discarded_cards.append(self.state.deck[deck_idx])
                    self.state.cards_discarded_total += 1

        self.state.discards_left -= 1
        self.state.selected_cards = []

        # Apply joker discard effects
        game_dict = self.state.to_dict()
        for jid in self.state.joker_ids:
            if jid not in JOKER_ID_TO_INFO:
                continue
            joker = JOKER_ID_TO_INFO[jid]
            ctx = {
                "phase": "discard",
                "discarded_cards": discarded_cards,
                "last_discarded_card": discarded_cards[-1] if discarded_cards else None,
                "is_first_discard": self.state.discards_left == self.state.max_discards - 1,
            }
            effect = self.joker_effects.apply(joker, "discard", ctx, game_dict)
            if effect:
                self.state.money += effect.get("money", 0)

        # Draw new cards
        self._deal_hand()

        progress = self.state.round_chips_scored / max(1, self.state.chips_needed)
        reward = self.reward_shaper.discard_reward(len(discarded_cards), progress, self.state.discards_left)

        return reward, False, {"discarded": len(discard_indices)}

    def _play_consumable(self, consumable_idx: int) -> Tuple[float, bool, Dict]:
        if consumable_idx >= len(self.state.consumables):
            return -1.0, False, {"error": "Invalid consumable index"}
        if len(self.state.selected_cards) == 0:
            return -1.0, False, {"error": "Need to select target cards first"}

        consumable_name = self.state.consumables[consumable_idx]

        target_cards = []
        for sidx in self.state.selected_cards:
            if sidx < len(self.state.hand_indexes):
                deck_idx = self.state.hand_indexes[sidx]
                if deck_idx < len(self.state.deck):
                    target_cards.append(self.state.deck[deck_idx])

        game_dict = self.state.to_dict()
        result = self.consumable_manager.use_consumable(consumable_name, game_dict, target_cards)

        if result.get("success"):
            self.state.consumables.pop(consumable_idx)
            self.state.tarots_used += 1

            if "planet_used" in result:
                ht = result.get("hand_type")
                if ht:
                    self.scorer.apply_planet(ht)
                    self.state.hand_levels[ht.name] = self.scorer.get_level(ht)
                    self.state.planets_used += 1

            self.state.money = game_dict.get("money", self.state.money)
            self.state.selected_cards = []

        reward = 1.0 if result.get("success") else -1.0
        return reward, False, result

    # ═══════════════════════════════════════════════════════════════
    # Phase: SHOP
    # ═══════════════════════════════════════════════════════════════

    def _step_shop(self, action: int) -> Tuple[float, bool, Dict]:
        if self.shop is None:
            self.shop = self._create_shop()

        if action == SHOP_SKIP:
            return self._on_shop_done()

        if action == SHOP_REROLL:
            reward, done, info = self.shop.step(ShopAct.REROLL)
            return reward, done, info

        if SHOP_BUY_BASE <= action < SHOP_BUY_BASE + SHOP_BUY_COUNT:
            item_idx = action - SHOP_BUY_BASE
            if self.shop and item_idx < len(self.shop.inventory):
                item = self.shop.inventory[item_idx]
                if item.item_type == ItemType.JOKER:
                    shop_act = ShopAct.BUY_JOKER
                elif item.item_type == ItemType.CARD:
                    shop_act = ShopAct.BUY_CARD
                elif item.item_type == ItemType.VOUCHER:
                    shop_act = ShopAct.BUY_VOUCHER
                else:
                    shop_act = ShopAct.BUY_PACK
                reward, done, info = self.shop.step(shop_act, item_idx)
                return reward, done, info
            return -1.0, False, {"error": "Invalid shop index"}

        if SHOP_SELL_BASE <= action < SHOP_SELL_BASE + SHOP_SELL_COUNT:
            joker_idx = action - SHOP_SELL_BASE
            if 0 <= joker_idx < len(self.state.joker_ids):
                sell_val = self.state.joker_sell_values.get(self.state.joker_ids[joker_idx], 1)
                self.state.money += sell_val
                self.state.joker_ids.pop(joker_idx)
                reward = sell_val / 5.0
                return reward, False, {"sold_joker": True}
            return -1.0, False, {"error": "Invalid joker slot"}

        return -1.0, False, {"error": "Unknown shop action"}

    def _is_joker_item(self, idx: int) -> bool:
        if self.shop and idx < len(self.shop.inventory):
            return self.shop.inventory[idx].item_type == ItemType.JOKER
        return False

    def _create_shop(self) -> Shop:
        player = PlayerState(
            money=self.state.money,
            jokers=list(self.state.joker_ids),
            joker_slots=self.state.joker_slots,
            consumables=list(self.state.consumables),
            consumable_slots=self.state.consumable_slots,
            vouchers=list(self.state.vouchers),
            deck=[c.card_id for c in self.state.deck],
            enhanced_count=self.state.enhanced_count,
            steel_count=self.state.steel_count,
        )
        return Shop(self.state.ante, player, seed=self.rng.randint(0, 2**31 - 1))

    def _on_shop_done(self):
        if self.shop:
            self.state.money = self.shop.player.money
            self.state.joker_ids = list(self.shop.player.jokers)
            self.state.vouchers = list(self.shop.player.vouchers)

        # Apply end-of-round joker effects
        game_dict = self.state.to_dict()
        for jid in self.state.joker_ids:
            if jid not in JOKER_ID_TO_INFO:
                continue
            joker = JOKER_ID_TO_INFO[jid]
            ctx = {"phase": "end_round"}
            effect = self.joker_effects.apply(joker, "end_round", ctx, game_dict)
            if effect:
                self.state.money += effect.get("money", 0)

        # Check game over conditions
        if self.state.ante >= 8 and self.state.round >= 3:
            self.state.won = True
            self.state.game_over = True
            self.state.phase = "game_over"
            return 100.0, True, {"won": True}

        # Advance to next blind/ante
        self._advance_round()
        return 10.0, False, {"shop_done": True}

    # ═══════════════════════════════════════════════════════════════
    # Round Management
    # ═══════════════════════════════════════════════════════════════

    def _on_blind_cleared(self, current_reward: float, info: Dict) -> Tuple[float, bool, Dict]:
        blind_name = BLIND_NAMES[self.state.round - 1] if self.state.round <= 3 else "Blind"
        reward = current_reward + self.reward_shaper.blind_clear_reward(self.state.ante)

        # Earn reward money
        reward_money = get_blind_reward(self.state.ante, blind_name)
        self.state.money += reward_money

        # Interest
        interest = min(5, self.state.money // 5)
        if self.state.has_joker("To the Moon"):
            interest = min(6, self.state.money // 5)
        self.state.money += interest

        self.state.rounds_played += 1
        self.state.phase = "shop"
        self.shop = self._create_shop()

        info["blind_cleared"] = True
        info["blind_name"] = blind_name
        return reward, False, info

    def _on_blind_failed(self, current_reward: float, info: Dict) -> Tuple[float, bool, Dict]:
        progress = min(1.0, self.state.round_chips_scored / max(1, self.state.chips_needed))
        penalty = self.reward_shaper.blind_fail_penalty(progress)

        if self.state.has_joker("Mr. Bones") and progress >= 0.25:
            self.state.round_chips_scored = self.state.chips_needed
            return self._on_blind_cleared(current_reward, info)

        self.state.game_over = True
        self.state.phase = "game_over"
        reward = current_reward + penalty
        info["blind_failed"] = True
        return reward, True, info

    def _advance_round(self):
        self.state.round += 1
        if self.state.round > 3:
            self.state.ante += 1
            self.state.round = 1

        blind_name = BLIND_NAMES[self.state.round - 1] if self.state.round <= 3 else "Small Blind"

        # Boss blind
        if blind_name == "Boss Blind":
            boss_name = self.boss_manager.select_boss(self.state.ante)
            self.state.active_boss_blind = boss_name
            self.state.boss_blind_active = True

        self.state.chips_needed = get_blind_chips(self.state.ante, blind_name)
        self.state.round_chips_scored = 0
        self.state.hands_left = 4
        self.state.discards_left = self.state.max_discards
        self.state.hands_played_this_round = 0
        self.played_hand_types_this_round = []

        # Apply boss blind round-start effects
        if self.state.boss_blind_active and self.boss_manager.active_blind:
            game_dict = self.state.to_dict()
            game_dict["chips_needed"] = self.state.chips_needed
            game_dict["discards_left"] = self.state.discards_left
            game_dict["hands_left"] = self.state.hands_left
            game_dict["hand_size"] = self.state.hand_size
            boss_effects = self.boss_manager.on_round_start(game_dict)
            self.state.chips_needed = game_dict.get("chips_needed", self.state.chips_needed)
            self.state.discards_left = game_dict.get("discards_left", self.state.discards_left)
            self.state.hands_left = game_dict.get("hands_left", self.state.hands_left)
            self.state.hand_size = game_dict.get("hand_size", self.state.hand_size)

        self.state.phase = "play"
        self._deal_hand()

    # ═══════════════════════════════════════════════════════════════
    # Card Dealing
    # ═══════════════════════════════════════════════════════════════

    def _deal_hand(self):
        current_hand = set(self.state.hand_indexes)
        available = [i for i, c in enumerate(self.state.deck) if i not in current_hand]

        if len(available) < self.state.hand_size:
            if len(self.state.deck) > 0:
                remaining = [i for i in range(len(self.state.deck)) if i not in current_hand]
                self.state.hand_indexes = remaining + list(current_hand)[:self.state.hand_size - len(remaining)]
                self.state.hand_indexes = self.state.hand_indexes[:self.state.hand_size]
            return

        self.rng.shuffle(available)
        self.state.hand_indexes = available[:self.state.hand_size]

    # ═══════════════════════════════════════════════════════════════
    # Observation Building
    # ═══════════════════════════════════════════════════════════════

    def _build_obs(self) -> Dict[str, np.ndarray]:
        hand = [self.state.deck[i] if i < len(self.state.deck) else None for i in self.state.hand_indexes]
        hand = [c for c in hand if c is not None]

        # Pad to 8 cards
        hand_padded = hand[:8] + [None] * (8 - len(hand[:8]))
        hand_ids = np.array([c.card_id if c else -1 for c in hand_padded], dtype=np.int32)
        hand_ranks = np.array([c.rank.value if c else 0 for c in hand_padded], dtype=np.int32)
        hand_suits = np.array([c.suit.value if c else 0 for c in hand_padded], dtype=np.int32)

        # One-hot hand
        hand_one_hot = np.zeros((8, 52), dtype=np.float32)
        for i, c in enumerate(hand_padded):
            if c:
                hand_one_hot[i, c.card_id] = 1.0

        # Selected cards
        selected_mask = np.zeros(8, dtype=np.int8)
        for idx in self.state.selected_cards:
            if idx < 8:
                selected_mask[idx] = 1

        # Joker IDs (padded)
        joker_arr = np.zeros(10, dtype=np.int32)
        for i, jid in enumerate(self.state.joker_ids[:10]):
            joker_arr[i] = jid

        # Hand features
        rank_counts, suit_counts, straight_pot, flush_pot = build_hand_features(hand_padded)

        # Hand levels
        hand_levels = np.zeros(12, dtype=np.int32)
        for i, name in enumerate(HAND_TYPE_ORDER):
            hand_levels[i] = self.state.hand_levels.get(name, 1)

        # Consumables
        consumable_arr = np.zeros(5, dtype=np.int32)
        for i, cname in enumerate(self.state.consumables[:5]):
            consumable_arr[i] = consumable_to_id(cname)

        # Shop items
        shop_item_arr = np.zeros(10, dtype=np.int32)
        shop_cost_arr = np.zeros(10, dtype=np.int32)
        shop_reroll = np.int32(0)
        if self.shop:
            for i, item in enumerate(self.shop.inventory[:10]):
                shop_item_arr[i] = int(item.item_type)
                shop_cost_arr[i] = item.cost
            shop_reroll = np.int32(self.shop.reroll_cost)

        # Boss blind
        boss_type = 0
        if self.state.boss_blind_active and self.state.active_boss_blind:
            from balarl.engine.boss_blinds import BOSS_BY_NAME
            boss_type = BOSS_BY_NAME.get(self.state.active_boss_blind, 0)

        face_down = np.zeros(8, dtype=np.int8)
        for idx in self.state.face_down_cards:
            if idx < 8:
                face_down[idx] = 1

        # Phase
        phase_map = {"play": 0, "shop": 1, "blinds": 2, "game_over": 0}
        phase_val = phase_map.get(self.state.phase, 0)

        # Action mask
        action_mask = np.zeros(action_space_size(), dtype=np.int8)
        if phase_val == 0:
            legal = get_legal_actions(Phase.PLAY, {"hand_size": len(hand_padded), "selected_cards": self.state.selected_cards,
                                                    "discards_left": self.state.discards_left, "consumables": self.state.consumables})
        elif phase_val == 1:
            shop_items = [i for i in range(len(self.shop.inventory))] if self.shop else []
            legal = get_legal_actions(Phase.SHOP, {"shop_items": shop_items,
                                                    "joker_ids": self.state.joker_ids})
        else:
            legal = get_legal_actions(Phase.BLIND_SELECT, {"round": self.state.round})
        for a in legal:
            if a < len(action_mask):
                action_mask[a] = 1

        # Progress
        progress = self.state.round_chips_scored / max(1, self.state.chips_needed)

        return {
            "hand": hand_ids,
            "hand_one_hot": hand_one_hot,
            "hand_ranks": hand_ranks,
            "hand_suits": hand_suits,
            "selected_cards": selected_mask,
            "money": np.int32(self.state.money),
            "ante": np.int32(self.state.ante),
            "round": np.int32(self.state.round),
            "hands_left": np.int32(self.state.hands_left),
            "discards_left": np.int32(self.state.discards_left),
            "hand_size": np.int32(self.state.hand_size),
            "chips_scored": np.int64(self.state.chips_scored),
            "chips_needed": np.int64(self.state.chips_needed),
            "progress_ratio": np.float32(progress),
            "joker_ids": joker_arr,
            "joker_count": np.int32(len(self.state.joker_ids)),
            "joker_slots": np.int32(self.state.joker_slots),
            "rank_counts": rank_counts,
            "suit_counts": suit_counts,
            "straight_potential": straight_pot,
            "flush_potential": flush_pot,
            "hand_levels": hand_levels,
            "consumable_count": np.int32(len(self.state.consumables)),
            "consumables": consumable_arr,
            "shop_items": shop_item_arr,
            "shop_costs": shop_cost_arr,
            "shop_reroll_cost": shop_reroll,
            "boss_blind_active": np.int32(1 if self.state.boss_blind_active else 0),
            "boss_blind_type": np.int32(boss_type),
            "face_down_cards": face_down,
            "phase": np.int32(phase_val),
            "action_mask": action_mask,
        }

    def _build_info(self) -> Dict:
        return {
            "ante": self.state.ante,
            "round": self.state.round,
            "money": self.state.money,
            "hands_left": self.state.hands_left,
            "jokers": [JOKER_ID_TO_INFO[jid].name for jid in self.state.joker_ids if jid in JOKER_ID_TO_INFO],
            "chips_scored": self.state.chips_scored,
            "chips_needed": self.state.chips_needed,
        }

    def _is_legal(self, action: int) -> bool:
        obs = self._build_obs()
        mask = obs.get("action_mask")
        if mask is None:
            return True
        if action < 0 or action >= len(mask):
            return False
        return bool(mask[action])

    def _render_text(self) -> str:
        hand = self.state.hand
        cards_str = " ".join(str(c) for c in hand)
        joker_str = ", ".join(
            JOKER_ID_TO_INFO[jid].name for jid in self.state.joker_ids if jid in JOKER_ID_TO_INFO
        ) or "none"
        selected = sorted(self.state.selected_cards)

        lines = [
            f"=== Ante {self.state.ante} | Round {self.state.round} | {self.state.phase.upper()} ===",
            f"Score: {self.state.round_chips_scored}/{self.state.chips_needed} | Money: ${self.state.money}",
            f"Hand: {cards_str}",
            f"Selected: {selected}",
            f"Jokers ({len(self.state.joker_ids)}): {joker_str}",
            f"Hands: {self.state.hands_left} | Discards: {self.state.discards_left}",
        ]
        if self.state.boss_blind_active:
            lines.append(f"Boss: {self.state.active_boss_blind}")
        return "\n".join(lines) + "\n"


BLIND_NAMES = ["Small Blind", "Big Blind", "Boss Blind"]
