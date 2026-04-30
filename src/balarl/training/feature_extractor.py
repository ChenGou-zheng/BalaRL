"""SB3 custom feature extractor for Balatro's Dict observation space.

Handles the full observation dict with batch-safe dimension management.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 0:
        return x.reshape(1, 1)
    if x.dim() == 1:
        return x.unsqueeze(-1)
    return x


def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0).unsqueeze(-1)
    if x.dim() == 2:
        return x.unsqueeze(-1)
    return x


class BalatroFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # Hand one-hot: (B, 8, 52) or (B, 8*52) depending on SB3 preprocessing
        self.hand_net = nn.Sequential(
            nn.Linear(8 * 52, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Rank embedding (0-14) and suit embedding (0-3)
        self.rank_emb = nn.Embedding(15, 16)
        self.suit_emb = nn.Embedding(4, 8)

        # Selection mask: (8,) → 8
        self.sel_linear = nn.Linear(8, 8)

        # Joker IDs: (10,) → 64 via embedding
        self.joker_emb = nn.Embedding(200, 32)
        self.joker_fc = nn.Sequential(
            nn.Linear(10 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Hand analysis: rank(13) + suit(4) + straight_pot + flush_pot → 32
        self.hand_analysis_net = nn.Sequential(
            nn.Linear(13 + 4 + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Hand levels: (12,) → 24
        self.hand_levels_net = nn.Sequential(
            nn.Linear(12, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
        )

        # Joker count info → 8
        self.joker_info_net = nn.Linear(2, 8)

        # Game state scalars → 32
        self.scalar_net = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Phase embedding
        self.phase_emb = nn.Embedding(3, 8)

        # Combined output
        combined_dim = 128 + 16*8 + 8*8 + 8 + 64 + 32 + 24 + 8 + 32 + 8  # = 496
        self.combined_net = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = observations["hand"].shape[0]

        # Hand one-hot
        ho = observations["hand_one_hot"].float()
        if ho.dim() > 2:
            ho = ho.reshape(batch_size, -1)
        hand_feat = self.hand_net(ho)

        # Rank/suit per slot
        ranks = observations["hand_ranks"].long().clamp(0, 14)
        suits = observations["hand_suits"].long().clamp(0, 3)
        rank_feat = self.rank_emb(ranks).reshape(batch_size, -1)
        suit_feat = self.suit_emb(suits).reshape(batch_size, -1)
        rank_suit_feat = torch.cat([rank_feat, suit_feat], dim=-1)

        # Selection mask
        sel = observations["selected_cards"].float()
        if sel.dim() < 2:
            sel = sel.unsqueeze(0)
        sel_feat = self.sel_linear(sel)

        # Jokers
        j_ids = observations["joker_ids"].long().clamp(0, 199)
        j_emb = self.joker_emb(j_ids).reshape(batch_size, -1)
        joker_feat = self.joker_fc(j_emb)

        # Hand analysis
        rc = _ensure_2d(observations["rank_counts"].float())
        sc = _ensure_2d(observations["suit_counts"].float())
        sp = _ensure_2d(observations["straight_potential"].float())
        fp = _ensure_2d(observations["flush_potential"].float())
        ha_input = torch.cat([rc, sc, sp, fp], dim=-1)
        ha_feat = self.hand_analysis_net(ha_input)

        # Hand levels
        hl = _ensure_2d(observations["hand_levels"].float())
        hl_feat = self.hand_levels_net(hl)

        # Joker info
        ji_input = torch.cat([
            _ensure_2d(observations["joker_count"].float()),
            _ensure_2d(observations["joker_slots"].float()),
        ], dim=-1)
        ji_feat = self.joker_info_net(ji_input)

        # Game state scalars: money, ante, round, hands_left, discards_left, hand_size,
        # chips_scored, chips_needed, progress_ratio, joker_count, joker_slots,
        # consumable_count, boss_blind_active
        scalar_keys = [
            "money", "ante", "round", "hands_left", "discards_left", "hand_size",
            "chips_scored", "chips_needed", "progress_ratio",
            "joker_count", "joker_slots", "consumable_count", "boss_blind_active",
        ]
        scalar_tensors = []
        for key in scalar_keys:
            val = observations[key].float()
            if key == "money":
                val = val / 100.0
            elif key in ("chips_scored", "chips_needed"):
                val = val / 1e6
            elif key == "ante":
                val = val / 10.0
            scalar_tensors.append(_ensure_2d(val))
        scalar_input = torch.cat(scalar_tensors, dim=-1)
        scalar_feat = self.scalar_net(scalar_input)

        # Phase
        phase = observations["phase"].long()
        if phase.dim() == 0:
            phase = phase.reshape(1)
        phase_feat = self.phase_emb(phase)
        if phase_feat.dim() == 3:
            phase_feat = phase_feat.squeeze(1)

        # Combine all
        combined = torch.cat([
            hand_feat, rank_suit_feat, sel_feat, joker_feat, ha_feat,
            hl_feat, ji_feat, scalar_feat, phase_feat,
        ], dim=-1)

        return self.combined_net(combined)
