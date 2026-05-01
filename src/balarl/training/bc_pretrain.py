"""Behavioral Cloning pretraining for SB3 PPO policy.

Loads expert trajectories and pre-trains the policy network (feature extractor +
action head) with supervised learning before PPO fine-tuning.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from stable_baselines3 import PPO


class TrajectoryDataset(Dataset):
    """PyTorch Dataset of (observation, action) pairs from expert trajectories."""

    def __init__(self, trajectories: List[Dict], action_space_size: int):
        self.observations: List[Dict[str, np.ndarray]] = []
        self.actions: List[int] = []

        for traj in trajectories:
            obs_list = traj.get("observations", traj.get("obs", []))
            act_list = traj.get("actions", traj.get("acts", []))
            for obs, act in zip(obs_list, act_list):
                # Ensure each obs value is a numpy array
                clean_obs = {k: np.asarray(v) for k, v in obs.items()}
                self.observations.append(clean_obs)
                self.actions.append(int(act))

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> tuple[Dict[str, np.ndarray], int]:
        return self.observations[idx], self.actions[idx]


def collate_obs(obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """Collate a list of observation dicts into a batch dict of tensors."""
    batch: Dict[str, List[np.ndarray]] = {}
    for obs in obs_list:
        for key, val in obs.items():
            batch.setdefault(key, []).append(val)

    result = {}
    for key, vals in batch.items():
        stacked = np.stack(vals)
        result[key] = torch.from_numpy(stacked)
    return result


def load_trajectories(path: str) -> List[Dict]:
    """Load pickled trajectories."""
    with open(path, "rb") as f:
        return pickle.load(f)


def pretrain_bc(
    model: PPO,
    trajectories: List[Dict],
    n_epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Pre-train PPO policy network via Behavioral Cloning.

    Args:
        model: SB3 PPO model (must be initialized on an env)
        trajectories: List of trajectory dicts from generate_trajectories.py
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for BC training
        device: "cpu" or "cuda"
        verbose: Print progress

    Returns:
        Dict with training stats: {"final_loss": float, "epochs": int, "samples": int}
    """
    action_space_size = model.action_space.n
    dataset = TrajectoryDataset(trajectories, action_space_size)
    n_samples = len(dataset)

    if n_samples == 0:
        print("  BC: No samples to train on")
        return {"final_loss": float("inf"), "epochs": 0, "samples": 0}

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda batch: (
            collate_obs([b[0] for b in batch]),
            torch.tensor([b[1] for b in batch], dtype=torch.long),
        ),
    )

    policy = model.policy
    policy.to(device)
    policy.train()

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    if verbose:
        print(f"  BC pretraining: {n_samples:,} samples, {n_epochs} epochs, "
              f"lr={learning_rate}, device={device}")

    t0 = time.perf_counter()
    best_loss = float("inf")

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for obs_batch, actions_batch in dataloader:
            obs_batch = {k: v.to(device) for k, v in obs_batch.items()}
            actions_batch = actions_batch.to(device)

            optimizer.zero_grad()

            # Forward through feature extractor + policy head
            features = policy.extract_features(obs_batch)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)

            loss = loss_fn(logits, actions_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        best_loss = min(best_loss, avg_loss)

        if verbose and (epoch + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            acc = _compute_accuracy(policy, dataloader, device, max_batches=20)
            print(f"    Epoch {epoch+1:>4}/{n_epochs}  "
                  f"loss={avg_loss:.4f}  "
                  f"acc={acc:.2%}  "
                  f"elapsed={elapsed:.0f}s")

    elapsed = time.perf_counter() - t0
    final_acc = _compute_accuracy(policy, dataloader, device, max_batches=50)

    if verbose:
        print(f"  BC complete: loss={best_loss:.4f}  "
              f"acc={final_acc:.2%}  "
              f"time={elapsed:.0f}s")

    policy.eval()
    return {"final_loss": best_loss, "epochs": n_epochs, "samples": n_samples, "accuracy": final_acc}


def _compute_accuracy(policy, dataloader, device, max_batches: int = 20) -> float:
    """Compute action prediction accuracy on a subset of the data."""
    policy.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (obs_batch, actions_batch) in enumerate(dataloader):
            if i >= max_batches:
                break
            obs_batch = {k: v.to(device) for k, v in obs_batch.items()}
            actions_batch = actions_batch.to(device)

            features = policy.extract_features(obs_batch)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)
            preds = logits.argmax(dim=-1)
            correct += (preds == actions_batch).sum().item()
            total += len(actions_batch)

    policy.train()
    return correct / max(1, total)
