#!/usr/bin/env python3
"""Generate expert trajectories for Behavioral Cloning pretraining.

Usage:
    python -m balarl.scripts.generate_trajectories --n-episodes 1000 --save trajectories/expert_trajectories.pkl

Output format: list of dicts, each dict:
    {
        "observations": [obs_dict_0, obs_dict_1, ...],   # per-step observations
        "actions": [action_0, action_1, ...],             # per-step actions (int)
        "final_ante": int,                                 # ante reached
        "total_reward": float,                              # cumulative reward
        "steps": int,                                      # episode length
    }
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from tqdm import tqdm

from balarl.env.balatro_env import BalatroEnv
from balarl.agents.expert import ExpertAgent


def collect_trajectory(env: BalatroEnv, agent: ExpertAgent, max_steps: int = 2000) -> Dict[str, Any] | None:
    """Run one full expert episode and collect (obs, action) trajectory.

    Returns None if the episode failed early (ante < 3).
    """
    obs, info = env.reset()
    agent._target_cards = None
    agent._select_idx = 0
    agent._shop_bought = False

    observations: List[Dict[str, np.ndarray]] = []
    actions: List[int] = []
    total_reward = 0.0
    steps = 0

    while steps < max_steps:
        action = agent.act(obs)
        observations.append(_serialize_obs(obs))
        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    final_ante = int(obs.get("ante", 1))
    won = info.get("won", False)

    # Only keep trajectories with meaningful progress (ante >= 3)
    if final_ante < 3 and not won:
        return None

    return {
        "observations": observations,
        "actions": actions,
        "final_ante": final_ante,
        "total_reward": total_reward,
        "steps": steps,
        "won": won,
    }


def _serialize_obs(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Copy observation dict to prevent mutation across steps."""
    return {k: np.asarray(v) for k, v in obs.items()}


def main():
    parser = argparse.ArgumentParser(description="Generate expert trajectories for BC")
    parser.add_argument("--n-episodes", type=int, default=1000, help="Number of episodes to attempt")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--min-ante", type=int, default=3, help="Minimum ante to keep trajectory")
    parser.add_argument("--save", type=str, default="trajectories/expert_trajectories.pkl", help="Output file")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    args = parser.parse_args()

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    trajectories = []
    attempts = 0
    kept = 0
    antes = []

    print(f"Generating trajectories (target: {args.n_episodes} kept, min_ante={args.min_ante})...")
    t0 = time.perf_counter()

    iterator = range(args.n_episodes)
    if not args.no_progress:
        try:
            iterator = tqdm(iterator, desc="Trajectories", unit="ep")
        except Exception:
            pass

    for i in iterator:
        env = BalatroEnv(seed=args.seed + i * 100)
        agent = ExpertAgent(seed=args.seed + i * 100 + 1)

        traj = collect_trajectory(env, agent, args.max_steps)
        attempts += 1

        if traj is not None:
            trajectories.append(traj)
            kept += 1
            antes.append(traj["final_ante"])

        # Progress update every 100 episodes
        if attempts % 100 == 0 and kept > 0:
            avg_ante = np.mean(antes[-min(20, len(antes)):])
            if not args.no_progress:
                print(f"  {attempts} attempted, {kept} kept, "
                      f"avg_ante={avg_ante:.1f}, "
                      f"win_rate={sum(1 for t in trajectories[-100:] if t['won']) / min(100, len(trajectories[-100:])):.1%}")

        if kept >= args.n_episodes:
            break

    elapsed = time.perf_counter() - t0

    # Save
    with open(save_path, "wb") as f:
        pickle.dump(trajectories, f)

    avg_ante = np.mean(antes) if antes else 0
    total_steps = sum(t["steps"] for t in trajectories)

    print(f"\n{'='*50}")
    print(f"  Trajectories saved: {save_path}")
    print(f"  Kept: {kept} / {attempts} attempted ({kept/max(1,attempts):.1%})")
    print(f"  Avg ante: {avg_ante:.1f}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Wins: {sum(1 for t in trajectories if t['won'])}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
