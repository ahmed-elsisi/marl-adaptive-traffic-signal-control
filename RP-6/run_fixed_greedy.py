"""
FixedGreedy heuristic baseline for HarvestEnv.

Policy ("collect every visible apple"):
  1. If standing on an apple cell, COLLECT.
  2. Else find the nearest apple within visibility_radius (Manhattan), break
     ties lexicographically by (row, col) of the apple. Move N/S/E/W in the
     direction that reduces Manhattan distance most. When |dr| == |dc|,
     row movement is chosen first (deterministic).
  3. If no apple is visible, STAY.

This policy queries env.apple_grid and env.agent_positions directly — it does
not run through the CNN obs pipeline. That matches Phase-1's heuristic
baselines (fixed-cycles, max-pressure) which read SUMO state directly rather
than through the 70-dim obs vector. The baseline's role is to anchor "what
does a simple rule do?", not to test partial observability.

CSV output format matches evaluate_harvest.py so the aggregate tables can be
read by the same downstream analysis tools.

Run from RP-6/:
    python run_fixed_greedy.py --episodes 5 --seed 42 --out-dir metrics/fixed_greedy
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from marl_env.harvest_env import (
    HarvestEnv,
    ACTION_NORTH, ACTION_SOUTH, ACTION_EAST, ACTION_WEST,
    ACTION_STAY, ACTION_COLLECT,
)
from marl_env.harvest_metrics import HarvestMetricsCollector


# ── Policy ────────────────────────────────────────────────────────────────────


def _nearest_visible_apple(
    apple_grid: np.ndarray,
    agent_pos: Tuple[int, int],
    visibility_radius: int,
) -> Tuple[int, int]:
    """Return (row, col) of the nearest apple within Manhattan visibility_radius.

    Tie-breaking: smallest Manhattan distance first; among ties, lexicographic
    by (row, col). Returns None if no apple in range.
    """
    r0, c0 = agent_pos
    h, w = apple_grid.shape
    r_min = max(0, r0 - visibility_radius)
    r_max = min(h, r0 + visibility_radius + 1)
    c_min = max(0, c0 - visibility_radius)
    c_max = min(w, c0 + visibility_radius + 1)

    best = None
    best_d = visibility_radius + 1
    for r in range(r_min, r_max):
        for c in range(c_min, c_max):
            if not apple_grid[r, c]:
                continue
            d = abs(r - r0) + abs(c - c0)
            if d < best_d or (d == best_d and best is not None and (r, c) < best):
                best = (r, c)
                best_d = d
    return best


def fixed_greedy_action(
    agent_pos: Tuple[int, int],
    apple_grid: np.ndarray,
    visibility_radius: int,
) -> int:
    r, c = agent_pos
    if apple_grid[r, c]:
        return ACTION_COLLECT

    target = _nearest_visible_apple(apple_grid, agent_pos, visibility_radius)
    if target is None:
        return ACTION_STAY

    tr, tc = target
    dr, dc = tr - r, tc - c
    # Move along the dominant axis; row first on ties (deterministic).
    if abs(dr) >= abs(dc):
        return ACTION_SOUTH if dr > 0 else ACTION_NORTH
    return ACTION_EAST if dc > 0 else ACTION_WEST


# ── Episode runner ────────────────────────────────────────────────────────────


def run_episode(env: HarvestEnv, episode_idx: int) -> Dict:
    obs, infos = env.reset()
    collector = HarvestMetricsCollector(env.agent_ids)
    max_capacity = env.grid_height * env.grid_width - env.num_agents
    collector.start_episode(env.initial_apple_count, max_capacity)
    visibility_radius = env.obs_window // 2

    done = False
    steps = 0
    episode_reward = {a: 0.0 for a in env.agent_ids}
    while not done:
        actions = {
            agent_id: fixed_greedy_action(
                env.agent_positions[agent_id], env.apple_grid, visibility_radius
            )
            for agent_id in env.agent_ids
        }
        obs, rewards, terms, truncs, infos = env.step(actions)
        collector.record_step(infos, rewards)
        for a in env.agent_ids:
            episode_reward[a] += float(rewards.get(a, 0.0))
        steps += 1
        done = bool(terms.get("__all__", False))

    summary = collector.summarize(env.episode_length)
    summary["episode_index"] = episode_idx
    summary["steps_run"] = steps
    summary["sum_per_agent_reward"] = float(sum(episode_reward.values()))
    summary["_collector"] = collector
    return summary


# ── CSV emission (kept format-compatible with evaluate_harvest.py) ────────────


_AGGREGATE_FIELDS = [
    "episode_index",
    "total_apples_collected",
    "gini_coefficient",
    "equality",
    "sustainability",
    "time_to_depletion",
    "initial_apples",
    "episode_length",
    "total_team_reward",
    "sum_per_agent_reward",
]


def _flatten_row(s: Dict, agent_ids: List[str]) -> Dict:
    row = {k: s.get(k, "") for k in _AGGREGATE_FIELDS}
    for a in agent_ids:
        row[f"apples_{a}"] = s["apples_per_agent"].get(a, 0)
    return row


def save_aggregate_csv(summaries: List[Dict], agent_ids: List[str], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = list(_AGGREGATE_FIELDS) + [f"apples_{a}" for a in agent_ids]
    rows = [_flatten_row(s, agent_ids) for s in summaries]
    mean_row = {"episode_index": "mean"}
    for k in header:
        if k == "episode_index":
            continue
        vals = [r[k] for r in rows if isinstance(r[k], (int, float)) and r[k] != ""]
        mean_row[k] = float(np.mean(vals)) if vals else ""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        writer.writerow(mean_row)


# ── Entry point ───────────────────────────────────────────────────────────────


def evaluate(
    num_episodes: int,
    seed: int,
    out_dir: str,
    shared_reward_weight: float,
    algo_tag: str,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg_template = {
        "grid_height": 8,
        "grid_width": 12,
        "num_agents": 4,
        "episode_length": 1000,
        "obs_window": 15,
        "initial_apple_density": 0.3,
        "apple_regrowth_base": 0.01,
        "apple_regrowth_radius": 2,
        "shared_reward_weight": shared_reward_weight,
    }

    print("=" * 80)
    print("FIXED-GREEDY HARVEST BASELINE")
    print("=" * 80)
    print(f"Episodes              : {num_episodes}")
    print(f"Base seed             : {seed}  (episode i uses seed+i)")
    print(f"shared_reward_weight  : {shared_reward_weight}")
    print(f"Out dir               : {out_dir}")
    print(f"Algo tag              : {algo_tag}")
    print("=" * 80)

    summaries: List[Dict] = []
    for ep in range(num_episodes):
        ep_cfg = dict(env_cfg_template)
        ep_cfg["seed"] = seed + ep
        env = HarvestEnv(ep_cfg)
        print(f"\n--- Episode {ep + 1}/{num_episodes} (env seed={ep_cfg['seed']}) ---")
        summary = run_episode(env, episode_idx=ep + 1)

        collector = summary.pop("_collector")
        collector.save_timeseries_csv(str(out_dir / f"{algo_tag}_ep{ep + 1}_timeseries.csv"))
        collector.save_episode_csv(summary, str(out_dir / f"{algo_tag}_ep{ep + 1}_summary.csv"))

        print(
            f"  total_apples={summary['total_apples_collected']}, "
            f"gini={summary['gini_coefficient']:.3f}, "
            f"sustain={summary['sustainability']:.3f}, "
            f"t_depl={summary['time_to_depletion']}"
        )
        summaries.append(summary)

    aggregate_path = out_dir / f"{algo_tag}_aggregate.csv"
    save_aggregate_csv(summaries, list(summaries[0]["apples_per_agent"].keys()), str(aggregate_path))
    print(f"\nWrote aggregate CSV: {aggregate_path}")

    print("\n" + "=" * 80)
    print(f"MEAN OVER {num_episodes} EPISODES")
    print("=" * 80)
    for k in ("total_apples_collected", "gini_coefficient", "equality",
              "sustainability", "time_to_depletion", "total_team_reward"):
        vals = [float(s[k]) for s in summaries]
        print(f"  {k:<25}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run FixedGreedy baseline on HarvestEnv")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42,
                        help="Base env seed; episode i uses seed+i.")
    parser.add_argument("--out-dir", type=str, default="metrics/fixed_greedy")
    parser.add_argument("--shared-reward-weight", type=float, default=0.0,
                        help="Env shared_reward_weight. FixedGreedy ignores it for action choice, but it affects the team_reward metric reported in CSVs.")
    parser.add_argument("--algo-tag", type=str, default="fixed_greedy",
                        help="Filename prefix for output CSVs.")
    args = parser.parse_args()

    evaluate(
        num_episodes=args.episodes,
        seed=args.seed,
        out_dir=args.out_dir,
        shared_reward_weight=args.shared_reward_weight,
        algo_tag=args.algo_tag,
    )


if __name__ == "__main__":
    main()
