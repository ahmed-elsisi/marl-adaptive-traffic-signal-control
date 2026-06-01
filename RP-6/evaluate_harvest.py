"""
Evaluate a trained Harvest checkpoint (MAPPO or IPPO) over N deterministic episodes.

Mirrors RP-5/evaluate.py's discipline:
  - Registers both Phase-2 model classes (mappo_cnn_centralized,
    ippo_cnn_decentralized) before PPO.from_checkpoint, so the same script
    handles either algorithm. The checkpoint's params.json picks the right one.
  - Manually applies the MeanStdFilter at action selection: with RLlib 2.35's
    enable_connectors=True default, compute_single_action does NOT run the
    filter. Without manual application, the deterministic policy collapses.
    (Same gotcha as Phase-1; see CLAUDE.md "Eval Gotchas".)

Outputs per episode (under --out-dir, default RP-6/metrics/):
  - {algo}_ep{i}_summary.csv   : one-row summary (apples, gini, sustainability,
                                 time-to-depletion, equality, per-agent counts)
  - {algo}_ep{i}_timeseries.csv: per-step (apples_on_grid, team_reward)
And a single aggregate CSV across episodes:
  - {algo}_aggregate.csv       : one row per episode + a mean row

Run:
    python evaluate_harvest.py --checkpoint results/mappo_harvest/PPO_harvest_<id>/checkpoint_000XYZ \\
                                --episodes 5 --seed 42
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from marl_env.harvest_env import HarvestEnv
from marl_env.harvest_metrics import HarvestMetricsCollector
from models.mappo_cnn_model import MAPPOCNNModelCentralizedCritic
from models.ippo_cnn_model import IPPOCNNModelDecentralizedCritic


# ── Filter helpers (mirrors RP-5/evaluate.py:413-446) ─────────────────────────


def _get_obs_filter(algo):
    """Find the shared_policy's MeanStdFilter on the local worker. Returns
    None if no filter (or NoFilter) is registered.

    Tries multiple attribute chains because RLlib 2.35 moved local_worker
    between Algorithm.workers and Algorithm.env_runner_group in different
    minor versions.
    """
    local_worker = None
    for attr_chain in [
        lambda a: a.env_runner_group.local_env_runner,
        lambda a: a.workers.local_worker(),
        lambda a: a.workers.local_env_runner,
    ]:
        try:
            local_worker = attr_chain(algo)
            if local_worker is not None:
                break
        except Exception:
            continue
    if local_worker is None:
        return None
    try:
        candidate = local_worker.filters.get("shared_policy")
        if candidate is not None and type(candidate).__name__ != "NoFilter":
            return candidate
    except Exception:
        return None
    return None


# ── Evaluation core ───────────────────────────────────────────────────────────


def run_episode(
    env: HarvestEnv,
    algo,
    obs_filter,
    episode_idx: int,
) -> Dict:
    """Run one deterministic episode. Returns the summary dict."""
    obs, infos = env.reset()
    collector = HarvestMetricsCollector(env.agent_ids)
    max_capacity = env.grid_height * env.grid_width - env.num_agents
    collector.start_episode(env.initial_apple_count, max_capacity)

    step = 0
    done = False
    episode_reward = {a: 0.0 for a in env.agent_ids}

    while not done:
        actions = {}
        for agent_id in env.agent_ids:
            agent_obs = obs[agent_id]
            if obs_filter is not None:
                agent_obs = obs_filter(agent_obs, update=False)
            actions[agent_id] = algo.compute_single_action(
                agent_obs,
                policy_id="shared_policy",
                explore=False,
            )
        obs, rewards, terms, truncs, infos = env.step(actions)
        collector.record_step(infos, rewards)
        for a in env.agent_ids:
            episode_reward[a] += float(rewards.get(a, 0.0))
        step += 1
        done = bool(terms.get("__all__", False))

    summary = collector.summarize(env.episode_length)
    summary["episode_index"] = episode_idx
    summary["steps_run"] = step
    summary["sum_per_agent_reward"] = float(sum(episode_reward.values()))
    summary["_collector"] = collector  # caller can save time-series
    return summary


# ── CSV emission ──────────────────────────────────────────────────────────────


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
    """One row per episode + a 'mean' row at the bottom."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = list(_AGGREGATE_FIELDS) + [f"apples_{a}" for a in agent_ids]
    rows = [_flatten_row(s, agent_ids) for s in summaries]

    # Build mean row (numeric columns only; episode_index becomes "mean")
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
    checkpoint_path: str,
    num_episodes: int,
    seed: int,
    out_dir: str,
    algo_tag: str,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Register both Phase-2 models so checkpoints from either algorithm load.
    tune.register_env("harvest", lambda cfg: HarvestEnv(cfg))
    ModelCatalog.register_custom_model("mappo_cnn_centralized", MAPPOCNNModelCentralizedCritic)
    ModelCatalog.register_custom_model("ippo_cnn_decentralized", IPPOCNNModelDecentralizedCritic)

    checkpoint_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)

    print("=" * 80)
    print("HARVEST EVALUATION")
    print("=" * 80)
    print(f"Checkpoint : {checkpoint_dir}")
    print(f"Episodes   : {num_episodes}")
    print(f"Seed       : {seed}")
    print(f"Out dir    : {out_dir}")
    print(f"Algo tag   : {algo_tag}")
    print("=" * 80)

    algo = PPO.from_checkpoint(checkpoint_dir)
    print("Checkpoint loaded.")

    obs_filter = _get_obs_filter(algo)
    if obs_filter is not None:
        rs = getattr(obs_filter, "running_stats", None) or getattr(obs_filter, "rs", None)
        n = getattr(rs, "num_pushes", None) if rs is not None else None
        if n is None:
            n = getattr(rs, "n", None) if rs is not None else None
        print(f"Obs filter detected: {type(obs_filter).__name__}, running_stats count={n}")
        if n in (None, 0):
            print("WARN: filter has no accumulated stats — checkpoint may not have synced filter state.")
    else:
        print("WARN: no MeanStdFilter detected on local worker. Obs passed raw.")

    # Pull env_config from the loaded algo so eval matches training.
    config_dict = algo.config.to_dict()
    env_cfg = dict(config_dict.get("env_config", {}))
    # Determinism: override seed for reproducibility across eval runs.
    env_cfg["seed"] = seed

    summaries: List[Dict] = []
    for ep in range(num_episodes):
        ep_cfg = dict(env_cfg)
        ep_cfg["seed"] = seed + ep   # vary seed per episode to sample env stochasticity
        env = HarvestEnv(ep_cfg)
        print(f"\n--- Episode {ep + 1}/{num_episodes} (env seed={ep_cfg['seed']}) ---")
        summary = run_episode(env, algo, obs_filter, episode_idx=ep + 1)

        collector = summary.pop("_collector")
        ts_path = out_dir / f"{algo_tag}_ep{ep + 1}_timeseries.csv"
        summary_path = out_dir / f"{algo_tag}_ep{ep + 1}_summary.csv"
        collector.save_timeseries_csv(str(ts_path))
        collector.save_episode_csv(summary, str(summary_path))

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

    # Print mean summary to stdout for quick reading.
    print("\n" + "=" * 80)
    print(f"MEAN OVER {num_episodes} EPISODES")
    print("=" * 80)
    for k in ("total_apples_collected", "gini_coefficient", "equality",
              "sustainability", "time_to_depletion", "total_team_reward"):
        vals = [float(s[k]) for s in summaries]
        print(f"  {k:<25}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
    print("=" * 80)

    return summaries


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Harvest checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory (or a checkpoint_XYZ subdir).")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of deterministic eval episodes.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base env seed; episode i uses seed+i.")
    parser.add_argument("--out-dir", type=str, default="metrics",
                        help="Directory to write per-episode CSVs into.")
    parser.add_argument("--algo", type=str, default=None,
                        choices=["mappo", "ippo"],
                        help="Algorithm tag for output filenames. Inferred from checkpoint path if omitted.")
    args = parser.parse_args()

    # Infer algo tag from checkpoint path if not given.
    algo_tag = args.algo
    if algo_tag is None:
        cp_lower = args.checkpoint.lower().replace("\\", "/")
        if "ippo_harvest" in cp_lower:
            algo_tag = "ippo"
        elif "mappo_harvest" in cp_lower:
            algo_tag = "mappo"
        else:
            algo_tag = "harvest"

    try:
        evaluate(
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes,
            seed=args.seed,
            out_dir=args.out_dir,
            algo_tag=algo_tag,
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
