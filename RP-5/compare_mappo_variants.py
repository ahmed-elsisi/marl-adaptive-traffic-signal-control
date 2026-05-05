"""
Compare two trained MAPPO variants on the same environment.

Designed for Task 1.1 (Semester 2): Our MAPPO vs Paper Baseline MAPPO
(Yu et al., 2021 — "The Surprising Effectiveness of PPO in Cooperative
Multi-Agent Games")

Both checkpoints are evaluated on the same SUMO environment with the same
seed so that only the trained policy weights differ.

Usage:
    python compare_mappo_variants.py \\
        --checkpoint-a results/mappo_traffic_control/<our_run> \\
        --label-a "Our MAPPO" \\
        --checkpoint-b results/mappo_traffic_control/<paper_baseline_run> \\
        --label-b "Paper Baseline MAPPO" \\
        --episodes 3 --seed 42

Outputs (saved to --results-dir, default: metrics/):
    variant_comparison_timeseries.png  — overlaid time-series (4 metrics)
    variant_comparison_summary.png     — bar chart summary
    variant_comparison_table.csv       — numeric results + improvement %
"""

import os
import sys
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marl_env.sumo_env import SUMOTrafficEnv
from models.mappo_model import MAPPOModelCentralizedCritic
from evaluate import MetricsCollector, EnvWrapperWithMetrics, load_config

try:
    import libsumo as traci
    print("Using libsumo")
except ImportError:
    import traci
    print("Using standard TraCI")


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def resolve_checkpoint(path: str) -> str:
    """Accept either a run directory or a checkpoint subdirectory."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")
    # If the user passed a run directory, pick the latest checkpoint inside it
    if p.is_dir() and not (p / "rllib_checkpoint.json").exists():
        checkpoints = sorted(p.glob("checkpoint_*"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint_* directories found in: {path}")
        return str(checkpoints[-1])
    return str(p)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_variant(
    checkpoint_path: str,
    label: str,
    env_config: dict,
    num_episodes: int,
    seed: int,
) -> dict:
    """
    Load a checkpoint and run num_episodes evaluation episodes.

    Returns a dict with time-series lists and summary stats.
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {label}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*70}")

    env_config = dict(env_config)
    env_config["sumo_seed"] = seed

    algo = PPO.from_checkpoint(checkpoint_path)
    print(f"  Checkpoint loaded.")

    # With enable_connectors=True (PPO default in RLlib 2.35),
    # Algorithm.compute_single_action does NOT apply MeanStdFilter — only the
    # ObsPreprocessorConnector runs. v2 (and paper_baseline) train with
    # observation_filter: "MeanStdFilter", so without manual application the
    # deterministic policy receives raw obs and collapses to a single argmax.
    obs_filter = None
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

    if local_worker is not None:
        try:
            candidate = local_worker.filters.get("shared_policy")
            if candidate is not None and type(candidate).__name__ != "NoFilter":
                obs_filter = candidate
        except Exception as e:
            print(f"  ⚠ Could not retrieve obs filter: {e}")

    if obs_filter is not None:
        rs = getattr(obs_filter, "running_stats", None) or getattr(obs_filter, "rs", None)
        n = getattr(rs, "num_pushes", None) if rs is not None else None
        if n is None:
            n = getattr(rs, "n", None) if rs is not None else None
        print(f"  ✓ Obs filter: {type(obs_filter).__name__}, running_stats count={n}")
        if n in (None, 0):
            print("  ⚠ Filter has no accumulated stats — checkpoint may not have synced filter state.")
    else:
        print("  ℹ No MeanStdFilter detected; observations passed through raw.")

    all_times = None
    all_halts = []
    all_arrivals = []
    all_wait = []
    all_active = []
    episode_rewards = []
    all_summary = []

    for ep in range(num_episodes):
        print(f"\n  Episode {ep + 1}/{num_episodes}")

        base_env = SUMOTrafficEnv(env_config)
        metrics = MetricsCollector(base_env.agent_ids)
        env = EnvWrapperWithMetrics(base_env, metrics)

        obs, _ = env.reset()
        ep_reward = {a: 0.0 for a in env.agent_ids}
        done = False

        while not done:
            actions = {}
            for a in env.agent_ids:
                agent_obs = obs[a]
                if obs_filter is not None:
                    agent_obs = obs_filter(agent_obs, update=False)
                actions[a] = algo.compute_single_action(
                    agent_obs, policy_id="shared_policy", explore=False
                )
            obs, rewards, terminateds, _, _ = env.step(actions)
            for a in env.agent_ids:
                ep_reward[a] += rewards[a]
            done = terminateds["__all__"]

        env.close()

        stats = metrics.get_summary_stats()
        total_reward = sum(ep_reward.values())
        episode_rewards.append(total_reward)
        all_summary.append(stats)

        # Accumulate time-series (align to shortest episode)
        if all_times is None:
            all_times = metrics.times[:]
        all_halts.append(metrics.total_halts)
        all_arrivals.append(metrics.cumulative_arrivals)
        all_wait.append(metrics.running_avg_wait)
        all_active.append(metrics.active_vehicles)

        print(f"    Reward:   {total_reward:.2f}")
        print(f"    Arrivals: {stats['total_arrivals']}")
        print(f"    Avg Wait: {stats['avg_waiting_time']:.2f}s")
        print(f"    Avg Halt: {stats['avg_halting']:.1f}")

    algo.stop()

    # Average time-series across episodes (trim to shortest)
    min_len = min(len(h) for h in all_halts)
    times = all_times[:min_len]
    avg_halts    = np.mean([h[:min_len] for h in all_halts], axis=0).tolist()
    avg_arrivals = np.mean([a[:min_len] for a in all_arrivals], axis=0).tolist()
    avg_wait     = np.mean([w[:min_len] for w in all_wait], axis=0).tolist()
    avg_active   = np.mean([a[:min_len] for a in all_active], axis=0).tolist()

    summary = {
        "mean_reward":   float(np.mean(episode_rewards)),
        "std_reward":    float(np.std(episode_rewards)),
        "mean_arrivals": float(np.mean([s["total_arrivals"]    for s in all_summary])),
        "mean_wait":     float(np.mean([s["avg_waiting_time"]  for s in all_summary])),
        "mean_avg_halt": float(np.mean([s["avg_halting"]       for s in all_summary])),
        "mean_peak_halt":float(np.mean([s["max_halting"]       for s in all_summary])),
    }

    return {
        "label":    label,
        "times":    times,
        "halts":    avg_halts,
        "arrivals": avg_arrivals,
        "wait":     avg_wait,
        "active":   avg_active,
        "summary":  summary,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {"a": "#2196F3", "b": "#F44336"}   # blue = A, red = B


def plot_timeseries(a: dict, b: dict, results_dir: str):
    """2×2 overlaid time-series comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Time-Series Comparison\n{a['label']}  vs  {b['label']}",
        fontsize=13, fontweight="bold"
    )

    panels = [
        (axes[0, 0], "halts",    "Total Halting Vehicles",   "Vehicles"),
        (axes[0, 1], "arrivals", "Cumulative Arrivals",       "Vehicles"),
        (axes[1, 0], "wait",     "Running Avg Wait Time (s)", "Seconds"),
        (axes[1, 1], "active",   "Active Vehicles",           "Vehicles"),
    ]

    for ax, key, title, ylabel in panels:
        ax.plot(a["times"], a[key], color=COLORS["a"], linewidth=1.4, label=a["label"])
        ax.plot(b["times"], b[key], color=COLORS["b"], linewidth=1.4,
                linestyle="--", label=b["label"], alpha=0.85)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "variant_comparison_timeseries.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_summary_bars(a: dict, b: dict, results_dir: str):
    """Bar chart comparing four key metrics."""
    sa, sb = a["summary"], b["summary"]
    labels = [a["label"], b["label"]]
    colors = [COLORS["a"], COLORS["b"]]

    metrics = [
        ("mean_arrivals",  "Total Arrivals",        "Vehicles",  True),
        ("mean_wait",      "Avg Waiting Time",       "Seconds",   False),
        ("mean_avg_halt",  "Avg Halting Vehicles",   "Vehicles",  False),
        ("mean_peak_halt", "Peak Halting Vehicles",  "Vehicles",  False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        f"Summary Comparison (avg over episodes)\n"
        f"{a['label']}  vs  {b['label']}",
        fontsize=12, fontweight="bold"
    )

    for ax, (key, title, ylabel, higher_better) in zip(axes, metrics):
        values = [sa[key], sb[key]]
        bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, rotation=10, ha="right")

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

        # Annotate which direction is better
        note = "↑ better" if higher_better else "↓ better"
        ax.text(0.98, 0.98, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="gray")

    plt.tight_layout()
    path = os.path.join(results_dir, "variant_comparison_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def save_table(a: dict, b: dict, results_dir: str):
    """Save numeric comparison table to CSV."""
    sa, sb = a["summary"], b["summary"]

    rows = [
        ("Metric",              a["label"],     b["label"],         "Improvement (A vs B)"),
        ("Mean Reward",         f"{sa['mean_reward']:.2f}",   f"{sb['mean_reward']:.2f}",
            _pct(sa["mean_reward"],   sb["mean_reward"],   higher_better=True)),
        ("Mean Arrivals",       f"{sa['mean_arrivals']:.0f}", f"{sb['mean_arrivals']:.0f}",
            _pct(sa["mean_arrivals"], sb["mean_arrivals"], higher_better=True)),
        ("Avg Wait (s)",        f"{sa['mean_wait']:.2f}",     f"{sb['mean_wait']:.2f}",
            _pct(sa["mean_wait"],     sb["mean_wait"],     higher_better=False)),
        ("Avg Halting",         f"{sa['mean_avg_halt']:.1f}", f"{sb['mean_avg_halt']:.1f}",
            _pct(sa["mean_avg_halt"], sb["mean_avg_halt"], higher_better=False)),
        ("Peak Halting",        f"{sa['mean_peak_halt']:.0f}",f"{sb['mean_peak_halt']:.0f}",
            _pct(sa["mean_peak_halt"],sb["mean_peak_halt"],higher_better=False)),
    ]

    path = os.path.join(results_dir, "variant_comparison_table.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    print(f"  Saved: {path}")
    return rows


def _pct(val_a: float, val_b: float, higher_better: bool) -> str:
    """Return improvement of A over B as a percentage string."""
    if val_b == 0:
        return "N/A"
    if higher_better:
        pct = (val_a - val_b) / abs(val_b) * 100
    else:
        pct = (val_b - val_a) / abs(val_b) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def print_console_table(rows: list, a_label: str, b_label: str):
    """Print a formatted summary table."""
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {a_label:<22} {b_label:<22} {'A improvement'}")
    print("-" * 70)
    for row in rows[1:]:   # skip header
        print(f"{row[0]:<20} {row[1]:<22} {row[2]:<22} {row[3]}")
    print("=" * 70)

    # Declare winner
    sa_rows = rows[1:]
    a_wins = sum(
        1 for r in sa_rows
        if r[3].startswith("+") and r[3] != "N/A"
    )
    b_wins = len(sa_rows) - a_wins
    print(f"\nA ({a_label}) wins: {a_wins}/{len(sa_rows)} metrics")
    print(f"B ({b_label}) wins: {b_wins}/{len(sa_rows)} metrics")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare two trained MAPPO checkpoints on the same environment"
    )
    parser.add_argument("--checkpoint-a", required=True,
                        help="Checkpoint path for variant A (our MAPPO)")
    parser.add_argument("--label-a", default="Our MAPPO",
                        help="Display label for variant A")
    parser.add_argument("--checkpoint-b", required=True,
                        help="Checkpoint path for variant B (paper baseline)")
    parser.add_argument("--label-b", default="Paper Baseline MAPPO",
                        help="Display label for variant B")
    parser.add_argument("--config", default="configs/mappo_config_v2.yaml",
                        help="Environment config (same for both variants). "
                             "Default = v2 because it carries the edge_connectivity "
                             "block needed for v2's neighbour-pressure obs features; "
                             "paper_baseline's actor architecture doesn't depend on it, "
                             "so this is the fair shared env.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="metrics")
    args = parser.parse_args()

    results_dir = ensure_dir(args.results_dir)

    # Resolve checkpoint directories
    ckpt_a = resolve_checkpoint(args.checkpoint_a)
    ckpt_b = resolve_checkpoint(args.checkpoint_b)

    # Load shared environment config
    config = load_config(args.config)
    env_config = {
        "agents":            config["agents"],
        "network_topology":  config["network_topology"],
        "detectors":         config["detectors"],
        "normalization":     config["normalization"],
        "reward_config":     config["reward_config"],
        # v2 obs builder reads neighbour outgoing/ingoing edges from this block;
        # missing → 24 of 70 obs dims silently zero out → policy collapse at eval.
        "edge_connectivity": config.get("edge_connectivity", {}),
        "network_file":      config["env_config"]["network_file"],
        "route_file":        config["env_config"]["route_file"],
        "use_gui":           False,
        "num_seconds":       config["env_config"].get("num_seconds", 3600),
        "delta_time":        config["env_config"].get("delta_time", 5),
        "yellow_time":       config["env_config"].get("yellow_time", 3),
        "min_green":         config["env_config"].get("min_green", 10),
        "max_green":         config["env_config"].get("max_green", 50),
        "enforce_min_green": config["env_config"].get("enforce_min_green", False),
        "enforce_min_red":   config["env_config"].get("enforce_min_red", True),
        "min_red":           config["env_config"].get("min_red", 1),
    }

    # Init Ray + register model
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    tune.register_env("sumo_traffic", lambda cfg: SUMOTrafficEnv(cfg))
    ModelCatalog.register_custom_model("mappo_centralized", MAPPOModelCentralizedCritic)

    print(f"\n{'='*70}")
    print("MAPPO VARIANT COMPARISON")
    print(f"  A: {args.label_a}")
    print(f"     {ckpt_a}")
    print(f"  B: {args.label_b}")
    print(f"     {ckpt_b}")
    print(f"  Episodes: {args.episodes}  |  Seed: {args.seed}")
    print(f"  Env config: {args.config}")
    print(f"{'='*70}")

    # Evaluate both variants
    result_a = evaluate_variant(ckpt_a, args.label_a, env_config, args.episodes, args.seed)
    result_b = evaluate_variant(ckpt_b, args.label_b, env_config, args.episodes, args.seed)

    # Generate outputs
    print(f"\n{'='*70}")
    print("GENERATING OUTPUTS")
    print(f"{'='*70}")
    plot_timeseries(result_a, result_b, results_dir)
    plot_summary_bars(result_a, result_b, results_dir)
    rows = save_table(result_a, result_b, results_dir)

    print_console_table(rows, args.label_a, args.label_b)

    print(f"\nAll outputs saved to: {results_dir}/")
    ray.shutdown()


if __name__ == "__main__":
    main()
