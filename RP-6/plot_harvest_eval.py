"""
Visualise Harvest evaluation runs from the CSVs emitted by
evaluate_harvest.py and run_fixed_greedy.py.

Inputs:
  One or more (label, metrics_dir) pairs. Each dir is expected to contain
    {algo_tag}_aggregate.csv
    {algo_tag}_ep{i}_timeseries.csv
  for one algo_tag (multiple algo_tags per dir are also handled).

Outputs (PNG, written to --out-dir):
  apples_on_grid_timeseries.png : the commons trajectory. Flat-near-zero
                                  = tragedy of the commons. High and
                                  stable = cooperation preserved.
  metrics_bars.png              : aggregate bars across the input dirs
                                  for total_apples, gini, sustainability,
                                  time_to_depletion (mean +- std across
                                  episodes per dir).
  per_agent_apples.png          : per-agent cumulative apples (best
                                  episode per dir) -- shows exploitation
                                  patterns (e.g. one agent hoards).

Run from RP-6/:
    python plot_harvest_eval.py \\
        --inputs "FixedGreedy:metrics/smoke_fixed_greedy" \\
                 "IPPO smoke:metrics/smoke_eval" \\
        --out-dir metrics/plots

    # Or simpler: one input gets per-dir-only plots (no cross-comparison)
    python plot_harvest_eval.py --inputs "FixedGreedy:metrics/smoke_fixed_greedy"
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── CSV discovery ────────────────────────────────────────────────────────────


_TS_RE   = re.compile(r"^(.+)_ep(\d+)_timeseries\.csv$")
_SUM_RE  = re.compile(r"^(.+)_ep(\d+)_summary\.csv$")
_AGG_RE  = re.compile(r"^(.+)_aggregate\.csv$")


def _discover(metrics_dir: Path) -> Dict[str, Dict]:
    """Group CSVs by algo_tag inside metrics_dir.

    Returns: {algo_tag: {"aggregate": path, "timeseries": {ep: path},
                          "summaries":  {ep: path}}}
    """
    tags: Dict[str, Dict] = defaultdict(lambda: {"timeseries": {}, "summaries": {}, "aggregate": None})
    if not metrics_dir.is_dir():
        raise FileNotFoundError(f"metrics dir not found: {metrics_dir}")
    for p in metrics_dir.iterdir():
        m = _TS_RE.match(p.name)
        if m:
            tags[m.group(1)]["timeseries"][int(m.group(2))] = p
            continue
        m = _SUM_RE.match(p.name)
        if m:
            tags[m.group(1)]["summaries"][int(m.group(2))] = p
            continue
        m = _AGG_RE.match(p.name)
        if m:
            tags[m.group(1)]["aggregate"] = p
    return dict(tags)


# ── Per-input panels ────────────────────────────────────────────────────────


def _load_timeseries(ts_paths: Dict[int, Path]) -> Dict[int, pd.DataFrame]:
    return {ep: pd.read_csv(p) for ep, p in sorted(ts_paths.items())}


def _stack_apples(ts_by_ep: Dict[int, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (steps array, apples matrix [n_episodes, n_steps]). Truncates
    to the shortest episode length so the stack is rectangular."""
    eps = sorted(ts_by_ep.keys())
    min_len = min(len(ts_by_ep[e]) for e in eps)
    apples = np.stack([ts_by_ep[e]["apples_on_grid"].values[:min_len] for e in eps])
    steps = ts_by_ep[eps[0]]["step"].values[:min_len]
    return steps, apples


# ── Plot 1: apples-on-grid timeseries (overlay across inputs) ───────────────


def plot_apples_on_grid(
    per_label_apples: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_path: Path,
):
    """One line per label = mean across episodes; shaded band = +-1 std."""
    fig, ax = plt.subplots(figsize=(9, 5))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (label, (steps, apples)) in enumerate(per_label_apples.items()):
        c = color_cycle[i % len(color_cycle)]
        mean = apples.mean(axis=0)
        std = apples.std(axis=0)
        ax.plot(steps, mean, label=f"{label} (n={apples.shape[0]} ep)", color=c, lw=2)
        ax.fill_between(steps, mean - std, mean + std, color=c, alpha=0.18)

    ax.set_xlabel("Step")
    ax.set_ylabel("Apples on grid")
    ax.set_title("Commons trajectory (apples on grid over time)")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── Plot 2: aggregate metric bars across inputs ─────────────────────────────


_AGG_METRICS = [
    ("total_apples_collected", "Total apples"),
    ("sustainability",         "Sustainability"),
    ("gini_coefficient",       "Gini (lower = more equal)"),
    ("time_to_depletion",      "Time to depletion"),
]


def plot_metric_bars(
    per_label_agg: Dict[str, pd.DataFrame],
    out_path: Path,
):
    """4-panel grid of metric bars (mean +- std) across inputs."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, (col, title) in zip(axes, _AGG_METRICS):
        labels = list(per_label_agg.keys())
        means, stds = [], []
        for label in labels:
            df = per_label_agg[label]
            # Skip the trailing "mean" row.
            episode_rows = df[df["episode_index"].apply(lambda x: str(x).isdigit())]
            vals = pd.to_numeric(episode_rows[col], errors="coerce").dropna().values
            means.append(vals.mean() if len(vals) else 0.0)
            stds.append(vals.std() if len(vals) else 0.0)
        xs = np.arange(len(labels))
        ax.bar(xs, means, yerr=stds, capsize=5,
               color=[color_cycle[i % len(color_cycle)] for i in range(len(labels))],
               edgecolor="black", linewidth=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Aggregate evaluation metrics (mean +- std across episodes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── Plot 3: per-agent cumulative apples per input ───────────────────────────


def plot_per_agent_apples(
    per_label_agg: Dict[str, pd.DataFrame],
    out_path: Path,
):
    """Grouped bar chart: per-agent mean apples (across episodes) per label.

    Shows the exploitation pattern: if one agent dominates the bar, the
    other agents are being out-competed."""
    fig, ax = plt.subplots(figsize=(9, 5))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    labels = list(per_label_agg.keys())
    agent_cols = None
    for df in per_label_agg.values():
        cs = [c for c in df.columns if c.startswith("apples_A")]
        if agent_cols is None or len(cs) > len(agent_cols):
            agent_cols = sorted(cs)
    if not agent_cols:
        print("WARN: no per-agent apples_A* columns in any input; skipping per-agent plot.")
        return

    n_labels = len(labels)
    n_agents = len(agent_cols)
    bar_w = 0.8 / n_agents
    xs = np.arange(n_labels)

    for ai, col in enumerate(agent_cols):
        means = []
        stds = []
        for label in labels:
            df = per_label_agg[label]
            episode_rows = df[df["episode_index"].apply(lambda x: str(x).isdigit())]
            vals = pd.to_numeric(episode_rows[col], errors="coerce").dropna().values
            means.append(vals.mean() if len(vals) else 0.0)
            stds.append(vals.std() if len(vals) else 0.0)
        offset = (ai - (n_agents - 1) / 2.0) * bar_w
        ax.bar(xs + offset, means, width=bar_w, yerr=stds,
               capsize=3, label=col.replace("apples_", ""),
               color=color_cycle[ai % len(color_cycle)],
               edgecolor="black", linewidth=0.4)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Apples (mean across episodes)")
    ax.set_title("Per-agent apple counts (exploitation patterns)")
    ax.legend(title="Agent", loc="best", frameon=True)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── Driver ───────────────────────────────────────────────────────────────────


def _parse_input(raw: str) -> Tuple[str, Path]:
    """Parse 'Label:path/to/dir'. If no colon, use the dir basename as label."""
    if ":" in raw and not raw[1:2] == ":":  # not a Windows drive like "C:..."
        label, path = raw.split(":", 1)
    else:
        # Windows-friendly fallback: search for last ':' after position 2
        idx = raw.rfind(":")
        if idx > 1:
            label, path = raw[:idx], raw[idx + 1:]
        else:
            path = raw
            label = Path(raw).name
    return label.strip(), Path(path.strip()).resolve()


def main():
    parser = argparse.ArgumentParser(description="Plot Harvest evaluation results from eval CSVs.")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="One or more 'Label:metrics_dir' pairs (e.g. 'FixedGreedy:metrics/fg').")
    parser.add_argument("--out-dir", type=str, default="metrics/plots",
                        help="Directory to write PNGs into.")
    parser.add_argument("--tag-filter", type=str, default=None,
                        help="If a dir has multiple algo_tags, restrict to this one.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_label_apples: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    per_label_agg: Dict[str, pd.DataFrame] = {}

    for raw in args.inputs:
        label, metrics_dir = _parse_input(raw)
        tags = _discover(metrics_dir)
        if not tags:
            print(f"WARN: no eval CSVs in {metrics_dir}; skipping {label}")
            continue
        if args.tag_filter:
            tags = {k: v for k, v in tags.items() if k == args.tag_filter}
        for tag, files in tags.items():
            full_label = label if len(tags) == 1 else f"{label} ({tag})"

            # Timeseries: stack per-episode apples-on-grid
            if files["timeseries"]:
                steps, apples = _stack_apples(_load_timeseries(files["timeseries"]))
                per_label_apples[full_label] = (steps, apples)
            else:
                print(f"WARN: no timeseries CSVs for {full_label}")

            # Aggregate
            if files["aggregate"]:
                per_label_agg[full_label] = pd.read_csv(files["aggregate"])
            else:
                print(f"WARN: no aggregate CSV for {full_label}")

    if not per_label_apples and not per_label_agg:
        print("ERROR: no usable data found in inputs.")
        return

    if per_label_apples:
        out = out_dir / "apples_on_grid_timeseries.png"
        plot_apples_on_grid(per_label_apples, out)
        print(f"wrote {out}")
    if per_label_agg:
        out = out_dir / "metrics_bars.png"
        plot_metric_bars(per_label_agg, out)
        print(f"wrote {out}")
        out = out_dir / "per_agent_apples.png"
        plot_per_agent_apples(per_label_agg, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
