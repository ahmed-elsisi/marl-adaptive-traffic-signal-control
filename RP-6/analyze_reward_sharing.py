"""
Phase-2 reward-sharing analysis (the load-bearing Harvest result).

Answers thesis question #1 -- "how do individual vs shared rewards affect
emergent cooperation?" -- using the completed Harvest training matrix.

Statistical unit of analysis = the TRAINING SEED. Each cell
(metrics/eval/<config>_seed<N>/<algo>_aggregate.csv) holds several deterministic
eval episodes for one trained policy; we collapse those to a per-seed mean, so
each condition contributes n = (#seeds) independent samples. Welch's t-test
(unequal variance) then compares conditions on each metric. With the full 5-seed
MAPPO matrix this comparison is fully powered (unlike the deadline-reduced IPPO
half).

Metrics (from harvest_metrics.py):
  total_apples_collected : productivity (higher = better)
  sustainability         : fraction of episode the commons stays productive
  gini_coefficient       : inequality (LOWER = more equal)
  time_to_depletion      : step the commons collapses (1000 = never -> good)

Conditions correspond to shared_reward_weight: individual=0.0, mixed=0.5,
team=1.0. The hypothesis: more sharing -> more cooperation -> higher
sustainability / time_to_depletion, lower gini.

Run from RP-6/:
    python analyze_reward_sharing.py --algo mappo
    python analyze_reward_sharing.py --algo mappo --eval-root metrics/eval \\
        --greedy metrics/fixed_greedy --out-dir metrics/reward_sharing
"""

import argparse
import csv
import itertools
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parent

_CONDITIONS = ["individual", "mixed", "team"]
_WEIGHT = {"individual": 0.0, "mixed": 0.5, "team": 1.0}
_METRICS = [
    ("total_apples_collected", "Total apples", "higher"),
    ("sustainability",         "Sustainability", "higher"),
    ("gini_coefficient",       "Gini (lower=fairer)", "lower"),
    ("time_to_depletion",      "Time to depletion", "higher"),
]


def _episode_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the trailing 'mean' row written by save_aggregate_csv."""
    return df[df["episode_index"].apply(lambda x: str(x).isdigit())].copy()


def _per_seed_means(eval_root: Path, algo: str) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Return {condition: {seed: {metric: mean-over-eval-episodes}}}."""
    out: Dict[str, Dict[int, Dict[str, float]]] = {c: {} for c in _CONDITIONS}
    for cond in _CONDITIONS:
        for cell in sorted(eval_root.glob(f"harvest_{algo}_{cond}_seed*")):
            agg = cell / f"{algo}_aggregate.csv"
            if not agg.exists():
                print(f"  WARN missing {agg}")
                continue
            seed = int(cell.name.split("seed")[-1])
            rows = _episode_rows(pd.read_csv(agg))
            out[cond][seed] = {
                m: float(pd.to_numeric(rows[m], errors="coerce").mean())
                for m, _, _ in _METRICS
            }
    return out


def _greedy_episode_metrics(greedy_dir: Path) -> Dict[str, np.ndarray]:
    """FixedGreedy has no training seed; its eval episodes are the samples."""
    agg = greedy_dir / "fixed_greedy_aggregate.csv"
    if not agg.exists():
        return {}
    rows = _episode_rows(pd.read_csv(agg))
    return {m: pd.to_numeric(rows[m], errors="coerce").dropna().values for m, _, _ in _METRICS}


def _condition_samples(per_seed: Dict[int, Dict[str, float]], metric: str) -> np.ndarray:
    return np.array([s[metric] for s in per_seed.values()], dtype=float)


def main():
    ap = argparse.ArgumentParser(description="Phase-2 reward-sharing analysis")
    ap.add_argument("--algo", default="mappo", choices=["mappo", "ippo"])
    ap.add_argument("--eval-root", default="metrics/eval")
    ap.add_argument("--greedy", default="metrics/fixed_greedy")
    ap.add_argument("--out-dir", default=None,
                    help="Default: metrics/reward_sharing_<algo>")
    args = ap.parse_args()

    eval_root = (_PROJECT_ROOT / args.eval_root).resolve()
    greedy_dir = (_PROJECT_ROOT / args.greedy).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else \
        (_PROJECT_ROOT / "metrics" / f"reward_sharing_{args.algo}")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed = _per_seed_means(eval_root, args.algo)
    greedy = _greedy_episode_metrics(greedy_dir)

    n_per_cond = {c: len(per_seed[c]) for c in _CONDITIONS}
    print("=" * 80)
    print(f"REWARD-SHARING ANALYSIS  ({args.algo.upper()})")
    print(f"seeds per condition: " + ", ".join(f"{c}={n_per_cond[c]}" for c in _CONDITIONS))
    if greedy:
        print(f"FixedGreedy episodes: {len(next(iter(greedy.values())))}")
    print("=" * 80)

    # ── Summary table: per-condition mean ± std across seeds ──────────────────
    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "shared_reward_weight", "n", "metric", "mean", "std"])
        for cond in _CONDITIONS:
            for metric, _, _ in _METRICS:
                vals = _condition_samples(per_seed[cond], metric)
                if len(vals) == 0:
                    continue
                w.writerow([cond, _WEIGHT[cond], len(vals), metric,
                            f"{vals.mean():.4f}", f"{vals.std(ddof=1) if len(vals) > 1 else 0.0:.4f}"])
        if greedy:
            for metric, _, _ in _METRICS:
                vals = greedy[metric]
                w.writerow(["FixedGreedy", "n/a", len(vals), metric,
                            f"{vals.mean():.4f}", f"{vals.std(ddof=1) if len(vals) > 1 else 0.0:.4f}"])
    print(f"\nwrote {summary_path}")

    # Pretty print
    for metric, label, direction in _METRICS:
        print(f"\n{label}  ({direction} is better)")
        for cond in _CONDITIONS:
            vals = _condition_samples(per_seed[cond], metric)
            if len(vals):
                print(f"  {cond:<12} (w={_WEIGHT[cond]:<3}): {vals.mean():9.3f} ± {vals.std(ddof=1) if len(vals)>1 else 0:6.3f}  (n={len(vals)})")
        if greedy:
            g = greedy[metric]
            print(f"  {'FixedGreedy':<12} (   ): {g.mean():9.3f} ± {g.std(ddof=1) if len(g)>1 else 0:6.3f}  (n={len(g)})")

    # ── Welch t-tests ────────────────────────────────────────────────────────
    ttest_path = out_dir / "ttests.csv"
    with open(ttest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "group_a", "group_b", "mean_a", "mean_b",
                    "t_stat", "p_value", "n_a", "n_b", "significant_p<0.05"])

        def emit(metric, a_name, a_vals, b_name, b_vals):
            if len(a_vals) < 2 or len(b_vals) < 2:
                t, p = float("nan"), float("nan")
            else:
                t, p = stats.ttest_ind(a_vals, b_vals, equal_var=False)
            w.writerow([metric, a_name, b_name,
                        f"{np.mean(a_vals):.4f}", f"{np.mean(b_vals):.4f}",
                        f"{t:.4f}", f"{p:.4f}", len(a_vals), len(b_vals),
                        "yes" if (p == p and p < 0.05) else "no"])

        print("\n" + "=" * 80)
        print("WELCH T-TESTS (per-seed means; * = p<0.05)")
        print("=" * 80)
        for metric, label, _ in _METRICS:
            print(f"\n{label}")
            cond_pairs = list(itertools.combinations(_CONDITIONS, 2))
            for a, b in cond_pairs:
                av = _condition_samples(per_seed[a], metric)
                bv = _condition_samples(per_seed[b], metric)
                emit(metric, a, av, b, bv)
                if len(av) >= 2 and len(bv) >= 2:
                    t, p = stats.ttest_ind(av, bv, equal_var=False)
                    star = " *" if p < 0.05 else ""
                    print(f"  {a:<11} vs {b:<11}: t={t:+7.3f}  p={p:.4f}{star}")
            if greedy:
                for cond in _CONDITIONS:
                    cv = _condition_samples(per_seed[cond], metric)
                    emit(metric, cond, cv, "FixedGreedy", greedy[metric])
                    if len(cv) >= 2 and len(greedy[metric]) >= 2:
                        t, p = stats.ttest_ind(cv, greedy[metric], equal_var=False)
                        star = " *" if p < 0.05 else ""
                        print(f"  {cond:<11} vs FixedGreedy: t={t:+7.3f}  p={p:.4f}{star}")
    print(f"\nwrote {ttest_path}")

    # ── Plot 1: metric bars across conditions (+ greedy) ─────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()
    labels = [f"{c}\n(w={_WEIGHT[c]})" for c in _CONDITIONS] + (["FixedGreedy"] if greedy else [])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, (metric, label, _) in zip(axes, _METRICS):
        means, stds = [], []
        for cond in _CONDITIONS:
            vals = _condition_samples(per_seed[cond], metric)
            means.append(vals.mean() if len(vals) else 0.0)
            stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        if greedy:
            g = greedy[metric]
            means.append(g.mean()); stds.append(g.std(ddof=1) if len(g) > 1 else 0.0)
        xs = np.arange(len(labels))
        ax.bar(xs, means, yerr=stds, capsize=5,
               color=[colors[i % len(colors)] for i in range(len(labels))],
               edgecolor="black", linewidth=0.6)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(label); ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(f"Reward-sharing sweep ({args.algo.upper()}): mean ± std across seeds")
    fig.tight_layout()
    p1 = out_dir / "metric_bars.png"
    fig.savefig(p1, dpi=120); plt.close(fig)
    print(f"wrote {p1}")

    # ── Plot 2: commons trajectory (apples on grid over time) per condition ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cond in enumerate(_CONDITIONS):
        curves = []
        for cell in sorted(eval_root.glob(f"harvest_{args.algo}_{cond}_seed*")):
            for ts in sorted(cell.glob(f"{args.algo}_ep*_timeseries.csv")):
                d = pd.read_csv(ts)
                if "apples_on_grid" in d.columns:
                    curves.append(d["apples_on_grid"].values)
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        mat = np.stack([c[:min_len] for c in curves])
        mean = mat.mean(axis=0); std = mat.std(axis=0)
        steps = np.arange(min_len)
        c = colors[i % len(colors)]
        ax.plot(steps, mean, label=f"{cond} (w={_WEIGHT[cond]}, n={mat.shape[0]})", color=c, lw=2)
        ax.fill_between(steps, mean - std, mean + std, color=c, alpha=0.15)
    # Greedy trajectory
    if greedy_dir.is_dir():
        gcurves = []
        for ts in sorted(greedy_dir.glob("fixed_greedy_ep*_timeseries.csv")):
            d = pd.read_csv(ts)
            if "apples_on_grid" in d.columns:
                gcurves.append(d["apples_on_grid"].values)
        if gcurves:
            min_len = min(len(c) for c in gcurves)
            mat = np.stack([c[:min_len] for c in gcurves])
            ax.plot(np.arange(min_len), mat.mean(axis=0), label=f"FixedGreedy (n={mat.shape[0]})",
                    color="black", lw=2, ls="--")
    ax.set_xlabel("Step"); ax.set_ylabel("Apples on grid")
    ax.set_title("Commons trajectory: high+stable = cooperation, collapse = tragedy of commons")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = out_dir / "commons_trajectory.png"
    fig.savefig(p2, dpi=120); plt.close(fig)
    print(f"wrote {p2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
