"""
Regenerate the dissertation result charts directly into HTML Report/assets/img.

Changes vs the original report/make_report_figures.py:
  - train_reward title no longer says "(v2)".
  - fair_comparison shows ONLY the matched 3s clearance (no 0s confound panel).
  - three new charts added from the existing data:
      three_way_comparison.png  -- MAPPO vs IPPO vs Paper-baseline across metrics
      per_seed_wait.png         -- per-seed avg-wait clouds (shows overlap)
      train_entropy_loss.png    -- entropy decay + value/policy loss
"""
import glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

APPLIED = Path(r"E:\Research\Emergent Social Behaviour and Dilemmas in MARL\Applied")
DATA = APPLIED / "Written" / "report" / "data"
OUT = APPLIED / "Written" / "HTML Report" / "assets" / "img"
OUT.mkdir(parents=True, exist_ok=True)
FA6AD = glob.glob(str(APPLIED / "reference" / "v2 256x256 - reworked - fa6ad" /
                      "PPO_sumo_traffic_*" / "progress.csv"))[0]
FAIRCSV = APPLIED / "RP-5" / "metrics" / "fair_comparison" / "fair_comparison.csv"

C = {"MAPPO": "#2c7fb8", "IPPO": "#41b6c4", "PaperBaseline": "#7a5195",
     "Max-Pressure": "#d95f0e", "Fixed-Time": "#999999"}
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#444"})


def load_summary():
    df = pd.read_csv(DATA / "eval_master_summary.csv")
    out = {}
    for _, r in df.iterrows():
        out.setdefault(r["controller"], {})[r["metric"]] = (
            float(r["mean"]), float(r["std"]),
            (None if r["controller"] == "MAPPO" else float(r["p_vs_MAPPO"])))
    return out


# ---- coordination-value punchline (MAPPO vs IPPO) ----
def fig_coordination_value(S):
    metrics = [("avg_wait", "Avg wait (s)"), ("avg_halt", "Avg halting"),
               ("max_halt", "Max halting"), ("switches", "Phase switches"),
               ("arrivals", "Arrivals (trips)")]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))
    for ax, (m, label) in zip(axes, metrics):
        mu = [S["MAPPO"][m][0], S["IPPO"][m][0]]
        sd = [S["MAPPO"][m][1], S["IPPO"][m][1]]
        p = S["IPPO"][m][2]
        ax.bar([0, 1], mu, yerr=sd, capsize=5, color=[C["MAPPO"], C["IPPO"]],
               edgecolor="black", linewidth=0.6, width=0.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["MAPPO", "IPPO"], fontsize=9)
        ax.set_title(label, fontsize=10)
        top = max(m_ + s_ for m_, s_ in zip(mu, sd))
        ax.set_ylim(0, top * 1.25)
        ax.text(0.5, top * 1.13,
                ("n.s. (p=%.2f)" % p) if (p is None or p >= 0.05) else ("p=%.3f *" % p),
                ha="center", fontsize=9, fontweight="bold",
                color=("#333333" if (p is None or p >= 0.05) else "#b30000"))
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Coordination value ≈ 0: MAPPO vs IPPO are indistinguishable on every metric "
                 "(2×2 grid, seeds 42–46)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "coordination_value.png", dpi=130); plt.close(fig)
    print("wrote coordination_value.png")


# ---- fair comparison, 3s clearance ONLY ----
def fig_fair_comparison(S):
    fair = pd.read_csv(FAIRCSV).set_index("controller")
    mp3 = float(fair.loc["Max-Pressure (3s)", "avg_wait_mean_s"])
    mp3e = float(fair.loc["Max-Pressure (3s)", "avg_wait_std_s"])
    ft3 = float(fair.loc["Fixed-Time (3s)", "avg_wait_mean_s"])
    ft3e = float(fair.loc["Fixed-Time (3s)", "avg_wait_std_s"])
    labels = ["MAPPO", "IPPO", "Paper\nbaseline", "Max-\nPressure", "Fixed-\nTime"]
    mu = [S["MAPPO"]["avg_wait"][0], S["IPPO"]["avg_wait"][0],
          S["PaperBaseline"]["avg_wait"][0], mp3, ft3]
    sd = [S["MAPPO"]["avg_wait"][1], S["IPPO"]["avg_wait"][1],
          S["PaperBaseline"]["avg_wait"][1], mp3e, ft3e]
    cols = [C["MAPPO"], C["IPPO"], C["PaperBaseline"], C["Max-Pressure"], C["Fixed-Time"]]
    xs = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(xs, mu, yerr=sd, capsize=6, color=cols, edgecolor="black", linewidth=0.7, width=0.62)
    for x, m_, s_ in zip(xs, mu, sd):
        ax.text(x, m_ + s_ + 0.8, f"{m_:.2f}", ha="center", fontsize=10, fontweight="bold")
    # learned vs heuristic separator + group brackets
    ax.axvspan(-0.5, 2.5, color=C["MAPPO"], alpha=0.06)
    ax.text(1.0, 51, "learned controllers", ha="center", fontsize=9, color="#2c7fb8", fontweight="bold")
    ax.text(3.5, 51, "heuristics", ha="center", fontsize=9, color="#d95f0e", fontweight="bold")
    # ratio annotations vs MAPPO
    ax.annotate("", xy=(3, mp3), xytext=(3, mu[0]),
                arrowprops=dict(arrowstyle="<->", color="#555"))
    ax.text(3.18, (mp3 + mu[0]) / 2, f"≈{mp3/mu[0]:.1f}×", fontsize=9, color="#555")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Average waiting time (s) — lower is better")
    ax.set_title("All controllers under matched 3 s safety clearance (seeds 42–46)")
    ax.set_ylim(0, 56)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fair_comparison.png", dpi=130); plt.close(fig)
    print("wrote fair_comparison.png")


# ---- three-way: MAPPO vs IPPO vs Paper-baseline ----
def fig_three_way(S):
    metrics = [("avg_wait", "Avg wait (s)"), ("avg_halt", "Avg halting"),
               ("max_halt", "Max halting"), ("switches", "Phase switches")]
    ctrls = ["MAPPO", "IPPO", "PaperBaseline"]
    names = ["MAPPO", "IPPO", "Paper-base"]
    cols = [C[c] for c in ctrls]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (m, label) in zip(axes, metrics):
        mu = [S[c][m][0] for c in ctrls]
        sd = [S[c][m][1] for c in ctrls]
        xs = np.arange(3)
        ax.bar(xs, mu, yerr=sd, capsize=5, color=cols, edgecolor="black", linewidth=0.6, width=0.66)
        ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=8.5, rotation=12)
        ax.set_title(label, fontsize=10)
        top = max(m_ + s_ for m_, s_ in zip(mu, sd))
        ax.set_ylim(0, top * 1.22)
        # p annotations vs MAPPO for the two comparators
        pieces = []
        for c, nm in ((("IPPO"), "I"), (("PaperBaseline"), "P")):
            p = S[c][m][2]
            pieces.append(f"{nm}:{'n.s.' if p >= 0.05 else 'p=%.3f*' % p}")
        ax.text(0.5, top * 1.10, "  ".join(pieces), ha="center", fontsize=8,
                color="#333", transform=ax.get_xaxis_transform() if False else ax.transData)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Internal and external comparison: MAPPO, IPPO and the paper-baseline are "
                 "statistically indistinguishable on outcomes (seeds 42–46)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "three_way_comparison.png", dpi=130); plt.close(fig)
    print("wrote three_way_comparison.png")


# ---- per-seed avg-wait clouds ----
def fig_per_seed():
    raw = pd.read_csv(DATA / "eval_master_raw.csv")
    ctrls = ["MAPPO", "IPPO", "PaperBaseline"]
    names = ["MAPPO", "IPPO", "Paper-baseline"]
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(0)
    for i, c in enumerate(ctrls):
        vals = raw[raw.controller == c]["avg_wait"].values
        jitter = (rng.random(len(vals)) - 0.5) * 0.18
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=70, color=C[c],
                   edgecolor="black", linewidth=0.6, zorder=3, alpha=0.9)
        mu, sd = vals.mean(), vals.std(ddof=1)
        ax.hlines(mu, i - 0.28, i + 0.28, color="black", lw=2, zorder=4)
        ax.add_patch(plt.Rectangle((i - 0.28, mu - sd), 0.56, 2 * sd,
                                   color=C[c], alpha=0.15, zorder=1))
        ax.text(i + 0.34, mu, f"{mu:.2f}±{sd:.2f}", va="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Average waiting time (s) per seed")
    ax.set_title("Per-seed average wait: the three controllers' clouds overlap\n"
                 "(each dot is one of seeds 42–46; bar = mean, band = ±1 SD)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "per_seed_wait.png", dpi=130); plt.close(fig)
    print("wrote per_seed_wait.png")


# ---- training curves ----
def fig_training():
    d = pd.read_csv(FA6AD)
    it = d["training_iteration"]
    r = d["env_runners/episode_reward_mean"]
    kl = d["info/learner/shared_policy/learner_stats/kl"]
    ev = d["info/learner/shared_policy/learner_stats/vf_explained_var"]
    ent = d["info/learner/shared_policy/learner_stats/entropy"]
    pl = d["info/learner/shared_policy/learner_stats/policy_loss"].abs()
    vl = d["info/learner/shared_policy/learner_stats/vf_loss"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(it, r, color=C["MAPPO"], lw=1.8)
    ax.set_xlabel("Training iteration"); ax.set_ylabel("Episode reward (mean)")
    ax.set_title("MAPPO training reward over iterations")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(OUT / "train_reward.png", dpi=130); plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(it, kl, color="#d95f0e", lw=1.5); a1.set_title("(a) KL divergence")
    a1.set_xlabel("Training iteration"); a1.set_ylabel("KL"); a1.grid(True, alpha=0.3)
    a2.plot(it, ev, color="#2c7fb8", lw=1.5); a2.set_title("(b) Explained variance")
    a2.set_xlabel("Training iteration"); a2.set_ylabel("Explained variance")
    a2.set_ylim(0, 1); a2.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "train_kl_explvar.png", dpi=130); plt.close(fig)

    # entropy + losses
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(it, ent, color="#7a5195", lw=1.6)
    a1.axhline(np.log(4), ls="--", color="#999", lw=1)
    a1.text(it.iloc[-1], np.log(4) - 0.04, "max ln(4)≈1.39", ha="right", va="top", fontsize=8, color="#777")
    a1.set_title("(a) Policy entropy"); a1.set_xlabel("Training iteration")
    a1.set_ylabel("Entropy"); a1.grid(True, alpha=0.3)
    a2.plot(it, vl, color="#2c7fb8", lw=1.5, label="value loss")
    a2.plot(it, pl, color="#d95f0e", lw=1.5, label="policy loss |.|")
    a2.set_yscale("log"); a2.set_title("(b) Value and policy loss")
    a2.set_xlabel("Training iteration"); a2.set_ylabel("Loss (log scale)")
    a2.legend(fontsize=9); a2.grid(True, alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(OUT / "train_entropy_loss.png", dpi=130); plt.close(fig)
    print("wrote train_reward.png, train_kl_explvar.png, train_entropy_loss.png")


if __name__ == "__main__":
    S = load_summary()
    fig_coordination_value(S)
    fig_fair_comparison(S)
    fig_three_way(S)
    fig_per_seed()
    fig_training()
    print("done ->", OUT)
