"""
Report Phase-B: generate the dissertation figures into report/figures/.

Reads:
  report/data/eval_master_raw.csv      (per controller x seed; from run_eval_sweep.py)
  report/data/eval_master_summary.csv  (means/std + p-vs-MAPPO)
  RP-5/metrics/fair_comparison/fair_comparison.csv  (heuristic avg-wait @ 0s and 3s)
  reference/.../fa6ad/.../progress.csv (v2 training curves)

Writes:
  figures/coordination_value.png  -- MAPPO vs IPPO across 5 metrics (the punchline)
  figures/fair_comparison.png     -- all controllers @ matched 3s clearance + confound
  figures/train_reward.png        -- MAPPO v2 episode reward over iterations
  figures/train_kl_explvar.png    -- KL divergence and explained variance
"""
import os, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

APPLIED = Path(r"E:\Research\Emergent Social Behaviour and Dilemmas in MARL\Applied")
DATA = APPLIED / "report" / "data"
FIG = APPLIED / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
FA6AD = glob.glob(str(APPLIED / "reference" / "v2 256x256 - reworked - fa6ad" /
                      "PPO_sumo_traffic_*" / "progress.csv"))[0]
FAIRCSV = APPLIED / "RP-5" / "metrics" / "fair_comparison" / "fair_comparison.csv"

C = {"MAPPO": "#2c7fb8", "IPPO": "#41b6c4", "PaperBaseline": "#7a5195",
     "Max-Pressure": "#d95f0e", "Fixed-Time": "#999999"}


def load_summary():
    df = pd.read_csv(DATA / "eval_master_summary.csv")
    out = {}
    for _, r in df.iterrows():
        out.setdefault(r["controller"], {})[r["metric"]] = (
            float(r["mean"]), float(r["std"]), (None if r["controller"] == "MAPPO" else float(r["p_vs_MAPPO"])))
    return out


# ---- Figure 1: coordination-value punchline (MAPPO vs IPPO) ----
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
        sig = "n.s." if (p is None or p >= 0.05) else f"p={p:.3f}*"
        tag = f"p={p:.3f}" if (p is not None and p >= 0.05) else sig
        top = max(m_ + s_ for m_, s_ in zip(mu, sd))
        ax.set_ylim(0, top * 1.25)
        ax.text(0.5, top * 1.13, ("n.s. (p=%.2f)" % p) if (p is not None and p >= 0.05) else ("p=%.3f *" % p),
                ha="center", fontsize=9, fontweight="bold",
                color=("#333333" if (p is None or p >= 0.05) else "#b30000"))
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Coordination value ≈ 0: MAPPO vs IPPO are indistinguishable on every metric "
                 "(2×2 grid, seeds 42–46)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "coordination_value.png", dpi=130); plt.close(fig)
    print("wrote coordination_value.png")


# ---- Figure 2: fair comparison incl. paper-baseline + confound ----
def fig_fair_comparison(S):
    fair = pd.read_csv(FAIRCSV).set_index("controller")
    def g(name): return (float(fair.loc[name, "avg_wait_mean_s"]), float(fair.loc[name, "avg_wait_std_s"]))
    mp3, mp3e = g("Max-Pressure (3s)"); ft3, ft3e = g("Fixed-Time (3s)")
    mp0, mp0e = g("Max-Pressure (0s)"); ft0, ft0e = g("Fixed-Time (0s)")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    # Left: matched 3s clearance, all controllers (RL from fresh sweep)
    labels = ["MAPPO", "IPPO", "Paper\nbaseline", "Max-\nPressure", "Fixed-\nTime"]
    mu = [S["MAPPO"]["avg_wait"][0], S["IPPO"]["avg_wait"][0], S["PaperBaseline"]["avg_wait"][0], mp3, ft3]
    sd = [S["MAPPO"]["avg_wait"][1], S["IPPO"]["avg_wait"][1], S["PaperBaseline"]["avg_wait"][1], mp3e, ft3e]
    cols = [C["MAPPO"], C["IPPO"], C["PaperBaseline"], C["Max-Pressure"], C["Fixed-Time"]]
    xs = np.arange(len(labels))
    axL.bar(xs, mu, yerr=sd, capsize=6, color=cols, edgecolor="black", linewidth=0.7)
    for x, m_, s_ in zip(xs, mu, sd):
        axL.text(x, m_ + s_ + 0.7, f"{m_:.1f}", ha="center", fontsize=9, fontweight="bold")
    axL.set_xticks(xs); axL.set_xticklabels(labels, fontsize=9)
    axL.set_ylabel("Avg waiting time (s) — lower is better")
    axL.set_title("Fair comparison: all controllers at 3s clearance")
    axL.grid(True, axis="y", alpha=0.3)
    # Right: confound (heuristics 0s vs 3s) with RL band
    hl = ["Max-Pressure", "Fixed-Time"]; xs2 = np.arange(2); bw = 0.35
    axR.bar(xs2 - bw/2, [mp0, ft0], bw, yerr=[mp0e, ft0e], capsize=5,
            label="0s clearance (unfair)", color="#a6cee3", edgecolor="black", linewidth=0.6)
    axR.bar(xs2 + bw/2, [mp3, ft3], bw, yerr=[mp3e, ft3e], capsize=5,
            label="3s clearance (fair)", color=C["Max-Pressure"], edgecolor="black", linewidth=0.6)
    rl_lo = min(S["MAPPO"]["avg_wait"][0], S["IPPO"]["avg_wait"][0])
    rl_hi = max(S["MAPPO"]["avg_wait"][0], S["IPPO"]["avg_wait"][0])
    axR.axhspan(rl_lo, rl_hi, color=C["MAPPO"], alpha=0.15)
    axR.axhline(rl_hi, color=C["MAPPO"], ls="--", lw=1.2, label="RL (MAPPO/IPPO) band")
    axR.set_xticks(xs2); axR.set_xticklabels(hl, fontsize=9)
    axR.set_ylabel("Avg waiting time (s)")
    axR.set_title("The confound: heuristics lose their edge\nonce charged the same clearance")
    axR.legend(fontsize=8, loc="upper left"); axR.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fair_comparison.png", dpi=130); plt.close(fig)
    print("wrote fair_comparison.png")


# ---- Figures 3-4: v2 training curves ----
def fig_training():
    d = pd.read_csv(FA6AD)
    it = d["training_iteration"]
    r = d["env_runners/episode_reward_mean"]
    kl = d["info/learner/shared_policy/learner_stats/kl"]
    ev = d["info/learner/shared_policy/learner_stats/vf_explained_var"]
    # reward
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(it, r, color=C["MAPPO"], lw=1.8)
    ax.set_xlabel("Training iteration"); ax.set_ylabel("Episode reward (mean)")
    ax.set_title("MAPPO (v2) training reward over iterations")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(FIG / "train_reward.png", dpi=130); plt.close(fig)
    # KL + explained variance
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(it, kl, color="#d95f0e", lw=1.5); a1.set_title("(a) KL divergence")
    a1.set_xlabel("Training iteration"); a1.set_ylabel("KL"); a1.grid(True, alpha=0.3)
    a2.plot(it, ev, color="#2c7fb8", lw=1.5); a2.set_title("(b) Explained variance")
    a2.set_xlabel("Training iteration"); a2.set_ylabel("Explained variance")
    a2.set_ylim(0, 1); a2.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "train_kl_explvar.png", dpi=130); plt.close(fig)
    print("wrote train_reward.png, train_kl_explvar.png")


if __name__ == "__main__":
    S = load_summary()
    fig_coordination_value(S)
    fig_fair_comparison(S)
    fig_training()
    print("All report figures written to", FIG)
