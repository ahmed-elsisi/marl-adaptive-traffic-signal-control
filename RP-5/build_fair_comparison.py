"""
Fair-comparison table + figure for the Phase-1 traffic controllers.

THE FAIRNESS ISSUE: the RL agents (MAPPO/IPPO) operate with min_red=3 -- a
3-second all-red clearance on every phase change (a realistic safety
constraint they were TRAINED under). The two heuristic baselines, however,
were originally evaluated with ZERO clearance (max-pressure.py / fixed-cycles.py
default YELLOW=0, ALL_RED=0), giving them free instantaneous switching. That
confounds the comparison: the heuristics' apparent edge was partly "no clearance
penalty," not controller quality.

This script collates a matched-clearance comparison: every controller scored
under the SAME 3-second clearance, over seeds 42-46. The heuristic numbers are
the per-seed avg-waiting-time values from re-running max-pressure.py /
fixed-cycles.py with --all-red 0 and --all-red 3 (this is reproducible: rerun
those scripts with the same flags). The RL numbers are the project's documented
5-seed deterministic eval (fa6ad / adfef checkpoints, already at min_red=3, i.e.
already matched-clearance).

Outputs (metrics/fair_comparison/):
  fair_comparison.csv  -- controller x clearance: mean, std, n, raw seeds
  fair_comparison.png  -- 2 panels: (L) matched 3s-clearance ranking,
                          (R) heuristics unfair(0s) vs fair(3s)

Run from RP-5/:  python build_fair_comparison.py
"""

from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "metrics" / "fair_comparison"
OUT.mkdir(parents=True, exist_ok=True)

# ── Raw per-seed avg waiting time (seconds), seeds 42-46 ─────────────────────
# Heuristics: from re-running the baseline scripts (this session, 2026-06-09).
MAXP_AR0 = [8.05, 8.14, 8.11, 8.24, 8.22]      # max-pressure.py --all-red 0
MAXP_AR3 = [22.41, 21.22, 20.12, 21.84, 22.56]  # max-pressure.py --all-red 3  (matched)
FIXED_AR0 = [38.25, 37.38, 38.84, 36.99, 38.65]  # fixed-cycles.py --all-red 0
FIXED_AR3 = [46.51, 47.00, 46.52, 47.16, 47.95]  # fixed-cycles.py --all-red 3  (matched)

# RL: project's documented 5-seed deterministic eval (seeds 42-46), already at
# min_red=3. Only mean/std are recorded (per-seed CSVs from the temporary-seed
# sweep were not retained), so they are entered directly with provenance.
RL = {
    "MAPPO (fa6ad)": (9.92, 0.65),
    "IPPO (adfef)":  (10.34, 0.31),
}


def ms(vals):
    a = np.array(vals, float)
    return a.mean(), a.std(ddof=1)


# ── CSV ──────────────────────────────────────────────────────────────────────
rows = []
for name, mn, sd, n, raw in [
    ("MAPPO (fa6ad)",      *RL["MAPPO (fa6ad)"], 5, ""),
    ("IPPO (adfef)",       *RL["IPPO (adfef)"],  5, ""),
    ("Max-Pressure (3s)",  *ms(MAXP_AR3),        5, " ".join(map(str, MAXP_AR3))),
    ("Fixed-Time (3s)",    *ms(FIXED_AR3),       5, " ".join(map(str, FIXED_AR3))),
    ("Max-Pressure (0s)",  *ms(MAXP_AR0),        5, " ".join(map(str, MAXP_AR0))),
    ("Fixed-Time (0s)",    *ms(FIXED_AR0),       5, " ".join(map(str, FIXED_AR0))),
]:
    rows.append([name, f"{mn:.2f}", f"{sd:.2f}", n, raw])

csv_path = OUT / "fair_comparison.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["controller", "avg_wait_mean_s", "avg_wait_std_s", "n_seeds", "raw_per_seed"])
    w.writerows(rows)
print(f"wrote {csv_path}")
for r in rows:
    print(f"  {r[0]:<20} {r[1]:>6} ± {r[2]:<5} s  (n={r[3]})")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

# Left: matched 3s-clearance ranking (the fair comparison)
labels = ["MAPPO", "IPPO", "Max-Pressure", "Fixed-Time"]
means = [RL["MAPPO (fa6ad)"][0], RL["IPPO (adfef)"][0], ms(MAXP_AR3)[0], ms(FIXED_AR3)[0]]
stds  = [RL["MAPPO (fa6ad)"][1], RL["IPPO (adfef)"][1], ms(MAXP_AR3)[1], ms(FIXED_AR3)[1]]
colors = ["#2c7fb8", "#41b6c4", "#d95f0e", "#999999"]
xs = np.arange(len(labels))
axL.bar(xs, means, yerr=stds, capsize=6, color=colors, edgecolor="black", linewidth=0.7)
for x, m, s in zip(xs, means, stds):
    axL.text(x, m + s + 0.6, f"{m:.1f}", ha="center", fontsize=10, fontweight="bold")
axL.set_xticks(xs); axL.set_xticklabels(labels, fontsize=10)
axL.set_ylabel("Avg waiting time (s)  — lower is better")
axL.set_title("Fair comparison: all controllers at 3s clearance\n(matched to RL min_red=3, seeds 42-46)")
axL.grid(True, alpha=0.3, axis="y")

# Right: heuristics unfair (0s) vs fair (3s) — illustrates the confound
hl = ["Max-Pressure", "Fixed-Time"]
a0 = [ms(MAXP_AR0)[0], ms(FIXED_AR0)[0]]
a0e = [ms(MAXP_AR0)[1], ms(FIXED_AR0)[1]]
a3 = [ms(MAXP_AR3)[0], ms(FIXED_AR3)[0]]
a3e = [ms(MAXP_AR3)[1], ms(FIXED_AR3)[1]]
xs2 = np.arange(len(hl)); bw = 0.35
axR.bar(xs2 - bw/2, a0, bw, yerr=a0e, capsize=5, label="0s clearance (original / unfair)",
        color="#a6cee3", edgecolor="black", linewidth=0.6)
axR.bar(xs2 + bw/2, a3, bw, yerr=a3e, capsize=5, label="3s clearance (matched / fair)",
        color="#d95f0e", edgecolor="black", linewidth=0.6)
# RL reference line
rl_min = min(RL["MAPPO (fa6ad)"][0], RL["IPPO (adfef)"][0])
rl_max = max(RL["MAPPO (fa6ad)"][0], RL["IPPO (adfef)"][0])
axR.axhspan(rl_min, rl_max, color="#2c7fb8", alpha=0.15)
axR.axhline(rl_max, color="#2c7fb8", ls="--", lw=1.2, label="RL (MAPPO/IPPO) band")
axR.set_xticks(xs2); axR.set_xticklabels(hl, fontsize=10)
axR.set_ylabel("Avg waiting time (s)")
axR.set_title("The confound: heuristics lose their edge\nonce charged the same clearance")
axR.legend(fontsize=8, loc="upper left")
axR.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
png = OUT / "fair_comparison.png"
fig.savefig(png, dpi=130)
plt.close(fig)
print(f"wrote {png}")
