"""
Report Phase-A: consistent multi-seed eval sweep for the three trained controllers
(MAPPO v2 fa6ad, IPPO adfef, paper-baseline e7611), seeds 42-46, at min_red=3.

Produces report/data/eval_master_raw.csv (one row per controller x seed) and
eval_master_summary.csv (mean/std per metric + Welch t-test p-values vs MAPPO).

Each eval runs in an ISOLATED subprocess (fresh Ray + SUMO) to avoid cross-run
state accumulation. The worker chdir's into RP-5 so evaluate.py's relative config
and sumo_network paths resolve.

NOTE: relies on the TEMPORARY one-line plumb in marl_env/sumo_env.py
(`--seed str(self.sumo_seed)`); revert that after the sweep.

Run from anywhere:  python report/run_eval_sweep.py
"""
import os, sys, json, subprocess, csv
from pathlib import Path

APPLIED = Path(r"E:\Research\Emergent Social Behaviour and Dilemmas in MARL\Applied")
RP5 = APPLIED / "RP-5"
REF = APPLIED / "reference"
DATA = APPLIED / "report" / "data"

CONTROLLERS = [
    ("MAPPO", REF / "v2 256x256 - reworked - fa6ad" /
     "PPO_sumo_traffic_fa6ad_00000_0_2026-05-08_02-53-42" / "checkpoint_000007",
     "configs/mappo_config_v2.yaml"),
    ("IPPO", REF / "ippo 256x256 - reworked - adfef" /
     "PPO_sumo_traffic_adfef_00000_0_2026-05-10_09-30-04" / "checkpoint_000003",
     "configs/ippo_config.yaml"),
    ("PaperBaseline", REF / "paper_baseline 512x512 - reworked - e7611" /
     "PPO_sumo_traffic_e7611_00000_0_2026-06-02_00-57-38" / "checkpoint_000200",
     "configs/mappo_baseline_paper.yaml"),
]
SEEDS = [42, 43, 44, 45, 46]
METRICS = ["avg_wait", "max_halt", "avg_halt", "switches", "arrivals"]


def worker(name, checkpoint, config, seed):
    """Run one eval episode and print a RESULT json line."""
    os.chdir(str(RP5))
    sys.path.insert(0, str(RP5))
    from evaluate import evaluate_mappo
    res = evaluate_mappo(checkpoint_path=checkpoint, num_episodes=1, use_gui=False,
                         config_path=config, results_dir="metrics/_sweep_tmp", seed=int(seed))
    st = res["episode_stats"][0]
    out = {
        "controller": name, "seed": int(seed),
        "avg_wait": float(st["avg_waiting_time"]),
        "max_halt": float(st["max_halting"]),
        "avg_halt": float(st["avg_halting"]),
        "switches": int(st["total_switches"]),
        "arrivals": int(st["total_arrivals"]),
    }
    print("@@RESULT@@" + json.dumps(out))


def driver():
    DATA.mkdir(parents=True, exist_ok=True)
    raw = []
    for name, cp, cfg in CONTROLLERS:
        for s in SEEDS:
            print(f"\n=== {name} seed={s} ===", flush=True)
            cmd = [sys.executable, __file__, "--worker", name, str(cp), cfg, str(s)]
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", env=env)
            line = next((l for l in proc.stdout.splitlines() if l.startswith("@@RESULT@@")), None)
            if line is None:
                print(proc.stdout[-2000:]); print(proc.stderr[-2000:])
                raise SystemExit(f"No result for {name} seed={s}")
            rec = json.loads(line[len("@@RESULT@@"):])
            raw.append(rec)
            print(f"  -> wait={rec['avg_wait']:.2f} maxHalt={rec['max_halt']:.0f} "
                  f"avgHalt={rec['avg_halt']:.2f} switches={rec['switches']} arrivals={rec['arrivals']}")

    # raw csv
    with open(DATA / "eval_master_raw.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["controller", "seed"] + METRICS)
        for r in raw:
            w.writerow([r["controller"], r["seed"]] + [r[m] for m in METRICS])

    # summary + Welch p-values vs MAPPO
    import numpy as np
    from scipy import stats
    names = ["MAPPO", "IPPO", "PaperBaseline"]
    vals = {n: {m: np.array([r[m] for r in raw if r["controller"] == n], float) for m in METRICS} for n in names}
    with open(DATA / "eval_master_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller", "metric", "mean", "std", "n", "p_vs_MAPPO"])
        for n in names:
            for m in METRICS:
                a = vals[n][m]
                if n == "MAPPO":
                    p = ""
                else:
                    _, p = stats.ttest_ind(vals["MAPPO"][m], a, equal_var=False)
                    p = f"{p:.4f}"
                w.writerow([n, m, f"{a.mean():.4f}", f"{a.std(ddof=1):.4f}", len(a), p])
    print(f"\nWrote {DATA/'eval_master_raw.csv'} and eval_master_summary.csv")
    # quick console summary
    for m in METRICS:
        line = f"{m:10}: " + "  ".join(f"{n} {vals[n][m].mean():.2f}±{vals[n][m].std(ddof=1):.2f}" for n in names)
        print(line)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        driver()
