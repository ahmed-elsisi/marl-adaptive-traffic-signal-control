"""
Multi-seed matrix driver for Phase-2 Harvest training.

Runs the full (6 RL configs × 5 seeds) = 30-run RL matrix sequentially on
the single GPU. Each run is invoked as a subprocess of the appropriate
train_*_harvest.py entry point, with --seed plumbed through to override
env_config.seed.

Two important behaviors:

  1. **Tolerates the known Windows pyarrow post-fit access violation**
     (documented in CLAUDE.md). The training itself succeeds and writes
     the checkpoint; Ray's ExperimentAnalysis crashes only after. This
     driver detects success by scanning for a new run dir containing a
     `checkpoint_000XYZ` subdir at the target iteration, not by the
     subprocess exit code.

  2. **Idempotent**: writes a stamp file under results/.matrix_done/
     once a run completes. Re-running the driver skips runs whose stamp
     exists. Delete the stamp to re-run a single cell.

FixedGreedy is NOT included here --it takes seconds per episode and is
fired with a single command:
    python run_fixed_greedy.py --episodes 5 --seed 42

Run from RP-6/:
    python run_matrix.py                    # full matrix (30 RL runs)
    python run_matrix.py --dry-run          # print the plan, exit
    python run_matrix.py --configs ippo     # only IPPO configs
    python run_matrix.py --seeds 42,43      # only first 2 seeds
    python run_matrix.py --iterations 200   # override per-run iter count
"""

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent
_RESULTS_DIR = _PROJECT_ROOT / "results"
_STAMP_DIR = _RESULTS_DIR / ".matrix_done"


# ── Matrix definition ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MatrixCell:
    algo: str        # "mappo" or "ippo"
    config: str      # config YAML basename (under configs/)
    seed: int

    @property
    def stamp_path(self) -> Path:
        config_stem = Path(self.config).stem
        return _STAMP_DIR / f"{config_stem}_seed{self.seed}.json"

    @property
    def label(self) -> str:
        return f"{self.algo:<5} {Path(self.config).stem:<27} seed={self.seed}"


def _build_matrix(configs_filter: Optional[List[str]], seeds: List[int]) -> List[MatrixCell]:
    # MAPPO order: team -> mixed -> individual (per 2026-06-02 decision) so the
    # decisive critic-health check (team, w=1.0 -> well-posed value target) ran
    # FIRST. (All 15 MAPPO cells are already complete/stamped, so this order is
    # now historical for MAPPO.)
    #
    # IPPO order: individual -> team -> mixed (2026-06-07 decision, deadline
    # triage). IPPO's decentralized critic was never affected by the bug-3
    # canary, so the team-first rationale doesn't apply. Under the reduced
    # 2-seed budget we front-load the *individual* (w=0.0) condition because
    # that is where the tragedy-of-commons / MAPPO-vs-IPPO contrast lives, so a
    # power-cut interrupt leaves the most decisive data on disk first. The
    # reward-sharing sweep is order-independent for the science.
    full = [
        ("mappo", "configs/harvest_mappo_team.yaml"),
        ("mappo", "configs/harvest_mappo_mixed.yaml"),
        ("mappo", "configs/harvest_mappo_individual.yaml"),
        ("ippo",  "configs/harvest_ippo_individual.yaml"),
        ("ippo",  "configs/harvest_ippo_team.yaml"),
        ("ippo",  "configs/harvest_ippo_mixed.yaml"),
    ]
    if configs_filter:
        keep = []
        for algo, cfg in full:
            stem = Path(cfg).stem
            if any(f in algo or f in stem for f in configs_filter):
                keep.append((algo, cfg))
        full = keep

    return [MatrixCell(algo=a, config=c, seed=s) for (a, c) in full for s in seeds]


# ── Subprocess invocation ────────────────────────────────────────────────────


_RUN_DIR_RE = re.compile(r"^PPO_harvest_[0-9a-f]+_\d+_\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def _experiment_dir(algo: str) -> Path:
    return _RESULTS_DIR / f"{algo}_harvest"


def _snapshot_run_dirs(algo: str) -> set:
    exp_dir = _experiment_dir(algo)
    if not exp_dir.exists():
        return set()
    return {p.name for p in exp_dir.iterdir() if p.is_dir() and _RUN_DIR_RE.match(p.name)}


def _wait_for_new_run_dir(algo: str, before: set, timeout_s: int = 60) -> Optional[Path]:
    """After a subprocess returns, find the run dir it created. Polls briefly
    because Ray's final-state flush is async on Windows."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        now = _snapshot_run_dirs(algo)
        new = now - before
        if new:
            return _experiment_dir(algo) / sorted(new)[0]
        time.sleep(1)
    return None


def _final_checkpoint(run_dir: Path) -> Optional[Path]:
    if not run_dir.exists():
        return None
    cps = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint_")])
    return cps[-1] if cps else None


def _final_iter_reward(run_dir: Path) -> Optional[float]:
    progress = run_dir / "progress.csv"
    if not progress.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(progress)
        col = "env_runners/episode_reward_mean"
        if col not in df.columns:
            return None
        return float(df.iloc[-1][col])
    except Exception:
        return None


def _run_one(cell: MatrixCell, iterations: int, checkpoint_freq: int) -> dict:
    """Invoke the training subprocess for one cell. Returns a result dict
    documenting success/failure."""
    entry = f"train_{cell.algo}_harvest.py"
    before = _snapshot_run_dirs(cell.algo)
    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()

    cmd = [
        sys.executable, entry,
        "--config", cell.config,
        "--iterations", str(iterations),
        "--checkpoint-freq", str(checkpoint_freq),
        "--seed", str(cell.seed),
    ]
    print(f"\n>>> {cell.label}  cmd: {' '.join(cmd)}")

    # We DON'T capture output --let the user watch progress live.
    proc = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    duration_s = time.time() - t0

    run_dir = _wait_for_new_run_dir(cell.algo, before)
    final_cp = _final_checkpoint(run_dir) if run_dir else None
    final_reward = _final_iter_reward(run_dir) if run_dir else None

    # Success criterion: a new run dir with a final checkpoint exists.
    # Exit code can be non-zero due to the known Windows pyarrow crash.
    success = run_dir is not None and final_cp is not None

    result = {
        "label": cell.label,
        "algo": cell.algo,
        "config": cell.config,
        "seed": cell.seed,
        "started": started,
        "duration_s": round(duration_s, 1),
        "exit_code": proc.returncode,
        "run_dir": str(run_dir) if run_dir else None,
        "final_checkpoint": str(final_cp) if final_cp else None,
        "final_reward_mean": final_reward,
        "success": success,
    }

    status = "OK" if success else "FAILED"
    rew = f"{final_reward:+.2f}" if final_reward is not None else "n/a"
    print(f"<<< {cell.label}  [{status}]  duration={duration_s/60:.1f}min  final_reward={rew}  exit={proc.returncode}")
    return result


# ── Stamps (idempotency) ─────────────────────────────────────────────────────


def _write_stamp(cell: MatrixCell, result: dict):
    _STAMP_DIR.mkdir(parents=True, exist_ok=True)
    with open(cell.stamp_path, "w") as f:
        json.dump(result, f, indent=2)


def _has_stamp(cell: MatrixCell) -> bool:
    return cell.stamp_path.exists()


# ── Driver ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase-2 Harvest matrix driver")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Training iterations per cell.")
    parser.add_argument("--checkpoint-freq", type=int, default=25)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46",
                        help="Comma-separated seed list.")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated substring filter on algo/config name (e.g. 'ippo,individual').")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit without running.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run cells even if a stamp file exists.")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    configs_filter = args.configs.split(",") if args.configs else None
    matrix = _build_matrix(configs_filter, seeds)

    print("=" * 80)
    print(f"PHASE-2 HARVEST MATRIX  ({len(matrix)} cells)")
    print("=" * 80)
    todo = []
    for cell in matrix:
        stamped = _has_stamp(cell)
        if stamped and not args.force:
            print(f"  SKIP  {cell.label}  (stamp at {cell.stamp_path.name})")
        else:
            print(f"  RUN   {cell.label}")
            todo.append(cell)

    print("-" * 80)
    print(f"To run : {len(todo)}    Skipped : {len(matrix) - len(todo)}")
    est_min = len(todo) * 180  # ~3 hr / 180 min per run from Week-5 projection
    print(f"Est. duration (sequential, ~3h/run): {est_min} min = {est_min/60:.1f} h = {est_min/60/24:.2f} days")
    print(f"Iterations/cell : {args.iterations}")
    print(f"Checkpoint freq : {args.checkpoint_freq}")
    print("=" * 80)

    if args.dry_run or not todo:
        print("\nDry run / nothing to do --exiting.")
        return

    results: List[dict] = []
    for i, cell in enumerate(todo, start=1):
        print(f"\n[{i}/{len(todo)}] {cell.label}")
        result = _run_one(cell, iterations=args.iterations, checkpoint_freq=args.checkpoint_freq)
        results.append(result)
        if result["success"]:
            _write_stamp(cell, result)
        else:
            print(f"  WARNING: no checkpoint found for {cell.label}; not stamping. "
                  f"Inspect {result['run_dir']} manually.")

    print("\n" + "=" * 80)
    print(f"MATRIX COMPLETE: {sum(r['success'] for r in results)}/{len(results)} cells succeeded")
    print("=" * 80)
    for r in results:
        ok = "OK    " if r["success"] else "FAILED"
        rew = f"{r['final_reward_mean']:+.2f}" if r["final_reward_mean"] is not None else "n/a"
        print(f"  [{ok}]  {r['label']}  reward={rew}  ({r['duration_s']/60:.1f} min)")


if __name__ == "__main__":
    main()
