"""
Evaluation-matrix driver for the completed Phase-2 Harvest training matrix.

Reads the matrix stamps under results/.matrix_done/ (written by run_matrix.py),
and for each completed cell runs evaluate_harvest.py over a FIXED deterministic
eval protocol (same env seeds for every policy), writing each cell's CSVs into
its own metrics/eval/<config>_seed<N>/ directory so nothing overwrites.

Why a fixed eval protocol: the training seed (42-46) is baked into each
checkpoint's weights. For a fair cross-condition / cross-algorithm comparison
every policy must be scored on the SAME set of environment episodes, so we
evaluate all of them at --seed 42 (env seeds 42..42+episodes-1), regardless of
which seed they trained on.

Idempotent: skips a cell whose output aggregate CSV already exists.

Run from RP-6/:
    python run_eval_matrix.py                       # all completed mappo + ippo cells
    python run_eval_matrix.py --algos mappo         # only mappo cells
    python run_eval_matrix.py --episodes 5 --seed 42
    python run_eval_matrix.py --dry-run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent
_STAMP_DIR = _PROJECT_ROOT / "results" / ".matrix_done"
_EVAL_ROOT = _PROJECT_ROOT / "metrics" / "eval"


def _load_cells(algos: Optional[List[str]]) -> List[dict]:
    """Read matrix stamps into a list of cell dicts (config_stem, seed, algo, checkpoint)."""
    cells = []
    if not _STAMP_DIR.is_dir():
        print(f"No stamp dir at {_STAMP_DIR} -- has the training matrix run?")
        return cells
    for stamp in sorted(_STAMP_DIR.glob("*.json")):
        try:
            data = json.loads(stamp.read_text())
        except Exception as e:
            print(f"  skip unreadable stamp {stamp.name}: {e}")
            continue
        if not data.get("success") or not data.get("final_checkpoint"):
            continue
        algo = data.get("algo", "")
        if algos and algo not in algos:
            continue
        cells.append({
            "config_stem": Path(data["config"]).stem,   # e.g. harvest_mappo_team
            "seed": data["seed"],
            "algo": algo,
            "checkpoint": data["final_checkpoint"],
        })
    return cells


def _run_one(cell: dict, episodes: int, eval_seed: int, force: bool) -> bool:
    out_dir = _EVAL_ROOT / f"{cell['config_stem']}_seed{cell['seed']}"
    aggregate = out_dir / f"{cell['algo']}_aggregate.csv"
    if aggregate.exists() and not force:
        print(f"  SKIP  {cell['config_stem']} seed={cell['seed']}  (aggregate exists)")
        return True

    if not Path(cell["checkpoint"]).exists():
        print(f"  MISS  {cell['config_stem']} seed={cell['seed']}  checkpoint gone: {cell['checkpoint']}")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "evaluate_harvest.py",
        "--checkpoint", cell["checkpoint"],
        "--algo", cell["algo"],
        "--episodes", str(episodes),
        "--seed", str(eval_seed),
        "--out-dir", str(out_dir),
    ]
    print(f"\n>>> {cell['config_stem']} seed={cell['seed']}  -> {out_dir.name}")
    # evaluate_harvest.py uses PPO.from_checkpoint (not ExperimentAnalysis), so
    # it is NOT subject to the post-fit pyarrow crash; exit code is meaningful.
    proc = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    ok = proc.returncode == 0 and aggregate.exists()
    print(f"<<< {cell['config_stem']} seed={cell['seed']}  [{'OK' if ok else 'FAILED'}]  exit={proc.returncode}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Run evaluate_harvest.py over the completed training matrix.")
    parser.add_argument("--algos", type=str, default="mappo,ippo",
                        help="Comma-separated algos to evaluate (mappo,ippo).")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Deterministic eval episodes per cell.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base env seed for the FIXED eval protocol (same for all cells).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate cells even if the aggregate CSV exists.")
    args = parser.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    cells = _load_cells(algos)

    print("=" * 80)
    print(f"HARVEST EVAL MATRIX  ({len(cells)} completed cells, algos={algos})")
    print(f"Protocol: {args.episodes} episodes, fixed env seeds {args.seed}..{args.seed + args.episodes - 1}")
    print("=" * 80)
    for c in cells:
        print(f"  {c['algo']:<5} {c['config_stem']:<26} seed={c['seed']}")
    print("-" * 80)

    if args.dry_run or not cells:
        print("Dry run / nothing to do -- exiting.")
        return

    results = []
    for i, cell in enumerate(cells, start=1):
        print(f"\n[{i}/{len(cells)}]")
        results.append(_run_one(cell, args.episodes, args.seed, args.force))

    print("\n" + "=" * 80)
    print(f"EVAL MATRIX COMPLETE: {sum(results)}/{len(results)} cells succeeded")
    print("=" * 80)


if __name__ == "__main__":
    main()
