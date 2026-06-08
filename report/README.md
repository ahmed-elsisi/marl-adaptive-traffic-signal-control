# Final Dissertation — Build & Regeneration Guide ("the brains")

This directory holds the **final traffic-MARL dissertation** and everything needed to
regenerate it from scratch. It is self-contained: given the trained reference
checkpoints, the commands below rebuild every number, every figure, and the PDF.

- **Output:** `main.pdf` (≈27 pages, committed).
- **Source:** `main.tex` + `sections/*.tex` + `references.bib` + `figures/`.
- **Data:** `data/eval_master_{raw,summary}.csv` (the numbers behind every result).
- **Scripts:** `run_eval_sweep.py` (data) → `make_report_figures.py` (figures).

---

## 1. The argument (narrative spine)

A "plot twist" structure. Each beat maps to a chapter:

1. **Setup** (Ch.1–2) — MARL + the cooperation question; traffic = pure-coordination
   testbed; why MAPPO (centralized critic, CTDE, GAE, neighbour-coupled reward)
   *should* win.
2. **Rising action** (Ch.5.1–5.2) — MAPPO converges; under **fair matched-3 s
   clearance** it beats Max-Pressure ~2× and Fixed-Time ~5×. The zero-clearance
   confound is exposed and corrected (methodological contribution).
3. **Credibility** (Ch.5.3) — our MAPPO ≈ the Yu et al. (2021) paper recipe, so it is
   not a strawman.
4. **The twist** (Ch.5.4) — **MAPPO ≈ IPPO: coordination value ≈ 0.** Indistinguishable
   on every metric; only +21 performance-neutral phase switches (p = 0.015).
5. **Understanding + future work** (Ch.6–7) — small-network incentive alignment ⇒
   scale hypothesis (Chu 2020 / CoLight); 5×5 scale-up + social dilemma left as
   compute-limited future work.

---

## 2. Data provenance

**Reference checkpoints** (trained earlier; live at the repo root `../reference/`,
**not** under `RP-5/`):

| Controller | Config | Checkpoint dir |
|---|---|---|
| MAPPO v2 (`fa6ad`) | `RP-5/configs/mappo_config_v2.yaml` | `reference/v2 256x256 - reworked - fa6ad/PPO_sumo_traffic_fa6ad_*/checkpoint_000007` |
| IPPO (`adfef`) | `RP-5/configs/ippo_config.yaml` | `reference/ippo 256x256 - reworked - adfef/PPO_sumo_traffic_adfef_*/checkpoint_000003` |
| Paper-baseline (`e7611`) | `RP-5/configs/mappo_baseline_paper.yaml` | `reference/paper_baseline 512x512 - reworked - e7611/PPO_sumo_traffic_e7611_*/checkpoint_000200` |

All three trained at `min_red = 3` (3 s all-red clearance), actor/critic as in
`sections/04-methodology.tex`. The eval is **deterministic** (`explore=False`) over
**seeds 42–46**.

**Heuristic baselines** (`RP-5/metrics/fair_comparison/fair_comparison.csv`, produced by
`RP-5/build_fair_comparison.py`): re-run of `max-pressure.py` / `fixed-cycles.py` at
both `--all-red 0` (unfair) and `--all-red 3` (matched/fair), seeds 42–46.

---

## 3. Regeneration pipeline

> Run from `Applied/` (repo root). On Windows, prefix with `PYTHONUTF8=1` — `evaluate.py`
> prints ✓ glyphs that crash under the default cp1252 codec when stdout is captured.

### Step 0 — temporary seed plumb (REQUIRED for the sweep, then REVERT)
`RP-5/marl_env/sumo_env.py` hard-codes the SUMO seed to 42 (line ~289). For a
multi-seed sweep, temporarily change:
```python
"--seed", str(42),            # ->
"--seed", str(self.sumo_seed),
```
`self.sumo_seed` is already read from `env_config` (set by `evaluate.py`). Run one
episode per invocation so the seed is constant within a run (avoids the per-episode
demand-offset bug — see memory `feedback_sumo_seed_hardcode`). **Revert to `str(42)`
afterward.**

### Step 1 — eval sweep → master CSVs
```bash
PYTHONUTF8=1 python report/run_eval_sweep.py
```
Runs 3 controllers × 5 seeds = 15 isolated subprocess evals (~25–35 min) and writes:
- `report/data/eval_master_raw.csv` — one row per (controller, seed).
- `report/data/eval_master_summary.csv` — mean/std per metric + Welch p-vs-MAPPO.

### Step 2 — figures
```bash
python report/make_report_figures.py
```
Writes to `report/figures/`:
- `coordination_value.png` — MAPPO vs IPPO across 5 metrics, n.s. annotations (the twist).
- `fair_comparison.png` — all controllers @ 3 s clearance + the 0 s-vs-3 s confound panel.
- `train_reward.png`, `train_kl_explvar.png` — from `fa6ad`'s `progress.csv`.

### Step 3 — compile
```bash
report/_tools/tectonic.exe report/main.tex      # local (tectonic auto-fetches packages)
# or: drop report/ into Overleaf (standard `report` class), or run pdflatex+bibtex twice.
```
`tectonic` is a single self-contained binary (gitignored under `_tools/`); download from
the tectonic-typesetting GitHub releases if absent.

---

## 4. The numbers (must match what the .tex cites)

From `data/eval_master_summary.csv` (mean ± std, seeds 42–46; p = Welch vs MAPPO):

| Metric | MAPPO | IPPO | p | Paper-baseline | p |
|---|---|---|---|---|---|
| Avg wait (s) | 9.92 ± 0.65 | 10.34 ± 0.31 | 0.236 | 10.21 ± 0.28 | 0.388 |
| Avg halting | 3.91 ± 0.11 | 4.00 ± 0.27 | 0.534 | 4.04 ± 0.27 | 0.355 |
| Max halting | 14.2 ± 1.6 | 14.6 ± 2.1 | 0.745 | 14.6 ± 1.7 | 0.713 |
| Completed trips | 1261.6 ± 0.9 | 1261.2 ± 1.1 | 0.545 | 1261.6 ± 0.5 | 1.000 |
| Phase switches | 1391 ± 12 | 1370 ± 8 | **0.015** | 1319 ± 23 | 0.0008 |

Fair baseline avg wait: **MAPPO 9.92** ≪ Max-Pressure **21.63 ± 1.00** ≪ Fixed-Time
**47.03 ± 0.59** (all at 3 s). Unfair 0 s: Max-Pressure 8.15, Fixed-Time 38.02.
Training (`fa6ad`): reward −2757 → last-10 **−49.76 ± 0.41**, explained var **0.907**,
KL **0.0034**.

> These reproduced the project's previously documented results exactly. If a rerun
> drifts, update the figures **and** the inline numbers in `sections/05-results.tex`,
> `00-abstract.tex`.

---

## 5. Corrections baked in vs. the interim report

- **Fair clearance only.** Never cite the 0 s heuristic numbers (8.05 / 38.25) without
  the caveat; the report uses matched-3 s throughout.
- **Config labelling.** Results are configuration **v2** (the one used for the
  MAPPO/IPPO/paper comparison). Do not present the Semester-1 v1 reward (−26.28) and v2
  (−49.76) as comparable — different reward weights + `min_red` change the scale.
- **No conflation.** The "98.5 % improvement" is *training-reward* improvement, not
  improvement over baselines.

---

## 6. Gotchas checklist

- [ ] Seed plumb applied for the sweep, **reverted** afterward (`git diff RP-5/marl_env/sumo_env.py` clean).
- [ ] `PYTHONUTF8=1` set or evaluate.py's ✓ prints crash on Windows.
- [ ] Reference checkpoints are at `../reference/` (root), not `RP-5/reference/`.
- [ ] `_tools/` (tectonic binary) stays gitignored.
- [ ] Title-page date in `main.tex` matches the actual submission date.
