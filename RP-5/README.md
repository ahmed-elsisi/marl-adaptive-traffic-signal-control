# RP-5: MAPPO & IPPO for Traffic Signal Control

Multi-Agent Reinforcement Learning for adaptive traffic signal control on a 2×2 grid network in SUMO. This directory holds the Semester-1 implementation (MAPPO with a centralized critic) and the Semester-2 IPPO comparator (decentralized critic) used to quantify the value of critic centralization.

This is the code-level README. For the research framing, methodology, and full Semester-2 plan, see `../CLAUDE.md`.

## What's here

```
RP-5/
├── train_mappo.py              # MAPPO training (centralized critic)
├── train_ippo.py               # IPPO training (decentralized critic)
├── evaluate.py                 # Evaluate a trained checkpoint, dump CSV + plots
├── compare_baseline.py         # 3-way: MAPPO vs Fixed-Time vs Max-Pressure
├── compare_mappo_variants.py   # MAPPO v1 vs v2 vs paper-baseline comparison
├── fixed-cycles.py             # Fixed-time baseline controller
├── max-pressure.py             # Max-pressure baseline controller
├── validate_edges.py           # SUMO edge connectivity diagnostic
├── configs/
│   ├── mappo_config_v2.yaml        # Canonical MAPPO (current)
│   ├── mappo_config.yaml           # Semester-1 frozen v1 baseline
│   ├── mappo_baseline_paper.yaml   # Yu et al. 2021, Hanabi adopted preset
│   └── ippo_config.yaml            # IPPO comparator (decentralized critic)
├── models/
│   ├── mappo_model.py          # MAPPOModelCentralizedCritic (RLlib custom model)
│   └── ippo_model.py           # IPPOModelDecentralizedCritic (RLlib custom model)
├── marl_env/
│   ├── sumo_env.py             # SUMOTrafficEnv (RLlib MultiAgentEnv)
│   ├── obs_builder.py          # 70-dim observation builder (local + neighbour)
│   └── reward_function.py      # 5-component reward (waiting/queue/throughput/pressure/neighbour-pressure)
├── sumo_network/               # 2x2 grid: marl-proj.{net,rou,sumocfg,ttl,add,nod}.xml
├── results/                    # Ray Tune output (checkpoints + per-iteration metrics)
├── metrics/                    # Evaluation outputs (CSV time-series + PNG plots)
├── logs/tensorboard/           # TensorBoard training logs
└── tests/                      # Validation and setup scripts
```

## Prerequisites

- Python 3.9+ (developed on Python 3.11, Windows 11)
- SUMO v1.x — Windows binaries (`sumo.exe` / `sumo-gui.exe`) on `PATH`, with `SUMO_HOME` set
- CUDA 11.8+ or 12.1+ for GPU training (developed on RTX 3060 Ti)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

**TraCI vs libsumo:** the code prefers `libsumo` (8× faster) and falls back to standard TraCI when libsumo fails to load. On the development machine libsumo fails at startup, so standard TraCI is what actually runs — do not force `LIBSUMO_AS_TRACI=1` and expect a speedup.

## Quick start

### Train MAPPO (canonical v2)

```bash
python train_mappo.py --config configs/mappo_config_v2.yaml --iterations 1000
```

The script defaults `--config` to the v1 baseline (`mappo_config.yaml`) for legacy reasons — **always pass `--config` explicitly** to use v2.

### Train IPPO (decentralized critic)

```bash
python train_ippo.py --config configs/ippo_config.yaml --iterations 1000
```

`ippo_config.yaml` is a verbatim copy of `mappo_config_v2.yaml` except for the model registration, so the only varied factor between the two experiments is critic centralization (parameter sharing, hyperparameters, actor architecture, and reward are all identical).

### Resume from a checkpoint

```bash
python train_mappo.py --config configs/mappo_config_v2.yaml \
    --resume results/mappo_traffic_control/PPO_sumo_traffic_<run_id>/checkpoint_000101
```

### Watch training metrics

```bash
tensorboard --logdir logs/tensorboard/
```

### Evaluate a trained policy

```bash
# Headless, 3 episodes
python evaluate.py --checkpoint results/mappo_traffic_control/<run_id> --episodes 3 --seed 42

# With SUMO GUI
python evaluate.py --checkpoint <path> --gui

# 3-way comparison: MAPPO vs Fixed-Time vs Max-Pressure
python compare_baseline.py --checkpoint <path> --episodes 1 --seed 42
```

`evaluate.py` and `compare_baseline.py` currently load the MAPPO model class. To evaluate IPPO checkpoints, an `--algo {mappo,ippo}` flag still needs to be wired in (tracked in CLAUDE.md as outstanding).

Outputs land in `metrics/`:
- `mappo_ep<N>_metrics.csv` — per-second time series (halts, arrivals, wait, speed)
- `mappo_ep<N>_halting.png`, `*_arrivals_wait.png`, `*_per_agent.png`
- `comparison_all_overlay.png`, `*_summary.png`, `*_heatmap.png` (from `compare_baseline.py`)

## Configs

All four configs share an identical environment block (SUMO files, agents, network topology, detectors, reward weights). They differ only in algorithm hyperparameters and model architecture.

| Config | Purpose | Critic | Notable hyperparameters |
| --- | --- | --- | --- |
| `mappo_config_v2.yaml` | Canonical MAPPO (current) | Centralized [512, 256, 128] | lr=5e-4, grad_clip=1.0, epochs=15, entropy=0.02 |
| `mappo_config.yaml` | Frozen Semester-1 baseline | Centralized [256, 128, 64] | lr=4e-4, grad_clip=0.5 |
| `mappo_baseline_paper.yaml` | Yu et al. 2021 Hanabi adopted preset | Centralized [512, 512] (ReLU) | lr=7e-4, grad_clip=10.0, vf_clip=0.2, entropy=0.015 |
| `ippo_config.yaml` | IPPO comparator | **Decentralized** [512, 256, 128] | All other knobs identical to MAPPO v2 |

See `../CLAUDE.md` for the full hyperparameter table and the detailed rationale behind each preset.

## Environment specifics (2×2 grid)

- **Junctions**: `J1` (top-left), `J2` (top-right), `J3` (bottom-left), `J4` (bottom-right)
- **Episode**: 3,600 simulation seconds, action every 5s (720 decisions/episode)
- **Observation**: 70-dim per agent — 28 local features + 21 features × 2 neighbours
- **Action space**: discrete, 4 phases (NS through+right, NS left, EW through+right, EW left)
- **Reward**: `r = -1.0·W -0.25·Q + 0.1·T - 0.4·P - 0.5·N`, clipped to `[-3.0, 1.0]`
  - W: cumulative waiting time, Q: queue length, T: throughput, P: positive pressure, N: neighbour pressure

The full SUMO network is in `sumo_network/marl-proj.*` and is committed to the repo — there is no setup script to run before training.

## Outputs

| Path | Contents |
| --- | --- |
| `results/mappo_traffic_control/` | MAPPO Ray Tune trial dirs, checkpoints, `progress.csv` |
| `results/ippo_traffic_control/` | IPPO trial dirs (created on first IPPO run) |
| `logs/tensorboard/` | Per-iteration scalars (reward, KL, entropy, value loss, explained variance) |
| `metrics/` | Evaluation CSVs and PNG plots produced by `evaluate.py` and `compare_baseline.py` |

Checkpoints are saved every 50 iterations and the 5 most recent are kept (configurable via `keep_checkpoints_num` in the config).

## Reproducibility

- Random seed defaults to **42** across NumPy, PyTorch, and SUMO.
- Override with `--seed` on `train_mappo.py`, `train_ippo.py`, `evaluate.py`, and `compare_baseline.py`.
- The Semester-1 result (98.5% reward improvement, converged at −26.28 ± 0.61 over iterations 82–101) was produced with `mappo_config.yaml`. To reproduce that exact run, use the v1 config — not v2.

## See also

- `../CLAUDE.md` — research methodology, Semester-2 plan, and full implementation reference
- `tests/` — environment and edge-connectivity validators
- `docs/` — additional design notes