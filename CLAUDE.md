# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project on emergent social behavior and dilemmas in Multi-Agent Reinforcement Learning (MARL), applied to adaptive traffic signal control. Four cooperative MAPPO agents control traffic lights at junctions J1–J4 in a 2×2 grid SUMO simulation.

All active code lives in `RP-5/`.

## Running Scripts

```bash
# Validate environment setup (SUMO, CUDA, dependencies)
python RP-5/tests/setup_validation.py

# Train MAPPO agents
python RP-5/train_mappo.py

# Evaluate a trained checkpoint
python RP-5/evaluate.py

# Compare MAPPO vs fixed-cycles vs max-pressure baselines
python RP-5/compare_baseline.py

# Run individual tests
python RP-5/tests/test_env.py
python RP-5/tests/test_sumo_connection.py
python RP-5/tests/test_pressure.py
```

No pytest integration — tests are standalone scripts that print validation output.

## Dependencies

Install via: `pip install -r RP-5/requirements.txt`

Key packages: `ray[rllib]==2.35.0`, `torch>=2.0.0`, `gymnasium>=0.28.1`, `sumo-rl>=1.4.3`, `traci>=1.19.0`. SUMO must be installed separately and available on PATH.

## Architecture

### Environment (`marl_env/`)

- **`sumo_env.py`** — Ray RLlib `MultiAgentEnv` wrapping SUMO via TraCI. Manages 4 agents (J1–J4), 4-action discrete phase space, yellow-time transitions, and min-green enforcement.
- **`obs_builder.py`** — `MAPPOObservationBuilderV2` produces 70-dim observations per agent:
  - **28-dim local:** queue lengths (12), current phase + elapsed time (3), movement pressures + derivatives (12), min-green flag (1)
  - **42-dim neighbor (21 × 2):** shared queue metrics, direction info, pressures, outgoing metrics (neighbor→agent), ingoing metrics (agent→neighbor)
  - Outgoing/ingoing metrics are the key research innovation enabling green-wave and backpressure learning.
- **`reward_function.py`** — 5-component weighted reward: queue penalty (−0.25), waiting time penalty (−1.0, primary), throughput bonus (+0.1), pressure penalty (−0.4), neighbor pressure (−0.5 with spatial discounting 0.9). Clipped to [−3.0, 1.0].

### Model (`models/mappo_model.py`)

`MAPPOModelCentralizedCritic` (extends `TorchModelV2`):
- **Actor (decentralized):** 70 → 128 → 64 → 4 actions per agent
- **Critic (centralized):** 280 (70 × 4 agents) → 256 → 128 → 64 → 1 value

Uses orthogonal initialization, value normalization via running mean/std.

### Training (`train_mappo.py`)

Uses Ray Tune + RLlib PPO with parameter sharing (`shared_policy`). Key settled hyperparameters: `train_batch_size=32768`, `sgd_minibatch_size=4096`, `num_sgd_iter=10` (NOT 15 — causes instability), `entropy_coeff=0.02`, `gamma=0.99`, `lambda=0.95`. Runs 3 rollout workers. Checkpoints saved to `RP-5/results/`.

### Baselines

- **`fixed-cycles.py`** — Fixed-time controller (30s NS / 30s EW, 3s yellow)
- **`max-pressure.py`** — Adaptive heuristic selecting phase that maximizes upstream−downstream queue pressure
- **`compare_baseline.py`** — Runs all three controllers with same seed (42) and generates comparative metrics/plots

### Configuration

All parameters in `RP-5/configs/mappo_config.yaml`: SUMO network paths, agent/junction definitions, detector-to-edge mappings, reward weights, network architecture, and training hyperparameters.

### Outputs

- `RP-5/results/` — Ray Tune checkpoints
- `RP-5/metrics/` — Evaluation CSVs and PNG plots
- `RP-5/sumo_network/` — SUMO simulation files (`marl-proj.*`)

## Key Implementation Details

- TraCI port management in `sumo_env.py` uses dynamic port allocation to support multi-worker Ray training without port conflicts.
- The `evaluate.py` script collects vehicle arrivals **during** the `delta_time` simulation loop (not after) — this is a critical correctness fix.
- Neighbor connectivity and detector-to-edge mappings are defined in `mappo_config.yaml` under `network_topology`; changing the SUMO network requires updating these mappings.
- Traffic demand: medium (`-p 2.0`) for training, heavy (`-p 1.0`) for evaluation stress testing.
