# Emergent Social Behaviour & Dilemmas in Multi-Agent Reinforcement Learning

**Project Type:** Graduation Research Project (AY 2025/2026)  
**Student:** Ahmed Wael Elsisi (214647)  
**Supervisor:** Dr. Randa Mohamed  
**Institution:** British University in Egypt - Electrical Engineering (Computer Engineering Programme)

---

## Project Mission

Investigate when and why cooperation emerges versus exploitation in multi-agent reinforcement learning (MARL) systems through systematic comparison across the coordination-dilemma spectrum.

### Core Research Questions

1. **Reward Structure Impact:** How do individual vs. shared rewards affect emergent cooperation patterns?
2. **Coordination Scaling:** Does coordination value increase as networks grow from small (2×2) to large (5×5) grids?
3. **Mechanism Differences:** How do cooperation mechanisms differ between pure coordination problems (traffic) and social dilemmas (resource exploitation)?
4. **Efficiency-Equity Trade-offs:** Can high system performance coexist with fair outcome distribution?

---

## Research Methodology

### Two-Semester Comparative Study

**Semester 1 (COMPLETED):** Coordination Baseline
- Environment: Traffic signal control (inherently cooperative)
- Network: 2×2 grid, 4 signalized intersections
- Algorithm: MAPPO (Multi-Agent Proximal Policy Optimization)
- Goal: Establish baseline where cooperation naturally benefits all agents

**Semester 2 (PLANNED):** Social Dilemma Testing
- Phase 1: On the existing 2×2 grid, compare the currently implemented MAPPO against the paper baseline MAPPO (Yu et al. 2021, "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games") and against IPPO. **5×5 grid scaling is deprioritized** — see "Scope changes" below.
- Phase 2: Implement social dilemma environment (Harvest/custom/predator-prey)
- Phase 3: Cross-environment analysis of cooperation mechanisms

**Scope changes (in-flight):**
- The originally planned 5×5 grid scale-up has been **set aside**. The MAPPO/IPPO comparison and the social-dilemma phase are now the priority — extending the SUMO network to 25 intersections costs implementation and training time without changing the conclusions the project is trying to draw about cooperation emergence. If time remains after Phases 1 and 2 land, 5×5 can be revisited.

### The Spectrum Being Studied
```
PURE COORDINATION ←――――――――――――――――→ SOCIAL DILEMMA
(Traffic Control)                    (Resource Sharing)

Cooperation benefits     Individual gain from
everyone                exploitation tempts
No exploitation          Collective harm if all
temptation              exploit
```

---

## Technical Stack

### Core Technologies

- **Simulator:** SUMO (Simulation of Urban MObility) v1.x — Windows binaries (`sumo.exe` / `sumo-gui.exe`)
- **Interface:** TraCI (standard only — libsumo fails to start on this setup, do NOT attempt to use it)
- **RL Framework:** Ray RLlib 2.35.0 (distributed training via Ray Tune)
- **Algorithm:** MAPPO (Multi-Agent PPO) with CTDE paradigm
- **Deep Learning:** PyTorch (via RLlib, framework="torch")
- **Visualization:** TensorBoard for training metrics
- **Hardware:** 1× RTX 3060Ti GPU + AMD Ryzen 5 3600 (3 CPU workers)
- **Platform:** Windows 11 (use forward slashes in paths, Unix shell via Git Bash)

### Key Dependencies
```python
# Core Dependencies
ray[rllib]==2.35.0
torch>=2.0.0
gymnasium>=0.28.1
numpy>=1.24.0
pandas>=2.0.0

# SUMO Integration
sumo-rl>=1.4.3
traci>=1.19.0

# Monitoring and Logging
tensorboard>=2.13.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
PyYAML>=6.0
tqdm>=4.65.0
scipy>=1.11.0

# Optional GPU Support (for RTX 3060 Ti)
# Ensure CUDA 11.8+ or 12.1+ is installed
# torch will use CUDA automatically if available
```

---

## Implementation Architecture

### MARL Setup (Semester 1)

**Environment:**
- 2×2 grid network (4 agents = 4 intersections: J1, J2, J3, J4)
- Roads: 150m between intersections, 100m entry/exit
- Lane configuration: 3-lane roads with strict directional assignments
  - Lane 0: Right turns
  - Lane 1: Straight
  - Lane 2: Left turns
- Episode: 3,600 simulation seconds, 5-second action frequency = 720 decisions/episode
- Traffic pattern: Light (0-600s) → Rush hour (600-2400s) → Light (2400-3600s)

**Observation Space (70 dimensions per agent):**

*Local features (28 dim):*
- Queue lengths: 4 edges × 3 movements = 12 detectors, normalized (12 dim)
- Current signal direction: NS or EW, one-hot encoded (2 dim)
- Elapsed phase time: Normalized by 60s (1 dim)
- Movement pressures: Incoming - outgoing, per movement (6 dim: NS right/straight/left, EW right/straight/left)
- Pressure derivatives: Rate of change per movement (6 dim)
- Min-green constraint flag: 1 if ≥10s elapsed, else 0 (1 dim)

*Neighbor features (42 dim = 21 per neighbor × 2 neighbors):*
Per neighbor (21 dim):
- Shared queues from neighbor (3 dim)
- Neighbor signal direction: one-hot (2 dim)
- Combined movement pressures: right/straight/left (3 dim)
- Total pressure scalar (1 dim)
- OUTGOING metrics — traffic from neighbor TO agent (6 dim): queue × 3 + avg wait × 3
- INGOING metrics — traffic from agent TO neighbor (6 dim): queue × 3 + available space × 3

Network topology (for neighbor lookup):
- J1 neighbors: J2, J3
- J2 neighbors: J1, J4
- J3 neighbors: J1, J4
- J4 neighbors: J2, J3

**Action Space (Discrete, 4 phases):**
```
A = {
  a₀ (Action 0) → Phase 0: NS through + right turns
  a₁ (Action 1) → Phase 6: NS left turns
  a₂ (Action 2) → Phase 2: EW through + right turns
  a₃ (Action 3) → Phase 4: EW left turns
}
Right turns are permissive for all phases.
enforce_min_green: false (agents learn optimal timing autonomously)
```

**Reward Function (Multi-component):**
```
r_t = -1.0·W_t - 0.25·Q_t + 0.1·T_t - 0.4·P_t - 0.5·N_t
Clipped to range: [-3.0, 1.0]

Where:
W_t = Normalized cumulative waiting time (primary objective)
Q_t = Normalized queue length (halted vehicles / max)
T_t = Normalized throughput (departed vehicles / expected)
P_t = Normalized positive pressure (incoming - outgoing, clipped at 0)
N_t = Normalized neighbor pressure (spatially discounted, γ=0.9)

Normalization constants:
  queue_max: 100 vehicles/lane
  phase_time_max: 60 seconds
  pressure_max: 100 vehicles
  waiting_time_max: 60 seconds
  spatial_discount: 0.9
```

**Neural Network Architecture:**

*Actor Network (Decentralized, class: MAPPOModelCentralizedCritic):*
- Input: 70 dim (local observations only)
- Hidden 1: 128 units, Tanh (orthogonal init)
- Hidden 2: 64 units, Tanh (orthogonal init)
- Output: 4 units (action logits, gain=0.01)

*Critic Network (Centralized):*
- Input: 280 dim (4 agents × 70 dim, global state)
- Hidden 1: 512 units, ReLU (orthogonal init)
- Hidden 2: 256 units, ReLU (orthogonal init)
- Hidden 3: 128 units, ReLU (orthogonal init)
- Output: 1 unit, Linear (value estimate)
- Value normalization enabled (running mean/std, momentum=0.99)

Actor activation is configurable via `actor_activation` in the model config (`models/mappo_model.py`); v2 uses Tanh.

*All 4 agents share a single policy ("shared_policy") — parameter sharing.*

**MAPPO Hyperparameters (from configs/mappo_config_v2.yaml — current canonical MAPPO):**
```python
{
    "lr": 5e-4,                    # Learning rate (v2; v1 was 4e-4)
    "gamma": 0.99,                 # Discount factor
    "lambda_": 0.95,               # GAE lambda
    "sgd_minibatch_size": 4096,
    "train_batch_size": 32768,
    "num_sgd_iter": 15,            # Epochs per update
    "clip_param": 0.2,             # PPO clip
    "vf_clip_param": 10.0,
    "grad_clip": 1.0,              # v2; v1 was 0.5
    "entropy_coeff": 0.02,
    "vf_loss_coeff": 1.0,
    "num_rollout_workers": 3,
    "rollout_fragment_length": 200,
    "batch_mode": "complete_episodes",
    "framework": "torch",
    "num_gpus": 1,
    "num_gpus_per_worker": 0,
    "vf_share_layers": false,      # Separate actor and critic networks
    "use_orthogonal_init": true,
    "use_value_normalization": true,
    "observation_filter": "MeanStdFilter"  # Running mean/std on obs (universal across all configs)
}
```

**MAPPO config variants (RP-5/configs/):**
- `mappo_config_v2.yaml` — **current canonical MAPPO** (improvised): lr=5e-4, grad_clip=1.0, critic=[512,256,128], richer per-junction `ns_edges`/`ew_edges` and an `edge_connectivity` block enabling directional neighbour pressure metrics.
- `mappo_config.yaml` — Semester-1 frozen v1 baseline (lr=4e-4, grad_clip=0.5, critic=[256,128,64]). Kept for reproducing the Semester-1 results.
- `mappo_baseline_paper.yaml` — Yu et al. (2021) "Surprising Effectiveness of PPO" reference hyperparameters, **Hanabi adopted preset** (Tables 11 + 18): lr=7e-4 (actor), epoch=15, mini-batch=1 → sgd_minibatch=train_batch_size=32768, clip=0.2 (policy + value), entropy=0.015, ReLU, MLP [512, 512] for both actor and critic, max_grad_norm=10.0. Hanabi is the only adopted MAPPO preset that uses MLP (no GRU), making it the closest paper-published config for our setup.
- `ippo_config.yaml` — **IPPO comparator**: copy of `mappo_config_v2.yaml` with two differences — `custom_model: "ippo_decentralized"` and `reward_config.neighbor_pressure_weight: 0.0`. All algorithm hyperparameters (lr, gamma, λ, clip, batches, entropy, grad_clip, actor/critic hidden sizes) are identical to MAPPO v2; what differs is the **full decentralized package**: decentralized critic + purely local reward. This makes IPPO a clean "fully independent agents" comparator against MAPPO's "fully cooperative CTDE" setup.

**IPPO scaffolding (Semester 2, Phase 1, Task 1.3):**
- `RP-5/models/ippo_model.py` — `IPPOModelDecentralizedCritic`. Same actor as MAPPO; critic input is the agent's own 70-dim local observation (vs MAPPO's 280-dim concatenated global state). No `centralized_critic_postprocessing` hook — RLlib's default PPO postprocessing computes GAE on local obs. Registered as `"ippo_decentralized"`.
- `RP-5/train_ippo.py` — IPPO training entry point. Mirrors `train_mappo.py` but with no `postprocess_fn` on the `PolicySpec`, model name swapped to `ippo_decentralized`, and Ray Tune experiment name `"ippo_traffic_control"` so checkpoints land in their own directory (`results/ippo_traffic_control/`). Defaults to `--config configs/ippo_config.yaml`.
- **Reward**: IPPO zeroes out `neighbor_pressure_weight` (the sole multi-agent coupling term in `reward_function.py`). Each agent optimizes only its own intersection metrics — queue, waiting time, throughput, and its own pressure. MAPPO keeps the neighbor coupling term active.
- **Parameter sharing**: still preserved — all 4 agents share a single `"shared_policy"` for IPPO, matching MAPPO setup. So the differences between MAPPO and IPPO are exactly: (1) centralized vs decentralized critic and (2) shared (with neighbour coupling) vs purely local reward. This is the "coordinated MARL" vs "independent learners" contrast, not a single-variable critic ablation.
- `evaluate.py` and `compare_baseline.py` still hardcode the MAPPO model — they need an `--algo {mappo,ippo}` flag (or a parallel `evaluate_ippo.py`) before IPPO checkpoints can be evaluated.

**What changed in v2 vs v1 (Semester-1 baseline):**
- **Hyperparameters:** lr 4e-4 → 5e-4; grad_clip 0.5 → 1.0.
- **Critic capacity:** [256, 128, 64] → [512, 256, 128] (≈2× wider first hidden).
- **Per-junction directional metadata:** new `ns_edges` / `ew_edges` lists inside the `detectors:` block, separating north/south from east/west incoming edges.
- **Neighbour edge connectivity:** new `edge_connectivity:` block giving the explicit outgoing/ingoing edges between every adjacent junction pair (J1↔J2, J1↔J3, J2↔J4, J3↔J4). Enables direction-aware neighbour pressure features.
- **Configurable actor activation:** `models/mappo_model.py` now reads `actor_activation` from the config (was hard-coded to Tanh).

### Training Configuration

- **Semester 1 Actual Run:** 101 iterations (stopped manually after convergence)
- **Config Target:** 1000 iterations (stopping criteria: `episode_reward_mean: -5`)
- **Duration:** ~28 hours (101 iterations)
- **Parallel Workers:** 3 rollout workers + 1 training process (GPU)
- **Evaluation:** Every 50 iterations (10 episodes, deterministic)
- **Checkpointing:** Every 50 iterations, keep 5 most recent
- **Results saved to:** `RP-5/results/mappo_traffic_control/`
- **TensorBoard logs:** `RP-5/logs/tensorboard/`
- **Seed:** 42 (NumPy, PyTorch, SUMO)

---

## Semester 1 Results (Achieved)

### Performance Metrics

- **Improvement:** 98.5% from baseline
- **Initial Reward:** -1,767.98 (iteration 1)
- **Final Reward:** -26.28 ± 0.61 (iterations 82-101, std dev across last 10)
- **Explained Variance:** 0.864 (critic prediction accuracy)
- **KL Divergence:** 0.00543 average (stable policy updates)
- **Entropy:** 0.728 final (retained stochasticity, max = ln(4) ≈ 1.39)

### Learning Phases

1. Initial learning (iter 1-10): -681.02 avg (high variance exploration)
2. Rapid improvement (iter 10-50): -71.09 avg (discovering coordination)
3. Fine-tuning (iter 50-82): -31.32 avg (policy refinement)
4. Convergence (iter 82-101): -27.08 ± 0.61 (stable performance)

### Emergent Behaviors Observed

- ✓ Offset signal phase timing (prevent downstream gridlock)
- ✓ Network-wide coordination without explicit communication
- ✓ Spatial awareness via neighbor pressure term
- ✓ Robust to traffic pattern variations

### Baseline Comparisons

Outperformed both heuristic baselines:
- Fixed-Time Controller: Pre-programmed cycles
- Max-Pressure Controller: Minimize incoming-outgoing imbalance

---

## Semester 2 Plan (In Progress)

### Phase 1: Traffic Scaling & IPPO Comparison (Weeks 1-3)

**Task 1.1:** Compare against baseline MAPPO
- Retrieve hyperparameters and neural network configuration from paper "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
- Apply the hyperparameters and configuration from the mentioned paper and compare against our currently implemented MAPPO
- Output comparison metrics and analyze the performance

**Task 1.2 (DEPRIORITIZED):** Scale environment 2×2 → 5×5 (25 intersections)
- Originally planned but **set aside**: the MAPPO-vs-paper-baseline and MAPPO-vs-IPPO comparisons (Tasks 1.1, 1.3, 1.4) and the social-dilemma phase carry the research conclusions, and the scale-up is implementation-heavy without changing those conclusions.
- If revisited later: maintain lane config and detector setup, adjust traffic demand proportionally, verify observation space (70 dim/agent, 1,750 dim centralized critic). The 5×5 SUMO network skeleton was prototyped earlier but reverted (see commits `fd14409`, `350242d`).

**Task 1.3:** Implement Independent PPO (IPPO) — *scaffolded*
- Status: model (`models/ippo_model.py`), config (`configs/ippo_config.yaml`), and training entry point (`train_ippo.py`) all created. Ready to train.
- **Decentralized critic**: each agent's value function sees only its own 70-dim local observation (vs MAPPO's 280-dim concatenated global state).
- **Local rewards**: `neighbor_pressure_weight` is zeroed in the IPPO config so each agent optimizes only its own intersection metrics — no cooperative / shared reward signal. MAPPO retains the neighbour coupling term.
- All algorithm hyperparameters identical to MAPPO v2 (lr, gamma, λ, clip, batches, entropy, grad_clip, actor/critic hidden sizes). The differences between MAPPO and IPPO are exactly the two MARL design choices: (1) centralized vs decentralized critic and (2) shared vs local reward — the "fully cooperative" vs "fully independent" contrast.
- Parameter sharing preserved: all 4 agents share a single `"shared_policy"`, matching MAPPO setup.
- Outstanding: extend `evaluate.py` and `compare_baseline.py` with `--algo {mappo,ippo}` flag before IPPO checkpoints can be evaluated against the baselines.

**Task 1.4:** Comparative evaluation
- Metrics: Network waiting time, throughput, queue lengths
- Analysis: Coordination value = (MAPPO performance - IPPO performance)

### Phase 2: Social Dilemma Environment (Weeks 4-9)

**Environment Candidates:**
1. **Harvest (PettingZoo):** Apple collection with sustainability dilemma
2. **Custom Traffic Dilemma:** Asymmetric tolling scenario
3. **Predator-Prey:** Cooperative hunting with exploitation opportunities

**Selection Criteria:**
- Measurable cooperation metrics
- Clear individual-vs-collective tension
- RLlib compatibility

**Experiments:**
- Train IPPO and MAPPO in selected dilemma
- Log cooperation-specific metrics (resource sustainability, collective welfare)
- Population-based training (if time permits)

### Phase 3: Cross-Environment Analysis (Weeks 10-15)

**Synthesis:**
- Compare learned behaviors: Traffic (coordination) vs Dilemma (conflict)
- Efficiency-equity analysis (Gini coefficient)
- Statistical significance testing
- Video demonstrations of emergent behaviors

---

## Code Organization

### Directory Structure
```
Applied/
├── CLAUDE.md                          # Project instructions (this file)
├── RP-5/                              # Semester 1 (MAPPO) + Semester 2 Phase 1 (IPPO) — 2×2 grid
│   ├── README.md                      # Code-level README for the RP-5 directory
│   ├── train_mappo.py                 # MAPPO training entry point (centralized critic)
│   ├── train_ippo.py                  # IPPO training entry point (decentralized critic)
│   ├── evaluate.py                    # MAPPO evaluation (arrival-tracking fix; needs --algo for IPPO)
│   ├── compare_baseline.py            # 3-way comparison: MAPPO vs Fixed vs MaxP
│   ├── compare_mappo_variants.py      # MAPPO v1 vs v2 vs paper-baseline comparison
│   ├── fixed-cycles.py                # Fixed-time baseline controller
│   ├── max-pressure.py                # Max-pressure baseline controller
│   ├── validate_edges.py              # SUMO edge connectivity validator
│   ├── configs/
│   │   ├── mappo_config_v2.yaml       # CURRENT MAPPO — improvised hyperparams + edge_connectivity
│   │   ├── mappo_config.yaml          # Semester-1 frozen baseline (legacy)
│   │   ├── mappo_baseline_paper.yaml  # Yu et al. (2021) paper-baseline comparator
│   │   └── ippo_config.yaml           # IPPO comparator (decentralized critic; rest = MAPPO v2)
│   ├── marl_env/
│   │   ├── sumo_env.py                # SUMOTrafficEnv (RLlib MultiAgentEnv)
│   │   ├── obs_builder.py             # MAPPOObservationBuilderV2 (70-dim)
│   │   └── reward_function.py         # MAPPORewardFunction (5-component)
│   ├── models/
│   │   ├── mappo_model.py             # MAPPOModelCentralizedCritic (custom RLlib model)
│   │   └── ippo_model.py              # IPPOModelDecentralizedCritic (custom RLlib model)
│   ├── sumo_network/
│   │   ├── marl-proj.net.xml          # 2×2 road network topology
│   │   ├── marl-proj.rou.xml          # Vehicle routes and demand
│   │   ├── marl-proj.sumocfg          # SUMO simulation config
│   │   ├── marl-proj.ttl.xml          # Traffic light logic (phase definitions)
│   │   ├── marl-proj.add.xml          # Detectors (E2 lanearea sensors)
│   │   └── marl-proj.nod.xml          # Node definitions
│   ├── results/
│   │   ├── mappo_traffic_control/     # MAPPO Ray Tune output (checkpoints + metrics)
│   │   └── ippo_traffic_control/      # IPPO Ray Tune output (created on first IPPO run)
│   ├── metrics/                       # Evaluation outputs (CSV + PNG plots)
│   ├── logs/
│   │   └── tensorboard/               # TensorBoard training logs
│   └── tests/                         # Validation and setup scripts
└── Emergent Social Behaviour... Interim.pdf  # Interim report (Semester 1)
```

### Key Files to Reference

- **Network file:** `RP-5/sumo_network/marl-proj.net.xml` — 2×2 road topology
- **Route file:** `RP-5/sumo_network/marl-proj.rou.xml` — vehicle demand
- **Config file:** `RP-5/sumo_network/marl-proj.sumocfg` — ties network + routes + params
- **TL logic:** `RP-5/sumo_network/marl-proj.ttl.xml` — phase index → signal string mapping
- **Detectors:** `RP-5/sumo_network/marl-proj.add.xml` — E2 lanearea detector definitions
- **Config (current MAPPO):** `RP-5/configs/mappo_config_v2.yaml` — improvised hyperparameters + extended environment metadata (ns/ew edge breakdown, neighbour edge_connectivity)
- **Config (legacy):** `RP-5/configs/mappo_config.yaml` — Semester-1 frozen baseline; kept to reproduce Semester-1 results
- **Config (paper comparator):** `RP-5/configs/mappo_baseline_paper.yaml` — Yu et al. (2021) hyperparameters
- **Config (IPPO comparator):** `RP-5/configs/ippo_config.yaml` — decentralized critic; all other hyperparameters identical to MAPPO v2
- **Training (MAPPO):** `RP-5/train_mappo.py` — run with `python train_mappo.py --config configs/mappo_config_v2.yaml` (script default is still v1 — always pass `--config` explicitly)
- **Training (IPPO):** `RP-5/train_ippo.py` — run with `python train_ippo.py --config configs/ippo_config.yaml`
- **Evaluation:** `RP-5/evaluate.py` — run with `python evaluate.py --checkpoint <path>` (currently MAPPO-only; needs `--algo` flag for IPPO checkpoints)
- **Checkpoints (MAPPO):** `RP-5/results/mappo_traffic_control/PPO_sumo_traffic_<run_id>/`
- **Checkpoints (IPPO):** `RP-5/results/ippo_traffic_control/PPO_sumo_traffic_<run_id>/`

---

## Common Development Tasks

### Training a New Model
```bash
# Train the current (improvised) MAPPO from scratch
python train_mappo.py --config configs/mappo_config_v2.yaml --iterations 1000

# Train the IPPO comparator (decentralized critic)
python train_ippo.py --config configs/ippo_config.yaml --iterations 1000

# Resume from checkpoint
python train_mappo.py --config configs/mappo_config_v2.yaml --resume results/mappo_traffic_control/<run_id>/checkpoint_000101
```

Note: `train_mappo.py` (and the other RP-5 scripts) currently still default to `mappo_config.yaml` (the v1 baseline). Always pass `--config configs/mappo_config_v2.yaml` explicitly to use the improvised MAPPO until the script defaults are flipped.

Key workflow:
1. Edit `configs/mappo_config_v2.yaml` to set hyperparameters
2. Run `train_mappo.py` — Ray initializes workers, each spawns a SUMO instance
3. Each worker gets a unique TraCI port (PID-based, 10000–65000 range)
4. TensorBoard: `tensorboard --logdir logs/tensorboard/`
5. Checkpoints saved every 50 iterations to `results/mappo_traffic_control/`

### Evaluation & Analysis
```bash
# Evaluate MAPPO vs baselines
python compare_baseline.py --checkpoint results/mappo_traffic_control/<run_id> --episodes 1 --seed 42

# Evaluate MAPPO only
python evaluate.py --checkpoint results/mappo_traffic_control/<run_id> --episodes 3 --seed 42

# Run with GUI
python evaluate.py --checkpoint <path> --gui
```

Critical implementation note: `evaluate.py` uses `EnvWrapperWithMetrics` to hook into
SUMO's delta_time loop and capture arrivals every simulation second (not just every 5s
RL step). This is required to correctly count ~1,200 vehicle completions per episode.

Outputs saved to `metrics/`:
- `mappo_ep<N>_metrics.csv` — time-series (halts, arrivals, wait, speed)
- `mappo_ep<N>_halting.png`, `*_arrivals_wait.png`, `*_per_agent.png`
- `comparison_all_overlay.png`, `comparison_all_summary.png`, `comparison_all_heatmap.png`

### Key Metrics to Track

**Training Metrics:**
- `episode_reward_mean`: Average reward across agents
- `policy_loss`: Actor network optimization
- `vf_loss`: Critic network optimization
- `kl`: Policy update magnitude (should stay low)
- `entropy`: Exploration level (should decay gradually)
- `explained_variance`: Critic prediction accuracy (target >0.8)

**Evaluation Metrics:**
- Total waiting time (primary optimization goal)
- Vehicle throughput (completed trips)
- Queue lengths (halted vehicle counts)
- Pressure imbalance (incoming - outgoing)

**Cooperation Metrics (S2):**
- Resource sustainability (for Harvest)
- Collective welfare indicators
- Gini coefficient (fairness/equity)
- Exploitation rates

---

## Research Context & Terminology

### MARL Core Concepts

**Centralized Training with Decentralized Execution (CTDE):**
- Training: Centralized critic accesses global state (all agents' obs + actions)
- Execution: Decentralized actors use only local observations
- Analogy: "Coach sees all during practice, players act independently in game"
- Resolves non-stationarity while maintaining scalability

**Non-Stationarity Problem:**
- Each agent sees others as part of environment
- But those "parts" are learning and changing
- Violates Markov assumption (stationary transition dynamics)
- CTDE addresses by conditioning critic on joint state

**Credit Assignment:**
- With shared rewards, which agent deserves credit?
- MAPPO's centralized critic helps attribute value
- Value decomposition methods (QMIX) provide alternative approach

**Independent Learning (IQL, IPPO):**
- Each agent treats others as environment
- No coordination mechanism
- Simple, scalable, but ignores multi-agent structure
- Serves as baseline for measuring coordination value

### Social Dilemma Concepts

**Prisoner's Dilemma:**
- Individual rational choice = defect
- Mutual cooperation yields better outcome
- But cooperation vulnerable to exploitation

**Tragedy of the Commons:**
- Shared resource with individual access
- Each benefits from exploitation, costs distributed
- Overexploitation destroys resource for all

**Sequential Social Dilemmas (SSD):**
- Multi-timestep grid world environments
- Dilemma structure persists over time
- Agents must balance immediate vs long-term
- Examples: Harvest (apple gathering), Cleanup

### Algorithm Comparisons

| Algorithm | Training | Execution | Strengths | Weaknesses |
|-----------|----------|-----------|-----------|------------|
| **MAPPO** | Centralized critic | Decentralized actor | Strong coordination, stable | Requires global state during training |
| **IPPO** | Independent | Decentralized | Simple, scalable | No coordination mechanism |
| **QMIX** | Centralized mixer | Decentralized | Value decomposition | Monotonicity constraint limits flexibility |
| **MADDPG** | Centralized critic | Decentralized actor | Handles continuous actions | Complex, less stable than PPO |

---

## Anticipated Challenges & Solutions

### Technical Challenges

**Challenge:** 5×5 grid computational cost
- Resolution: deprioritized for Semester 2 (see Scope changes). If revisited, maintain 3 workers and extend training time as needed.

**Challenge:** Social dilemma environment selection
- Solution: Allocated Weeks 3-4 for evaluation, have fallback to traffic-only

**Challenge:** Fair IPPO-MAPPO comparison
- Solution: Keep all algorithm hyperparameters and the actor architecture identical; vary only the two MARL design choices that define the contrast — centralized vs decentralized critic, and shared/neighbour-coupled vs purely local reward. This is a "fully cooperative MARL package" vs "fully independent learners" comparison, not a single-variable critic ablation.

### Experimental Challenges

**Challenge:** Measuring cooperation in dilemmas
- Solution: Design environment-specific metrics (resource sustainability, collective welfare)

**Challenge:** Ensuring reproducibility
- Solution: Fixed random seeds, comprehensive logging, checkpoint versioning

### Time Management

**Challenge:** Ambitious S2 scope
- Solution: Prioritize core experiments (traffic scaling + dilemma baseline), mark population training as optional

---

## Key References

1. **Sutton & Barto (2018):** RL fundamentals, Bellman equations, MDP framework
2. **Schulman et al. (2017):** PPO algorithm (clipping, stable updates)
3. **Schulman et al. (2016):** GAE for advantage estimation
4. **Yu et al. (2021):** MAPPO effectiveness in cooperative games
5. **Lowe et al. (2017):** MADDPG, CTDE paradigm
6. **Rashid et al. (2018):** QMIX value decomposition
7. **Tan (1993):** IQL baseline, independent learning
8. **Leibo et al. (2017):** Sequential social dilemmas
9. **Wei et al. (2019):** CoLight, pressure-based methods
10. **Chu et al. (2020):** Large-scale MARL for traffic

---

## Development Conventions

### Git Workflow

**After every meaningful unit of work, commit and push to GitHub.** This provides a safe revert point at all times — critical for a project with 28-hour training runs where a bad change may only surface much later.

Rules:
- Commit after each logical change: new file, config addition, bug fix, model edit, new script
- Write descriptive commit messages that say *what* changed and *why* (not just "update files")
- Always push immediately after committing — local-only commits offer no protection
- Never batch unrelated changes into one commit
- Use `git status` before committing to catch untracked files

```bash
cd "E:/Research/Emergent Social Behaviour and Dilemmas in MARL/Applied"
git add <specific files>
git commit -m "Short description of what and why"
git push
```

### Code Style
- PEP 8 for Python
- Type hints for function signatures
- Docstrings for all classes/functions
- Comprehensive inline comments for MARL-specific logic

### Experimentation
- Each experiment = one config file under `configs/`. `mappo_config_v2.yaml` is the current canonical MAPPO; `mappo_config.yaml`, `mappo_baseline_paper.yaml`, and `ippo_config.yaml` are kept for legacy / comparator runs.
- Unique run names with timestamps (auto-generated by Ray Tune)
- TensorBoard logs in `RP-5/logs/tensorboard/`
- Checkpoints in `RP-5/results/mappo_traffic_control/` (MAPPO) and `RP-5/results/ippo_traffic_control/` (IPPO)

### Documentation
- README for each environment directory
- Config file comments explaining all hyperparameters
- Analysis scripts with markdown cells explaining methodology

---

## Success Criteria

### Semester 1 (Achieved ✓)
- [x] MAPPO implementation trains successfully
- [x] Convergence demonstrated (reward plateaus)
- [x] Outperforms baselines (fixed-time, max-pressure)
- [x] Emergent coordination behaviors observed
- [x] Interim report submitted

### Semester 2 (Targets)
- [ ] Compare against baseline MAPPO from "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
- [ ] IPPO baseline trained for comparison
- [ ] Coordination value quantified (MAPPO - IPPO)
- [ ] Social dilemma environment implemented
- [ ] Cross-environment comparison complete
- [ ] Statistical analysis of cooperation mechanisms
- [ ] Final thesis submitted
- [ ] Defense presentation delivered
- [~] 5×5 traffic environment functional — **deprioritized** (see Scope changes); revisit only if time remains after Phases 1–2.

---

## Contact & Resources

**Student:** Ahmed Wael Elsisi (214647)  
**Supervisor:** Dr. Randa Mohamed  
**Institution:** British University in Egypt

**Key Resources:**
- SUMO Documentation: https://sumo.dlr.de/docs/
- Ray RLlib Docs: https://docs.ray.io/en/latest/rllib/
- PettingZoo (Multi-Agent Envs): https://pettingzoo.farama.org/

---

## Critical Implementation Details

### SUMO Edge Naming Convention
- Edge `E1` goes from A→B; edge `-E1` goes from B→A (SUMO negation convention)
- Outgoing edges = negation of incoming edges (used in reward function)
- Detector IDs follow pattern: `det_{edge}_{movement}_stop` (e.g., `det_-E6_0_stop`)
- J1 incoming edges: `-E6`, `E0`, `E16`, `-E1`
- J2 incoming edges: `E1`, `-E7`, `-E11`, `-E10`
- J3 incoming edges: `-E17`, `-E16`, `-E18`, `-E15`
- J4 incoming edges: `E15`, `E11`, `-E8`, `-E9`

### TraCI Port Management
- Each worker process gets a unique TraCI port: `10000 + (PID % 55000) + random(0,100)`
- Up to 5 retry attempts with port increment if collision occurs
- **Always use standard TraCI — libsumo fails to start on this machine and must NOT be used**

### Policy Sharing
- All 4 agents share a single policy `"shared_policy"` (parameter sharing)
- This means one set of actor/critic weights is trained across all agents
- Each agent still uses its own local observation for actor forward pass
- Centralized critic constructs 280-dim global state = concatenation of all 4 agents' 70-dim obs

### Phase Index Mapping
The SUMO `.ttl.xml` file defines 8 phases. Only 4 "green" phases are used as actions:
- Phase 0 (`GGrgrrGGrgrr`): NS through + right
- Phase 2 (`GrrGgrGrrGGr`): EW through + right
- Phase 4 (`GrrGrGGrrGrG`): EW left turns
- Phase 6 (`GrGGrrGrGGrr`): NS left turns
Phases 1, 3, 5, 7 are yellow transitions (handled automatically by SUMO).

### Min-Green Enforcement
`enforce_min_green: false` in config (default). Agents freely choose any phase each step.
If set to `true`, phase changes are blocked until 10s have elapsed (hard constraint).

---

## Notes for AI Assistants

When helping with this project:

1. **Understand the dual-environment methodology** - Traffic is baseline, dilemmas are the research contribution
2. **Respect the comparative framework** - MAPPO vs IPPO contrasts the full cooperative MARL package (centralized critic + shared/neighbour-coupled reward) against the full independent-learners package (decentralized critic + purely local reward). Coordination value = MAPPO performance − IPPO performance.
3. **Recognize CTDE is central** - Centralized training, decentralized execution
4. **Traffic is cooperative** - Network effects align incentives
5. **Dilemmas create conflict** - Individual gain from exploitation
6. **S2 is exploratory** - Environment selection still pending
7. **Time constraints matter** - 15-week S2 timeline is tight
8. **Non-specialist audience** - Presentations must be accessible