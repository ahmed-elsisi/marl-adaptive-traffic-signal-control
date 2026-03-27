#!/usr/bin/env python3
"""
fixed-cycles.py (SEED-AWARE VERSION)

Run a fixed 4-phase cycle (simultaneous at all TLs) and produce CSV + plots.
NOW SUPPORTS --seed parameter for fair comparison with MAPPO!
"""

import os, sys, argparse, csv
from collections import defaultdict

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)

try:
    import traci
    import traci.constants as tc
except Exception as e:
    raise RuntimeError("traci import failed. Make sure SUMO tools are on PYTHONPATH.") from e

# plotting (post-run)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- PARAMETERS ----------------
STEP_LENGTH = 1.0        # sumo step length (s)
YELLOW = 0               # yellow seconds (applied simultaneously)
ALL_RED = 0             # all-red seconds (simultaneous)
# green durations for the 4 phases (seconds) - tune as desired
PHASE_SEQUENCE = [
    (0, 25.0),  # action index, green duration
    (1, 25.0),
    (2, 8.0),
    (3, 8.0),
]
LOG_CSV = "fixed_cycle_log.csv"

# 4 hardcoded action strings (must match your controlled_lanes ordering)
ACTION_STATES = [
    "ggrgrrggrgrr".lower(),
    "grrggrgrrggr".lower(),
    "grrgrggrrgrg".lower(),
    "grggrrgrggrr".lower()
]

# ---------------- HELPERS ----------------
def make_yellow(cur, nxt):
    return "".join(
        "y" if c.lower() == "g" and n.lower() != "g" else
        ("g" if c.lower() == "g" else "r")
        for c, n in zip(cur, nxt)
    )

def set_states_for_all(tl_ids, states_map):
    """states_map: tl_id -> state_string. Sets each TL state (no stepping)."""
    for tl in tl_ids:
        st = states_map.get(tl)
        if st is None: continue
        traci.trafficlight.setRedYellowGreenState(tl, st)

# ---------------- METRIC HELPERS ----------------
def sample_network_metrics(lane_list, tl_models, last_wait, total_wait_accum, total_arrived):
    """
    Sample per-step metrics:
     - total halting vehicles (sum lane.getLastStepHaltingNumber)
     - update last_wait cache for active vehicles
     - handle arrivals using last_wait (pop)
    Returns tuple: (th, total_arrived, total_wait_accum, arrived_count)
    """
    # update last_wait for active vehicles (one getIDList call)
    active = traci.vehicle.getIDList()
    for vid in active:
        try:
            last_wait[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)
        except Exception:
            last_wait[vid] = last_wait.get(vid, 0.0)

    # arrived vehicles this step
    arrived_ids = traci.simulation.getArrivedIDList()
    arrived_count = 0
    if arrived_ids:
        for vid in arrived_ids:
            wait = last_wait.pop(vid, 0.0)
            total_wait_accum += wait
            arrived_count += 1
        total_arrived += arrived_count

    # total halting vehicles
    th = 0
    for lane in lane_list:
        try:
            th += traci.lane.getLastStepHaltingNumber(lane)
        except Exception:
            pass

    return th, total_arrived, total_wait_accum, arrived_count

# ---------------- MAIN ----------------
def run_fixed_cycle(sumocfg, steps, use_gui=False, seed=42):
    """
    Run fixed-time control with optional seed.
    
    Args:
        sumocfg: Path to SUMO config file
        steps: Number of simulation steps
        use_gui: Whether to use GUI
        seed: Random seed for SUMO (None = random)
    """
    sumo_bin = "sumo-gui.exe" if use_gui else "sumo.exe"
    sumo_cmd = [sumo_bin, "-c", sumocfg]
    
    # ADD SEED PARAMETER (for fair comparison with MAPPO!)
    print(f'USING SEED-FIXED: {seed}')
    if seed is not None:
        sumo_cmd.extend(["--seed", str(seed)])
        print(f"SUMO starting with seed {seed}...")
    else:
        print("SUMO starting with random seed...")
    
    traci.start(sumo_cmd)
    print("SUMO started.")

    tl_ids = traci.trafficlight.getIDList()
    print("Traffic lights:", tl_ids)
    lane_list = traci.lane.getIDList()
    print(lane_list)

    # Build simple models (controlled_lanes used for sanity)
    tl_models = {}
    for tl in tl_ids:
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(tl)
        except Exception:
            controlled_lanes = []
        tl_models[tl] = {"controlled_lanes": controlled_lanes}
        
    print(controlled_lanes)
    # prepare CSV
    csvf = open(LOG_CSV, "w", newline="")
    writer = csv.writer(csvf)
    writer.writerow(["time", "total_halts", "cumulative_arrivals", "running_avg_wait", "per_tl_halts", "active_veh"])

    # metrics storage
    times = []
    total_halts = []
    cumulative_arrivals = []
    running_avg_wait = []
    per_tl_halts = {tl: [] for tl in tl_ids}
    switch_counts = {tl: 0 for tl in tl_ids}

    # arrival/wait bookkeeping
    last_wait = {}       # vid -> last observed accumulated waiting time
    total_arrived = 0
    total_wait_accum = 0.0

    sim_time = 0.0
    step = 0

    try:
        # initial sampling step 0
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        # main loop: iterate phases repeatedly until steps exhausted
        phase_idx = 0
        while step < steps:
            action_idx, green_dur = PHASE_SEQUENCE[phase_idx]
            # get current states for all TLs
            cur_states = {tl: traci.trafficlight.getRedYellowGreenState(tl).lower() for tl in tl_ids}
            target = ACTION_STATES[action_idx].lower()

            # build yellow states map
            yellow_map = {}
            for tl in tl_ids:
                cur = cur_states[tl]
                # if lengths mismatch, pad/truncate target to length of cur (safe fallback)
                tgt = target
                if len(tgt) != len(cur):
                    if len(tgt) < len(cur):
                        tgt = tgt + "r" * (len(cur) - len(tgt))
                    else:
                        tgt = tgt[:len(cur)]
                yellow_map[tl] = make_yellow(cur, tgt)

            # 1) set all yellow (only where needed)
            set_states_for_all(tl_ids, yellow_map)
            steps_y = int(round(YELLOW / STEP_LENGTH))
            for _ in range(max(1, steps_y)):
                # sample metrics each simulation second
                th, total_arrived, total_wait_accum, _ = sample_network_metrics(lane_list, tl_models, last_wait, total_wait_accum, total_arrived)
                times.append(traci.simulation.getTime()); total_halts.append(th); cumulative_arrivals.append(total_arrived)
                running_avg_wait.append((total_wait_accum / total_arrived) if total_arrived>0 else 0.0)
                for tl in tl_ids:
                    h = 0
                    for ln in tl_models[tl]["controlled_lanes"]:
                        try: h += traci.lane.getLastStepHaltingNumber(ln)
                        except: pass
                    per_tl_halts[tl].append(h)
                writer.writerow([traci.simulation.getTime(), th, total_arrived, running_avg_wait[-1], ";".join(str(per_tl_halts[tl][-1]) for tl in tl_ids), len(traci.vehicle.getIDList())])
                traci.simulationStep(); step += 1
                if step >= steps: break
            if step >= steps: break

            # 2) all-red
            if ALL_RED > 0:
                all_red_map = {tl: "r" * len(traci.trafficlight.getRedYellowGreenState(tl)) for tl in tl_ids}
                set_states_for_all(tl_ids, all_red_map)
                steps_r = int(round(ALL_RED / STEP_LENGTH))
                for _ in range(max(1, steps_r)):
                    th, total_arrived, total_wait_accum, _ = sample_network_metrics(lane_list, tl_models, last_wait, total_wait_accum, total_arrived)
                    times.append(traci.simulation.getTime()); total_halts.append(th); cumulative_arrivals.append(total_arrived)
                    running_avg_wait.append((total_wait_accum / total_arrived) if total_arrived>0 else 0.0)
                    for tl in tl_ids:
                        h = 0
                        for ln in tl_models[tl]["controlled_lanes"]:
                            try: h += traci.lane.getLastStepHaltingNumber(ln)
                            except: pass
                        per_tl_halts[tl].append(h)
                    writer.writerow([traci.simulation.getTime(), th, total_arrived, running_avg_wait[-1], ";".join(str(per_tl_halts[tl][-1]) for tl in tl_ids), len(traci.vehicle.getIDList())])
                    traci.simulationStep(); step += 1
                    if step >= steps: break
            if step >= steps: break

            # 3) set all target green simultaneously
            target_map = {}
            for tl in tl_ids:
                cur = traci.trafficlight.getRedYellowGreenState(tl).lower()  # length reference
                tgt = ACTION_STATES[action_idx].lower()
                if len(tgt) != len(cur):
                    if len(tgt) < len(cur): tgt = tgt + "r"*(len(cur)-len(tgt))
                    else: tgt = tgt[:len(cur)]
                target_map[tl] = tgt
            set_states_for_all(tl_ids, target_map)
            # increment switch counts
            for tl in tl_ids:
                switch_counts[tl] += 1

            # 4) run green duration while sampling per-second
            steps_g = int(round(green_dur / STEP_LENGTH))
            for _ in range(max(1, steps_g)):
                th, total_arrived, total_wait_accum, _ = sample_network_metrics(lane_list, tl_models, last_wait, total_wait_accum, total_arrived)
                times.append(traci.simulation.getTime()); total_halts.append(th); cumulative_arrivals.append(total_arrived)
                running_avg_wait.append((total_wait_accum / total_arrived) if total_arrived>0 else 0.0)
                for tl in tl_ids:
                    h = 0
                    for ln in tl_models[tl]["controlled_lanes"]:
                        try: h += traci.lane.getLastStepHaltingNumber(ln)
                        except: pass
                    per_tl_halts[tl].append(h)
                writer.writerow([traci.simulation.getTime(), th, total_arrived, running_avg_wait[-1], ";".join(str(per_tl_halts[tl][-1]) for tl in tl_ids), len(traci.vehicle.getIDList())])
                traci.simulationStep(); step += 1
                if step >= steps: break
            if step >= steps: break

            # rotate to next phase
            phase_idx = (phase_idx + 1) % len(PHASE_SEQUENCE)

    finally:
        csvf.close()
        traci.close()
        print("Run finished. CSV ->", LOG_CSV)

    # Print summary
    print(f"\n{'='*80}")
    print(f"FIXED-TIME CONTROL SUMMARY")
    print(f"{'='*80}")
    print(f"Total arrivals: {total_arrived}")
    print(f"Avg waiting time: {(total_wait_accum / total_arrived) if total_arrived > 0 else 0:.2f}s")
    print(f"Max halting: {max(total_halts) if total_halts else 0}")
    print(f"Avg halting: {sum(total_halts) / len(total_halts) if total_halts else 0:.2f}")
    print(f"Seed used: {seed if seed is not None else 'random'}")
    print(f"{'='*80}\n")

    # ---------------- PLOTS ----------------
    # 1 Total halting vehicles over time
    plt.figure(figsize=(10,3))
    plt.plot(times, total_halts, marker='.', markersize=3)
    plt.title("Total halting vehicles over time")
    plt.xlabel("Simulation time (s)"); plt.ylabel("Network total halting vehicles"); plt.grid(True); plt.tight_layout()
    plt.savefig("metrics/fixed_plots_total_halts.png", dpi=150); plt.close()

    # 2 throughput + running avg wait
    fig, ax1 = plt.subplots(figsize=(10,3))
    ax1.plot(times, cumulative_arrivals, label="Cumulative arrivals")
    ax1.set_xlabel("Simulation time (s)"); ax1.set_ylabel("Cumulative arrivals"); ax1.grid(True)
    ax2 = ax1.twinx(); ax2.plot(times, running_avg_wait, linestyle='--', label="Running avg waiting (s)"); ax2.set_ylabel("Running avg waiting time (s)")
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right'); fig.tight_layout()
    fig.savefig("metrics/fixed_plots_arrivals_wait.png", dpi=150); plt.close()

    # 3 per TL halts
    plt.figure(figsize=(10,3))
    for tl in tl_ids:
        plt.plot(times, per_tl_halts[tl], label=tl, linewidth=0.8)
    plt.legend(); plt.title("Per-intersection halting (per-sample)"); plt.xlabel("Simulation time (s)"); plt.ylabel("Halting vehicles per TL"); plt.grid(True); plt.tight_layout()
    plt.savefig("metrics/fixed_plots_per_tl_halts.png", dpi=150); plt.close()

    # 4 switch counts
    plt.figure(figsize=(8,3))
    tlist = list(tl_ids); counts = [switch_counts.get(t,0) for t in tlist]
    plt.bar(tlist, counts); plt.title("Switch counts per intersection"); plt.xlabel("Traffic Light ID"); plt.ylabel("Number of phase switches during run"); plt.tight_layout()
    plt.savefig("metrics/fixed_plots_switch_counts.png", dpi=150); plt.close()

    print("Plots saved: fixed_plots_total_halts.png, fixed_plots_arrivals_wait.png, fixed_plots_per_tl_halts.png, fixed_plots_switch_counts.png")

# ---------------- ENTRYPOINT ----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sumo-cfg", default="sumo_network/marl-proj.sumocfg")
    p.add_argument("--steps", type=int, default=3600)
    p.add_argument("--sumo-mode", choices=("gui","cli"), default="cli")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for SUMO (default: random). Use 11 to match MAPPO training!")
    args = p.parse_args()
    use_gui = (args.sumo_mode == "gui")
    
    print(f"\n{'='*80}")
    print(f"FIXED-TIME CONTROL EVALUATION")
    print(f"{'='*80}")
    print(f"Config: {args.sumo_cfg}")
    print(f"Steps: {args.steps}")
    print(f"GUI: {use_gui}")
    print(f"Seed: {args.seed if args.seed is not None else '42'}")
    print(f"{'='*80}\n")
    
    run_fixed_cycle(args.sumo_cfg, steps=args.steps, use_gui=use_gui, seed=args.seed)