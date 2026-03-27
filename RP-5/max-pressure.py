#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   MAX-PRESSURE BASELINE CONTROLLER                            ║
║                 Production-Ready - Fully Integrated                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Max-Pressure Control Strategy:
  At each decision point (every CONTROL_INTERVAL seconds):
    1. For each traffic light:
       - Compute pressure for each possible action (phase)
       - Pressure = Upstream queue - Downstream queue
       - Select action with maximum pressure
    2. Apply action if MIN_GREEN constraint satisfied
    3. Handle yellow and all-red transitions properly

This adaptive strategy responds to real-time traffic by prioritizing movements
with the highest queue imbalance, similar to back-pressure routing.

Features:
  ✓ Matches fixed-cycles.py methodology exactly
  ✓ Seed parameter for fair comparison
  ✓ Proper metrics sampling DURING simulation steps
  ✓ Same CSV format as fixed-cycles.py
  ✓ All outputs to metrics/ folder
  ✓ Comprehensive logging and visualization

Usage:
    python max-pressure.py --steps 3600 --seed 42
    python max-pressure.py --steps 3600 --seed 42 --gui
    python max-pressure.py --steps 3600 --seed 42 --control-interval 10
"""

import os
import sys
import argparse
import csv
from collections import defaultdict

# SUMO imports
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)

try:
    import traci
except Exception as e:
    raise RuntimeError("traci import failed. Make sure SUMO tools are on PYTHONPATH.") from e

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# PARAMETERS
# ============================================================================

STEP_LENGTH = 1.0        # SUMO simulation step (seconds)
CONTROL_INTERVAL = 5.0   # Decision frequency (seconds)
MIN_GREEN = 10.0          # Minimum green time before switching (seconds)
YELLOW = 0               # Yellow phase duration (seconds)
ALL_RED = 0              # All-red phase duration (seconds)

# 4 hardcoded action strings (matching your TLS configuration)
ACTION_STATES = [
    "ggrgrrggrgrr".lower(),   # Action 0: NS through + right
    "grrggrgrrggr".lower(),   # Action 1: NS left turn
    "grrgrggrrgrg".lower(),   # Action 2: EW through + right
    "grggrrgrggrr".lower()    # Action 3: EW left turn
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def make_yellow(cur, nxt):
    """Generate yellow state for transition from cur to nxt."""
    return "".join(
        "y" if c.lower() == "g" and n.lower() != "g" else
        ("g" if c.lower() == "g" else "r")
        for c, n in zip(cur, nxt)
    )


def build_tl_model(tl_id):
    """
    Build traffic light model for max-pressure calculation.
    
    Returns dict with:
      - controlled_lanes: List of lanes controlled by this TL
      - links: Dict of (from_lane, to_lane) connections
      - action_fromlanes: Dict mapping action_idx -> set of lanes served (green)
    """
    model = {}
    
    # Get controlled lanes
    try:
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
    except:
        controlled_lanes = []
    
    # Get connections (links)
    try:
        raw_links = traci.trafficlight.getControlledLinks(tl_id)
    except:
        raw_links = []
    
    links = {}
    for group in raw_links:
        for conn in group:
            if len(conn) >= 2:
                from_lane, to_lane = conn[0], conn[1]
                links[(from_lane, to_lane)] = {
                    "from": from_lane,
                    "to": to_lane
                }
    
    model["controlled_lanes"] = controlled_lanes
    model["links"] = links
    
    # Map each action to lanes it serves (green)
    action_fromlanes = {}
    for action_idx, action_state in enumerate(ACTION_STATES):
        served_lanes = set()
        for i, signal in enumerate(action_state):
            if signal.lower() == "g" and i < len(controlled_lanes):
                served_lanes.add(controlled_lanes[i])
        action_fromlanes[action_idx] = served_lanes
    
    model["action_fromlanes"] = action_fromlanes
    
    return model


def get_lane_queue(lane_id):
    """Get queue length (halting vehicles) on a lane."""
    try:
        return traci.lane.getLastStepHaltingNumber(lane_id)
    except:
        return 0


def compute_pressure(tl_model, action_idx):
    """
    Compute pressure for a given action.
    
    Pressure = Sum(upstream queues) - Sum(downstream queues)
    
    Returns:
      (pressure, upstream_total)
    """
    served_lanes = tl_model["action_fromlanes"].get(action_idx, set())
    
    # Upstream: total queue on lanes served by this action
    upstream_total = sum(get_lane_queue(lane) for lane in served_lanes)
    
    # Downstream: total queue on destination lanes
    downstream_total = 0
    for (from_lane, to_lane), info in tl_model["links"].items():
        if from_lane in served_lanes:
            downstream_total += get_lane_queue(to_lane)
    
    pressure = upstream_total - downstream_total
    
    return pressure, upstream_total


def apply_action_prepare(tl_id, action_idx):
    """
    Prepare state transition plan for tl_id given action_idx.
    
    Returns list of (state_string, duration_seconds) tuples.
    The final green phase has duration 0 (handled by main loop).
    """
    target = ACTION_STATES[action_idx]
    cur = traci.trafficlight.getRedYellowGreenState(tl_id).lower()
    
    # Adjust target length to match current
    if len(cur) != len(target):
        if len(target) < len(cur):
            target = target + "r" * (len(cur) - len(target))
        else:
            target = target[:len(cur)]
    
    # Already at target
    if cur == target:
        return []
    
    plan = []
    
    # Yellow transition (if needed)
    yellow_state = make_yellow(cur, target)
    if yellow_state != cur:
        plan.append((yellow_state, YELLOW))
    
    # All-red
    if ALL_RED > 0:
        all_red = "r" * len(cur)
        plan.append((all_red, ALL_RED))
    
    # Final green (duration 0, handled by main loop)
    plan.append((target, 0.0))
    
    return plan


# ============================================================================
# MAIN MAX-PRESSURE CONTROLLER
# ============================================================================

def run_max_pressure(sumocfg, steps, use_gui=False, seed=42, results_dir="metrics"):
    """
    Run max-pressure traffic control.
    
    Args:
        sumocfg: Path to SUMO configuration file
        steps: Number of simulation steps
        use_gui: Whether to use SUMO GUI
        seed: Random seed for reproducibility
        results_dir: Directory for saving results
    """
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Start SUMO
    sumo_bin = "sumo-gui.exe" if use_gui else "sumo.exe"
    sumo_cmd = [sumo_bin, "-c", sumocfg]
    
    if seed is not None:
        sumo_cmd.extend(["--seed", str(seed)])
        print(f"SUMO starting with seed {seed}...")
    else:
        print("SUMO starting with random seed...")
    
    traci.start(sumo_cmd)
    print("SUMO started.")
    
    # Get traffic lights and lanes
    tl_ids = traci.trafficlight.getIDList()
    print(f"Traffic lights: {tl_ids}")
    lane_list = traci.lane.getIDList()
    print(f"Total lanes: {len(lane_list)}")
    
    # Build TL models for pressure calculation
    tl_models = {}
    for tl in tl_ids:
        tl_models[tl] = build_tl_model(tl)
        print(f"  {tl}: {len(tl_models[tl]['controlled_lanes'])} controlled lanes")
    
    # Initialize controller state
    current_action = {}
    last_switch_time = {}
    switch_counts = {tl: 0 for tl in tl_ids}
    
    # Initialize current action for each TL
    for tl in tl_ids:
        last_switch_time[tl] = -9999.0
        try:
            cur_state = traci.trafficlight.getRedYellowGreenState(tl).lower()
            # Try to match current state to one of our actions
            found = None
            for idx, action_state in enumerate(ACTION_STATES):
                comp = action_state
                if len(comp) != len(cur_state):
                    if len(comp) < len(cur_state):
                        comp = comp + "r" * (len(cur_state) - len(comp))
                    else:
                        comp = comp[:len(cur_state)]
                if comp == cur_state:
                    found = idx
                    break
            current_action[tl] = found
        except:
            current_action[tl] = None
    
    # Initialize metrics storage
    times = []
    total_halts = []
    cumulative_arrivals = []
    running_avg_wait = []
    active_vehicles_list = []
    per_tl_halts = {tl: [] for tl in tl_ids}
    
    # Arrival/wait tracking (matches fixed-cycles.py exactly)
    last_wait = {}
    total_arrived = 0
    total_wait_accum = 0.0
    
    # CSV logging
    csv_path = os.path.join(results_dir, "max_pressure_log.csv")
    csvf = open(csv_path, "w", newline="")
    writer = csv.writer(csvf)
    writer.writerow([
        "time", "total_halts", "cumulative_arrivals", "running_avg_wait",
        "per_tl_halts", "active_veh"
    ])
    
    # Decision log CSV (for detailed analysis)
    decision_csv_path = os.path.join(results_dir, "max_pressure_decisions.csv")
    decision_csvf = open(decision_csv_path, "w", newline="")
    decision_writer = csv.writer(decision_csvf)
    decision_writer.writerow([
        "time", "tl_id", "chosen_action", "pressure", "upstream_sum", "active_veh"
    ])
    
    # Main simulation variables
    sim_time = 0.0
    step = 0
    next_decision = 0.0
    
    # ========================================================================
    # NESTED FUNCTION: Sample metrics and step simulation
    # This ensures metrics are captured DURING simulation (corrected method)
    # ========================================================================
    def sample_and_step():
        """Sample metrics and advance simulation by one step."""
        nonlocal total_arrived, total_wait_accum, sim_time, step
        
        # Advance simulation
        traci.simulationStep()
        sim_time = traci.simulation.getTime()
        
        # Update waiting times for active vehicles
        active_veh_ids = traci.vehicle.getIDList()
        for vid in active_veh_ids:
            try:
                last_wait[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)
            except:
                last_wait[vid] = last_wait.get(vid, 0.0)
        
        # Handle arrived vehicles THIS STEP
        # CRITICAL: Called every step, so we capture ALL arrivals!
        arrived_ids = traci.simulation.getArrivedIDList()
        if arrived_ids:
            for vid in arrived_ids:
                wait = last_wait.pop(vid, 0.0)
                total_wait_accum += wait
            total_arrived += len(arrived_ids)
        
        # Total halting vehicles across network
        th = 0
        for lane in lane_list:
            try:
                th += traci.lane.getLastStepHaltingNumber(lane)
            except:
                pass
        
        # Store metrics
        times.append(sim_time)
        total_halts.append(th)
        cumulative_arrivals.append(total_arrived)
        avg_wait = (total_wait_accum / total_arrived) if total_arrived > 0 else 0.0
        running_avg_wait.append(avg_wait)
        active_vehicles_list.append(len(active_veh_ids))
        
        # Per-TL halting
        for tl in tl_ids:
            halts = 0
            for lane in tl_models[tl]["controlled_lanes"]:
                try:
                    halts += traci.lane.getLastStepHaltingNumber(lane)
                except:
                    pass
            per_tl_halts[tl].append(halts)
        
        # Write CSV row
        writer.writerow([
            sim_time, th, total_arrived, avg_wait,
            ";".join(str(per_tl_halts[tl][-1]) for tl in tl_ids),
            len(active_veh_ids)
        ])
        
        step += 1
    
    # ========================================================================
    # MAIN SIMULATION LOOP
    # ========================================================================
    try:
        print(f"\n{'='*80}")
        print("MAX-PRESSURE CONTROL STARTED")
        print(f"{'='*80}")
        print(f"Control interval: {CONTROL_INTERVAL}s")
        print(f"Min green: {MIN_GREEN}s")
        print(f"Yellow: {YELLOW}s, All-red: {ALL_RED}s")
        print(f"Total steps: {steps}")
        print(f"Seed: {seed}")
        print(f"{'='*80}\n")
        
        # Initial step
        sample_and_step()
        next_decision = sim_time + CONTROL_INTERVAL
        
        # Main loop
        while step < steps:
            # ================================================================
            # DECISION POINT: Compute and apply max-pressure actions
            # ================================================================
            if sim_time + 1e-9 >= next_decision:
                next_decision = sim_time + CONTROL_INTERVAL
                
                # ============================================================
                # Step 1: Compute best action for each TL (using same snapshot)
                # ============================================================
                decisions = {}
                pressures = {}
                upstreams = {}
                
                for tl in tl_ids:
                    model = tl_models[tl]
                    best_action = None
                    best_pressure = -1e9
                    best_upstream = 0
                    
                    # Evaluate all actions
                    for action_idx in range(len(ACTION_STATES)):
                        pressure, upstream = compute_pressure(model, action_idx)
                        
                        # Select action with max pressure
                        # Tie-break by upstream (prefer serving more vehicles)
                        if pressure > best_pressure or \
                           (abs(pressure - best_pressure) < 1e-6 and upstream > best_upstream):
                            best_pressure = pressure
                            best_action = action_idx
                            best_upstream = upstream
                    
                    decisions[tl] = best_action
                    pressures[tl] = best_pressure
                    upstreams[tl] = best_upstream
                
                # ============================================================
                # Step 2: Apply actions for each TL
                # ============================================================
                for tl in tl_ids:
                    best_action = decisions[tl]
                    now = sim_time
                    
                    # If no current action, apply immediately
                    if current_action[tl] is None:
                        plan = apply_action_prepare(tl, best_action)
                        if plan:
                            # Execute plan (yellow/all-red transitions)
                            for state_str, dur in plan:
                                try:
                                    traci.trafficlight.setRedYellowGreenState(tl, state_str)
                                except:
                                    pass
                                
                                # Step through transition duration
                                if dur > 0:
                                    nsteps = int(round(dur / STEP_LENGTH))
                                    for _ in range(max(1, nsteps)):
                                        if step >= steps:
                                            break
                                        sample_and_step()
                            
                            # Set final green state
                            final_state = plan[-1][0] if plan else None
                            if final_state:
                                try:
                                    traci.trafficlight.setRedYellowGreenState(tl, final_state)
                                except:
                                    pass
                            
                            last_switch_time[tl] = sim_time
                            current_action[tl] = best_action
                            switch_counts[tl] += 1
                        else:
                            # No change needed
                            current_action[tl] = best_action
                    
                    # If different action, check MIN_GREEN constraint
                    elif best_action != current_action[tl]:
                        time_since_switch = now - last_switch_time[tl]
                        
                        if time_since_switch >= MIN_GREEN:
                            plan = apply_action_prepare(tl, best_action)
                            if plan:
                                # Execute plan
                                for state_str, dur in plan:
                                    try:
                                        traci.trafficlight.setRedYellowGreenState(tl, state_str)
                                    except:
                                        pass
                                    
                                    if dur > 0:
                                        nsteps = int(round(dur / STEP_LENGTH))
                                        for _ in range(max(1, nsteps)):
                                            if step >= steps:
                                                break
                                            sample_and_step()
                                
                                # Set final green state
                                final_state = plan[-1][0] if plan else None
                                if final_state:
                                    try:
                                        traci.trafficlight.setRedYellowGreenState(tl, final_state)
                                    except:
                                        pass
                                
                                last_switch_time[tl] = sim_time
                                current_action[tl] = best_action
                                switch_counts[tl] += 1
                        # else: MIN_GREEN not satisfied, keep current action
                    
                    # else: Same action, no change needed
                    
                    # Log decision
                    try:
                        decision_writer.writerow([
                            sim_time, tl, current_action[tl],
                            pressures[tl], upstreams[tl],
                            len(traci.vehicle.getIDList())
                        ])
                    except:
                        pass
                    
                    # Check if we've exhausted steps
                    if step >= steps:
                        break
                
                # After processing all TLs, continue to next iteration
                continue
            
            # ================================================================
            # NON-DECISION STEP: Just sample and step
            # ================================================================
            sample_and_step()
            
            # Periodic progress
            if step % 600 == 0:
                print(f"Progress: t={sim_time:.0f}s, arrivals={total_arrived}, "
                      f"halting={total_halts[-1] if total_halts else 0}, "
                      f"active={len(traci.vehicle.getIDList())}")
    
    finally:
        csvf.close()
        decision_csvf.close()
        traci.close()
        print(f"\nSimulation finished.")
    
    # ========================================================================
    # PRINT SUMMARY
    # ========================================================================
    print(f"\n{'='*80}")
    print("MAX-PRESSURE CONTROL SUMMARY")
    print(f"{'='*80}")
    print(f"Total arrivals: {total_arrived}")
    print(f"Avg waiting time: {(total_wait_accum / total_arrived) if total_arrived > 0 else 0:.2f}s")
    print(f"Max halting: {max(total_halts) if total_halts else 0}")
    print(f"Avg halting: {sum(total_halts) / len(total_halts) if total_halts else 0:.2f}")
    print(f"Seed used: {seed if seed is not None else 'random'}")
    print(f"\nPer-Intersection Switches:")
    for tl in tl_ids:
        print(f"  {tl}: {switch_counts[tl]} switches")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # GENERATE PLOTS (Matching fixed-cycles.py style)
    # ========================================================================
    print("Generating plots...")
    
    # 1. Total halting vehicles
    plt.figure(figsize=(10, 3))
    plt.plot(times, total_halts, marker='.', markersize=3)
    plt.title("Max-Pressure: Total Halting Vehicles", fontweight='bold')
    plt.xlabel("Simulation time (s)")
    plt.ylabel("Halting vehicles")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "max_pressure_halting.png"), dpi=150)
    plt.close()
    
    # 2. Arrivals and wait time
    fig, ax1 = plt.subplots(figsize=(10, 3))
    ax1.plot(times, cumulative_arrivals, 'b-', label="Cumulative arrivals", linewidth=1.2)
    ax1.set_xlabel("Simulation time (s)")
    ax1.set_ylabel("Cumulative arrivals", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(times, running_avg_wait, 'r--', label="Running avg wait", linewidth=1.2)
    ax2.set_ylabel("Running avg waiting time (s)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    fig.suptitle('Max-Pressure: Arrivals and Wait Time', fontweight='bold')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "max_pressure_arrivals_wait.png"), dpi=150)
    plt.close()
    
    # 3. Per-TL halting
    plt.figure(figsize=(10, 3))
    for tl in tl_ids:
        plt.plot(times, per_tl_halts[tl], label=tl, linewidth=0.8, alpha=0.7)
    plt.legend()
    plt.title("Max-Pressure: Per-Junction Halting", fontweight='bold')
    plt.xlabel("Simulation time (s)")
    plt.ylabel("Halting vehicles")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "max_pressure_per_tl_halts.png"), dpi=150)
    plt.close()
    
    # 4. Switch counts
    plt.figure(figsize=(8, 3))
    tlist = list(tl_ids)
    counts = [switch_counts[tl] for tl in tlist]
    plt.bar(tlist, counts, color='green', alpha=0.7)
    plt.title("Max-Pressure: Phase Switch Counts", fontweight='bold')
    plt.xlabel("Traffic Light ID")
    plt.ylabel("Number of switches")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "max_pressure_switches.png"), dpi=150)
    plt.close()
    
    print(f"✓ CSV logged: {csv_path}")
    print(f"✓ Decisions logged: {decision_csv_path}")
    print(f"✓ Plots saved to {results_dir}/")
    
    return {
        'total_arrivals': total_arrived,
        'avg_waiting_time': (total_wait_accum / total_arrived) if total_arrived > 0 else 0.0,
        'max_halting': max(total_halts) if total_halts else 0,
        'avg_halting': sum(total_halts) / len(total_halts) if total_halts else 0.0,
        'switch_counts': switch_counts
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Max-Pressure Baseline Controller - Production Ready'
    )
    parser.add_argument('--sumo-cfg', default='sumo_network/marl-proj.sumocfg',
                       help='Path to SUMO configuration file')
    parser.add_argument('--steps', type=int, default=3600,
                       help='Number of simulation steps (default: 3600)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for SUMO (default: 42)')
    parser.add_argument('--gui', action='store_true',
                       help='Use SUMO GUI for visualization')
    parser.add_argument('--results-dir', type=str, default='metrics',
                       help='Directory for saving results (default: metrics/)')
    parser.add_argument('--control-interval', type=float, default=CONTROL_INTERVAL,
                       help=f'Control decision interval in seconds (default: {CONTROL_INTERVAL})')
    parser.add_argument('--min-green', type=float, default=MIN_GREEN,
                       help=f'Minimum green time in seconds (default: {MIN_GREEN})')
    parser.add_argument('--yellow', type=float, default=YELLOW,
                       help=f'Yellow phase duration in seconds (default: {YELLOW})')
    parser.add_argument('--all-red', type=float, default=ALL_RED,
                       help=f'All-red phase duration in seconds (default: {ALL_RED})')
    
    args = parser.parse_args()
    
    # Update global parameters
    CONTROL_INTERVAL = args.control_interval
    MIN_GREEN = args.min_green
    YELLOW = args.yellow
    ALL_RED = args.all_red
    
    print(f"\n{'='*80}")
    print("MAX-PRESSURE BASELINE CONTROLLER")
    print(f"{'='*80}")
    print(f"Config: {args.sumo_cfg}")
    print(f"Steps: {args.steps}")
    print(f"Seed: {args.seed}")
    print(f"GUI: {args.gui}")
    print(f"Results Dir: {args.results_dir}/")
    print(f"Control Interval: {CONTROL_INTERVAL}s")
    print(f"Min Green: {MIN_GREEN}s")
    print(f"Yellow: {YELLOW}s, All-Red: {ALL_RED}s")
    print(f"{'='*80}\n")
    
    # Run max-pressure controller
    stats = run_max_pressure(
        sumocfg=args.sumo_cfg,
        steps=args.steps,
        use_gui=args.gui,
        seed=args.seed,
        results_dir=args.results_dir
    )
    
    print("\n✓ Max-Pressure evaluation completed successfully!")
    print(f"📁 All outputs saved to: {args.results_dir}/")