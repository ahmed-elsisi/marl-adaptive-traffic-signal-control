"""
Test Script: Visualize Reward Function Using ACTUAL Code Functions

This script uses the ACTUAL methods from MAPPORewardFunction to test
and visualize reward calculations. This ensures you're testing the real
implementation, not a reimplementation.

Usage:
    python test_reward.py --steps 20 --gui
    python test_reward.py --steps 100
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
import numpy as np
import traci

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from marl_env.sumo_env import SUMOTrafficEnv


def print_separator(char='=', length=80):
    """Print a separator line."""
    print(char * length)


def print_edge_details(env, agent_id):
    """
    Print detailed edge vehicle counts using ACTUAL reward function method.
    """
    print(f"\n  {agent_id} - Edge Vehicle Counts (using _get_edge_vehicle_count):")
    print(f"  " + "-" * 76)
    
    edge_config = env.reward_function.edge_mappings.get(agent_id, {})
    
    # Incoming edges - using actual function
    print(f"  Incoming Edges:")
    total_incoming = 0
    for edge_id in edge_config.get('incoming', []):
        # Use ACTUAL method from reward function
        count = env.reward_function._get_edge_vehicle_count(edge_id, traci)
        total_incoming += count
        
        # Also show lane breakdown
        try:
            num_lanes = traci.edge.getLaneNumber(edge_id)
            lane_details = []
            for lane_idx in range(num_lanes):
                lane_id = f"{edge_id}_{lane_idx}"
                try:
                    lane_veh = traci.lane.getLastStepVehicleNumber(lane_id)
                    lane_details.append(f"L{lane_idx}:{lane_veh}")
                except:
                    lane_details.append(f"L{lane_idx}:ERR")
            print(f"    {edge_id:8s}: {count:2d} veh [{', '.join(lane_details)}]")
        except:
            print(f"    {edge_id:8s}: {count:2d} veh")
    
    print(f"  {'TOTAL':8s}: {total_incoming:2d} vehicles")
    
    # Outgoing edges - using actual function
    print(f"\n  Outgoing Edges:")
    total_outgoing = 0
    for edge_id in edge_config.get('outgoing', []):
        # Use ACTUAL method from reward function
        count = env.reward_function._get_edge_vehicle_count(edge_id, traci)
        total_outgoing += count
        
        # Also show lane breakdown
        try:
            num_lanes = traci.edge.getLaneNumber(edge_id)
            lane_details = []
            for lane_idx in range(num_lanes):
                lane_id = f"{edge_id}_{lane_idx}"
                try:
                    lane_veh = traci.lane.getLastStepVehicleNumber(lane_id)
                    lane_details.append(f"L{lane_idx}:{lane_veh}")
                except:
                    lane_details.append(f"L{lane_idx}:ERR")
            print(f"    {edge_id:8s}: {count:2d} veh [{', '.join(lane_details)}]")
        except:
            print(f"    {edge_id:8s}: {count:2d} veh")
    
    print(f"  {'TOTAL':8s}: {total_outgoing:2d} vehicles")
    
    # Pressure (raw calculation)
    raw_pressure = total_incoming - total_outgoing
    positive_pressure = max(0, raw_pressure)
    
    print(f"\n  Raw Pressure: {total_incoming} - {total_outgoing} = {raw_pressure}")
    print(f"  Positive Pressure (penalized): {positive_pressure}")
    
    return total_incoming, total_outgoing, raw_pressure, positive_pressure


def print_detector_details(env, agent_id):
    """
    Print detailed detector information.
    """
    print(f"\n  {agent_id} - Detector Information:")
    print(f"  " + "-" * 76)
    
    detectors = env.detector_mappings.get(agent_id, [])
    total_queue = 0
    total_waiting = 0
    
    for det_id in detectors:
        try:
            queue = traci.lanearea.getJamLengthVehicle(det_id)
            vehicle_ids = traci.lanearea.getLastStepVehicleIDs(det_id)
            num_vehicles = len(vehicle_ids)
            
            # Calculate waiting time for vehicles in this detector
            det_waiting = 0
            for veh_id in vehicle_ids:
                try:
                    det_waiting += traci.vehicle.getAccumulatedWaitingTime(veh_id)
                except:
                    pass
            
            total_queue += queue
            total_waiting += det_waiting
            
            print(f"    {det_id:25s}: Queue={queue:2.0f}, Vehicles={num_vehicles}, Wait={det_waiting:5.1f}s")
        except Exception as e:
            print(f"    {det_id:25s}: ERROR - {e}")
    
    print(f"  {'TOTAL':25s}: Queue={total_queue:2.0f}, Wait={total_waiting:5.1f}s")
    
    return total_queue, total_waiting


def print_reward_components(env, agent_id):
    """
    Print reward components using ACTUAL reward function methods.
    """
    print(f"\n  {agent_id} - Reward Components (using ACTUAL functions):")
    print(f"  " + "-" * 76)
    
    # Get normalization config
    norm_config = env.reward_function.norm_config
    reward_config = env.reward_function.reward_config
    
    # 1. Queue penalty - ACTUAL function
    queue_norm = env.reward_function._calculate_queue_penalty(agent_id, traci)
    queue_reward = queue_norm * reward_config['queue_weight']
    max_queue = len(env.detector_mappings[agent_id]) * norm_config['queue_max']
    print(f"    Queue Penalty:")
    print(f"      Normalized: {queue_norm:.4f}")
    print(f"      Weight: {reward_config['queue_weight']}")
    print(f"      Component: {queue_reward:.4f}")
    print(f"      (max normalization: {max_queue})")
    
    # 2. Waiting time penalty - ACTUAL function
    waiting_norm = env.reward_function._calculate_waiting_time_penalty(agent_id, traci)
    waiting_reward = waiting_norm * reward_config['waiting_time_weight']
    max_waiting = len(env.detector_mappings[agent_id]) * norm_config['waiting_time_max']
    print(f"    Waiting Time Penalty:")
    print(f"      Normalized: {waiting_norm:.4f}")
    print(f"      Weight: {reward_config['waiting_time_weight']}")
    print(f"      Component: {waiting_reward:.4f}")
    print(f"      (max normalization: {max_waiting})")
    
    # 3. Throughput bonus - ACTUAL function
    throughput_norm = env.reward_function._calculate_throughput_bonus(agent_id, traci)
    throughput_reward = throughput_norm * reward_config['throughput_weight']
    print(f"    Throughput Bonus:")
    print(f"      Normalized: {throughput_norm:.4f}")
    print(f"      Weight: {reward_config['throughput_weight']}")
    print(f"      Component: {throughput_reward:.4f}")
    
    # 4. Pressure penalty - ACTUAL function
    pressure_norm = env.reward_function._calculate_pressure_penalty(agent_id, traci)
    pressure_reward = pressure_norm * reward_config['pressure_weight']
    edge_config = env.reward_function.edge_mappings[agent_id]
    max_pressure = len(edge_config['incoming']) * 3 * norm_config['pressure_max']
    print(f"    Pressure Penalty:")
    print(f"      Normalized: {pressure_norm:.4f}")
    print(f"      Weight: {reward_config['pressure_weight']}")
    print(f"      Component: {pressure_reward:.4f}")
    print(f"      (max normalization: {max_pressure})")
    
    # 5. Neighbor pressure - ACTUAL function
    neighbor_norm = env.reward_function._calculate_neighbor_pressure(agent_id, traci)
    neighbor_reward = neighbor_norm * reward_config['neighbor_pressure_weight']
    neighbors = env.reward_function.topology.get(agent_id, [])
    print(f"    Neighbor Pressure Penalty:")
    print(f"      Normalized: {neighbor_norm:.4f}")
    print(f"      Weight: {reward_config['neighbor_pressure_weight']}")
    print(f"      Component: {neighbor_reward:.4f}")
    print(f"      (neighbors: {neighbors})")
    
    # Total - ACTUAL function
    total_reward = env.reward_function.calculate_reward(agent_id, traci)
    
    # Manual sum for verification
    manual_sum = queue_reward + waiting_reward + throughput_reward + pressure_reward + neighbor_reward
    
    print(f"\n    TOTAL REWARD:")
    print(f"      From calculate_reward(): {total_reward:.4f}")
    print(f"      Manual sum (verify):     {manual_sum:.4f}")
    print(f"      Clipping range: [{reward_config['clip_min']}, {reward_config['clip_max']}]")
    
    if abs(total_reward - manual_sum) < 0.001:
        print(f"      ✓ Rewards match!")
    else:
        print(f"      ⚠ Difference: {abs(total_reward - manual_sum):.4f}")
    
    return {
        'queue': queue_reward,
        'waiting': waiting_reward,
        'throughput': throughput_reward,
        'pressure': pressure_reward,
        'neighbor': neighbor_reward,
        'total': total_reward
    }


def test_reward_functions(config_path: str, num_steps: int = 20, use_gui: bool = False):
    """
    Run test simulation using ACTUAL reward function methods.
    """
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override for testing
    config['env_config']['use_gui'] = use_gui
    config['env_config']['num_seconds'] = num_steps * config['env_config']['delta_time']
    
    # Create environment
    env_config = {
        'agents': config['agents'],
        'network_topology': config['network_topology'],
        'detectors': config['detectors'],
        'normalization': config['normalization'],
        'reward_config': config['reward_config'],
        'network_file': os.path.join(project_root, config['env_config']['network_file']),
        'route_file': os.path.join(project_root, config['env_config']['route_file']),
        'use_gui': use_gui,
        'num_seconds': num_steps * config['env_config']['delta_time'],
        'delta_time': config['env_config']['delta_time'],
        'yellow_time': config['env_config']['yellow_time'],
        'min_green': config['env_config']['min_green'],
        'max_green': config['env_config']['max_green'],
        'sumo_seed': config['env_config']['sumo_seed'],
        'enforce_min_green': config['env_config'].get('enforce_min_green', False),
    }
    
    print_separator()
    print("TESTING REWARD FUNCTION - Using ACTUAL Code Functions")
    print_separator()
    print(f"Simulation: {num_steps} steps × {env_config['delta_time']}s = {num_steps * env_config['delta_time']}s total")
    print(f"GUI: {'Enabled' if use_gui else 'Disabled'}")
    print_separator()
    
    env = SUMOTrafficEnv(env_config)
    
    # Reset environment
    obs, info = env.reset()
    
    print("\n✓ Environment initialized")
    print(f"✓ Agents: {env.agent_ids}")
    print(f"✓ Observation dimension: {len(obs[env.agent_ids[0]])} per agent")
    print(f"✓ Reward function: {type(env.reward_function).__name__}")
    print(f"✓ Observation builder: {type(env.obs_builder).__name__}")
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Normalization:")
    print(f"    queue_max: {env.reward_function.norm_config['queue_max']}")
    print(f"    waiting_time_max: {env.reward_function.norm_config['waiting_time_max']}")
    print(f"    pressure_max: {env.reward_function.norm_config['pressure_max']}")
    print(f"  Reward Weights:")
    print(f"    queue_weight: {env.reward_function.reward_config['queue_weight']}")
    print(f"    waiting_time_weight: {env.reward_function.reward_config['waiting_time_weight']}")
    print(f"    pressure_weight: {env.reward_function.reward_config['pressure_weight']}")
    print(f"    neighbor_pressure_weight: {env.reward_function.reward_config['neighbor_pressure_weight']}")
    print(f"    throughput_weight: {env.reward_function.reward_config['throughput_weight']}")
    
    # Run simulation
    step = 0
    done = False
    
    # Track rewards over time
    all_rewards = {agent_id: [] for agent_id in env.agent_ids}
    
    while not done and step < num_steps:
        step += 1
        
        # Random actions for testing
        actions = {agent_id: env.action_space_dict[agent_id].sample() 
                   for agent_id in env.agent_ids}
        
        # Step environment
        obs, rewards, dones, truncated, info = env.step(actions)
        done = dones.get('__all__', False)
        
        # Store rewards
        for agent_id in env.agent_ids:
            all_rewards[agent_id].append(rewards[agent_id])
        
        # Print detailed information every 10 steps
        if step % 10 == 0 or step == 1:
            print_separator('=')
            print(f"STEP {step} (t={step * env_config['delta_time']}s)")
            print_separator('=')
            
            # Print for each agent
            for agent_id in env.agent_ids:
                print_separator('-')
                print(f"Agent: {agent_id}")
                print_separator('-')
                
                # 1. Edge details (using ACTUAL function)
                incoming, outgoing, raw_pressure, pos_pressure = print_edge_details(env, agent_id)
                
                # 2. Detector details
                queue, waiting = print_detector_details(env, agent_id)
                
                # 3. Reward components (using ACTUAL functions)
                reward_components = print_reward_components(env, agent_id)
                
                # 4. Action taken
                print(f"\n  {agent_id} - Action Taken: {actions[agent_id]}")
                
                # Action mapping
                action_map = {
                    0: "NS through + right (Phase 0)",
                    1: "NS left turn (Phase 6)",
                    2: "EW through + right (Phase 2)",
                    3: "EW left turn (Phase 4)"
                }
                print(f"    → {action_map.get(actions[agent_id], 'Unknown')}")
            
            print()
    
    # Final summary
    print_separator('=')
    print(f"SIMULATION COMPLETE - {step} steps")
    print_separator('=')
    
    print("\nReward Statistics (across all steps):")
    print_separator('-')
    for agent_id in env.agent_ids:
        rewards_array = np.array(all_rewards[agent_id])
        print(f"  {agent_id}:")
        print(f"    Mean:   {rewards_array.mean():7.4f}")
        print(f"    Median: {np.median(rewards_array):7.4f}")
        print(f"    Min:    {rewards_array.min():7.4f}")
        print(f"    Max:    {rewards_array.max():7.4f}")
        print(f"    Std:    {rewards_array.std():7.4f}")
    
    # Overall mean
    all_rewards_flat = [r for rewards in all_rewards.values() for r in rewards]
    print(f"\n  Overall Mean Reward: {np.mean(all_rewards_flat):7.4f}")
    
    print_separator('=')
    
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Test reward function using ACTUAL code implementation"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/mappo_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=20,
        help='Number of simulation steps'
    )
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Enable SUMO GUI'
    )
    
    args = parser.parse_args()
    
    try:
        test_reward_functions(
            config_path=args.config,
            num_steps=args.steps,
            use_gui=args.gui
        )
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nError during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()