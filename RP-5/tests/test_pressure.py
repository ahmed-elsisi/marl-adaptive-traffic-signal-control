"""
Test Script: Verify Pressure Calculation Fix

This script helps you verify that the pressure calculation is working correctly
by checking if pressure values can be both positive and negative.
"""

import numpy as np
import sys
import os

# Add your project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your environment
from marl_env.sumo_env import SUMOTrafficEnv
from configs.env_config import get_env_config


def test_pressure_values():
    """
    Test that pressure values are properly calculated.
    
    Expected behavior:
    - Pressure should be able to be positive (more incoming than outgoing)
    - Pressure should be able to be negative (more outgoing than incoming)
    - Pressure should change over time based on traffic light states
    """
    print("=" * 70)
    print("PRESSURE CALCULATION VERIFICATION TEST")
    print("=" * 70)
    
    # Create environment
    env_config = get_env_config()
    env = SUMOTrafficEnv(env_config)
    
    # Reset environment
    obs, info = env.reset()
    
    print("\n✓ Environment created and reset successfully")
    
    # Run for a few steps and collect pressure values
    pressure_values = {agent_id: [] for agent_id in env.agent_ids}
    
    print("\nRunning simulation for 100 steps...")
    print("Collecting pressure values...\n")
    
    for step in range(100):
        # Take random actions
        actions = {agent_id: np.random.randint(0, 4) for agent_id in env.agent_ids}
        obs, rewards, dones, truncs, infos = env.step(actions)
        
        # Extract pressure values from observations
        # Pressure is at indices 17-20 (after 12 queue + 4 phase + 1 elapsed)
        for agent_id in env.agent_ids:
            agent_obs = obs[agent_id]
            pressure_features = agent_obs[17:21]  # 4 pressure values
            pressure_values[agent_id].extend(pressure_features.tolist())
        
        if step % 20 == 0:
            print(f"Step {step}:")
            for agent_id in env.agent_ids:
                p_ns = agent_obs[17]  # NS pressure
                p_ew = agent_obs[19]  # EW pressure
                print(f"  {agent_id}: NS pressure={p_ns:+.3f}, EW pressure={p_ew:+.3f}")
    
    env.close()
    
    # Analyze results
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    
    all_pressures = []
    for agent_id in env.agent_ids:
        all_pressures.extend(pressure_values[agent_id])
    
    all_pressures = np.array(all_pressures)
    
    # Remove zero values (initialization)
    non_zero_pressures = all_pressures[all_pressures != 0]
    
    print(f"\nTotal pressure samples: {len(all_pressures)}")
    print(f"Non-zero pressure samples: {len(non_zero_pressures)}")
    
    if len(non_zero_pressures) > 0:
        positive_count = np.sum(non_zero_pressures > 0)
        negative_count = np.sum(non_zero_pressures < 0)
        
        print(f"\nPositive pressure values: {positive_count} ({positive_count/len(non_zero_pressures)*100:.1f}%)")
        print(f"Negative pressure values: {negative_count} ({negative_count/len(non_zero_pressures)*100:.1f}%)")
        
        print(f"\nPressure statistics:")
        print(f"  Min: {np.min(non_zero_pressures):+.3f}")
        print(f"  Max: {np.max(non_zero_pressures):+.3f}")
        print(f"  Mean: {np.mean(non_zero_pressures):+.3f}")
        print(f"  Std: {np.std(non_zero_pressures):.3f}")
        
        # Verdict
        print("\n" + "=" * 70)
        print("VERDICT")
        print("=" * 70)
        
        if positive_count > 0 and negative_count > 0:
            print("✓ PASS: Pressure values include both positive and negative")
            print("✓ This indicates proper calculation (incoming - outgoing)")
            print("\nYour pressure calculation is working correctly!")
        elif positive_count > 0 and negative_count == 0:
            print("✗ FAIL: Only positive pressure values detected")
            print("✗ This suggests you're only counting incoming vehicles")
            print("\nYou need to implement the outgoing vehicle count!")
        elif negative_count > 0 and positive_count == 0:
            print("⚠ WARNING: Only negative pressure values detected")
            print("⚠ This is unusual - check your calculation logic")
        else:
            print("✗ FAIL: All pressure values are zero")
            print("✗ Pressure calculation is not working at all")
    else:
        print("\n✗ FAIL: No non-zero pressure values detected")
        print("✗ Pressure calculation is not working")
    
    print("\n" + "=" * 70)


def test_outgoing_edge_mapping():
    """
    Test that outgoing edge mappings are correct.
    
    This helps verify that the edge names in _get_outgoing_edges()
    actually exist in your SUMO network.
    """
    print("\n" + "=" * 70)
    print("OUTGOING EDGE MAPPING VERIFICATION")
    print("=" * 70)
    
    # Import SUMO
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        print("✗ SUMO_HOME not set!")
        return
    
    import traci
    
    # Start SUMO
    env_config = get_env_config()
    network_file = env_config['network_file']
    
    sumo_cmd = [
        "sumo",
        "-c", network_file.replace('.net.xml', '.sumocfg'),
        "--no-step-log",
        "--no-warnings"
    ]
    
    traci.start(sumo_cmd)
    
    print("\n✓ SUMO started successfully")
    
    # Get all edges in network
    all_edges = traci.edge.getIDList()
    print(f"\nTotal edges in network: {len(all_edges)}")
    print(f"Edge IDs: {all_edges[:20]}...")  # Show first 20
    
    # Test outgoing edge mappings
    from marl_env.obs_builder import MAPPOObservationBuilder
    
    obs_builder = MAPPOObservationBuilder(
        agent_ids=env_config['agents'],
        network_topology=env_config['network_topology'],
        detector_config=env_config['detectors'],
        normalization_config=env_config['normalization']
    )
    
    print("\nVerifying outgoing edge mappings:")
    all_valid = True
    
    for agent_id in env_config['agents']:
        print(f"\n{agent_id}:")
        for direction in ['NS', 'EW']:
            outgoing_edges = obs_builder._get_outgoing_edges(agent_id, direction)
            print(f"  {direction} direction: {outgoing_edges}")
            
            for edge in outgoing_edges:
                if edge in all_edges:
                    print(f"    ✓ {edge} exists")
                else:
                    print(f"    ✗ {edge} NOT FOUND in network!")
                    all_valid = False
    
    traci.close()
    
    print("\n" + "=" * 70)
    if all_valid:
        print("✓ All outgoing edges are valid!")
    else:
        print("✗ Some outgoing edges not found in network!")
        print("You need to update the outgoing_map in _get_outgoing_edges()")
    print("=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MAPPO PRESSURE CALCULATION VERIFICATION")
    print("=" * 70)
    
    print("\nThis script will:")
    print("1. Test if pressure values can be positive AND negative")
    print("2. Verify outgoing edge mappings are correct")
    print("\nPress Ctrl+C to abort at any time.\n")
    
    try:
        # Test 1: Pressure values
        test_pressure_values()
        
        # Test 2: Edge mappings
        test_outgoing_edge_mapping()
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest complete!")