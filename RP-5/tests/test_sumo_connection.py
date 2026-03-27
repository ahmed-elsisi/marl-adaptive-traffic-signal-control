"""
Test SUMO Connection and Configuration

Validates:
- SUMO installation
- Network files
- Detector configuration
- Basic simulation
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set SUMO_HOME
if 'SUMO_HOME' not in os.environ:
    print("ERROR: SUMO_HOME environment variable not set!")
    print("Please set SUMO_HOME to your SUMO installation directory.")
    sys.exit(1)

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

import traci


def test_sumo_installation():
    """Test if SUMO is properly installed."""
    print("="*80)
    print("Testing SUMO Installation")
    print("="*80)
    
    print(f"SUMO_HOME: {os.environ.get('SUMO_HOME')}")
    
    try:
        import sumolib
        print("✓ sumolib imported successfully")
    except ImportError:
        print("✗ Failed to import sumolib")
        return False
    
    try:
        import traci
        print("✓ TraCI imported successfully")
    except ImportError:
        print("✗ Failed to import TraCI")
        return False
    
    print("\nSUMO installation: OK")
    return True


def test_network_files():
    """Test if network files exist and are valid."""
    print("\n" + "="*80)
    print("Testing Network Files")
    print("="*80)
    
    network_dir = project_root / "sumo_network"
    
    required_files = [
        "marl-proj.net.xml",
        "marl-proj.rou.xml",
        "marl-proj.add.xml",
        "marl-proj.sumocfg"
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = network_dir / filename
        if filepath.exists():
            print(f"✓ {filename} exists")
        else:
            print(f"✗ {filename} NOT FOUND")
            all_exist = False
    
    if all_exist:
        print("\nNetwork files: OK")
    else:
        print("\nNetwork files: MISSING FILES")
    
    return all_exist


def test_sumo_simulation():
    """Test basic SUMO simulation."""
    print("\n" + "="*80)
    print("Testing SUMO Simulation")
    print("="*80)
    
    network_dir = project_root / "sumo_network"
    sumocfg = network_dir / "marl-proj.sumocfg"
    
    if not sumocfg.exists():
        print(f"✗ Config file not found: {sumocfg}")
        return False
    
    try:
        # Start SUMO
        sumo_cmd = [
            "sumo",
            "-c", str(sumocfg),
            "--no-step-log",
            "--no-warnings",
            "--quit-on-end"
        ]
        
        print(f"Starting SUMO with config: {sumocfg}")
        traci.start(sumo_cmd)
        
        # Run for 10 steps
        print("Running simulation for 10 steps...")
        for step in range(10):
            traci.simulationStep()
        
        # Get some basic info
        num_vehicles = traci.vehicle.getIDCount()
        sim_time = traci.simulation.getTime()
        
        print(f"✓ Simulation step: {sim_time}")
        print(f"✓ Number of vehicles: {num_vehicles}")
        
        # Close
        traci.close()
        
        print("\nSUMO simulation: OK")
        return True
        
    except Exception as e:
        print(f"✗ Simulation error: {e}")
        try:
            traci.close()
        except:
            pass
        return False


def test_traffic_lights():
    """Test traffic light control."""
    print("\n" + "="*80)
    print("Testing Traffic Lights")
    print("="*80)
    
    network_dir = project_root / "sumo_network"
    sumocfg = network_dir / "marl-proj.sumocfg"
    
    try:
        # Start SUMO
        sumo_cmd = [
            "sumo",
            "-c", str(sumocfg),
            "--no-step-log",
            "--no-warnings",
            "--quit-on-end"
        ]
        
        traci.start(sumo_cmd)
        
        # Get traffic light IDs
        tl_ids = traci.trafficlight.getIDList()
        print(f"Traffic light IDs: {tl_ids}")
        
        expected_tls = ['J1', 'J2', 'J3', 'J4']
        all_present = all(tl_id in tl_ids for tl_id in expected_tls)
        
        if all_present:
            print("✓ All expected traffic lights present")
        else:
            print("✗ Some traffic lights missing")
        
        # Test controlling traffic lights
        for tl_id in expected_tls:
            if tl_id in tl_ids:
                # Get current state
                state = traci.trafficlight.getRedYellowGreenState(tl_id)
                print(f"  {tl_id} state: {state}")
                
                # Try to set phase
                traci.trafficlight.setPhase(tl_id, 0)
                new_state = traci.trafficlight.getRedYellowGreenState(tl_id)
                print(f"  {tl_id} new state: {new_state}")
        
        traci.close()
        
        print("\nTraffic lights: OK")
        return True
        
    except Exception as e:
        print(f"✗ Traffic light error: {e}")
        try:
            traci.close()
        except:
            pass
        return False


def test_detectors():
    """Test detector configuration."""
    print("\n" + "="*80)
    print("Testing Detectors")
    print("="*80)
    
    network_dir = project_root / "sumo_network"
    sumocfg = network_dir / "marl-proj.sumocfg"
    
    try:
        # Start SUMO
        sumo_cmd = [
            "sumo",
            "-c", str(sumocfg),
            "--no-step-log",
            "--no-warnings",
            "--quit-on-end"
        ]
        
        traci.start(sumo_cmd)
        
        # Run for a bit to get vehicles
        for _ in range(50):
            traci.simulationStep()
        
        # Get detector IDs
        detector_ids = traci.lanearea.getIDList()
        print(f"Number of detectors: {len(detector_ids)}")
        
        # We expect 48 detectors (4 junctions × 12 detectors)
        if len(detector_ids) == 48:
            print("✓ Correct number of detectors (48)")
        else:
            print(f"✗ Expected 48 detectors, found {len(detector_ids)}")
        
        # Test a few detectors
        test_detectors = [
            'det_-E6_0_stop',
            'det_E0_1_stop',
            'det_E16_2_stop',
        ]
        
        for det_id in test_detectors:
            if det_id in detector_ids:
                jam_length = traci.lanearea.getJamLengthVehicle(det_id)
                veh_count = traci.lanearea.getLastStepVehicleNumber(det_id)
                print(f"  {det_id}: {veh_count} vehicles, {jam_length} jammed")
            else:
                print(f"  ✗ {det_id} not found")
        
        traci.close()
        
        print("\nDetectors: OK")
        return True
        
    except Exception as e:
        print(f"✗ Detector error: {e}")
        import traceback
        traceback.print_exc()
        try:
            traci.close()
        except:
            pass
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("SUMO CONNECTION AND CONFIGURATION TESTS")
    print("="*80 + "\n")
    
    tests = [
        ("SUMO Installation", test_sumo_installation),
        ("Network Files", test_network_files),
        ("SUMO Simulation", test_sumo_simulation),
        ("Traffic Lights", test_traffic_lights),
        ("Detectors", test_detectors),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("="*80)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*80)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)