# test_env.py
import sys
sys.path.insert(0, '.')

from marl_env.sumo_env import SUMOTrafficEnv
import yaml

# Load config
with open('configs/mappo_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Build env config
env_config = {
    'agents': config['agents'],
    'network_topology': config['network_topology'],
    'detectors': config['detectors'],
    'normalization': config['normalization'],
    'reward_config': config['reward_config'],
    'network_file': config['env_config']['network_file'],
    'route_file': config['env_config']['route_file'],
    'use_gui': False,
    'num_seconds': 100,  # Short test
    'delta_time': 5,
    'yellow_time': 3,
    'min_green': 10,
    'max_green': 50,
    'sumo_seed': 42,
}

print("Creating environment...")
try:
    env = SUMOTrafficEnv(env_config)
    print("✓ Environment created")
    
    print("\nTrying reset...")
    obs, info = env.reset()
    print("✓ Reset successful!")
    
    print(f"\nObservation keys: {obs.keys()}")
    print(f"Observation shapes:")
    for agent_id, agent_obs in obs.items():
        print(f"  {agent_id}: {agent_obs.shape} - dtype: {agent_obs.dtype}")
        print(f"    Min: {agent_obs.min():.3f}, Max: {agent_obs.max():.3f}")
        
        # Check for NaN or Inf
        import numpy as np
        if np.isnan(agent_obs).any():
            print(f"    ❌ Contains NaN!")
        if np.isinf(agent_obs).any():
            print(f"    ❌ Contains Inf!")
    
    print("\n✓ Environment works correctly!")
    env.close()
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()