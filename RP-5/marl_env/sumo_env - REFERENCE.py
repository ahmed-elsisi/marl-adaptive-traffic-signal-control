"""
FINAL CORRECTED: SUMO Traffic Environment

Your action space:
- Action 0: NS through + right (GGgrrrGGgrrr)
- Action 1: NS left turn (rrrGGGrrrGGG)
- Action 2: EW through + right (GrGGrrGrGGrr)
- Action 3: EW left turn (rrrGrGrrrGrG)

Key fixes:
1. ✓ Action space: Discrete(4) - matches your setup
2. ✓ Direct state string setting via setRedYellowGreenState
3. ✓ Observation space: 32 dimensions
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional

# SUMO imports
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

try:
    if os.environ.get('LIBSUMO_AS_TRACI', '0') == '1':
        import libsumo as traci
        print("Using libsumo (8x faster)")
    else:
        import traci
        print("Using standard TraCI")
except ImportError:
    import traci
    print("Using standard TraCI")

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from marl_env.obs_builder import MAPPOObservationBuilder
from marl_env.reward_function import MAPPORewardFunction


class SUMOTrafficEnv(MultiAgentEnv):
    """
    CORRECTED: SUMO Multi-Agent Environment with 4-action space.
    
    Actions directly set traffic light state strings.
    """
    
    # Phase configurations
    PHASE_CONFIGS = {
        0: "GGgrrrGGgrrr",  # NS through + right
        1: "rrrGGGrrrGGG",  # NS left turn
        2: "GrGGrrGrGGrr",  # EW through + right
        3: "rrrGrGrrrGrG",  # EW left turn
    }
    
    def __init__(self, env_config: Dict):
        super().__init__()
        
        self.config = env_config
        
        # Agent configuration
        self.agent_ids = env_config['agents']
        self._agent_ids = set(self.agent_ids)
        
        # Network topology
        self.network_topology = env_config['network_topology']
        
        # SUMO configuration
        self.network_file = env_config['network_file']
        self.route_file = env_config['route_file']
        self.use_gui = env_config.get('use_gui', False)
        self.num_seconds = env_config.get('num_seconds', 3600)
        self.delta_time = env_config.get('delta_time', 5)
        self.yellow_time = env_config.get('yellow_time', 3)
        self.min_green = env_config.get('min_green', 10)
        self.max_green = env_config.get('max_green', 50)
        self.sumo_seed = env_config.get('sumo_seed', 42)
        
        # Build observation builder
        self.obs_builder = MAPPOObservationBuilder(
            agent_ids=self.agent_ids,
            network_topology=self.network_topology,
            detector_config=env_config['detectors'],
            normalization_config=env_config['normalization']
        )
        
        # Build reward function
        detector_mappings = self.obs_builder.detector_mappings
        self.reward_function = MAPPORewardFunction(
            agent_ids=self.agent_ids,
            network_topology=self.network_topology,
            detector_mappings=detector_mappings,
            reward_config=env_config['reward_config'],
            normalization_config=env_config['normalization']
        )
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Episode tracking
        self.current_step = 0
        self.episode_count = 0
        
        # SUMO connection
        self.sumo = None
        self.sumo_running = False
        
        # Metrics tracking
        self.metrics = {
            'total_queue': [],
            'total_waiting_time': [],
            'throughput': [],
            'avg_speed': []
        }
    
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        # Action space: 4 discrete actions
        self.action_space_dict = {
            agent_id: spaces.Discrete(4)
            for agent_id in self.agent_ids
        }
        
        # Observation space: 32 dimensions
        self.observation_space_dict = {
            agent_id: spaces.Box(
                low=-1.0,
                high=1.0,
                shape=self.obs_builder.get_observation_space_shape(agent_id),
                dtype=np.float32
            )
            for agent_id in self.agent_ids
        }
        
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Dict, Dict]:
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)
        
        # Close existing SUMO instance
        if self.sumo_running:
            try:
                traci.close()
            except:
                pass
            self.sumo_running = False
        
        # Start SUMO
        self._start_sumo()
        
        # Reset builders
        self.obs_builder.reset()
        self.reward_function.reset()
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_count += 1
        
        # Reset metrics
        self.metrics = {
            'total_queue': [],
            'total_waiting_time': [],
            'throughput': [],
            'avg_speed': []
        }
        
        # Get initial observations
        observations = self._get_observations()
        infos = {agent_id: {} for agent_id in self.agent_ids}
        
        return observations, infos
    
    def step(
        self,
        action_dict: Dict[str, int]
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute one step."""
        # Apply actions
        self._apply_actions(action_dict)
        
        # Run simulation
        for _ in range(self.delta_time):
            traci.simulationStep()
        
        self.current_step += 1
        
        # Get observations
        observations = self._get_observations()
        
        # Calculate rewards
        rewards = self._get_rewards()
        
        # Check if done
        sim_time = traci.simulation.getTime()
        done = sim_time >= self.num_seconds
        
        terminateds = {agent_id: done for agent_id in self.agent_ids}
        terminateds['__all__'] = done
        
        truncateds = {agent_id: False for agent_id in self.agent_ids}
        truncateds['__all__'] = False
        
        # Collect metrics
        infos = self._get_infos()
        
        return observations, rewards, terminateds, truncateds, infos
    
    def _start_sumo(self):
        """Start SUMO simulation."""
        sumo_binary = "sumo-gui.exe" if self.use_gui else "sumo.exe"
        
        sumo_cmd = [
            sumo_binary,
            "-c", self.network_file.replace('.net.xml', '.sumocfg'),
            "--no-step-log",
            "--no-warnings",
            "--time-to-teleport", "-1",
            "--seed", str(self.sumo_seed),
            "--quit-on-end"
        ]
        
        if self.use_gui:
            sumo_cmd.extend(["--start", "--delay", "100"])
        
        traci.start(sumo_cmd)
        self.sumo_running = True
    
    def _apply_actions(self, action_dict: Dict[str, int]):
        """
        Apply traffic signal actions.
        
        Maps actions to phase configuration strings and sets them directly.
        
        Action 0: GGgrrrGGgrrr (NS through + right)
        Action 1: rrrGGGrrrGGG (NS left turn)
        Action 2: GrGGrrGrGGrr (EW through + right)
        Action 3: rrrGrGrrrGrG (EW left turn)
        """
        for agent_id, action in action_dict.items():
            try:
                # Get phase configuration string
                phase_state = self.PHASE_CONFIGS.get(action, self.PHASE_CONFIGS[0])
                
                # Set traffic light state directly
                traci.trafficlight.setRedYellowGreenState(agent_id, phase_state)
                
            except Exception as e:
                print(f"Error applying action for {agent_id}: {e}")
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents."""
        observations = {}
        
        for agent_id in self.agent_ids:
            obs = self.obs_builder.get_observation(agent_id, traci)
            observations[agent_id] = obs
        
        return observations
    
    def _get_rewards(self) -> Dict[str, float]:
        """Calculate rewards for all agents."""
        rewards = {}
        
        for agent_id in self.agent_ids:
            reward = self.reward_function.calculate_reward(agent_id, traci)
            rewards[agent_id] = reward
        
        return rewards
    
    def _get_infos(self) -> Dict[str, Dict]:
        """Collect metrics."""
        total_queue = 0
        total_waiting = 0
        total_vehicles = 0
        total_speed = 0
        
        for agent_id in self.agent_ids:
            detectors = self.obs_builder.detector_mappings[agent_id]
            
            for det_id in detectors:
                try:
                    queue = traci.lanearea.getJamLengthVehicle(det_id)
                    total_queue += queue
                    
                    veh_ids = traci.lanearea.getLastStepVehicleIDs(det_id)
                    for veh_id in veh_ids:
                        waiting = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                        speed = traci.vehicle.getSpeed(veh_id)
                        total_waiting += waiting
                        total_speed += speed
                        total_vehicles += 1
                except:
                    pass
        
        # Calculate averages
        avg_queue = total_queue / (4 * 12) if (4 * 12) > 0 else 0
        avg_waiting = total_waiting / max(total_vehicles, 1)
        avg_speed = total_speed / max(total_vehicles, 1)
        
        try:
            throughput = traci.simulation.getArrivedNumber()
        except:
            throughput = 0
        
        # Update metrics
        self.metrics['total_queue'].append(total_queue)
        self.metrics['total_waiting_time'].append(avg_waiting)
        self.metrics['throughput'].append(throughput)
        self.metrics['avg_speed'].append(avg_speed)
        
        # Create info dict
        infos = {}
        for agent_id in self.agent_ids:
            infos[agent_id] = {
                'avg_queue': avg_queue,
                'max_queue': max(self.metrics['total_queue']) if self.metrics['total_queue'] else 0,
                'avg_waiting_time': avg_waiting,
                'throughput': throughput,
                'avg_speed': avg_speed,
                'total_pressure': total_queue,
            }
        
        return infos
    
    def close(self):
        """Close SUMO simulation."""
        if self.sumo_running:
            try:
                traci.close()
            except:
                pass
            self.sumo_running = False
    
    def render(self):
        """Render handled by SUMO GUI."""
        pass
    
    # RLlib MultiAgentEnv compatibility
    def observation_space_sample(self, agent_ids: list = None) -> Dict:
        if agent_ids is None:
            agent_ids = self.agent_ids
        return {
            agent_id: self.observation_space_dict[agent_id].sample()
            for agent_id in agent_ids
        }
    
    def action_space_sample(self, agent_ids: list = None) -> Dict:
        if agent_ids is None:
            agent_ids = self.agent_ids
        return {
            agent_id: self.action_space_dict[agent_id].sample()
            for agent_id in agent_ids
        }
    
    def action_space_contains(self, action: Dict) -> bool:
        return all(
            self.action_space_dict[agent_id].contains(act)
            for agent_id, act in action.items()
        )
    
    def observation_space_contains(self, obs: Dict) -> bool:
        return all(
            self.observation_space_dict[agent_id].contains(ob)
            for agent_id, ob in obs.items()
        )
    
    @property
    def get_agent_ids(self) -> set:
        return self._agent_ids