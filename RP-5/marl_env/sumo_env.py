"""
CORRECTED: SUMO Traffic Environment with Enhanced 70-dim Observations

Features:
- 70-dim observations from obs_builder_v2 (neighbor outgoing/ingoing metrics)
- Optional min-green enforcement via config
- Centralized critic support
- 4 discrete actions with phase indices
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional
import random 

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

from marl_env.obs_builder import MAPPOObservationBuilderV2  # CORRECTED: 70-DIM ENHANCED OBS
from marl_env.reward_function import MAPPORewardFunction


class SUMOTrafficEnv(MultiAgentEnv):
    """
    SUMO Multi-Agent Environment with:
    - 70-dim enhanced observations (obs_builder_v2)
    - Optional min-green enforcement
    - 4-action space with phase indices
    """
    
    # Action to Phase Index mapping (based on your marl-proj.ttl.xml)
    ACTION_TO_PHASE = {
        0: 0,  # NS through + right → Phase 0: "GGrgrrGGrgrr"
        1: 6,  # NS left turn → Phase 6: "GrGGrrGrGGrr"
        2: 2,  # EW through + right → Phase 2: "GrrGgrGrrGGr"
        3: 4,  # EW left turn → Phase 4: "GrrGrGGrrGrG"
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
        self.sumo_seed = env_config.get('sumo_seed', 11)
        self.sumo_seed_base = self.sumo_seed
        self.episode_seed_offset = 0

        # Optional min-green enforcement
        self.enforce_min_green = env_config.get('enforce_min_green', False)
        
        pid = os.getpid()
        # Use PID to generate unique port in range 10000-65000
        # Hash the PID to get consistent port per worker
        self.traci_port = 10000 + (pid % 55000)
        
        # Add small random offset to avoid collisions if PIDs are sequential
        self.traci_port += random.randint(0, 100)
        
        print(f"[PID {pid}] Assigned TraCI port: {self.traci_port}")

        # Build ENHANCED observation builder (70-DIM with neighbor metrics)
        self.detector_config = env_config['detectors']  # Store for reward function
        self.obs_builder = MAPPOObservationBuilderV2(
            agent_ids=self.agent_ids,
            network_topology=self.network_topology,
            detector_config=self.detector_config,
            normalization_config=env_config['normalization'],
            env_config=env_config
        )
        
        # Build reward function
        self.detector_mappings = self.obs_builder.detector_mappings
        self.reward_function = MAPPORewardFunction(
            agent_ids=self.agent_ids,
            network_topology=self.network_topology,
            detector_config=self.detector_config,
            detector_mappings=self.detector_mappings,
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
        
        # Track current phase and timing for min-green enforcement
        self.current_phases = {}
        self.phase_start_times = {}
        
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
        
        # Observation space: 70 dimensions (ENHANCED OBS BUILDER)
        self.observation_space_dict = {
            agent_id: spaces.Box(
                low=-1.0,
                high=1.0,
                shape=self.obs_builder.get_observation_space_shape(agent_id),  # (70,)
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
        self.obs_builder.validate_edges(traci)
        
        # Initialize traffic lights to prevent auto-progression
        self._initialize_traffic_lights()
        
        # Reset builders
        self.obs_builder.reset()
        self.reward_function.reset()
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_count += 1
        self.episode_seed_offset += 1

        self.current_phases = {agent_id: 0 for agent_id in self.agent_ids}
        self.phase_start_times = {agent_id: 0.0 for agent_id in self.agent_ids}
        
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
    
    def _initialize_traffic_lights(self):
        """
        Initialize traffic lights to prevent auto-progression.
        
        Sets phase duration to very long so SUMO doesn't auto-advance.
        """
        for agent_id in self.agent_ids:
            try:
                # Set to Phase 0 initially
                traci.trafficlight.setPhase(agent_id, 0)
                
                # Set duration to 1000s to prevent auto-progression
                traci.trafficlight.setPhaseDuration(agent_id, 1000.0)
                
            except Exception as e:
                print(f"Warning: Could not initialize TL for {agent_id}: {e}")
    
    def step(
        self,
        action_dict: Dict[str, int]
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute one step."""
        # Apply actions (with optional min-green enforcement)
        if self.enforce_min_green:
            self._apply_actions_with_min_green(action_dict)
        else:
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
        """
        Start SUMO simulation with UNIQUE PORT per worker.
        
        This prevents port conflicts when using multiple rollout workers.
        """
        sumo_binary = "sumo-gui.exe" if self.use_gui else "sumo.exe"
        # SUMO seed is hardcoded to 42, not derived from sumo_seed_base + episode_seed_offset.
        # Per-episode seed offsetting previously caused inconsistent demand sampling across
        # workers, so we keep a single deterministic seed for the whole run.
        sumo_cmd = [
            sumo_binary,
            "-c", self.network_file.replace('.net.xml', '.sumocfg'),
            "--no-step-log",
            "--no-warnings",
            "--time-to-teleport", "-1",
            "--seed", str(42),
            "--quit-on-end"
        ]
        
        if self.use_gui:
            sumo_cmd.extend(["--start", "--delay", "100"])
    
        # Standard TraCI - use auto-assigned unique port
        max_retries = 5
        for attempt in range(max_retries):
            try:
                traci.start(
                    sumo_cmd,
                    port=self.traci_port,
                    numRetries=2
                )
                self.sumo_running = True
                print(f"[PID {os.getpid()}] SUMO started on port {self.traci_port}")
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # Port collision - try different port
                    print(f"[PID {os.getpid()}] Port {self.traci_port} busy, retrying...")
                    self.traci_port += random.randint(10, 100)
                else:
                    raise RuntimeError(f"Failed to start SUMO after {max_retries} attempts: {e}")
            
    def _apply_actions(self, action_dict: Dict[str, int]):
        """
        Apply traffic signal actions using phase indices (NO min-green enforcement).
        
        Agent learns optimal timing on its own.
        
        ACTION MAPPING:
        - Action 0 → Phase 0: NS through + right turns
        - Action 1 → Phase 6: NS left turns
        - Action 2 → Phase 2: EW through + right turns
        - Action 3 → Phase 4: EW left turns
        """
        current_time = traci.simulation.getTime()
        
        for agent_id, action in action_dict.items():
            try:
                # Map action to phase index
                new_phase = self.ACTION_TO_PHASE.get(action, 0)
                current_phase = self.current_phases.get(agent_id, 0)
                
                # Track phase changes
                if new_phase != current_phase:
                    self.current_phases[agent_id] = new_phase
                    self.phase_start_times[agent_id] = current_time
                
                # Set phase with long duration (prevents auto-advance)
                traci.trafficlight.setPhase(agent_id, new_phase)
                traci.trafficlight.setPhaseDuration(agent_id, 1000.0)
                
            except Exception as e:
                print(f"Error applying action for {agent_id}: {e}")
    
    def _apply_actions_with_min_green(self, action_dict: Dict[str, int]):
        """
        Apply actions WITH min-green enforcement.
        
        Only allows phase changes if min_green time has elapsed.
        Otherwise, maintains current phase.
        
        Use this if you want hard timing constraints.
        """
        current_time = traci.simulation.getTime()
        
        for agent_id, action in action_dict.items():
            try:
                new_phase = self.ACTION_TO_PHASE.get(action, 0)
                current_phase = self.current_phases.get(agent_id, 0)
                
                # Check if phase change requested
                phase_change = (new_phase != current_phase)
                
                if phase_change:
                    # Check if min-green is satisfied
                    phase_start = self.phase_start_times.get(agent_id, current_time)
                    elapsed = current_time - phase_start
                    
                    if elapsed < self.min_green:
                        # Min-green NOT satisfied - maintain current phase
                        new_phase = current_phase
                        phase_change = False
                    else:
                        # Min-green satisfied - allow change
                        self.current_phases[agent_id] = new_phase
                        self.phase_start_times[agent_id] = current_time
                
                # Set phase in SUMO
                traci.trafficlight.setPhase(agent_id, new_phase)
                traci.trafficlight.setPhaseDuration(agent_id, 1000.0)
                
            except Exception as e:
                print(f"Error applying action for {agent_id}: {e}")
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get 70-dim enhanced observations for all agents."""
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