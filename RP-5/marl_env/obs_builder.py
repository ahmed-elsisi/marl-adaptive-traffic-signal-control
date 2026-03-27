"""
ENHANCED: MAPPO Observation Builder with Neighbor Lane Metrics

Key enhancements for green wave and backpressure learning:
1. OUTGOING metrics (neighbor → agent): Traffic neighbor is sending TO agent
   - Enables green wave learning and platoon anticipation
2. INGOING metrics (agent → neighbor): Traffic agent is sending TO neighbor  
   - Enables backpressure control to avoid saturating neighbors

New observation dimensions:
- Local features: 28 (unchanged)
  - Queue lengths: 12
  - Current direction: 2
  - Elapsed time: 1
  - Movement pressures: 6
  - Pressure derivatives: 6
  - Min green flag: 1

- Neighbor features: 42 (21 per neighbor × 2 neighbors) ← ENHANCED!
  Per neighbor (21 features):
  - Shared queues: 3
  - Direction: 2
  - Movement pressures: 3
  - Total pressure: 1
  - OUTGOING (neighbor→agent): 6 ← NEW!
    - Queue: 3 (right, straight, left)
    - Waiting time: 3 (right, straight, left)
  - INGOING (agent→neighbor): 6 ← NEW!
    - Queue: 3 (right, straight, left)
    - Available space: 3 (right, straight, left)

Total: 28 + 42 = 70 dimensions per agent
Global state for centralized critic: 70 × 4 agents = 280 dimensions
"""

import numpy as np
from typing import Dict, List, Tuple
try:
    import libsumo as traci
    print("Using libsumo (8x faster)")
except ImportError:
    import traci
    print("Using standard TraCI")


class MAPPOObservationBuilderV2:
    """
    Enhanced observation builder with neighbor outgoing/ingoing lane metrics.
    
    Enables agents to learn:
    - Green wave coordination (outgoing metrics)
    - Backpressure control (ingoing metrics)
    - Better multi-agent cooperation
    """
    
    def __init__(
        self,
        agent_ids: List[str],
        network_topology: Dict[str, List[str]],
        detector_config: Dict[str, Dict],
        normalization_config: Dict[str, float]
    ):
        self.agent_ids = agent_ids
        self.topology = network_topology
        self.detector_config = detector_config
        self.norm_config = normalization_config
        
        # Build detector mappings
        self._build_detector_mappings()
        
        # Build edge connectivity for outgoing/ingoing metrics
        self._build_edge_connectivity()
        
        # Direction tracking
        self.current_directions = {agent_id: 0 for agent_id in agent_ids}
        self.phase_start_times = {agent_id: 0.0 for agent_id in agent_ids}
        
        # Movement-specific pressure tracking
        self.prev_pressure = {
            agent_id: {
                'NS_right': 0.0, 'NS_straight': 0.0, 'NS_left': 0.0,
                'EW_right': 0.0, 'EW_straight': 0.0, 'EW_left': 0.0
            } 
            for agent_id in agent_ids
        }
        
        # Dimensions
        self.local_dim = 28  # unchanged
        self.neighbor_dim_per = 21  # 9 (old) + 6 (outgoing) + 6 (ingoing)
        
    def _build_detector_mappings(self):
        """Build detector ID mappings."""
        self.detector_mappings = {}
        
        for agent_id in self.agent_ids:
            config = self.detector_config[agent_id]
            detectors = []
            
            for edge in config['incoming_edges']:
                for movement in config['movements']:
                    det_id = f"det_{edge}_{movement}_stop"
                    detectors.append(det_id)
            
            self.detector_mappings[agent_id] = detectors
            assert len(detectors) == 12, f"{agent_id} has {len(detectors)} detectors, expected 12"
    
    def _build_edge_connectivity(self):
        """
        Build edge connectivity for outgoing/ingoing metrics.
        
        CRITICAL: Verify these edges match your SUMO network!
        Check with: sumo-gui -n sumo_network/marl-proj.net.xml
        
        Structure:
        {
            'J1': {
                'J2': {
                    'outgoing': [edges from J2 to J1],  # Traffic J2 is sending TO J1
                    'ingoing': [edges from J1 to J2]    # Traffic J1 is sending TO J2
                }
            }
        }
        """
        self.edge_connectivity = {
            'J1': {
                'J2': {
                    'outgoing': ['-E1'],  # J2→J1: Traffic coming from J2
                    'ingoing': ['E1']     # J1→J2: Traffic going to J2
                },
                'J3': {
                    'outgoing': ['E16'],  # J3→J1
                    'ingoing': ['-E16']     # J1→J3
                }
            },
            'J2': {
                'J1': {
                    'outgoing': ['E1'],   # J1→J2
                    'ingoing': ['-E1']    # J2→J1
                },
                'J4': {
                    'outgoing': ['-E11'],  # J4→J2
                    'ingoing': ['E11']     # J2→J4
                }
            },
            'J3': {
                'J1': {
                    'outgoing': ['-E16'],  # J1→J3
                    'ingoing': ['E16']   # J3→J1
                },
                'J4': {
                    'outgoing': ['-E15'],  # J4→J3
                    'ingoing': ['E15']     # J3→J4
                }
            },
            'J4': {
                'J2': {
                    'outgoing': ['E11'],  # J2→J4
                    'ingoing': ['-E11']   # J4→J2
                },
                'J3': {
                    'outgoing': ['E15'],  # J3→J4
                    'ingoing': ['-E15']   # J4→J3
                }
            }
        }
    
    def get_observation(self, agent_id: str, traci_conn) -> np.ndarray:
        """Build enhanced observation vector (70 dims)."""
        obs_parts = []
        
        # LOCAL FEATURES (28 dims) - unchanged
        # 1. Queue lengths (12)
        queue_lengths = self._get_queue_lengths(agent_id, traci_conn)
        obs_parts.append(queue_lengths)
        
        # 2. Current direction (2)
        direction_onehot = self._get_direction_onehot(agent_id, traci_conn)
        obs_parts.append(direction_onehot)
        
        # 3. Elapsed time (1)
        elapsed_time = self._get_elapsed_phase_time(agent_id, traci_conn)
        obs_parts.append(np.array([elapsed_time], dtype=np.float32))
        
        # 4. Movement-specific pressure (6)
        pressures = self._get_movement_pressures(agent_id, traci_conn)
        obs_parts.append(pressures)
        
        # 5. Pressure derivatives (6)
        pressure_deriv = self._get_pressure_derivatives(agent_id, pressures)
        obs_parts.append(pressure_deriv)
        
        # 6. Min green satisfied (1)
        min_green = self._get_min_green_flag(agent_id, traci_conn)
        obs_parts.append(np.array([min_green], dtype=np.float32))
        
        # NEIGHBOR FEATURES (42 dims) - ENHANCED!
        # 7. Enhanced neighbor features with outgoing/ingoing metrics
        neighbor_features = self._get_enhanced_neighbor_features(agent_id, traci_conn)
        obs_parts.append(neighbor_features)
        
        observation = np.concatenate(obs_parts).astype(np.float32)
        
        # Verify dimensions
        expected_dim = self.get_observation_space_shape(agent_id)[0]
        assert observation.shape[0] == expected_dim, \
            f"Observation dim mismatch: got {observation.shape[0]}, expected {expected_dim}"
        
        return observation
    
    # ========== LOCAL FEATURE METHODS (Unchanged) ==========
    
    def _get_queue_lengths(self, agent_id: str, traci_conn) -> np.ndarray:
        """Get normalized queue lengths (12 lanes)."""
        detectors = self.detector_mappings[agent_id]
        queue_lengths = []
        
        for det_id in detectors:
            try:
                halted = traci_conn.lanearea.getJamLengthVehicle(det_id)
                queue_lengths.append(halted)
            except:
                queue_lengths.append(0.0)
        
        queue_array = np.array(queue_lengths, dtype=np.float32)
        normalized = queue_array / self.norm_config['queue_max']
        return np.clip(normalized, 0.0, 1.0)
    
    def _get_direction_onehot(self, agent_id: str, traci_conn) -> np.ndarray:
        """Get current direction (NS or EW)."""
        try:
            tl_state = traci_conn.trafficlight.getRedYellowGreenState(agent_id)
            
            if tl_state.startswith('GGg') or tl_state.startswith('rrrGGG'):
                direction_idx = 0  # NS
            elif tl_state.startswith('GrGG') or tl_state.startswith('rrrGrG'):
                direction_idx = 1  # EW
            else:
                direction_idx = self.current_directions.get(agent_id, 0)
            
            if direction_idx != self.current_directions.get(agent_id, 0):
                self.current_directions[agent_id] = direction_idx
                self.phase_start_times[agent_id] = traci_conn.simulation.getTime()
        except:
            direction_idx = 0
        
        onehot = np.zeros(2, dtype=np.float32)
        onehot[direction_idx] = 1.0
        return onehot
    
    def _get_elapsed_phase_time(self, agent_id: str, traci_conn) -> float:
        """Get normalized elapsed time."""
        try:
            current_time = traci_conn.simulation.getTime()
            start_time = self.phase_start_times.get(agent_id, current_time)
            elapsed = current_time - start_time
        except:
            elapsed = 0.0
        
        normalized = elapsed / self.norm_config['phase_time_max']
        return np.clip(normalized, 0.0, 1.0)
    
    def _get_movement_pressures(self, agent_id: str, traci_conn) -> np.ndarray:
        """Calculate movement-specific pressure (6 values)."""
        pressures = np.zeros(6, dtype=np.float32)
        
        try:
            detectors = self.detector_mappings[agent_id]
            
            # Count incoming vehicles by movement type
            incoming_ns_right = incoming_ns_straight = incoming_ns_left = 0
            incoming_ew_right = incoming_ew_straight = incoming_ew_left = 0
            
            for i, det_id in enumerate(detectors):
                try:
                    count = traci_conn.lanearea.getLastStepVehicleNumber(det_id)
                    movement_idx = i % 3
                    
                    if i < 6:  # NS lanes
                        if movement_idx == 0:
                            incoming_ns_right += count
                        elif movement_idx == 1:
                            incoming_ns_straight += count
                        else:
                            incoming_ns_left += count
                    else:  # EW lanes
                        if movement_idx == 0:
                            incoming_ew_right += count
                        elif movement_idx == 1:
                            incoming_ew_straight += count
                        else:
                            incoming_ew_left += count
                except:
                    pass
            
            # Count outgoing vehicles
            outgoing_ns = self._count_outgoing_vehicles(agent_id, 'NS', traci_conn)
            outgoing_ew = self._count_outgoing_vehicles(agent_id, 'EW', traci_conn)
            
            # Distribute outgoing proportionally
            total_ns = incoming_ns_right + incoming_ns_straight + incoming_ns_left
            if total_ns > 0:
                outgoing_ns_right = outgoing_ns * (incoming_ns_right / total_ns)
                outgoing_ns_straight = outgoing_ns * (incoming_ns_straight / total_ns)
                outgoing_ns_left = outgoing_ns * (incoming_ns_left / total_ns)
            else:
                outgoing_ns_right = outgoing_ns_straight = outgoing_ns_left = 0
            
            total_ew = incoming_ew_right + incoming_ew_straight + incoming_ew_left
            if total_ew > 0:
                outgoing_ew_right = outgoing_ew * (incoming_ew_right / total_ew)
                outgoing_ew_straight = outgoing_ew * (incoming_ew_straight / total_ew)
                outgoing_ew_left = outgoing_ew * (incoming_ew_left / total_ew)
            else:
                outgoing_ew_right = outgoing_ew_straight = outgoing_ew_left = 0
            
            # Calculate pressure
            pressures[0] = incoming_ns_right - outgoing_ns_right
            pressures[1] = incoming_ns_straight - outgoing_ns_straight
            pressures[2] = incoming_ns_left - outgoing_ns_left
            pressures[3] = incoming_ew_right - outgoing_ew_right
            pressures[4] = incoming_ew_straight - outgoing_ew_straight
            pressures[5] = incoming_ew_left - outgoing_ew_left
            
        except Exception as e:
            pass
        
        normalized = pressures / self.norm_config['pressure_max']
        return np.clip(normalized, -1.0, 1.0)
    
    def _count_outgoing_vehicles(self, agent_id: str, direction: str, traci_conn) -> float:
        """Count vehicles on outgoing lanes."""
        try:
            outgoing_edges = self._get_outgoing_edges(agent_id, direction)
            
            outgoing_count = 0
            for edge in outgoing_edges:
                try:
                    num_lanes = traci_conn.edge.getLaneNumber(edge)
                    for lane_idx in range(num_lanes):
                        lane_id = f"{edge}_{lane_idx}"
                        veh_count = traci_conn.lane.getLastStepVehicleNumber(lane_id)
                        outgoing_count += veh_count
                except:
                    pass
            
            return outgoing_count
        except:
            return 0.0
    
    def _get_outgoing_edges(self, agent_id: str, direction: str) -> List[str]:
        """Get outgoing edges for agent."""
        outgoing_map = {
            'J1': {
                'NS': ['E1', '-E0'],
                'EW': ['E6', '-E16']
            },
            'J2': {
                'NS': ['-E1', 'E10'],
                'EW': ['E7', 'E11']
            },
            'J3': {
                'NS': ['E15', 'E17'],
                'EW': ['E16', 'E18']
            },
            'J4': {
                'NS': ['E9', '-E15'],
                'EW': ['-E11', 'E8']
            }
        }
        
        return outgoing_map.get(agent_id, {}).get(direction, [])
    
    def _get_pressure_derivatives(self, agent_id: str, current_pressures: np.ndarray) -> np.ndarray:
        """Calculate pressure rate of change (6 values)."""
        derivatives = np.zeros(6, dtype=np.float32)
        
        try:
            derivatives[0] = current_pressures[0] - self.prev_pressure[agent_id]['NS_right']
            derivatives[1] = current_pressures[1] - self.prev_pressure[agent_id]['NS_straight']
            derivatives[2] = current_pressures[2] - self.prev_pressure[agent_id]['NS_left']
            derivatives[3] = current_pressures[3] - self.prev_pressure[agent_id]['EW_right']
            derivatives[4] = current_pressures[4] - self.prev_pressure[agent_id]['EW_straight']
            derivatives[5] = current_pressures[5] - self.prev_pressure[agent_id]['EW_left']
            
            # Update previous
            self.prev_pressure[agent_id]['NS_right'] = float(current_pressures[0])
            self.prev_pressure[agent_id]['NS_straight'] = float(current_pressures[1])
            self.prev_pressure[agent_id]['NS_left'] = float(current_pressures[2])
            self.prev_pressure[agent_id]['EW_right'] = float(current_pressures[3])
            self.prev_pressure[agent_id]['EW_straight'] = float(current_pressures[4])
            self.prev_pressure[agent_id]['EW_left'] = float(current_pressures[5])
        except:
            pass
        
        return np.clip(derivatives, -1.0, 1.0)
    
    def _get_min_green_flag(self, agent_id: str, traci_conn) -> float:
        """Check if minimum green time satisfied."""
        try:
            current_time = traci_conn.simulation.getTime()
            start_time = self.phase_start_times.get(agent_id, current_time)
            elapsed = current_time - start_time
            
            min_green = 10.0
            flag = 1.0 if elapsed >= min_green else 0.0
        except:
            flag = 0.0
        
        return flag
    
    # ========== ENHANCED NEIGHBOR FEATURES (NEW!) ==========
    
    def _get_enhanced_neighbor_features(self, agent_id: str, traci_conn) -> np.ndarray:
        """
        Get enhanced neighbor features with outgoing/ingoing metrics.
        
        Per neighbor (21 features):
        - Shared queues: 3
        - Direction: 2
        - Movement pressures: 3
        - Total pressure: 1
        - OUTGOING (neighbor→agent): 6
        - INGOING (agent→neighbor): 6
        """
        neighbors = self.topology.get(agent_id, [])
        neighbor_features = []
        
        for neighbor_id in neighbors:
            # Original features (9)
            shared_queues = self._get_shared_queues(neighbor_id, traci_conn)
            neighbor_direction = self._get_direction_onehot(neighbor_id, traci_conn)
            neighbor_pressures = self._get_movement_pressures(neighbor_id, traci_conn)
            
            # Combined pressures (3)
            pressure_right = (neighbor_pressures[0] + neighbor_pressures[3]) / 2
            pressure_straight = (neighbor_pressures[1] + neighbor_pressures[4]) / 2
            pressure_left = (neighbor_pressures[2] + neighbor_pressures[5]) / 2
            
            # Total pressure (1)
            total_pressure = np.sum(np.abs(neighbor_pressures))
            
            # NEW: Outgoing metrics (neighbor→agent) (6)
            outgoing_metrics = self._get_neighbor_outgoing_metrics(
                agent_id, neighbor_id, traci_conn
            )
            
            # NEW: Ingoing metrics (agent→neighbor) (6)
            ingoing_metrics = self._get_neighbor_ingoing_metrics(
                agent_id, neighbor_id, traci_conn
            )
            
            # Concatenate all features (3 + 2 + 3 + 1 + 6 + 6 = 21)
            neighbor_feat = np.concatenate([
                shared_queues,  # 3
                neighbor_direction,  # 2
                np.array([pressure_right, pressure_straight, pressure_left], dtype=np.float32),  # 3
                np.array([total_pressure], dtype=np.float32),  # 1
                outgoing_metrics,  # 6
                ingoing_metrics  # 6
            ])
            
            neighbor_features.append(neighbor_feat)
        
        # Concatenate all neighbors or return zeros
        if neighbor_features:
            neighbor_array = np.concatenate(neighbor_features)
        else:
            num_neighbors = len(self.topology.get(agent_id, []))
            neighbor_array = np.zeros(self.neighbor_dim_per * num_neighbors, dtype=np.float32)
        
        return neighbor_array
    
    def _get_shared_queues(self, neighbor_id: str, traci_conn) -> np.ndarray:
        """Get shared queue lengths from neighbor (3 values)."""
        neighbor_detectors = self.detector_mappings.get(neighbor_id, [])
        shared_queues = []
        
        for i in range(3):
            if i < len(neighbor_detectors):
                try:
                    queue = traci_conn.lanearea.getJamLengthVehicle(neighbor_detectors[i])
                    shared_queues.append(queue / self.norm_config['queue_max'])
                except:
                    shared_queues.append(0.0)
            else:
                shared_queues.append(0.0)
        
        return np.array(shared_queues, dtype=np.float32)
    
    def _get_neighbor_outgoing_metrics(
        self, 
        agent_id: str, 
        neighbor_id: str, 
        traci_conn
    ) -> np.ndarray:
        """
        Get OUTGOING metrics: Traffic FROM neighbor TO agent.
        
        Returns 6 values:
        - queue_right: Queued right-turn vehicles from neighbor to agent
        - queue_straight: Queued straight vehicles from neighbor to agent
        - queue_left: Queued left-turn vehicles from neighbor to agent
        - wait_right: Average waiting time for right turns
        - wait_straight: Average waiting time for straight
        - wait_left: Average waiting time for left turns
        
        Purpose: Enables green wave learning and platoon anticipation
        """
        metrics = np.zeros(6, dtype=np.float32)
        
        try:
            # Get edges FROM neighbor TO agent
            connectivity = self.edge_connectivity.get(agent_id, {}).get(neighbor_id, {})
            outgoing_edges = connectivity.get('outgoing', [])
            
            if not outgoing_edges:
                return metrics
            
            # Count vehicles and waiting times by movement type
            queue_by_movement = {0: 0, 1: 0, 2: 0}  # right, straight, left
            wait_by_movement = {0: [], 1: [], 2: []}
            
            for edge in outgoing_edges:
                try:
                    num_lanes = traci_conn.edge.getLaneNumber(edge)
                    
                    for lane_idx in range(num_lanes):
                        lane_id = f"{edge}_{lane_idx}"
                        
                        # Get vehicles in lane
                        veh_ids = traci_conn.lane.getLastStepVehicleIDs(lane_id)
                        
                        for veh_id in veh_ids:
                            try:
                                # Determine movement type from lane
                                movement_type = lane_idx % 3  # 0=right, 1=straight, 2=left
                                
                                # Count queue
                                speed = traci_conn.vehicle.getSpeed(veh_id)
                                if speed < 0.1:  # Stopped vehicle
                                    queue_by_movement[movement_type] += 1
                                
                                # Get waiting time
                                wait_time = traci_conn.vehicle.getAccumulatedWaitingTime(veh_id)
                                if wait_time > 0:
                                    wait_by_movement[movement_type].append(wait_time)
                                    
                            except:
                                continue
                except:
                    continue
            
            # Normalize queues
            metrics[0] = queue_by_movement[0] / self.norm_config['queue_max']  # right
            metrics[1] = queue_by_movement[1] / self.norm_config['queue_max']  # straight
            metrics[2] = queue_by_movement[2] / self.norm_config['queue_max']  # left
            
            # Average waiting times
            metrics[3] = (np.mean(wait_by_movement[0]) if wait_by_movement[0] else 0.0) / self.norm_config['waiting_time_max']
            metrics[4] = (np.mean(wait_by_movement[1]) if wait_by_movement[1] else 0.0) / self.norm_config['waiting_time_max']
            metrics[5] = (np.mean(wait_by_movement[2]) if wait_by_movement[2] else 0.0) / self.norm_config['waiting_time_max']
            
        except Exception as e:
            pass
        
        return np.clip(metrics, 0.0, 1.0)
    
    def _get_neighbor_ingoing_metrics(
        self, 
        agent_id: str, 
        neighbor_id: str, 
        traci_conn
    ) -> np.ndarray:
        """
        Get INGOING metrics: Traffic FROM agent TO neighbor.
        
        Returns 6 values:
        - queue_right: Queued right-turn vehicles from agent to neighbor
        - queue_straight: Queued straight vehicles from agent to neighbor
        - queue_left: Queued left-turn vehicles from agent to neighbor
        - space_right: Available space for right turns
        - space_straight: Available space for straight
        - space_left: Available space for left turns
        
        Purpose: Enables backpressure control to avoid saturating neighbors
        """
        metrics = np.zeros(6, dtype=np.float32)
        
        try:
            # Get edges FROM agent TO neighbor
            connectivity = self.edge_connectivity.get(agent_id, {}).get(neighbor_id, {})
            ingoing_edges = connectivity.get('ingoing', [])
            
            if not ingoing_edges:
                return metrics
            
            # Count vehicles and calculate available space
            queue_by_movement = {0: 0, 1: 0, 2: 0}
            capacity_by_movement = {0: 0, 1: 0, 2: 0}
            occupied_by_movement = {0: 0, 1: 0, 2: 0}
            
            for edge in ingoing_edges:
                try:
                    num_lanes = traci_conn.edge.getLaneNumber(edge)
                    
                    for lane_idx in range(num_lanes):
                        lane_id = f"{edge}_{lane_idx}"
                        movement_type = lane_idx % 3
                        
                        # Get lane capacity (length / avg vehicle length)
                        lane_length = traci_conn.lane.getLength(lane_id)
                        capacity = lane_length / 7.5  # Assume 7.5m per vehicle
                        capacity_by_movement[movement_type] += capacity
                        
                        # Get current occupancy
                        veh_ids = traci_conn.lane.getLastStepVehicleIDs(lane_id)
                        occupied_by_movement[movement_type] += len(veh_ids)
                        
                        # Count stopped vehicles
                        for veh_id in veh_ids:
                            try:
                                speed = traci_conn.vehicle.getSpeed(veh_id)
                                if speed < 0.1:
                                    queue_by_movement[movement_type] += 1
                            except:
                                continue
                except:
                    continue
            
            # Normalize queues
            metrics[0] = queue_by_movement[0] / self.norm_config['queue_max']
            metrics[1] = queue_by_movement[1] / self.norm_config['queue_max']
            metrics[2] = queue_by_movement[2] / self.norm_config['queue_max']
            
            # Calculate available space (1.0 = full capacity available, 0.0 = no space)
            for i, movement in enumerate([0, 1, 2]):
                if capacity_by_movement[movement] > 0:
                    space = 1.0 - (occupied_by_movement[movement] / capacity_by_movement[movement])
                    metrics[3 + i] = max(0.0, space)
                else:
                    metrics[3 + i] = 0.0
                    
        except Exception as e:
            pass
        
        return np.clip(metrics, 0.0, 1.0)
    
    def get_observation_space_shape(self, agent_id: str) -> Tuple[int]:
        """Get observation space shape (70 dims)."""
        num_neighbors = len(self.topology.get(agent_id, []))
        total_dim = self.local_dim + (self.neighbor_dim_per * num_neighbors)
        return (total_dim,)
    
    def validate_edges(self, traci_conn):
        """Validate edge connectivity matches SUMO network."""
        for agent_id, neighbors in self.edge_connectivity.items():
            for neighbor_id, edges in neighbors.items():
                for direction in ['outgoing', 'ingoing']:
                    for edge in edges[direction]:
                        try:
                            traci_conn.edge.getLaneNumber(edge)
                        except:
                            print(f"ERROR: Edge '{edge}' not found!")
                            print(f"  In: {agent_id} → {neighbor_id} ({direction})")
                            raise ValueError(f"Invalid edge: {edge}")
                    
    def reset(self):
        """Reset for new episode."""
        self.current_directions = {agent_id: 0 for agent_id in self.agent_ids}
        self.phase_start_times = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.prev_pressure = {
            agent_id: {
                'NS_right': 0.0, 'NS_straight': 0.0, 'NS_left': 0.0,
                'EW_right': 0.0, 'EW_straight': 0.0, 'EW_left': 0.0
            } 
            for agent_id in self.agent_ids
        }