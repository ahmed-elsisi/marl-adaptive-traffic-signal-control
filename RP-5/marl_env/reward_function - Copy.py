"""
MAPPO Reward Function for Traffic Signal Control

Research-backed reward design:
- Queue minimization (primary objective)
- Waiting time minimization (key metric)
- Throughput maximization (efficiency)
- Pressure-based coordination (multi-agent)
- Spatial discounting for neighbor rewards
"""

import numpy as np
from typing import Dict, List
try:
    # if os.environ.get('LIBSUMO_AS_TRACI', '0') == '1':
    import libsumo as traci
    print("Using libsumo (8x faster)")
except ImportError:
    import traci
    print("Using standard TraCI")


class MAPPORewardFunction:
    """
    Calculates rewards for MAPPO traffic signal control.
    
    Reward components (research-backed):
    1. Queue penalty (w=-0.25): Minimize stopped vehicles
    2. Waiting time penalty (w=-1.0): Primary metric for user experience
    3. Throughput bonus (w=+0.1): Encourage vehicle movement
    4. Pressure penalty (w=-0.5): Balance traffic flow
    5. Neighbor pressure (w=-0.3, discounted): Multi-agent coordination
    """
    
    def __init__(
        self,
        agent_ids: List[str],
        network_topology: Dict[str, List[str]],
        detector_config: Dict[str, Dict],
        detector_mappings: Dict[str, List[str]],
        reward_config: Dict[str, float],
        normalization_config: Dict[str, float]
    ):
        """
        Initialize reward function.
        
        Args:
            agent_ids: List of agent IDs
            network_topology: Neighbor mapping
            detector_mappings: Detector IDs per agent
            reward_config: Reward coefficients
            normalization_config: Normalization constants
        """
        self.agent_ids = agent_ids
        self.topology = network_topology
        self.detector_config = detector_config
        self.detector_mappings = detector_mappings
        self.reward_config = reward_config
        self.norm_config = normalization_config

        self.edge_mappings = self._build_edge_mappings()
        
        # Previous state tracking for deltas
        self.prev_metrics = {agent_id: {} for agent_id in agent_ids}
        
        # Throughput tracking
        self.prev_arrived_count = 0

    def _build_edge_mappings(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Build incoming/outgoing edge mappings from detector config.
        
        In SUMO edge naming:
        - If edge "E1" goes from A→B, then edge "-E1" goes from B→A
        - Incoming edges to a junction have their opposite as outgoing edges
        
        Args:
            detector_mappings: Dict with 'incoming_edges' per junction
            
        Returns:
            Dict mapping agent_id to {'incoming': [...], 'outgoing': [...]}
        """
        edge_mappings = {}
        
        for agent_id, config in self.detector_config.items():
            incoming_edges = config.get('incoming_edges', [])
            
            # Outgoing edges are the reverse direction of incoming edges
            outgoing_edges = []
            for edge in incoming_edges:
                if edge.startswith('-'):
                    # "-E1" → "E1"
                    outgoing_edges.append(edge[1:])
                else:
                    # "E1" → "-E1"
                    outgoing_edges.append('-' + edge)
            
            edge_mappings[agent_id] = {
                'incoming': incoming_edges,
                'outgoing': outgoing_edges
            }
        
        return edge_mappings
    

    def calculate_reward(
        self,
        agent_id: str,
        traci_conn,
        global_metrics: Dict = None
    ) -> float:
        """
        Calculate reward for a single agent.
        
        Args:
            agent_id: Agent identifier
            traci_conn: TraCI connection
            global_metrics: Optional global network metrics
            
        Returns:
            float: Clipped reward in range [clip_min, clip_max]
        """
        reward_components = {}
        
        # 1. Queue penalty
        queue_penalty = self._calculate_queue_penalty(agent_id, traci_conn)
        reward_components['queue'] = queue_penalty * self.reward_config['queue_weight']
        
        # 2. Waiting time penalty (PRIMARY METRIC)
        waiting_penalty = self._calculate_waiting_time_penalty(agent_id, traci_conn)
        reward_components['waiting'] = waiting_penalty * self.reward_config['waiting_time_weight']
        
        # 3. Throughput bonus
        throughput_bonus = self._calculate_throughput_bonus(agent_id, traci_conn)
        reward_components['throughput'] = throughput_bonus * self.reward_config['throughput_weight']
        
        # 4. Pressure penalty
        pressure_penalty = self._calculate_pressure_penalty(agent_id, traci_conn)
        reward_components['pressure'] = pressure_penalty * self.reward_config['pressure_weight']
        
        # 5. Neighbor pressure (spatial discounting)
        neighbor_penalty = self._calculate_neighbor_pressure(agent_id, traci_conn)
        reward_components['neighbor'] = neighbor_penalty * self.reward_config['neighbor_pressure_weight']
        
        # Total reward
        total_reward = sum(reward_components.values())
        
        # Clip to prevent instability
        clipped_reward = np.clip(
            total_reward,
            self.reward_config['clip_min'],
            self.reward_config['clip_max']
        )

        # print(f"{agent_id} Reward Components:")
        # print(f"  Queue: {reward_components['queue']:.3f}")
        # print(f"  Waiting: {reward_components['waiting']:.3f}")
        # print(f"  Throughput: {reward_components['throughput']:.3f}")
        # print(f"  Pressure: {reward_components['pressure']:.3f}")
        # print(f"  Neighbor: {reward_components['neighbor']:.3f}")
        # print(f"  TOTAL: {total_reward:.3f}")

        return clipped_reward
    
    def _get_edge_vehicle_count(self, edge_id: str, traci_conn) -> int:
        """
        Get total vehicle count on all lanes of an edge.
        
        Args:
            edge_id: SUMO edge ID
            traci_conn: TraCI connection
            
        Returns:
            Total number of vehicles on this edge
        """
        try:
            # Get all lanes on this edge
            lane_ids = traci_conn.edge.getLaneNumber(edge_id)
            total_vehicles = 0
            
            for lane_idx in range(lane_ids):
                lane_id = f"{edge_id}_{lane_idx}"
                try:
                    total_vehicles += traci_conn.lane.getLastStepVehicleNumber(lane_id)
                except:
                    pass
            
            return total_vehicles
        except:
            return 0
        
    def _calculate_queue_penalty(self, agent_id: str, traci_conn) -> float:
        """
        Calculate queue length penalty.
        
        Returns negative normalized queue length.
        """
        detectors = self.detector_mappings[agent_id]
        total_queue = 0.0
        
        for det_id in detectors:
            try:
                queue = traci_conn.lanearea.getJamLengthVehicle(det_id)
                total_queue += queue
            except:
                pass
        
        # Normalize by (num_lanes * queue_max)
        max_queue = len(detectors) * self.norm_config['queue_max']
        normalized_queue = total_queue / max_queue if max_queue > 0 else 0.0
        
        return normalized_queue
    
    def _calculate_waiting_time_penalty(self, agent_id: str, traci_conn) -> float:
        """
        Calculate waiting time penalty (PRIMARY METRIC).
        
        Waiting time = accumulated time vehicles spend stopped.
        Research: This is the most important metric for user experience.
        """
        detectors = self.detector_mappings[agent_id]
        total_waiting_time = 0.0
        
        for det_id in detectors:
            try:
                # Get vehicles in detector
                vehicle_ids = traci_conn.lanearea.getLastStepVehicleIDs(det_id)
                
                for veh_id in vehicle_ids:
                    # Get waiting time for each vehicle
                    waiting_time = traci_conn.vehicle.getAccumulatedWaitingTime(veh_id)
                    total_waiting_time += waiting_time
            except:
                pass
        
        # Normalize by (num_lanes * waiting_time_max)
        max_waiting = len(detectors) * self.norm_config['waiting_time_max']
        normalized_waiting = total_waiting_time / max_waiting if max_waiting > 0 else 0.0
        
        return normalized_waiting

    def _calculate_throughput_bonus(self, agent_id: str, traci_conn) -> float:
        """
        Calculate throughput bonus (CORRECTED).
        
        Throughput = vehicles that ARRIVED (completed trips) this step.
        Uses GLOBAL tracking to avoid 4× overcounting (each vehicle counted once).
        All agents share same throughput signal for cooperation.
        
        Fixed bugs:
        - Use getArrivedNumber() not getDepartedNumber() (arrived = completed trips)
        - Use global counter not per-agent (prevents overcounting)
        """
        try:
            # Get global arrived count (vehicles that completed trips)
            current_arrived = traci_conn.simulation.getArrivedNumber()
            
            # Calculate delta from previous step (global tracking)
            delta_arrived = current_arrived - self.prev_arrived_count
            
            # Update global counter
            self.prev_arrived_count = current_arrived
            
            # Normalize by expected throughput per agent per step
            # Assume ~1-5 vehicles per agent per 5-second step
            normalized_throughput = delta_arrived / (len(self.agent_ids) * 5.0)
        except:
            normalized_throughput = 0.0
        
        return normalized_throughput
    
    def _calculate_pressure_penalty(self, agent_id: str, traci_conn) -> float:
        """
        Calculate pressure penalty using ACCURATE edge-level counts.
        
        Pressure = Σ(vehicles on incoming edges) - Σ(vehicles on outgoing edges)
        
        This measures traffic imbalance at the junction:
        - High positive pressure: More vehicles arriving than leaving → Congestion building
        - Near-zero pressure: Balanced flow
        - Negative pressure: More vehicles leaving than arriving → Demand decreasing
        
        Network topology:
        - J1: incoming=["-E6", "E0", "E16", "-E1"], outgoing=["E6", "-E0", "-E16", "E1"]
        - J2: incoming=["E1", "-E7", "-E11", "-E10"], outgoing=["-E1", "E7", "E11", "E10"]
        - J3: incoming=["-E17", "-E16", "-E18", "-E15"], outgoing=["E17", "E16", "E18", "E15"]
        - J4: incoming=["E15", "E11", "-E8", "-E9"], outgoing=["-E15", "-E11", "E8", "E9"]
        """
        edge_config = self.edge_mappings[agent_id]
        
        # Count vehicles on incoming edges
        incoming_count = 0
        for edge_id in edge_config['incoming']:
            incoming_count += self._get_edge_vehicle_count(edge_id, traci_conn)
        
        # Count vehicles on outgoing edges
        outgoing_count = 0
        for edge_id in edge_config['outgoing']:
            outgoing_count += self._get_edge_vehicle_count(edge_id, traci_conn)
        
        # Pressure = incoming - outgoing
        # Use absolute value to penalize any imbalance (high or low)
        pressure = incoming_count - outgoing_count
        positive_pressure = max(0, pressure)

        # Normalize
        # Max pressure = all lanes full on incoming, empty on outgoing
        # Assume ~4 incoming edges × 3 lanes × max_vehicles_per_lane
        max_pressure = len(edge_config['incoming']) * 3 * self.norm_config['pressure_max']
        normalized_pressure = positive_pressure / max_pressure if max_pressure > 0 else 0.0

        return normalized_pressure
    
    def _calculate_neighbor_pressure(self, agent_id: str, traci_conn) -> float:
        """
        Calculate neighbor pressure with spatial discounting.
        
        Research: Spatial discounting (0.9) helps multi-agent coordination.
        Agents care about neighbors' pressure but with reduced weight.
        
        Network topology:
        - J1 neighbors: J2, J3
        - J2 neighbors: J1, J4
        - J3 neighbors: J1, J4
        - J4 neighbors: J2, J3
        
        Returns normalized sum of neighbor pressures (incoming - outgoing).
        """
        neighbors = self.topology.get(agent_id, [])
        total_neighbor_pressure = 0.0
        
        for neighbor_id in neighbors:
            neighbor_config = self.edge_mappings.get(neighbor_id, {})
            
            if not neighbor_config:
                continue
            
            # Count vehicles on neighbor's incoming edges
            neighbor_incoming = 0
            for edge_id in neighbor_config.get('incoming', []):
                neighbor_incoming += self._get_edge_vehicle_count(edge_id, traci_conn)
            
            # Count vehicles on neighbor's outgoing edges
            neighbor_outgoing = 0
            for edge_id in neighbor_config.get('outgoing', []):
                neighbor_outgoing += self._get_edge_vehicle_count(edge_id, traci_conn)
            
            # Neighbor pressure = |incoming - outgoing|
            neighbor_pressure = neighbor_incoming - neighbor_outgoing
            positive_neighbor_pressure = max(0, neighbor_pressure)

            # Apply spatial discount (0.9)
            discounted_pressure = positive_neighbor_pressure * self.norm_config['spatial_discount']
            total_neighbor_pressure += discounted_pressure

        # Normalize
        if neighbors:
            # Max neighbor pressure = num_neighbors × (4 edges × 3 lanes × max_vehicles)
            max_single_neighbor = 4 * 3 * self.norm_config['pressure_max']
            max_neighbor_pressure = len(neighbors) * max_single_neighbor
            normalized = total_neighbor_pressure / max_neighbor_pressure if max_neighbor_pressure > 0 else 0.0
        else:
            normalized = 0.0
        
        return normalized
    
    def calculate_global_reward(self, traci_conn) -> float:
        """
        Calculate global network reward (for analysis/logging).
        
        This is NOT used for training in MAPPO, but useful for monitoring.
        """
        total_reward = 0.0
        
        for agent_id in self.agent_ids:
            agent_reward = self.calculate_reward(agent_id, traci_conn)
            total_reward += agent_reward
        
        return total_reward / len(self.agent_ids)
    
    def reset(self):
        """Reset internal state for new episode."""
        self.prev_metrics = {agent_id: {} for agent_id in self.agent_ids}
        self.prev_arrived_count = 0