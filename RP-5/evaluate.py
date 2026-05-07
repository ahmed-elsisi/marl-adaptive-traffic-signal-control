"""
CORRECTED EVALUATION - PROPER ARRIVAL TRACKING

Critical Fix: Collects arrivals DURING delta_time loop, not after.

This ensures we capture ALL vehicle arrivals, matching how baselines work.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import csv
from typing import Dict, List
from datetime import datetime
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from marl_env.sumo_env import SUMOTrafficEnv
from models.mappo_model import MAPPOModelCentralizedCritic
from models.ippo_model import IPPOModelDecentralizedCritic

try:
    import libsumo as traci
    print("✓ Using libsumo (fast)")
except ImportError:
    import traci
    print("✓ Using standard TraCI")


def ensure_results_dir(base_dir: str = "metrics") -> str:
    """Ensure results directory exists."""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class EnvWrapperWithMetrics:
    """
    Wrapper that collects arrivals DURING delta_time loop.
    
    CRITICAL: This is the ONLY way to capture all arrivals without
    bypassing environment logic.
    """
    
    def __init__(self, env, metrics_collector):
        self.env = env
        self.metrics = metrics_collector
        
        
        # Expose environment attributes
        self.agent_ids = env.agent_ids
        self.observation_space_dict = env.observation_space_dict
        self.action_space_dict = env.action_space_dict
    
    def step(self, action_dict):
        """
        Instrumented step that collects metrics DURING simulation.
        
        This is the KEY FIX: We hook into the environment's step method
        to collect arrivals during the delta_time loop without bypassing
        any environment logic.
        """
        # Track actions for switch counting
        for agent_id, action in action_dict.items():
            self.metrics.track_action(agent_id, action)
        
        # Apply actions (this sets phases, handles yellow time, etc.)
        # Pass arrival-sampling callback so any min_red clearance steps are also sampled.
        self.env._apply_actions(
            action_dict,
            on_sim_step=self.metrics.sample_arrivals_this_step,
        )

        # CRITICAL FIX: Run simulation and collect metrics DURING loop
        for _ in range(self.env.delta_time):
            traci.simulationStep()
            # Collect arrivals THIS STEP (before they're cleared)
            self.metrics.sample_arrivals_this_step()
        
        # After delta_time loop, collect other metrics
        self.metrics.sample_other_metrics()
        
        self.env.current_step += 1
        
        # Get observations and rewards using environment's methods
        observations = self.env._get_observations()
        rewards = self.env._get_rewards()
        
        # Check termination
        sim_time = traci.simulation.getTime()
        done = sim_time >= self.env.num_seconds
        
        terminateds = {agent_id: done for agent_id in self.env.agent_ids}
        terminateds['__all__'] = done
        
        truncateds = {agent_id: False for agent_id in self.env.agent_ids}
        truncateds['__all__'] = False
        
        infos = self.env._get_infos()
        
        return observations, rewards, terminateds, truncateds, infos
    
    def reset(self, **kwargs):
        """Reset environment."""
        return self.env.reset(**kwargs)
    
    def close(self):
        """Close environment."""
        self.env.close()


class MetricsCollector:
    """
    Metrics collector with SPLIT sampling:
    - Arrivals: Sampled EVERY simulation step (during delta_time loop)
    - Other metrics: Sampled ONCE per RL step (after delta_time loop)
    """
    
    def __init__(self, agent_ids: List[str]):
        self.agent_ids = agent_ids
        
        # Time-series data
        self.times = []
        self.total_halts = []
        self.cumulative_arrivals = []
        self.cumulative_departed = []
        self.running_avg_wait = []
        self.active_vehicles = []
        
        # Per-agent data
        self.per_agent_halts = {agent: [] for agent in agent_ids}
        self.switch_counts = {agent: 0 for agent in agent_ids}
        self.last_actions = {agent: None for agent in agent_ids}
        
        # Tracking
        self.last_wait = {}  # vehicle_id -> waiting time
        self.total_arrivals = 0
        self.total_departed = 0
        self.total_wait_sum = 0.0
        self.total_teleported = 0
    
    def sample_arrivals_this_step(self):
        """
        CRITICAL: Sample arrivals THIS simulation step.
        
        Must be called EVERY simulationStep() to avoid missing arrivals.
        """
        try:
            # Update waiting times for active vehicles
            active_veh = traci.vehicle.getIDList()
            for vid in active_veh:
                try:
                    self.last_wait[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)
                except:
                    pass
            
            # Get arrivals THIS STEP (gets cleared after each call!)
            arrived = traci.simulation.getArrivedIDList()
            if arrived:
                for vid in arrived:
                    wait = self.last_wait.pop(vid, 0.0)
                    self.total_wait_sum += wait
                    self.total_arrivals += 1
            
        except Exception as e:
            pass  # Silently handle SUMO errors
    
    def sample_other_metrics(self):
        """
        Sample other metrics ONCE per RL step (after delta_time loop).
        
        These don't need to be sampled every second.
        """
        try:
            current_time = traci.simulation.getTime()
            
            # Get lanes
            try:
                all_lanes = traci.lane.getIDList()
            except:
                all_lanes = []
            
            # Total halting
            total_halt = 0
            for lane in all_lanes:
                try:
                    total_halt += traci.lane.getLastStepHaltingNumber(lane)
                except:
                    pass
            
            # Per-agent halting
            for agent_id in self.agent_ids:
                agent_halt = 0
                try:
                    controlled_lanes = traci.trafficlight.getControlledLanes(agent_id)
                    for lane in set(controlled_lanes):
                        try:
                            agent_halt += traci.lane.getLastStepHaltingNumber(lane)
                        except:
                            pass
                except:
                    pass
                self.per_agent_halts[agent_id].append(agent_halt)
            
            # Departed count
            try:
                self.total_departed = traci.simulation.getDepartedNumber()
            except:
                pass
            
            # Active vehicles
            try:
                active_count = len(traci.vehicle.getIDList())
            except:
                active_count = 0
            
            # Store metrics
            self.times.append(current_time)
            self.total_halts.append(total_halt)
            self.cumulative_arrivals.append(self.total_arrivals)
            self.cumulative_departed.append(self.total_departed)
            
            avg_wait = (self.total_wait_sum / self.total_arrivals) if self.total_arrivals > 0 else 0.0
            self.running_avg_wait.append(avg_wait)
            
            self.active_vehicles.append(active_count)
            
        except Exception as e:
            pass
    
    def track_action(self, agent_id: str, action: int):
        """Track phase switches."""
        if self.last_actions[agent_id] is not None:
            if self.last_actions[agent_id] != action:
                self.switch_counts[agent_id] += 1
        self.last_actions[agent_id] = action
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics."""
        return {
            'total_arrivals': self.total_arrivals,
            'total_departed': self.total_departed,
            'total_teleported': self.total_teleported,
            'avg_waiting_time': (self.total_wait_sum / self.total_arrivals) if self.total_arrivals > 0 else 0.0,
            'max_halting': max(self.total_halts) if self.total_halts else 0,
            'avg_halting': np.mean(self.total_halts) if self.total_halts else 0.0,
            'total_switches': sum(self.switch_counts.values()),
            'per_agent_switches': dict(self.switch_counts),
        }
    
    def save_csv(self, filename: str, results_dir: str = "metrics"):
        """Save time-series metrics to CSV."""
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'time', 'total_halts', 'cumulative_arrivals', 
                'cumulative_departed', 'running_avg_wait', 'active_veh'
            ])
            
            for i in range(len(self.times)):
                writer.writerow([
                    self.times[i],
                    self.total_halts[i],
                    self.cumulative_arrivals[i],
                    self.cumulative_departed[i],
                    self.running_avg_wait[i],
                    self.active_vehicles[i]
                ])
        
        print(f"  ✓ CSV saved: {filename}")
        return filepath
    
    def save_plots(self, prefix: str = "mappo", results_dir: str = "metrics"):
        """Generate visualization plots."""
        # 1. Halting vehicles
        plt.figure(figsize=(10, 3))
        plt.plot(self.times, self.total_halts, 'b-', linewidth=0.8, alpha=0.7)
        plt.title("MAPPO: Total Halting Vehicles", fontsize=12, fontweight='bold')
        plt.xlabel("Simulation time (s)")
        plt.ylabel("Halting vehicles")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filepath = os.path.join(results_dir, f"{prefix}_halting.png")
        plt.savefig(filepath, dpi=150)
        plt.close()
        
        # 2. Arrivals and wait time
        fig, ax1 = plt.subplots(figsize=(10, 3))
        
        color = 'tab:blue'
        ax1.plot(self.times, self.cumulative_arrivals, color=color, 
                linewidth=1.5, label='Arrivals')
        ax1.set_xlabel("Simulation time (s)")
        ax1.set_ylabel("Cumulative arrivals", color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.plot(self.times, self.running_avg_wait, color=color, 
                linestyle='--', linewidth=1.2, label='Avg wait')
        ax2.set_ylabel("Running avg wait (s)", color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.suptitle('MAPPO: Arrivals and Wait Time', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        fig.tight_layout()
        filepath = os.path.join(results_dir, f"{prefix}_arrivals_wait.png")
        fig.savefig(filepath, dpi=150)
        plt.close()
        
        # 3. Per-agent halting
        plt.figure(figsize=(10, 3))
        for agent_id in self.agent_ids:
            if self.per_agent_halts[agent_id]:
                plt.plot(self.times, self.per_agent_halts[agent_id], 
                        label=agent_id, linewidth=0.8, alpha=0.7)
        plt.legend()
        plt.title("MAPPO: Per-Junction Halting", fontsize=12, fontweight='bold')
        plt.xlabel("Simulation time (s)")
        plt.ylabel("Halting vehicles")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filepath = os.path.join(results_dir, f"{prefix}_per_agent.png")
        plt.savefig(filepath, dpi=150)
        plt.close()
        
        print(f"  ✓ Plots saved: {prefix}_*.png")


def evaluate_mappo(
    checkpoint_path: str,
    num_episodes: int = 3,
    use_gui: bool = False,
    config_path: str = 'configs/mappo_config.yaml',
    results_dir: str = "metrics",
    seed: int = 42
) -> Dict:
    """
    Evaluate MAPPO with CORRECT arrival tracking.
    
    Critical fix: Samples arrivals DURING delta_time loop.
    """
    results_dir = ensure_results_dir(results_dir)
    config = load_config(config_path)
    config['env_config']['use_gui'] = use_gui
    config['env_config']['sumo_seed'] = seed
    
    print(f"⚙️  Configuration:")
    print(f"  Seed: {seed}")
    print(f"  Delta time: {config['env_config'].get('delta_time', 5)}s")
    
    # Add all required config
    config['env_config']['agents'] = config['agents']
    config['env_config']['network_topology'] = config['network_topology']
    config['env_config']['detectors'] = config['detectors']
    config['env_config']['normalization'] = config['normalization']
    config['env_config']['reward_config'] = config['reward_config']
    config['env_config']['edge_connectivity'] = config.get('edge_connectivity', {})
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    tune.register_env("sumo_traffic", lambda cfg: SUMOTrafficEnv(cfg))
    # Register both Phase-1 model classes so checkpoints from either algo load.
    # The checkpoint's params.json names which one to use; we just need both
    # registered before PPO.from_checkpoint() instantiates the policy.
    ModelCatalog.register_custom_model("mappo_centralized", MAPPOModelCentralizedCritic)
    ModelCatalog.register_custom_model("ippo_decentralized", IPPOModelDecentralizedCritic)
    
    print(f"\n{'='*80}")
    print(f"MAPPO EVALUATION (CORRECT ARRIVAL TRACKING)")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes: {num_episodes}")
    print(f"{'='*80}\n")
    
    # Load checkpoint
    if os.path.isdir(checkpoint_path):
        checkpoint_dir = checkpoint_path
    else:
        checkpoint_dir = os.path.dirname(checkpoint_path) or checkpoint_path
    
    try:
        algo = PPO.from_checkpoint(checkpoint_dir)
        print("✓ Checkpoint loaded!\n")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return None

    # With enable_connectors=True (PPO default in RLlib 2.35), Algorithm.compute_single_action
    # only invokes ObsPreprocessorConnector — it does NOT apply MeanStdFilter. If the policy
    # was trained with observation_filter: "MeanStdFilter", we must apply it manually here,
    # otherwise the policy receives raw obs and collapses to a single argmax under explore=False.
    obs_filter = None
    local_worker = None
    for attr_chain in [
        lambda a: a.env_runner_group.local_env_runner,
        lambda a: a.workers.local_worker(),
        lambda a: a.workers.local_env_runner,
    ]:
        try:
            local_worker = attr_chain(algo)
            if local_worker is not None:
                break
        except Exception:
            continue

    if local_worker is None:
        print("⚠ Could not locate local worker on algo (tried env_runner_group / workers).")
    else:
        try:
            candidate = local_worker.filters.get("shared_policy")
            if candidate is not None and type(candidate).__name__ != "NoFilter":
                obs_filter = candidate
        except Exception as e:
            print(f"⚠ Could not retrieve obs filter from local worker: {e}")

    if obs_filter is not None:
        rs = getattr(obs_filter, "running_stats", None) or getattr(obs_filter, "rs", None)
        n = getattr(rs, "num_pushes", None) if rs is not None else None
        if n is None:
            n = getattr(rs, "n", None) if rs is not None else None
        print(f"✓ Obs filter: {type(obs_filter).__name__}, running_stats count={n}")
        if n in (None, 0):
            print("⚠ Filter has no accumulated stats — checkpoint may not have synced filter state.")
    else:
        print("ℹ No MeanStdFilter detected; observations passed through raw.")
    
    episode_rewards = []
    episode_stats = []
    
    # Run episodes
    for ep in range(num_episodes):
        print(f"\n{'─'*80}")
        print(f"Episode {ep + 1}/{num_episodes}")
        print(f"{'─'*80}")
        
        # Create base environment
        base_env = SUMOTrafficEnv(config['env_config'])
        
        # Create metrics collector
        metrics = MetricsCollector(base_env.agent_ids)
        
        # CRITICAL: Wrap environment to collect arrivals during delta loop
        env = EnvWrapperWithMetrics(base_env, metrics)
        
        # Reset
        obs, info = env.reset()
        
        episode_reward = {agent: 0.0 for agent in env.agent_ids}
        step_count = 0
        done = False
        
        # Run episode
        while not done:
            # Get actions
            actions = {}
            for agent_id in env.agent_ids:
                agent_obs = obs[agent_id]
                # Manually apply MeanStdFilter (skipped by connector-enabled compute_single_action)
                if obs_filter is not None:
                    agent_obs = obs_filter(agent_obs, update=False)
                action = algo.compute_single_action(
                    agent_obs,
                    policy_id="shared_policy",
                    explore=False
                )
                actions[agent_id] = action
            
            # Step (arrivals collected DURING step)
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            
            # Accumulate rewards
            for agent_id in env.agent_ids:
                episode_reward[agent_id] += rewards[agent_id]
            
            step_count += 1
            done = terminateds['__all__']
        
        # Get statistics
        stats = metrics.get_summary_stats()
        total_reward = sum(episode_reward.values())
        
        episode_rewards.append(total_reward)
        episode_stats.append({
            'episode': ep + 1,
            'total_reward': total_reward,
            'per_agent_reward': dict(episode_reward),
            'steps': step_count,
            **stats
        })
        
        # Print summary
        print(f"\n📊 Episode {ep + 1} Results:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {step_count}")
        print(f"  Arrivals: {stats['total_arrivals']} ← Should be ~1,200!")
        print(f"  Departed: {stats['total_departed']}")
        print(f"  Avg Wait: {stats['avg_waiting_time']:.2f}s")
        print(f"  Max Halt: {stats['max_halting']}")
        print(f"  Total Switches: {stats['total_switches']}")
        
        # Save files
        metrics.save_csv(f"mappo_ep{ep + 1}_metrics.csv", results_dir)
        metrics.save_plots(f"mappo_ep{ep + 1}", results_dir)
        
        env.close()
    
    # Aggregate results
    arrivals = [s['total_arrivals'] for s in episode_stats]
    wait_times = [s['avg_waiting_time'] for s in episode_stats]
    
    results = {
        'num_episodes': num_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_arrivals': np.mean(arrivals),
        'std_arrivals': np.std(arrivals),
        'mean_wait_time': np.mean(wait_times),
        'std_wait_time': np.std(wait_times),
        'episode_stats': episode_stats,
    }
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Mean Arrivals: {results['mean_arrivals']:.0f} ← Should be ~1,200!")
    print(f"Mean Wait: {results['mean_wait_time']:.2f}s")
    print(f"Mean Reward: {results['mean_reward']:.2f}")
    print(f"{'='*80}\n")
    
    algo.stop()
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate MAPPO (CORRECT arrival tracking)')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--config', type=str, default='configs/mappo_config_v2.yaml')
    parser.add_argument('--results-dir', type=str, default='metrics')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    results = evaluate_mappo(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        use_gui=args.gui,
        config_path=args.config,
        results_dir=args.results_dir,
        seed=args.seed
    )
    
    if results is None:
        sys.exit(1)
    
    print("\n✓ Evaluation completed!")


if __name__ == "__main__":
    main()