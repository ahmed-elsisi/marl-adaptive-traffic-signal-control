"""
MAPPO Training Script for Traffic Signal Control

Trains MAPPO agents using Ray RLlib 2.35.0 with:
- CENTRALIZED CRITIC with global state (184-dim)
- Decentralized actors
- Research-backed hyperparameters
- TensorBoard logging
- Checkpoint management
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import numpy as np
import torch

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from marl_env.sumo_env import SUMOTrafficEnv
from models.mappo_model import MAPPOModelCentralizedCritic, centralized_critic_postprocessing


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_env(env_config: dict) -> SUMOTrafficEnv:
    """Environment creator function for RLlib."""
    return SUMOTrafficEnv(env_config)


def get_policy_mapping_fn(agent_ids: list):
    """
    Create policy mapping function.
    
    All agents share the same policy (parameter sharing).
    """
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"
    
    return policy_mapping_fn


def build_mappo_config(config: dict) -> PPOConfig:
    """
    Build MAPPO algorithm configuration with CENTRALIZED CRITIC.
    
    Uses PPO as base with multi-agent settings for MAPPO.
    """
    env_config = {
        'agents': config['agents'],
        'network_topology': config['network_topology'],
        'detectors': config['detectors'],
        'normalization': config['normalization'],
        'reward_config': config['reward_config'],
        'network_file': os.path.join(project_root, config['env_config']['network_file']),
        'route_file': os.path.join(project_root, config['env_config']['route_file']),
        'use_gui': config['env_config'].get('use_gui', False),
        'num_seconds': config['env_config'].get('num_seconds', 3600),
        'delta_time': config['env_config'].get('delta_time', 5),
        'yellow_time': config['env_config'].get('yellow_time', 3),
        'min_green': config['env_config'].get('min_green', 10),
        'max_green': config['env_config'].get('max_green', 50),
        'sumo_seed': config['env_config'].get('sumo_seed', 42),
        'enforce_min_green': config['env_config'].get('enforce_min_green', False),  # NEW!
    }
    
    # Register environment
    register_env("sumo_traffic", lambda cfg: create_env(cfg))
    
    # Get sample environment for space definitions
    temp_env = create_env(env_config)
    obs_space = temp_env.observation_space_dict['J1']
    act_space = temp_env.action_space_dict['J1']
    temp_env.close()
    
    # Define policies with CENTRALIZED CRITIC POSTPROCESSING
    policies = {
        "shared_policy": PolicySpec(
            policy_class=None,  # Use default
            observation_space=obs_space,
            action_space=act_space,
            config={
                'postprocess_fn': centralized_critic_postprocessing,  # CRITICAL!
            },
        )
    }
    
    # MAPPO configuration
    mappo_cfg = config['mappo_config']
    
    # Model configuration with num_agents
    model_config = {
        'custom_model': 'mappo_centralized',  # CHANGED!
        'custom_model_config': {
            'num_agents': 4,  # CRITICAL for global state construction
            'actor_hiddens': config['model_config']['custom_model_config'].get('actor_hiddens', [64, 64]),
            'critic_hiddens': config['model_config']['custom_model_config'].get('critic_hiddens', [256, 128]),
            'use_lstm': config['model_config']['custom_model_config'].get('use_lstm', False),
            'lstm_cell_size': config['model_config']['custom_model_config'].get('lstm_cell_size', 64),
            'critic_activation': config['model_config']['custom_model_config'].get('critic_activation', 'relu'),
            'use_orthogonal_init': config['model_config']['custom_model_config'].get('use_orthogonal_init', True),
            'orthogonal_gain': config['model_config']['custom_model_config'].get('orthogonal_gain', 0.01),
            'use_value_normalization': config['model_config']['custom_model_config'].get('use_value_normalization', True),
        }
    }
    
    # Build PPO config
    algo_config = (
        PPOConfig()
        .environment(
            env="sumo_traffic",
            env_config=env_config,
        )
        .framework(
            framework=mappo_cfg.get('framework', 'torch')
        )
        .training(
            # Learning rate
            lr=mappo_cfg.get('lr', 3e-4),
            
            # PPO-specific (CRITICAL for MAPPO stability)
            gamma=mappo_cfg.get('gamma', 0.99),
            lambda_=mappo_cfg.get('lambda_', 0.95),
            
            # CRITICAL: 5-10 epochs
            num_sgd_iter=mappo_cfg.get('num_sgd_iter', 10),
            
            # Batch sizes
            sgd_minibatch_size=mappo_cfg.get('sgd_minibatch_size', 4096),
            train_batch_size=mappo_cfg.get('train_batch_size', 32768),
            
            # Clipping
            clip_param=mappo_cfg.get('clip_param', 0.2),
            vf_clip_param=mappo_cfg.get('vf_clip_param', 10.0),
            grad_clip=mappo_cfg.get('grad_clip', 0.5),
            
            # Regularization
            entropy_coeff=mappo_cfg.get('entropy_coeff', 0.01),
            vf_loss_coeff=mappo_cfg.get('vf_loss_coeff', 1.0),
            
            # CRITICAL: Separate actor and critic
            vf_share_layers=False,
            use_critic=True,
            use_gae=True,
            
            # Model configuration
            model=model_config,
        )
        .rollouts(
            num_rollout_workers=mappo_cfg.get('num_rollout_workers', 4),
            num_envs_per_worker=mappo_cfg.get('num_envs_per_worker', 1),
            rollout_fragment_length=mappo_cfg.get('rollout_fragment_length', 200),
            batch_mode=mappo_cfg.get('batch_mode', 'complete_episodes'),
        )
        .resources(
            num_gpus=mappo_cfg.get('num_gpus', 1),
            num_gpus_per_worker=mappo_cfg.get('num_gpus_per_worker', 0),
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=get_policy_mapping_fn(config['agents']),
            policies_to_train=["shared_policy"],
        )
        .evaluation(
            evaluation_interval=mappo_cfg.get('evaluation_interval', 50),
            evaluation_duration=mappo_cfg.get('evaluation_duration', 10),
            evaluation_num_workers=0,
            evaluation_duration_unit=mappo_cfg.get('evaluation_duration_unit', 'episodes'),
            evaluation_config={
                "explore": False,
                "batch_mode": "complete_episodes",
                "env_config": {**env_config, "use_gui": False},
            },
        )
    )
    
    return algo_config


def train_mappo(
    config_path: str,
    num_iterations: int = 2000,
    checkpoint_freq: int = 100,
    resume_path: str = None
):
    """
    Train MAPPO agents with centralized critic.
    
    Args:
        config_path: Path to configuration YAML
        num_iterations: Number of training iterations
        checkpoint_freq: Checkpoint frequency
        resume_path: Path to checkpoint for resuming
    """
    # Load configuration
    config = load_config(config_path)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            num_cpus=config['mappo_config'].get('num_rollout_workers', 4) + 1,
            num_gpus=config['mappo_config'].get('num_gpus', 1),
            ignore_reinit_error=True,
        )
    
    # Build algorithm configuration
    algo_config = build_mappo_config(config)
    
    # Create results directory
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Training configuration
    training_config = config.get('training', {})
    
    # Run training
    tuner = tune.Tuner(
        "PPO",
        param_space=algo_config.to_dict(),
        run_config=air.RunConfig(
            name="mappo_traffic_control",
            storage_path=str(results_dir),
            stop={
                "training_iteration": num_iterations,
            },
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=checkpoint_freq,
                checkpoint_at_end=True,
                num_to_keep=training_config.get('keep_checkpoints_num', 5),
            ),
            verbose=1,
        ),
    )
    
    # Execute training
    results = tuner.fit()
    
    # Get best result - handle different RLlib metric names
    best_result = None
    reward_metric_names = [
        "episode_reward_mean",
        "sampler_results/episode_reward_mean", 
        "env_runners/episode_reward_mean",
        "episode_return_mean"
    ]
    
    for metric_name in reward_metric_names:
        try:
            best_result = results.get_best_result(metric=metric_name, mode="max")
            print(f"\n✓ Using metric: {metric_name}")
            break
        except Exception:
            continue
    
    # If still no result, try to get any result
    if best_result is None:
        try:
            all_results = results.get_dataframe()
            if not all_results.empty:
                print("\n⚠ Could not find standard metrics")
        except:
            pass
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    
    if best_result:
        # Print checkpoint if available
        if hasattr(best_result, 'checkpoint') and best_result.checkpoint:
            print(f"Best checkpoint: {best_result.checkpoint}")
        elif hasattr(best_result, 'path'):
            print(f"Results path: {best_result.path}")
        
        # Try to print reward
        if hasattr(best_result, 'metrics'):
            for metric_name in reward_metric_names:
                if metric_name in best_result.metrics:
                    print(f"Best reward: {best_result.metrics[metric_name]:.2f}")
                    break
            else:
                # Show available metrics
                available = [k for k in best_result.metrics.keys() if 'reward' in k.lower() or 'return' in k.lower()]
                if available:
                    print(f"Available reward metrics: {available}")
                    print(f"  {available[0]}: {best_result.metrics[available[0]]:.2f}")
                else:
                    print("Note: Standard reward metrics not found")
                    print(f"Available metrics: {list(best_result.metrics.keys())[:5]}...")
    else:
        print("⚠ Could not retrieve best result")
        print("Check results/mappo_traffic_control for checkpoints")
    
    print("="*80)
    
    return best_result


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train MAPPO for traffic signal control with centralized critic"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mappo_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50,
        help="Checkpoint frequency"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set environment variable for libsumo (8x speedup)
    os.environ['LIBSUMO_AS_TRACI'] = '1'
    
    # Train
    print("="*80)
    print("MAPPO Training for Traffic Signal Control")
    print("WITH CENTRALIZED CRITIC")
    print("="*80)
    print(f"Configuration: {args.config}")
    print(f"Iterations: {args.iterations}")
    print(f"Checkpoint frequency: {args.checkpoint_freq}")
    print(f"Random seed: {args.seed}")
    print("="*80)
    
    try:
        best_result = train_mappo(
            config_path=args.config,
            num_iterations=args.iterations,
            checkpoint_freq=args.checkpoint_freq,
            resume_path=args.resume
        )
        
        print("\nTraining successful!")
        if best_result and hasattr(best_result, 'checkpoint'):
            print(f"Best checkpoint saved at: {best_result.checkpoint}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Checkpoints have been saved.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Shutdown Ray
        ray.shutdown()


if __name__ == "__main__":
    main()