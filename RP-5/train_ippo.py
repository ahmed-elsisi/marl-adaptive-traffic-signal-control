"""
IPPO Training Script for Traffic Signal Control.

Mirrors train_mappo.py but with two differences:
- The policy uses the ippo_decentralized model (critic input = 70-dim local obs).
- No centralized_critic_postprocessing hook on the PolicySpec — RLlib's default
  PPO postprocessing handles GAE on local observations.

All hyperparameters are loaded from configs/ippo_config.yaml, which is a
verbatim copy of mappo_config_v2.yaml apart from the custom_model entry, so
the only varied factor between the two experiments is critic centralization.
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
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from marl_env.sumo_env import SUMOTrafficEnv
from models.ippo_model import IPPOModelDecentralizedCritic  # noqa: F401 (registers model)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_env(env_config: dict) -> SUMOTrafficEnv:
    return SUMOTrafficEnv(env_config)


def get_policy_mapping_fn(agent_ids: list):
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"
    return policy_mapping_fn


def build_ippo_config(config: dict) -> PPOConfig:
    """Build IPPO algorithm configuration with decentralized critic."""
    env_config = {
        "agents": config["agents"],
        "network_topology": config["network_topology"],
        "detectors": config["detectors"],
        "normalization": config["normalization"],
        "reward_config": config["reward_config"],
        "network_file": os.path.join(project_root, config["env_config"]["network_file"]),
        "route_file": os.path.join(project_root, config["env_config"]["route_file"]),
        "use_gui": config["env_config"].get("use_gui", False),
        "num_seconds": config["env_config"].get("num_seconds", 3600),
        "delta_time": config["env_config"].get("delta_time", 5),
        "yellow_time": config["env_config"].get("yellow_time", 3),
        "min_green": config["env_config"].get("min_green", 10),
        "max_green": config["env_config"].get("max_green", 50),
        "sumo_seed": config["env_config"].get("sumo_seed", 42),
        "enforce_min_green": config["env_config"].get("enforce_min_green", False),
        # All-red clearance between phase changes. Defaults match SUMOTrafficEnv
        # (enforce_min_red=True, min_red=1) so existing configs that don't
        # declare these keys keep their previous behavior.
        "enforce_min_red": config["env_config"].get("enforce_min_red", True),
        "min_red": config["env_config"].get("min_red", 1),
        "tl_program_id": config["env_config"].get("tl_program_id", "0"),
        "edge_connectivity": config.get("edge_connectivity", {}),
    }

    register_env("sumo_traffic", lambda cfg: create_env(cfg))

    temp_env = create_env(env_config)
    obs_space = temp_env.observation_space_dict["J1"]
    act_space = temp_env.action_space_dict["J1"]
    temp_env.close()

    # IPPO PolicySpec: NO postprocess_fn — default PPO postprocessing handles GAE.
    policies = {
        "shared_policy": PolicySpec(
            policy_class=None,
            observation_space=obs_space,
            action_space=act_space,
            config={},
        )
    }

    ippo_cfg = config["mappo_config"]  # section name kept for config-loader parity

    model_cfg_in = config["model_config"]["custom_model_config"]
    model_config = {
        "custom_model": "ippo_decentralized",
        "custom_model_config": {
            "num_agents": len(config["agents"]),
            "agent_ids": config["agents"],
            "actor_hiddens": model_cfg_in.get("actor_hiddens", [64, 64]),
            "actor_activation": model_cfg_in.get("actor_activation", "tanh"),
            "critic_hiddens": model_cfg_in.get("critic_hiddens", [256, 128]),
            "critic_activation": model_cfg_in.get("critic_activation", "relu"),
            "use_lstm": model_cfg_in.get("use_lstm", False),
            "lstm_cell_size": model_cfg_in.get("lstm_cell_size", 64),
            "use_orthogonal_init": model_cfg_in.get("use_orthogonal_init", True),
            "orthogonal_gain": model_cfg_in.get("orthogonal_gain", 0.01),
            "use_value_normalization": model_cfg_in.get("use_value_normalization", True),
        },
    }

    algo_config = (
        PPOConfig()
        .environment(
            env="sumo_traffic",
            env_config=env_config,
        )
        .framework(
            framework=ippo_cfg.get("framework", "torch"),
        )
        .training(
            lr=ippo_cfg.get("lr", 3e-4),
            gamma=ippo_cfg.get("gamma", 0.99),
            lambda_=ippo_cfg.get("lambda_", 0.95),
            num_sgd_iter=ippo_cfg.get("num_sgd_iter", 10),
            sgd_minibatch_size=ippo_cfg.get("sgd_minibatch_size", 4096),
            train_batch_size=ippo_cfg.get("train_batch_size", 32768),
            clip_param=ippo_cfg.get("clip_param", 0.2),
            vf_clip_param=ippo_cfg.get("vf_clip_param", 10.0),
            grad_clip=ippo_cfg.get("grad_clip", 0.5),
            entropy_coeff=ippo_cfg.get("entropy_coeff", 0.01),
            vf_loss_coeff=ippo_cfg.get("vf_loss_coeff", 1.0),
            vf_share_layers=False,
            use_critic=True,
            use_gae=True,
            model=model_config,
        )
        .rollouts(
            num_rollout_workers=ippo_cfg.get("num_rollout_workers", 4),
            num_envs_per_worker=ippo_cfg.get("num_envs_per_worker", 1),
            rollout_fragment_length=ippo_cfg.get("rollout_fragment_length", 200),
            batch_mode=ippo_cfg.get("batch_mode", "complete_episodes"),
            sample_timeout_s=ippo_cfg.get("sample_timeout_s", 120),
            observation_filter=ippo_cfg.get("observation_filter", "MeanStdFilter"),
        )
        .resources(
            num_gpus=ippo_cfg.get("num_gpus", 1),
            num_gpus_per_worker=ippo_cfg.get("num_gpus_per_worker", 0),
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=get_policy_mapping_fn(config["agents"]),
            policies_to_train=["shared_policy"],
        )
        .evaluation(
            evaluation_interval=ippo_cfg.get("evaluation_interval", 50),
            evaluation_duration=ippo_cfg.get("evaluation_duration", 10),
            evaluation_num_workers=0,
            evaluation_duration_unit=ippo_cfg.get("evaluation_duration_unit", "episodes"),
            evaluation_config={
                "explore": False,
                "batch_mode": "complete_episodes",
                "env_config": {**env_config, "use_gui": False},
            },
        )
    )

    return algo_config


def train_ippo(
    config_path: str,
    num_iterations: int = 1000,
    checkpoint_freq: int = 50,
    resume_path: str = None,
):
    config = load_config(config_path)

    if not ray.is_initialized():
        ray.init(
            num_cpus=config["mappo_config"].get("num_rollout_workers", 4) + 1,
            num_gpus=config["mappo_config"].get("num_gpus", 1),
            ignore_reinit_error=True,
        )

    algo_config = build_ippo_config(config)

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    training_config = config.get("training", {})

    run_config = air.RunConfig(
        name="ippo_traffic_control",
        storage_path=str(results_dir),
        stop={
            "training_iteration": num_iterations,
        },
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=checkpoint_freq,
            checkpoint_at_end=True,
            num_to_keep=training_config.get("keep_checkpoints_num", 5),
        ),
        verbose=1,
    )

    if resume_path:
        checkpoint = Path(resume_path)
        if checkpoint.name.startswith("checkpoint_"):
            experiment_dir = checkpoint.parent.parent
        else:
            experiment_dir = checkpoint.parent
        print(f"Restoring experiment from: {experiment_dir}")
        tuner = tune.Tuner.restore(
            path=str(experiment_dir),
            trainable="PPO",
            resume_unfinished=True,
            restart_errored=False,
            param_space=algo_config.to_dict(),
        )
    else:
        tuner = tune.Tuner(
            "PPO",
            param_space=algo_config.to_dict(),
            run_config=run_config,
        )

    results = tuner.fit()

    best_result = None
    reward_metric_names = [
        "episode_reward_mean",
        "sampler_results/episode_reward_mean",
        "env_runners/episode_reward_mean",
        "episode_return_mean",
    ]

    for metric_name in reward_metric_names:
        try:
            best_result = results.get_best_result(metric=metric_name, mode="max")
            print(f"\nUsing metric: {metric_name}")
            break
        except Exception:
            continue

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)

    if best_result:
        if hasattr(best_result, "checkpoint") and best_result.checkpoint:
            print(f"Best checkpoint: {best_result.checkpoint}")
        elif hasattr(best_result, "path"):
            print(f"Results path: {best_result.path}")

        if hasattr(best_result, "metrics"):
            for metric_name in reward_metric_names:
                if metric_name in best_result.metrics:
                    print(f"Best reward: {best_result.metrics[metric_name]:.2f}")
                    break
            else:
                available = [
                    k for k in best_result.metrics.keys()
                    if "reward" in k.lower() or "return" in k.lower()
                ]
                if available:
                    print(f"Available reward metrics: {available}")
                    print(f"  {available[0]}: {best_result.metrics[available[0]]:.2f}")
                else:
                    print("Note: Standard reward metrics not found")
                    print(f"Available metrics: {list(best_result.metrics.keys())[:5]}...")
    else:
        print("Could not retrieve best result")
        print("Check results/ippo_traffic_control for checkpoints")

    print("=" * 80)
    return best_result


def main():
    parser = argparse.ArgumentParser(
        description="Train IPPO for traffic signal control with decentralized critic"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ippo_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50,
        help="Checkpoint frequency",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Force standard TraCI — libsumo fails to start on this setup. Matches
    # train_mappo.py, which intentionally never sets LIBSUMO_AS_TRACI.
    os.environ.pop("LIBSUMO_AS_TRACI", None)

    print("=" * 80)
    print("IPPO Training for Traffic Signal Control")
    print("WITH DECENTRALIZED CRITIC")
    print("=" * 80)
    print(f"Configuration: {args.config}")
    print(f"Iterations: {args.iterations}")
    print(f"Checkpoint frequency: {args.checkpoint_freq}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)

    try:
        best_result = train_ippo(
            config_path=args.config,
            num_iterations=args.iterations,
            checkpoint_freq=args.checkpoint_freq,
            resume_path=args.resume,
        )

        print("\nTraining successful!")
        if best_result and hasattr(best_result, "checkpoint"):
            print(f"Best checkpoint saved at: {best_result.checkpoint}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Checkpoints have been saved.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
