"""
MAPPO training entry point for HarvestEnv (Phase-2).

Mirrors RP-5/train_mappo.py / RP-5/train_ippo.py structure so the workflow
is consistent across phases. Differences from Phase-1:

  - Env: HarvestEnv (RLlib MultiAgentEnv) instead of SUMOTrafficEnv.
  - Model: 'mappo_cnn_centralized' (CNN actor + CNN critic on full grid)
           instead of 'mappo_centralized' (MLP).
  - Postprocessing: harvest_centralized_critic_postprocessing lifts
                    info['global_state'] into the SampleBatch (vs the
                    SUMO version which concats other agents' obs).
  - PolicySpec includes the postprocess_fn directly (RLlib 2.35 pattern).

PPO hyperparameters are loaded from the YAML config (default
configs/harvest_mappo_team.yaml — the cooperative end of the sweep, used
as the Week-5 smoke condition).

Run:
    python train_mappo_harvest.py --config configs/harvest_mappo_team.yaml \\
                                   --iterations 200
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

# Make RP-6/ importable when invoked as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from marl_env.harvest_env import HarvestEnv
from models.mappo_cnn_model import (   # noqa: F401  (registers model)
    MAPPOCNNModelCentralizedCritic,
    harvest_centralized_critic_postprocessing,
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_env(env_config: dict) -> HarvestEnv:
    return HarvestEnv(env_config)


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "shared_policy"


def build_algo_config(config: dict) -> PPOConfig:
    """Translate the YAML config into an RLlib PPOConfig."""
    yaml_env = config.get("env_config", {})
    env_config = {
        "grid_height":           yaml_env.get("grid_height", 8),
        "grid_width":            yaml_env.get("grid_width", 12),
        "num_agents":            yaml_env.get("num_agents", 4),
        "episode_length":        yaml_env.get("episode_length", 1000),
        "obs_window":            yaml_env.get("obs_window", 15),
        "initial_apple_density": yaml_env.get("initial_apple_density", 0.3),
        "apple_regrowth_base":   yaml_env.get("apple_regrowth_base", 0.01),
        "apple_regrowth_radius": yaml_env.get("apple_regrowth_radius", 2),
        "shared_reward_weight":  yaml_env.get("shared_reward_weight", 0.0),
        "seed":                  yaml_env.get("seed", 42),
    }

    register_env("harvest", lambda cfg: create_env(cfg))

    # Probe a temporary env for spaces — same pattern as RP-5/train_ippo.py.
    probe = create_env(env_config)
    obs_space = probe.observation_space_dict[probe.agent_ids[0]]
    act_space = probe.action_space_dict[probe.agent_ids[0]]
    # No probe.close() — HarvestEnv has no resources to release.

    # PolicySpec with the global-state postprocessing hook attached.
    policies = {
        "shared_policy": PolicySpec(
            policy_class=None,
            observation_space=obs_space,
            action_space=act_space,
            config={"postprocess_fn": harvest_centralized_critic_postprocessing},
        )
    }

    mappo_cfg = config["mappo_config"]
    model_cfg_in = config["model_config"]["custom_model_config"]

    model_config = {
        "custom_model": "mappo_cnn_centralized",
        "custom_model_config": {
            "global_state_height": model_cfg_in.get("global_state_height", env_config["grid_height"]),
            "global_state_width":  model_cfg_in.get("global_state_width",  env_config["grid_width"]),
            "actor_conv_specs":   [tuple(s) for s in model_cfg_in.get("actor_conv_specs",  [(16, 3), (32, 3)])],
            "actor_hiddens":      model_cfg_in.get("actor_hiddens",      [128, 64]),
            "actor_activation":   model_cfg_in.get("actor_activation",   "tanh"),
            "critic_conv_specs":  [tuple(s) for s in model_cfg_in.get("critic_conv_specs", [(32, 3), (64, 3)])],
            "critic_hiddens":     model_cfg_in.get("critic_hiddens",     [512, 256, 128]),
            "critic_activation":  model_cfg_in.get("critic_activation",  "relu"),
            "use_orthogonal_init": model_cfg_in.get("use_orthogonal_init", True),
            "orthogonal_gain":    model_cfg_in.get("orthogonal_gain", 0.01),
            "use_value_normalization": model_cfg_in.get("use_value_normalization", True),
        },
    }

    algo_config = (
        PPOConfig()
        .environment(env="harvest", env_config=env_config)
        .framework(framework=mappo_cfg.get("framework", "torch"))
        .training(
            lr=mappo_cfg.get("lr", 5e-4),
            gamma=mappo_cfg.get("gamma", 0.99),
            lambda_=mappo_cfg.get("lambda_", 0.95),
            num_sgd_iter=mappo_cfg.get("num_sgd_iter", 10),
            sgd_minibatch_size=mappo_cfg.get("sgd_minibatch_size", 32768),
            train_batch_size=mappo_cfg.get("train_batch_size", 32768),
            clip_param=mappo_cfg.get("clip_param", 0.2),
            vf_clip_param=mappo_cfg.get("vf_clip_param", 10.0),
            grad_clip=mappo_cfg.get("grad_clip", 1.0),
            entropy_coeff=mappo_cfg.get("entropy_coeff", 0.02),
            vf_loss_coeff=mappo_cfg.get("vf_loss_coeff", 1.0),
            vf_share_layers=False,
            use_critic=True,
            use_gae=True,
            model=model_config,
        )
        .rollouts(
            num_rollout_workers=mappo_cfg.get("num_rollout_workers", 3),
            num_envs_per_worker=mappo_cfg.get("num_envs_per_worker", 1),
            rollout_fragment_length=mappo_cfg.get("rollout_fragment_length", 200),
            batch_mode=mappo_cfg.get("batch_mode", "complete_episodes"),
            sample_timeout_s=mappo_cfg.get("sample_timeout_s", 120),
            observation_filter=mappo_cfg.get("observation_filter", "MeanStdFilter"),
        )
        .resources(
            num_gpus=mappo_cfg.get("num_gpus", 1),
            num_gpus_per_worker=mappo_cfg.get("num_gpus_per_worker", 0),
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"],
        )
        .evaluation(
            evaluation_interval=mappo_cfg.get("evaluation_interval", 10),
            evaluation_duration=mappo_cfg.get("evaluation_duration", 5),
            evaluation_num_workers=0,
            evaluation_duration_unit=mappo_cfg.get("evaluation_duration_unit", "episodes"),
            evaluation_config={
                "explore": False,
                "batch_mode": "complete_episodes",
                "env_config": dict(env_config),
            },
        )
    )
    return algo_config


def train(config_path: str, num_iterations: int, checkpoint_freq: int, resume: str = None):
    config = load_config(config_path)

    if not ray.is_initialized():
        ray.init(
            num_cpus=config["mappo_config"].get("num_rollout_workers", 3) + 1,
            num_gpus=config["mappo_config"].get("num_gpus", 1),
            ignore_reinit_error=True,
        )

    algo_config = build_algo_config(config)

    results_dir = _PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    training_cfg = config.get("training", {})
    run_config = air.RunConfig(
        name="mappo_harvest",
        storage_path=str(results_dir),
        stop={"training_iteration": num_iterations},
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=checkpoint_freq,
            checkpoint_at_end=True,
            num_to_keep=training_cfg.get("keep_checkpoints_num", 5),
        ),
        verbose=1,
    )

    if resume:
        ckpt = Path(resume)
        experiment_dir = ckpt.parent.parent if ckpt.name.startswith("checkpoint_") else ckpt.parent
        print(f"Restoring experiment from: {experiment_dir}")
        tuner = tune.Tuner.restore(
            path=str(experiment_dir),
            trainable="PPO",
            resume_unfinished=True,
            restart_errored=False,
            param_space=algo_config.to_dict(),
        )
    else:
        tuner = tune.Tuner("PPO", param_space=algo_config.to_dict(), run_config=run_config)

    return tuner.fit()


def main():
    parser = argparse.ArgumentParser(description="Train MAPPO on HarvestEnv")
    parser.add_argument("--config", type=str, default="configs/harvest_mappo_team.yaml")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--checkpoint-freq", type=int, default=25)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Force standard TraCI off — HarvestEnv doesn't use TraCI but mirroring
    # train_ippo.py's defensive pop in case the env var leaks in.
    os.environ.pop("LIBSUMO_AS_TRACI", None)

    print("=" * 80)
    print("MAPPO Training on HarvestEnv (CENTRALIZED CRITIC, CNN ENCODER)")
    print("=" * 80)
    print(f"Config:           {args.config}")
    print(f"Iterations:       {args.iterations}")
    print(f"Checkpoint freq:  {args.checkpoint_freq}")
    print(f"Seed:             {args.seed}")
    print("=" * 80)

    try:
        train(args.config, args.iterations, args.checkpoint_freq, args.resume)
        print("\nTraining complete.")
    except KeyboardInterrupt:
        print("\nInterrupted; checkpoints preserved.")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
