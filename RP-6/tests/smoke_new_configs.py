"""
Lightweight integration smoke for the 5 new Phase-2 configs.

For each YAML, this script:
  1. Loads the file
  2. Builds the RLlib PPOConfig (= probes env, registers model, validates schema)
  3. Confirms the env's shared_reward_weight matches the YAML
  4. Tears the probe env down without invoking Ray's training loop

This catches YAML schema errors, missing model registrations, env-config
plumbing bugs, and reward-weight passthrough issues — at a fraction of the
cost of a 1-iter Ray training smoke. The full training-loop integration
is already verified for MAPPO via the Week-5 smoke run (c99e2) and the
IPPO entry point gets its own full 1-iter run in smoke_ippo_train.py.

Run from RP-6/:
    python tests/smoke_new_configs.py
"""

from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
_RP6 = _HERE.parent
if str(_RP6) not in sys.path:
    sys.path.insert(0, str(_RP6))

# Import the build functions from the actual training scripts.
import train_mappo_harvest as mappo_entry
import train_ippo_harvest as ippo_entry


CASES = [
    ("MAPPO individual", "configs/harvest_mappo_individual.yaml", 0.0, mappo_entry, "mappo_cnn_centralized"),
    ("MAPPO mixed",      "configs/harvest_mappo_mixed.yaml",      0.5, mappo_entry, "mappo_cnn_centralized"),
    ("MAPPO team",       "configs/harvest_mappo_team.yaml",       1.0, mappo_entry, "mappo_cnn_centralized"),
    ("IPPO  individual", "configs/harvest_ippo_individual.yaml",  0.0, ippo_entry,  "ippo_cnn_decentralized"),
    ("IPPO  mixed",      "configs/harvest_ippo_mixed.yaml",       0.5, ippo_entry,  "ippo_cnn_decentralized"),
    ("IPPO  team",       "configs/harvest_ippo_team.yaml",        1.0, ippo_entry,  "ippo_cnn_decentralized"),
]


def smoke_one(name: str, config_path: str, expected_w: float, entry, expected_model: str):
    print(f"\n--- {name}  ({config_path}) ---")
    cfg = entry.load_config(config_path)

    # 1. shared_reward_weight plumbed from YAML
    actual_w = cfg["env_config"]["shared_reward_weight"]
    assert actual_w == expected_w, f"  shared_reward_weight mismatch: {actual_w} != {expected_w}"
    print(f"  shared_reward_weight = {actual_w}  ok")

    # 2. custom_model name correct
    actual_model = cfg["model_config"]["custom_model"]
    assert actual_model == expected_model, f"  custom_model mismatch: {actual_model} != {expected_model}"
    print(f"  custom_model = {actual_model}  ok")

    # 3. Builds a PPOConfig (= probes env, registers model)
    algo_cfg = entry.build_algo_config(cfg)
    assert algo_cfg is not None
    print(f"  PPOConfig built  ok")

    # 4. Sanity: env_config in the built config carries the same weight
    built_env_cfg = algo_cfg.to_dict()["env_config"]
    assert built_env_cfg["shared_reward_weight"] == expected_w
    print(f"  env_config.shared_reward_weight passthrough  ok")


def main():
    print("=" * 70)
    print("SMOKE: 6 Phase-2 configs (lightweight, no training)")
    print("=" * 70)
    for case in CASES:
        smoke_one(*case)
    print("\n" + "=" * 70)
    print("ALL CONFIG SMOKES PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
