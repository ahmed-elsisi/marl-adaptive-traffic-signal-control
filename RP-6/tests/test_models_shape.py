"""
Week-5 CPU shape-check for HarvestEnv + CNN models + postprocessing hook.

No training, no GPU. Just exercises:
  1. Both model classes import and register cleanly.
  2. Forward pass through actor produces (batch, num_actions) logits.
  3. value_function() produces (batch,) values for IPPO (decentralized) and
     for MAPPO when global_state is/isn't injected.
  4. harvest_centralized_critic_postprocessing lifts info['global_state']
     into a SampleBatch column with the right shape.

Run from RP-6/ with:
    python tests/test_models_shape.py
"""

from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
_RP6 = _HERE.parent
if str(_RP6) not in sys.path:
    sys.path.insert(0, str(_RP6))

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete

from marl_env.harvest_env import HarvestEnv, NUM_ACTIONS
from models.ippo_cnn_model import IPPOCNNModelDecentralizedCritic
from models.mappo_cnn_model import (
    MAPPOCNNModelCentralizedCritic,
    harvest_centralized_critic_postprocessing,
)
from ray.rllib.policy.sample_batch import SampleBatch


# ── Test 1: env produces obs and global_state of expected shape ───────────────


def test_env_shapes():
    print("\n" + "=" * 70)
    print("TEST 1: HarvestEnv obs + global_state shapes")
    print("=" * 70)

    env = HarvestEnv({"shared_reward_weight": 1.0, "seed": 42})
    obs, infos = env.reset()
    for a in env.agent_ids:
        assert obs[a].shape == (env.obs_window, env.obs_window, 3), \
            f"obs[{a}] shape: {obs[a].shape}"
        assert obs[a].dtype == np.uint8
        gs = infos[a]["global_state"]
        assert gs.shape == (env.grid_height, env.grid_width, 3), \
            f"info[{a}].global_state shape: {gs.shape}"
        assert gs.dtype == np.uint8
    print(f"  obs:          (15, 15, 3) uint8  ok ({len(env.agent_ids)} agents)")
    print(f"  global_state: ({env.grid_height}, {env.grid_width}, 3) uint8  ok")
    print("TEST 1 PASSED")


# ── Test 2: IPPO model forward + value ────────────────────────────────────────


def test_ippo_model_forward():
    print("\n" + "=" * 70)
    print("TEST 2: IPPOCNNModelDecentralizedCritic forward + value")
    print("=" * 70)

    obs_space = Box(low=0, high=255, shape=(15, 15, 3), dtype=np.uint8)
    act_space = Discrete(NUM_ACTIONS)
    model = IPPOCNNModelDecentralizedCritic(
        obs_space=obs_space,
        action_space=act_space,
        num_outputs=NUM_ACTIONS,
        model_config={"custom_model_config": {}},
        name="ippo_test",
    )
    model.eval()
    batch = 7
    obs_torch = torch.from_numpy(
        np.random.randint(0, 256, size=(batch, 15, 15, 3), dtype=np.uint8)
    )
    logits, _ = model({"obs": obs_torch}, [], None)
    values = model.value_function()
    assert logits.shape == (batch, NUM_ACTIONS), f"logits shape: {logits.shape}"
    assert values.shape == (batch,), f"values shape: {values.shape}"
    print(f"  logits: {tuple(logits.shape)}  ok")
    print(f"  values: {tuple(values.shape)}  ok")
    print("TEST 2 PASSED")


# ── Test 3: MAPPO model forward + value (with and without global_state) ───────


def test_mappo_model_forward_with_global_state():
    print("\n" + "=" * 70)
    print("TEST 3: MAPPOCNNModelCentralizedCritic forward + value (with global_state)")
    print("=" * 70)

    obs_space = Box(low=0, high=255, shape=(15, 15, 3), dtype=np.uint8)
    act_space = Discrete(NUM_ACTIONS)
    model = MAPPOCNNModelCentralizedCritic(
        obs_space=obs_space,
        action_space=act_space,
        num_outputs=NUM_ACTIONS,
        model_config={
            "custom_model_config": {
                "global_state_height": 8,
                "global_state_width": 12,
            }
        },
        name="mappo_test",
    )
    model.eval()
    batch = 5
    obs_torch = torch.from_numpy(
        np.random.randint(0, 256, size=(batch, 15, 15, 3), dtype=np.uint8)
    )
    gs_torch = torch.from_numpy(
        np.random.randint(0, 256, size=(batch, 8, 12, 3), dtype=np.uint8)
    )
    logits, _ = model({"obs": obs_torch, "global_state": gs_torch}, [], None)
    values = model.value_function()
    assert logits.shape == (batch, NUM_ACTIONS), f"logits shape: {logits.shape}"
    assert values.shape == (batch,), f"values shape: {values.shape}"
    print(f"  logits: {tuple(logits.shape)}  ok")
    print(f"  values: {tuple(values.shape)}  ok (centralized critic on (8,12,3) global state)")

    # Also test fallback when global_state is missing — should not error.
    logits2, _ = model({"obs": obs_torch}, [], None)
    values2 = model.value_function()
    assert values2.shape == (batch,)
    print(f"  fallback (no global_state): values shape {tuple(values2.shape)}  ok")
    print("TEST 3 PASSED")


# ── Test 4: postprocessing hook lifts global_state from infos ─────────────────


def test_postprocessing_hook():
    print("\n" + "=" * 70)
    print("TEST 4: harvest_centralized_critic_postprocessing lifts info['global_state']")
    print("=" * 70)

    batch = 4

    class _StubPolicy:
        def __init__(self):
            self.config = {"env_config": {"grid_height": 8, "grid_width": 12}}

    sample_batch = SampleBatch({
        SampleBatch.OBS: np.zeros((batch, 15, 15, 3), dtype=np.uint8),
        SampleBatch.INFOS: np.array([
            {"global_state": np.full((8, 12, 3), 42, dtype=np.uint8)}
            for _ in range(batch)
        ], dtype=object),
    })
    out = harvest_centralized_critic_postprocessing(_StubPolicy(), sample_batch)
    assert "global_state" in out, "postprocessing did not add global_state column"
    gs = out["global_state"]
    assert gs.shape == (batch, 8, 12, 3), f"global_state column shape: {gs.shape}"
    assert (gs == 42).all(), "global_state values not preserved from infos"
    print(f"  injected column shape: {tuple(gs.shape)}  ok")
    print(f"  values preserved (==42): ok")

    # Eval-time fallback when infos are missing
    sample_batch_no_infos = SampleBatch({
        SampleBatch.OBS: np.zeros((batch, 15, 15, 3), dtype=np.uint8),
    })
    out2 = harvest_centralized_critic_postprocessing(_StubPolicy(), sample_batch_no_infos)
    assert out2["global_state"].shape == (batch, 8, 12, 3), \
        f"fallback shape: {out2['global_state'].shape}"
    assert (out2["global_state"] == 0).all(), "fallback should be zeros"
    print(f"  eval-time fallback (no infos): zeros of correct shape  ok")
    print("TEST 4 PASSED")


# ── Test 5: end-to-end env step + postprocessing flow ────────────────────────


def test_env_to_postprocessing_pipeline():
    print("\n" + "=" * 70)
    print("TEST 5: end-to-end env.step -> infos -> postprocessing -> model")
    print("=" * 70)

    env = HarvestEnv({"shared_reward_weight": 1.0, "seed": 7})
    obs, infos = env.reset()
    actions = {a: 4 for a in env.agent_ids}  # all stay
    obs2, rewards, terms, truncs, infos2 = env.step(actions)

    # Single-step batch from agent A0
    a = env.agent_ids[0]
    batch = SampleBatch({
        SampleBatch.OBS: np.expand_dims(obs2[a], 0),
        SampleBatch.INFOS: np.array([infos2[a]], dtype=object),
    })

    class _StubPolicy:
        def __init__(self):
            self.config = {
                "env_config": {
                    "grid_height": env.grid_height,
                    "grid_width": env.grid_width,
                }
            }

    out = harvest_centralized_critic_postprocessing(_StubPolicy(), batch)
    gs = out["global_state"]
    assert gs.shape == (1, env.grid_height, env.grid_width, 3)

    obs_space = env.observation_space_dict[a]
    act_space = env.action_space_dict[a]
    model = MAPPOCNNModelCentralizedCritic(
        obs_space=obs_space,
        action_space=act_space,
        num_outputs=NUM_ACTIONS,
        model_config={
            "custom_model_config": {
                "global_state_height": env.grid_height,
                "global_state_width": env.grid_width,
            }
        },
        name="mappo_pipeline",
    )
    model.eval()
    obs_torch = torch.from_numpy(out[SampleBatch.OBS])
    gs_torch = torch.from_numpy(out["global_state"])
    logits, _ = model({"obs": obs_torch, "global_state": gs_torch}, [], None)
    values = model.value_function()
    assert logits.shape == (1, NUM_ACTIONS)
    assert values.shape == (1,)
    print(f"  pipeline produced logits {tuple(logits.shape)}, values {tuple(values.shape)}  ok")
    print("TEST 5 PASSED")


if __name__ == "__main__":
    test_env_shapes()
    test_ippo_model_forward()
    test_mappo_model_forward_with_global_state()
    test_postprocessing_hook()
    test_env_to_postprocessing_pipeline()
    print("\n" + "=" * 70)
    print("ALL SHAPE TESTS PASSED")
    print("=" * 70)
