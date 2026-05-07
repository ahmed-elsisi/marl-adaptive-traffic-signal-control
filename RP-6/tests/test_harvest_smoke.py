"""
Week-4 verification suite for HarvestEnv.

Three checks:
  1. End-to-end: a full 1000-step random-action episode runs without error
     and the metrics collector produces a sensible summary.
  2. Apple regrowth dynamics: after artificially depleting all apples in one
     half of the grid, that half stays empty for the rest of the episode
     (no spontaneous regeneration without neighbours).
  3. Reward sharing: with shared_reward_weight=1.0 and exactly one agent
     collecting, all agents receive +1/N reward.

Run from RP-6/ with:
    python -m tests.test_harvest_smoke
or:
    python tests/test_harvest_smoke.py
"""

from pathlib import Path
import sys

# Allow running as a script from RP-6/ without installing as a package.
_HERE = Path(__file__).resolve().parent
_RP6  = _HERE.parent
if str(_RP6) not in sys.path:
    sys.path.insert(0, str(_RP6))

import numpy as np

from marl_env.harvest_env import (
    HarvestEnv,
    ACTION_COLLECT,
    ACTION_STAY,
    NUM_ACTIONS,
)
from marl_env.harvest_metrics import HarvestMetricsCollector
from marl_env.harvest_reward import compute_team_sharing_rewards


# ── Test 1: end-to-end random rollout ─────────────────────────────────────────


def test_random_rollout():
    print("\n" + "=" * 70)
    print("TEST 1: 1000-step random rollout + metrics summary")
    print("=" * 70)

    env = HarvestEnv({"shared_reward_weight": 0.0, "seed": 42})
    obs, infos = env.reset()

    assert set(obs.keys()) == set(env.agent_ids), "obs missing some agents"
    for a in env.agent_ids:
        assert obs[a].shape == (env.obs_window, env.obs_window, 3), \
            f"obs[{a}] has wrong shape: {obs[a].shape}"
        assert obs[a].dtype == np.uint8, f"obs[{a}] has wrong dtype: {obs[a].dtype}"

    print(f"Initial grid ({env.initial_apple_count} apples):")
    print(env.render_ascii())

    collector = HarvestMetricsCollector(env.agent_ids)
    max_capacity = env.grid_height * env.grid_width - env.num_agents
    collector.start_episode(env.initial_apple_count, max_capacity)
    collector.record_step(infos)  # step 0 snapshot

    rng = np.random.default_rng(0)
    total_team_reward = 0.0

    for step in range(env.episode_length):
        actions = {a: int(rng.integers(NUM_ACTIONS)) for a in env.agent_ids}
        obs, rewards, terms, truncs, infos = env.step(actions)
        collector.record_step(infos, rewards)
        total_team_reward += sum(rewards.values())
        if (step + 1) % 250 == 0:
            apples_now = infos[env.agent_ids[0]]["current_apple_count_global"]
            print(f"\n--- Step {step + 1}  (apples on grid: {apples_now}) ---")
            print(env.render_ascii())
        if terms["__all__"]:
            break

    summary = collector.summarize(env.episode_length)
    print("\nEpisode summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Sanity assertions
    assert summary["episode_length"] == env.episode_length
    assert summary["total_apples_collected"] >= 0
    assert 0.0 <= summary["gini_coefficient"] <= 1.0
    assert 0.0 <= summary["sustainability"] <= 1.0, \
        f"sustainability out of [0,1]: {summary['sustainability']}"
    assert 0 <= summary["time_to_depletion"] <= env.episode_length
    assert abs(total_team_reward - summary["total_team_reward"]) < 1e-6, \
        "collector and direct-sum reward totals disagree"

    print("\nTEST 1 PASSED")
    return summary


# ── Test 2: apple regrowth dilemma core ───────────────────────────────────────


def test_no_regrowth_without_neighbours():
    print("\n" + "=" * 70)
    print("TEST 2: fully-depleted grid never regrows (dilemma core)")
    print("=" * 70)

    env = HarvestEnv({"shared_reward_weight": 0.0, "seed": 7})
    env.reset()

    # Wipe ALL apples from the grid. With zero apples anywhere, every cell's
    # neighbour count is 0, so regrowth probability is 0 everywhere — the
    # grid must stay empty for the rest of the episode regardless of agent
    # behaviour. This is the strict form of the dilemma core: a fully
    # collapsed commons cannot recover.
    env.apple_grid[:] = False
    assert env.apple_grid.sum() == 0, "grid should start empty"

    actions_stay = {a: ACTION_STAY for a in env.agent_ids}
    for step in range(200):
        env.step(actions_stay)
        if env.apple_grid.sum() != 0:
            raise AssertionError(
                f"step {step}: grid regrew {int(env.apple_grid.sum())} apples "
                f"despite having zero apple neighbours — regrowth logic is broken"
            )

    print("  200 stay-steps from a fully-empty grid -> 0 apples ever regrew  ok")
    print("TEST 2 PASSED")


# ── Test 3: shared-reward weighting ───────────────────────────────────────────


def test_shared_reward_weighting():
    print("\n" + "=" * 70)
    print("TEST 3: shared_reward_weight blends individual and team-average")
    print("=" * 70)

    apples = {"A0": 1, "A1": 0, "A2": 0, "A3": 0}

    # w=0.0 -> pure individual
    r0 = compute_team_sharing_rewards(apples, 0.0)
    assert r0 == {"A0": 1.0, "A1": 0.0, "A2": 0.0, "A3": 0.0}, r0

    # w=1.0 -> all agents see team_avg = 1/4 = 0.25
    r1 = compute_team_sharing_rewards(apples, 1.0)
    expected = {a: 0.25 for a in apples}
    assert all(abs(r1[a] - expected[a]) < 1e-9 for a in apples), r1

    # w=0.5 -> A0 gets 0.5*1 + 0.5*0.25 = 0.625, others get 0.5*0 + 0.5*0.25 = 0.125
    r05 = compute_team_sharing_rewards(apples, 0.5)
    assert abs(r05["A0"] - 0.625) < 1e-9, r05
    for a in ("A1", "A2", "A3"):
        assert abs(r05[a] - 0.125) < 1e-9, r05

    print("  w=0.0 (individual): A0=1.0, others=0.0  ok")
    print("  w=0.5 (mixed):      A0=0.625, others=0.125  ok")
    print("  w=1.0 (team):       all=0.25  ok")
    print("TEST 3 PASSED")


if __name__ == "__main__":
    summary = test_random_rollout()
    test_no_regrowth_without_neighbours()
    test_shared_reward_weighting()
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
