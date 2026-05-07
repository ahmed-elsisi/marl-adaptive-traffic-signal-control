"""
HarvestEnv: a sequential social dilemma environment for MARL.

Mirrors RP-5/marl_env/sumo_env.py's MultiAgentEnv contract so the same
training infrastructure can be reused for Phase 2. Implementation follows
Leibo et al. 2017 ("Multi-agent Reinforcement Learning in Sequential
Social Dilemmas") with the following scoping decisions:

  - 12 x 8 grid, 4 agents (matches Phase 1 SUMO setup for cross-env synthesis).
  - 1000-step episodes, no early termination.
  - Actions: Discrete(6) — N/S/E/W movement, stay, collect-apple. (Tag action
    deferred; would extend to Discrete(7) when punishment dynamics are studied.)
  - Observations: 15 x 15 x 3 RGB egocentric patch per agent (uint8). All
    agents render identically — no self/other distinction in the patch.
  - Apple regrowth (the dilemma core):
        P(regrow at empty cell) = base * neighbours_within_radius_2
    with no regrowth where there are zero apple neighbours within radius. This
    means depleting an area destroys it permanently for the rest of the episode.
  - Reward: sparse +1 per apple collected, blended with a team-average signal
    via shared_reward_weight (computed in harvest_reward.py).

Public surface mirrors SUMOTrafficEnv:
  - MultiAgentEnv subclass
  - reset() -> (obs_dict, info_dict)
  - step(actions) -> (obs, rewards, terminateds, truncateds, infos)
  - observation_space_dict, action_space_dict (per agent)
  - agent_ids
  - _get_global_state() -> (H, W, 3) uint8: extra method for the centralized
    critic used by MAPPO. Not part of the standard MultiAgentEnv contract.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from gymnasium.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .harvest_obs import (
    build_observations,
    render_global_rgb,
)
from .harvest_reward import compute_team_sharing_rewards


# ── Action ids ────────────────────────────────────────────────────────────────
ACTION_NORTH   = 0
ACTION_SOUTH   = 1
ACTION_EAST    = 2
ACTION_WEST    = 3
ACTION_STAY    = 4
ACTION_COLLECT = 5
NUM_ACTIONS = 6

# (row, col) deltas
_MOVE_DELTAS = {
    ACTION_NORTH: (-1, 0),
    ACTION_SOUTH: ( 1, 0),
    ACTION_EAST:  (0,  1),
    ACTION_WEST:  (0, -1),
}


class HarvestEnv(MultiAgentEnv):
    """4-agent Harvest sequential social dilemma on a 12 x 8 grid."""

    def __init__(self, env_config: Optional[Dict] = None):
        super().__init__()
        env_config = dict(env_config or {})

        # Grid + agent population
        self.grid_height = int(env_config.get("grid_height", 8))
        self.grid_width  = int(env_config.get("grid_width", 12))
        self.num_agents  = int(env_config.get("num_agents", 4))
        self.episode_length = int(env_config.get("episode_length", 1000))

        # Observation
        self.obs_window = int(env_config.get("obs_window", 15))
        if self.obs_window % 2 == 0:
            raise ValueError(f"obs_window must be odd, got {self.obs_window}")

        # Apple dynamics
        self.initial_apple_density = float(env_config.get("initial_apple_density", 0.3))
        self.regrowth_base   = float(env_config.get("apple_regrowth_base", 0.01))
        self.regrowth_radius = int(env_config.get("apple_regrowth_radius", 2))

        # Reward
        self.shared_reward_weight = float(env_config.get("shared_reward_weight", 0.0))

        # Reproducibility
        self.seed_value = int(env_config.get("seed", 42))
        self._rng = np.random.default_rng(self.seed_value)

        # Agents
        self.agent_ids: List[str] = [f"A{i}" for i in range(self.num_agents)]
        self._agents_set = set(self.agent_ids)

        # Spaces — exposed as dicts (mirrors RP-5 sumo_env.py:148-165) and as
        # single attributes for RLlib's per-policy space inference.
        obs_shape = (self.obs_window, self.obs_window, 3)
        self.observation_space_dict: Dict[str, Box] = {
            a: Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
            for a in self.agent_ids
        }
        self.action_space_dict: Dict[str, Discrete] = {
            a: Discrete(NUM_ACTIONS) for a in self.agent_ids
        }
        self.observation_space = self.observation_space_dict[self.agent_ids[0]]
        self.action_space = self.action_space_dict[self.agent_ids[0]]

        # Per-episode state — initialised in reset()
        self.apple_grid: np.ndarray = None      # bool (H, W)
        self.agent_positions: Dict[str, Tuple[int, int]] = {}
        self.step_count: int = 0
        self.initial_apple_count: int = 0
        self.last_apples_collected: Dict[str, int] = {a: 0 for a in self.agent_ids}

    # ── RLlib MultiAgentEnv API ───────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Place apples at random
        self.apple_grid = (
            self._rng.random((self.grid_height, self.grid_width))
            < self.initial_apple_density
        )

        # Place agents in random non-apple cells (clear the cell if dense layout
        # forces an overlap).
        empty_cells = list(zip(*np.where(~self.apple_grid)))
        self._rng.shuffle(empty_cells)
        if len(empty_cells) < self.num_agents:
            # Fallback: spawn anywhere
            all_cells = [
                (r, c)
                for r in range(self.grid_height)
                for c in range(self.grid_width)
            ]
            self._rng.shuffle(all_cells)
            empty_cells = all_cells

        self.agent_positions = {}
        for i, agent_id in enumerate(self.agent_ids):
            r, c = empty_cells[i]
            self.agent_positions[agent_id] = (int(r), int(c))
            self.apple_grid[r, c] = False  # no apple under spawned agent

        self.initial_apple_count = int(self.apple_grid.sum())
        self.step_count = 0
        self.last_apples_collected = {a: 0 for a in self.agent_ids}

        obs = build_observations(
            self.apple_grid, self.agent_positions, self.obs_window
        )
        infos = {a: self._build_info(a) for a in self.agent_ids}
        return obs, infos

    def step(
        self,
        action_dict: Dict[str, int],
    ) -> Tuple[
        Dict[str, np.ndarray],   # obs
        Dict[str, float],        # rewards
        Dict[str, bool],         # terminateds (with __all__)
        Dict[str, bool],         # truncateds  (with __all__)
        Dict[str, Dict],         # infos
    ]:
        # 1. Resolve movement (collision-aware, randomized order).
        self._apply_movements(action_dict)

        # 2. Resolve collect actions.
        apples_collected = self._apply_collects(action_dict)

        # 3. Apple regrowth.
        self._regrow_apples()

        # 4. Bookkeeping.
        self.last_apples_collected = apples_collected
        self.step_count += 1

        # 5. Build outputs.
        obs = build_observations(
            self.apple_grid, self.agent_positions, self.obs_window
        )
        rewards = compute_team_sharing_rewards(
            apples_collected, self.shared_reward_weight
        )
        terminated = self.step_count >= self.episode_length
        terminateds = {a: terminated for a in self.agent_ids}
        terminateds["__all__"] = terminated
        truncateds = {a: False for a in self.agent_ids}
        truncateds["__all__"] = False
        infos = {a: self._build_info(a) for a in self.agent_ids}
        return obs, rewards, terminateds, truncateds, infos

    # ── Centralized-critic helper ─────────────────────────────────────────────

    def _get_global_state(self) -> np.ndarray:
        """Full grid as (H, W, 3) uint8 — input to the MAPPO centralized critic."""
        return render_global_rgb(self.apple_grid, self.agent_positions)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _apply_movements(self, action_dict: Dict[str, int]):
        # Shuffle order so no agent has a deterministic priority advantage.
        order = list(self.agent_ids)
        self._rng.shuffle(order)

        occupied = set(self.agent_positions.values())
        for agent_id in order:
            action = int(action_dict.get(agent_id, ACTION_STAY))
            if action not in _MOVE_DELTAS:
                continue  # STAY or COLLECT — no movement
            old_pos = self.agent_positions[agent_id]
            dr, dc = _MOVE_DELTAS[action]
            new_r, new_c = old_pos[0] + dr, old_pos[1] + dc
            if not (0 <= new_r < self.grid_height and 0 <= new_c < self.grid_width):
                continue
            new_pos = (new_r, new_c)
            if new_pos in occupied:
                continue
            occupied.discard(old_pos)
            occupied.add(new_pos)
            self.agent_positions[agent_id] = new_pos

    def _apply_collects(self, action_dict: Dict[str, int]) -> Dict[str, int]:
        apples_collected = {a: 0 for a in self.agent_ids}
        for agent_id in self.agent_ids:
            if int(action_dict.get(agent_id, ACTION_STAY)) != ACTION_COLLECT:
                continue
            r, c = self.agent_positions[agent_id]
            if self.apple_grid[r, c]:
                self.apple_grid[r, c] = False
                apples_collected[agent_id] = 1
        return apples_collected

    def _regrow_apples(self):
        """P(regrow at empty cell) = base * count_neighbours_within_radius.

        Neighbours that are themselves apples (excluding the cell itself) drive
        regrowth. No regrowth in cells with zero apple neighbours — this is the
        dilemma core. Cells under an agent are also blocked from regrowth.
        """
        # Vectorised neighbour-count via shifted-and-summed slicing.
        radius = self.regrowth_radius
        h, w = self.apple_grid.shape
        apples_int = self.apple_grid.astype(np.int32)
        neighbour_counts = np.zeros((h, w), dtype=np.int32)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                src_r0 = max(0, dr); src_r1 = min(h, h + dr)
                src_c0 = max(0, dc); src_c1 = min(w, w + dc)
                dst_r0 = max(0, -dr); dst_r1 = dst_r0 + (src_r1 - src_r0)
                dst_c0 = max(0, -dc); dst_c1 = dst_c0 + (src_c1 - src_c0)
                neighbour_counts[dst_r0:dst_r1, dst_c0:dst_c1] += (
                    apples_int[src_r0:src_r1, src_c0:src_c1]
                )

        empty = ~self.apple_grid
        # Block regrowth under any agent.
        agent_mask = np.zeros((h, w), dtype=bool)
        for (r, c) in self.agent_positions.values():
            agent_mask[r, c] = True
        empty = empty & ~agent_mask

        prob = self.regrowth_base * neighbour_counts.astype(np.float32)
        prob[~empty] = 0.0
        prob[neighbour_counts == 0] = 0.0  # explicit: no regrowth without neighbours

        regrow = self._rng.random(prob.shape) < prob
        self.apple_grid |= regrow

    def _build_info(self, agent_id: str) -> Dict:
        # `global_state` is included in every agent's info so the MAPPO
        # centralized-critic postprocessing hook can pull it into the
        # SampleBatch without needing to coordinate across agents. The cost
        # is that the same (H, W, 3) array is duplicated num_agents times
        # per step in the rollout buffer — tolerable at the small grid sizes
        # used here. IPPO's decentralized critic ignores this field.
        return {
            "apples_collected_this_step": int(self.last_apples_collected[agent_id]),
            "position": tuple(self.agent_positions[agent_id]),
            "current_apple_count_global": int(self.apple_grid.sum()),
            "global_state": self._get_global_state(),
        }

    # ── Debug rendering ───────────────────────────────────────────────────────

    def render_ascii(self) -> str:
        """ASCII grid: '.' empty, 'a' apple, '0'..'N' agent index. Useful for smoke tests."""
        chars = np.full((self.grid_height, self.grid_width), ".", dtype="<U1")
        chars[self.apple_grid] = "a"
        for i, agent_id in enumerate(self.agent_ids):
            r, c = self.agent_positions[agent_id]
            chars[r, c] = str(i)
        return "\n".join("".join(row) for row in chars)
