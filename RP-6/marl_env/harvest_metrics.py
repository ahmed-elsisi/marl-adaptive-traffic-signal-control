"""
Per-episode metrics for HarvestEnv evaluation.

Mirrors the design of RP-5/evaluate.py's MetricsCollector + EnvWrapperWithMetrics
so that Phase-2 evaluation reuses the same patterns. Metric definitions follow
the SSD literature (Leibo 2017, Hughes 2018) so results are comparable to
published Harvest results.

Metrics produced per episode:

  - total_apples_collected:  collective welfare (efficiency proxy)
  - apples_per_agent[i]:     per-agent raw counts
  - gini_coefficient:        equity (0 = perfect equality, 1 = max inequality)
  - equality:                1 - 2 * gini  (often more readable in SSD papers)
  - sustainability:          min over last 25% of episode of
                              apples_on_grid / initial_apples
                              (tragedy-of-commons indicator)
  - time_to_depletion:       first step where apples_on_grid == 0,
                              or episode_length if never depleted

CSV column convention is kept Phase-1-compatible so that
shared/cross_env_synthesis.py can ingest both phases identically.
"""

from typing import Dict, List, Optional
import csv
import os
import numpy as np


# ── Pure-function metric primitives ───────────────────────────────────────────


def gini_coefficient(values: List[float]) -> float:
    """Standard Gini coefficient on a list of non-negative values.

    Returns 0.0 for empty inputs or all-zeros (perfectly equal nothing).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    # Standard formula: G = (2 * sum_i (i+1) * x_i) / (n * sum(x)) - (n + 1) / n
    cumulative = np.arange(1, n + 1) * arr
    return float((2.0 * cumulative.sum()) / (n * arr.sum()) - (n + 1) / n)


def sustainability(
    apples_on_grid_per_step: List[int],
    max_capacity: int,
    tail_fraction: float = 0.25,
) -> float:
    """Mean apples-on-grid / max_capacity over the last `tail_fraction` of the
    episode. Bounded in [0, 1].

    `max_capacity` should be (grid_cells - num_agents) — the maximum number of
    cells that could hold apples simultaneously. We deliberately use the mean
    (not min) over the tail so a single transient depletion doesn't dominate;
    the tragedy-of-commons signal is "is the commons systematically depleted
    in steady state?", which is a mean question.

    Low values mean the commons collapsed (tragedy of the commons).
    High values mean apples remained plentiful through the episode.
    """
    if max_capacity <= 0 or len(apples_on_grid_per_step) == 0:
        return 0.0
    tail_len = max(1, int(len(apples_on_grid_per_step) * tail_fraction))
    tail = apples_on_grid_per_step[-tail_len:]
    # Clamp to [0, 1] — apple count can briefly exceed (H*W - N) when agents
    # transiently sit on top of apples between collect and regrowth steps.
    return float(min(1.0, np.mean(tail) / max_capacity))


def time_to_depletion(
    apples_on_grid_per_step: List[int],
    episode_length: int,
) -> int:
    """First step index where apples_on_grid is 0, else episode_length."""
    for t, count in enumerate(apples_on_grid_per_step):
        if count == 0:
            return t
    return episode_length


# ── Per-episode collector ─────────────────────────────────────────────────────


class HarvestMetricsCollector:
    """Accumulates per-step state and produces summary stats at episode end.

    Usage (mirrors RP-5/evaluate.py:51-123 pattern):

        collector = HarvestMetricsCollector(env.agent_ids)
        obs, _ = env.reset()
        collector.start_episode(env.initial_apple_count)
        while not done:
            actions = ...
            obs, rewards, terms, truncs, infos = env.step(actions)
            collector.record_step(infos, rewards)
            done = terms["__all__"]
        summary = collector.summarize(episode_length=env.episode_length)
    """

    def __init__(self, agent_ids: List[str]):
        self.agent_ids = list(agent_ids)
        self._reset_state()

    def _reset_state(self):
        self.apples_collected_total: Dict[str, int] = {a: 0 for a in self.agent_ids}
        self.apples_on_grid_per_step: List[int] = []
        self.reward_per_step: List[float] = []   # team-summed reward
        self.initial_apples: int = 0
        self.max_capacity: int = 0

    def start_episode(self, initial_apple_count: int, max_capacity: int):
        """Reset state at the start of each new episode.

        `max_capacity` should be (grid_cells - num_agents) so the
        sustainability metric is bounded in [0, 1].
        """
        self._reset_state()
        self.initial_apples = int(initial_apple_count)
        self.max_capacity = int(max_capacity)

    def record_step(
        self,
        infos: Dict[str, Dict],
        rewards: Optional[Dict[str, float]] = None,
    ):
        # Apples on grid this step — pulled from any agent's info (they all see
        # the same global count).
        any_agent = self.agent_ids[0]
        if any_agent in infos and "current_apple_count_global" in infos[any_agent]:
            self.apples_on_grid_per_step.append(
                int(infos[any_agent]["current_apple_count_global"])
            )

        # Per-agent apples this step
        for a in self.agent_ids:
            if a in infos and "apples_collected_this_step" in infos[a]:
                self.apples_collected_total[a] += int(
                    infos[a]["apples_collected_this_step"]
                )

        if rewards is not None:
            self.reward_per_step.append(float(sum(rewards.values())))

    def summarize(self, episode_length: int) -> Dict:
        per_agent = [self.apples_collected_total[a] for a in self.agent_ids]
        total = int(sum(per_agent))
        gini = gini_coefficient(per_agent)
        return {
            "total_apples_collected": total,
            "apples_per_agent": dict(self.apples_collected_total),
            "gini_coefficient": gini,
            "equality": 1.0 - 2.0 * gini,
            "sustainability": sustainability(
                self.apples_on_grid_per_step, self.max_capacity
            ),
            "time_to_depletion": time_to_depletion(
                self.apples_on_grid_per_step, episode_length
            ),
            "initial_apples": self.initial_apples,
            "episode_length": episode_length,
            "total_team_reward": float(sum(self.reward_per_step)),
        }

    # ── CSV output (Phase-1-compatible column convention) ─────────────────────

    def save_episode_csv(self, summary: Dict, csv_path: str):
        """Write a one-row summary CSV. Column names are stable across episodes
        so cross_env_synthesis.py can ingest a directory of these uniformly.
        """
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        # Flatten apples_per_agent into per-column entries
        flat = {
            "total_apples_collected": summary["total_apples_collected"],
            "gini_coefficient": summary["gini_coefficient"],
            "equality": summary["equality"],
            "sustainability": summary["sustainability"],
            "time_to_depletion": summary["time_to_depletion"],
            "initial_apples": summary["initial_apples"],
            "episode_length": summary["episode_length"],
            "total_team_reward": summary["total_team_reward"],
        }
        for a, v in summary["apples_per_agent"].items():
            flat[f"apples_{a}"] = v

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(flat.keys()))
            writer.writerow(list(flat.values()))

    def save_timeseries_csv(self, csv_path: str):
        """Write per-step apples-on-grid + per-step team reward time-series.

        Used by compare_algos_harvest.py for cross-seed averaging plots.
        """
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "apples_on_grid", "team_reward"])
            n = max(len(self.apples_on_grid_per_step), len(self.reward_per_step))
            for t in range(n):
                apples = (
                    self.apples_on_grid_per_step[t]
                    if t < len(self.apples_on_grid_per_step)
                    else ""
                )
                rew = (
                    self.reward_per_step[t]
                    if t < len(self.reward_per_step)
                    else ""
                )
                writer.writerow([t, apples, rew])
