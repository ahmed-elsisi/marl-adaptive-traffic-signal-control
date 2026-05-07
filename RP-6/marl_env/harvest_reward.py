"""
Reward computation for HarvestEnv.

Sparse +1 per apple collected, blended with a team-average signal under a
configurable weight. The blend weight is the load-bearing experimental
dimension for the Phase-2 sharing sweep:

    w = 0.0  -> pure individual (canonical SSD; defection-permitting)
    w = 0.5  -> mixed
    w = 1.0  -> fully team-shared (defection has no individual gain)

No throughput / pressure / shaping terms — those were SUMO-specific in
Phase 1 and would muddy the dilemma signal.
"""

from typing import Dict


def compute_team_sharing_rewards(
    apples_collected: Dict[str, int],
    shared_reward_weight: float,
) -> Dict[str, float]:
    """Blend per-agent collected-apple counts with a team-average signal.

    Args:
        apples_collected:     {agent_id: apples_collected_this_step (0 or 1)}.
        shared_reward_weight: blend weight in [0.0, 1.0]. 0 = individual,
                              1 = team average.

    Returns:
        {agent_id: float reward}.
    """
    if not 0.0 <= shared_reward_weight <= 1.0:
        raise ValueError(
            f"shared_reward_weight must be in [0, 1], got {shared_reward_weight}"
        )

    n = len(apples_collected)
    if n == 0:
        return {}

    team_avg = sum(apples_collected.values()) / n

    return {
        agent_id: (1.0 - shared_reward_weight) * float(indiv)
                  + shared_reward_weight * team_avg
        for agent_id, indiv in apples_collected.items()
    }
