"""
Egocentric partial-observation builder for HarvestEnv.

Produces a (window, window, 3) uint8 RGB patch centered on each agent.
Out-of-bounds cells are encoded as walls.

Color scheme (matches Leibo et al. 2017 conventions, simplified):
  empty cell  -> black  (0,   0,   0)
  apple       -> green  (0,   255, 0)
  agent       -> blue   (0,   0,   255)   (all agents identical, no self/other)
  out-of-grid -> red    (255, 0,   0)     (wall padding)
"""

from typing import Dict, Tuple
import numpy as np


# RGB encodings — uint8
COLOR_EMPTY = np.array((0,   0,   0),   dtype=np.uint8)
COLOR_APPLE = np.array((0,   255, 0),   dtype=np.uint8)
COLOR_AGENT = np.array((0,   0,   255), dtype=np.uint8)
COLOR_WALL  = np.array((255, 0,   0),   dtype=np.uint8)


def render_global_rgb(
    apple_grid: np.ndarray,
    agent_positions: Dict[str, Tuple[int, int]],
) -> np.ndarray:
    """Render the global grid state as an (H, W, 3) uint8 RGB image.

    Apples and agents both occupy single cells; if an agent is on an apple
    cell, the agent color wins (the env removes apples when an agent steps
    onto and collects them, but transient overlap can occur during a single
    step's resolution).
    """
    h, w = apple_grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[apple_grid] = COLOR_APPLE
    for (r, c) in agent_positions.values():
        rgb[r, c] = COLOR_AGENT
    return rgb


def build_egocentric_window(
    global_rgb: np.ndarray,
    agent_position: Tuple[int, int],
    window_size: int,
) -> np.ndarray:
    """Slice an (window_size, window_size, 3) patch centered on `agent_position`.

    Pads with wall color (red) for cells outside the grid bounds. The agent
    itself appears at the geometric center of the returned patch.

    Args:
        global_rgb:    (H, W, 3) uint8 RGB image of the whole grid.
        agent_position: (row, col) of the agent in grid coords.
        window_size:    odd integer side length of the egocentric window.

    Returns:
        (window_size, window_size, 3) uint8 array.
    """
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")

    half = window_size // 2
    h, w, _ = global_rgb.shape

    padded = np.empty((h + 2 * half, w + 2 * half, 3), dtype=np.uint8)
    padded[:] = COLOR_WALL
    padded[half:half + h, half:half + w] = global_rgb

    r, c = agent_position
    # Agent at (r, c) sits at (r+half, c+half) in padded coords; window is
    # (r+half - half : r+half + half + 1) = (r : r + window_size).
    return padded[r:r + window_size, c:c + window_size].copy()


def build_observations(
    apple_grid: np.ndarray,
    agent_positions: Dict[str, Tuple[int, int]],
    window_size: int,
) -> Dict[str, np.ndarray]:
    """Build per-agent egocentric observations from the current grid state."""
    global_rgb = render_global_rgb(apple_grid, agent_positions)
    return {
        agent_id: build_egocentric_window(global_rgb, pos, window_size)
        for agent_id, pos in agent_positions.items()
    }
