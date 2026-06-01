"""
Visualize Harvest policy BEHAVIOR as animated GIFs.

Replays a policy (a trained MAPPO/IPPO checkpoint, or the FixedGreedy rule) for
one episode and renders the grid over time — apples depleting/regrowing and the
4 agents moving, each drawn in a distinct colour — so you can literally watch
"greed depletes the commons, sharing sustains it".

Reuses the existing rollout machinery (no changes to the training/eval pipeline):
  - evaluate_harvest._get_obs_filter   (MeanStdFilter retrieval — MANDATORY at
    action time, else the deterministic policy collapses; same Phase-1 gotcha)
  - run_fixed_greedy.fixed_greedy_action
  - HarvestEnv exposes apple_grid / agent_positions / info["apples_collected_this_step"]
  - trained env_config is read straight from the checkpoint (algo.config)

Modes
-----
Single trained policy:
    python visualize_harvest.py --checkpoint <ckpt_dir> --label "MAPPO team" --out out.gif
Greedy baseline (no checkpoint needed):
    python visualize_harvest.py --greedy --out greedy.gif
Auto (after the matrix is trained) — emits 7 per-policy GIFs + 2 panels:
    python visualize_harvest.py --auto --results-dir results --out-dir metrics/viz

Common flags: --seed (default 42, applied to every rollout), --max-frames
(default 250), --fps (default 12).
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from marl_env.harvest_env import HarvestEnv
from run_fixed_greedy import fixed_greedy_action

# ── Visual constants ──────────────────────────────────────────────────────────
CELL = 36                      # px per grid cell
BANNER_H = 66                  # px for the text header
BG = (18, 18, 18)
BANNER_BG = (32, 34, 40)
GRIDLINE = (45, 45, 50)
APPLE = (45, 200, 70)
DIVIDER = (60, 60, 66)         # gap colour between panel quadrants
# Distinct, colour-blind-friendly-ish agent palette (indexed by agent order)
AGENT_COLORS = [
    (66, 135, 245),   # blue
    (245, 145, 32),   # orange
    (224, 70, 190),   # magenta
    (40, 210, 210),   # cyan
]

# Greedy env params (mirror run_fixed_greedy.evaluate's template).
GREEDY_ENV_TEMPLATE = {
    "grid_height": 8, "grid_width": 12, "num_agents": 4,
    "episode_length": 1000, "obs_window": 15, "initial_apple_density": 0.3,
    "apple_regrowth_base": 0.01, "apple_regrowth_radius": 2,
    "shared_reward_weight": 0.0,
}

# Conditions for --auto. (config stem under configs/, output tag, pretty label)
_CONDITIONS = [
    ("harvest_mappo_individual", "mappo_individual", "MAPPO  individual (w=0.0)"),
    ("harvest_mappo_mixed",      "mappo_mixed",      "MAPPO  mixed (w=0.5)"),
    ("harvest_mappo_team",       "mappo_team",       "MAPPO  team (w=1.0)"),
    ("harvest_ippo_individual",  "ippo_individual",  "IPPO  individual (w=0.0)"),
    ("harvest_ippo_mixed",       "ippo_mixed",       "IPPO  mixed (w=0.5)"),
    ("harvest_ippo_team",        "ippo_team",        "IPPO  team (w=1.0)"),
]


def _get_font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


_FONT = _get_font(13)
_FONT_SMALL = _get_font(11)


# ── Rendering ─────────────────────────────────────────────────────────────────


def render_frame(env: HarvestEnv, cumulative: Dict[str, int], step: int,
                 label: str) -> np.ndarray:
    """One RGB frame: header banner + the grid with apples and coloured agents."""
    gh, gw = env.grid_height, env.grid_width
    grid_px_h, grid_px_w = gh * CELL, gw * CELL
    W, H = grid_px_w, BANNER_H + grid_px_h
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Banner
    d.rectangle([0, 0, W, BANNER_H], fill=BANNER_BG)
    eplen = env.episode_length
    on_grid = int(env.apple_grid.sum())
    d.text((6, 4), label, fill=(235, 235, 235), font=_FONT)
    d.text((6, 22), f"step {step}/{eplen}    apples on grid: {on_grid}",
           fill=(180, 180, 185), font=_FONT_SMALL)
    # Per-agent cumulative collected, coloured to match the markers.
    x = 6
    for i, aid in enumerate(env.agent_ids):
        col = AGENT_COLORS[i % len(AGENT_COLORS)]
        d.rectangle([x, 40, x + 10, 50], fill=col)
        txt = f"{cumulative.get(aid, 0)}"
        d.text((x + 14, 39), txt, fill=col, font=_FONT_SMALL)
        x += 14 + 8 * max(2, len(txt)) + 12

    # Grid offset
    oy = BANNER_H
    # Gridlines
    for r in range(gh + 1):
        d.line([(0, oy + r * CELL), (grid_px_w, oy + r * CELL)], fill=GRIDLINE)
    for c in range(gw + 1):
        d.line([(c * CELL, oy), (c * CELL, oy + grid_px_h)], fill=GRIDLINE)

    # Apples
    ag = env.apple_grid
    m = 5
    for r in range(gh):
        for c in range(gw):
            if ag[r, c]:
                x0, y0 = c * CELL + m, oy + r * CELL + m
                d.rectangle([x0, y0, x0 + CELL - 2 * m, y0 + CELL - 2 * m],
                            fill=APPLE)

    # Agents (drawn last, on top)
    am = 4
    for i, aid in enumerate(env.agent_ids):
        r, c = env.agent_positions[aid]
        col = AGENT_COLORS[i % len(AGENT_COLORS)]
        x0, y0 = c * CELL + am, oy + r * CELL + am
        d.ellipse([x0, y0, x0 + CELL - 2 * am, y0 + CELL - 2 * am], fill=col)

    return np.asarray(img, dtype=np.uint8)


def write_gif(frames: List[np.ndarray], path: str, fps: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=False)
    print(f"  wrote {path}  ({len(frames)} frames, {imgs[0].size[0]}x{imgs[0].size[1]})")


def make_panel(rollouts: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Composite 4 same-sized rollouts into a synchronized 2x2 grid."""
    n = max(len(r) for r in rollouts)
    padded = [r + [r[-1]] * (n - len(r)) for r in rollouts]  # pad with last frame
    fh, fw = padded[0][0].shape[:2]
    g = 6
    hgap = np.full((fh, g, 3), DIVIDER, dtype=np.uint8)
    vgap = np.full((g, fw * 2 + g, 3), DIVIDER, dtype=np.uint8)
    out = []
    for t in range(n):
        top = np.hstack([padded[0][t], hgap, padded[1][t]])
        bot = np.hstack([padded[2][t], hgap, padded[3][t]])
        out.append(np.vstack([top, vgap, bot]))
    return out


# ── Rollouts ──────────────────────────────────────────────────────────────────


def _run_and_capture(env_cfg: Dict, seed: int, max_frames: int, label: str,
                     act_fn) -> List[np.ndarray]:
    """Run one episode; act_fn(env, obs) -> action_dict. Capture subsampled frames."""
    cfg = dict(env_cfg)
    cfg["seed"] = seed
    env = HarvestEnv(cfg)
    obs, infos = env.reset()
    eplen = env.episode_length
    stride = max(1, math.ceil(eplen / max_frames))
    cumulative = {a: 0 for a in env.agent_ids}

    frames = [render_frame(env, cumulative, 0, label)]
    done, step = False, 0
    while not done:
        actions = act_fn(env, obs)
        obs, rewards, terms, truncs, infos = env.step(actions)
        step += 1
        for aid in env.agent_ids:
            cumulative[aid] += int(infos[aid].get("apples_collected_this_step", 0))
        done = bool(terms.get("__all__", False))
        if step % stride == 0 or done:
            frames.append(render_frame(env, cumulative, step, label))
    total = sum(cumulative.values())
    print(f"  [{label}] total apples={total}  per-agent={list(cumulative.values())}")
    return frames


def rollout_greedy(seed: int, max_frames: int, label: str = "FixedGreedy") -> List[np.ndarray]:
    radius = GREEDY_ENV_TEMPLATE["obs_window"] // 2

    def act(env, obs):
        return {a: fixed_greedy_action(env.agent_positions[a], env.apple_grid, radius)
                for a in env.agent_ids}

    return _run_and_capture(GREEDY_ENV_TEMPLATE, seed, max_frames, label, act)


def rollout_trained(checkpoint_dir: str, seed: int, max_frames: int,
                    label: str) -> List[np.ndarray]:
    from ray.rllib.algorithms.ppo import PPO
    from evaluate_harvest import _get_obs_filter

    algo = PPO.from_checkpoint(checkpoint_dir)
    obs_filter = _get_obs_filter(algo)
    if obs_filter is None:
        print(f"  WARN: no MeanStdFilter for {label} — obs passed raw (policy may collapse).")
    env_cfg = dict(algo.config.to_dict().get("env_config", {}))

    def act(env, obs):
        actions = {}
        for aid in env.agent_ids:
            o = obs[aid]
            if obs_filter is not None:
                o = obs_filter(o, update=False)
            actions[aid] = algo.compute_single_action(o, policy_id="shared_policy",
                                                      explore=False)
        return actions

    try:
        frames = _run_and_capture(env_cfg, seed, max_frames, label, act)
    finally:
        algo.stop()
    return frames


# ── Checkpoint discovery (for --auto) ─────────────────────────────────────────


def _find_checkpoint(results_dir: Path, config_stem: str, seed: int) -> Optional[str]:
    """Read the run_matrix stamp for this condition/seed -> final checkpoint path."""
    stamp = results_dir / ".matrix_done" / f"{config_stem}_seed{seed}.json"
    if not stamp.exists():
        return None
    try:
        d = json.load(open(stamp))
        cp = d.get("final_checkpoint")
        if cp and os.path.isdir(cp):
            return cp
    except Exception:
        return None
    return None


# ── Ray bootstrap ─────────────────────────────────────────────────────────────


def _init_ray_and_models():
    import ray
    from ray import tune
    from ray.rllib.models import ModelCatalog
    from models.mappo_cnn_model import MAPPOCNNModelCentralizedCritic
    from models.ippo_cnn_model import IPPOCNNModelDecentralizedCritic
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)
    tune.register_env("harvest", lambda cfg: HarvestEnv(cfg))
    ModelCatalog.register_custom_model("mappo_cnn_centralized", MAPPOCNNModelCentralizedCritic)
    ModelCatalog.register_custom_model("ippo_cnn_decentralized", IPPOCNNModelDecentralizedCritic)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Visualize Harvest policy behaviour as GIFs")
    p.add_argument("--checkpoint", type=str, help="Trained checkpoint dir (single mode).")
    p.add_argument("--greedy", action="store_true", help="Render the FixedGreedy baseline.")
    p.add_argument("--auto", action="store_true",
                   help="Render all trained conditions (from results stamps) + greedy + 2 panels.")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--label", type=str, default=None, help="Banner label (single mode).")
    p.add_argument("--out", type=str, default="harvest_behavior.gif", help="Output GIF (single mode).")
    p.add_argument("--out-dir", type=str, default="metrics/viz", help="Output dir (--auto).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-frames", type=int, default=250)
    p.add_argument("--fps", type=int, default=12)
    args = p.parse_args()

    if args.auto:
        results_dir = Path(args.results_dir)
        out_dir = Path(args.out_dir)
        _init_ray_and_models()

        cache: Dict[str, List[np.ndarray]] = {}
        for stem, tag, label in _CONDITIONS:
            cp = _find_checkpoint(results_dir, stem, args.seed)
            if cp is None:
                print(f"SKIP {tag}: no checkpoint stamp for seed {args.seed}.")
                continue
            print(f"\n=== {label} ===\n  checkpoint: {cp}")
            frames = rollout_trained(cp, args.seed, args.max_frames, label)
            cache[tag] = frames
            write_gif(frames, str(out_dir / f"{tag}.gif"), args.fps)

        print("\n=== FixedGreedy ===")
        greedy_frames = rollout_greedy(args.seed, args.max_frames)
        cache["greedy"] = greedy_frames
        write_gif(greedy_frames, str(out_dir / "greedy.gif"), args.fps)

        # Panels: one per algorithm (individual | mixed | team | greedy).
        for algo in ("mappo", "ippo"):
            quad = [f"{algo}_individual", f"{algo}_mixed", f"{algo}_team", "greedy"]
            if all(q in cache for q in quad):
                panel = make_panel([cache[q] for q in quad])
                write_gif(panel, str(out_dir / f"{algo}_panel.gif"), args.fps)
            else:
                missing = [q for q in quad if q not in cache]
                print(f"SKIP {algo}_panel: missing {missing}")
        print(f"\nDone. GIFs in {out_dir}/")
        return

    if args.greedy:
        frames = rollout_greedy(args.seed, args.max_frames, args.label or "FixedGreedy")
        write_gif(frames, args.out, args.fps)
        return

    if args.checkpoint:
        _init_ray_and_models()
        label = args.label or Path(args.checkpoint).parts[-2]
        frames = rollout_trained(args.checkpoint, args.seed, args.max_frames, label)
        write_gif(frames, args.out, args.fps)
        return

    p.error("Specify one of --auto, --greedy, or --checkpoint.")


if __name__ == "__main__":
    main()
