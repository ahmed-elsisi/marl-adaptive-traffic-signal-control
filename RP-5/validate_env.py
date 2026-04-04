"""
Environment Validation Script
==============================
Runs one short simulation episode and prints a full per-step, per-agent
breakdown: raw SUMO values, normalized observations, and reward components.

Purpose: sanity-check that queue readings, pressure calculations, neighbor
         features, and reward arithmetic are all correct before training.

Usage:
    # 2x2 network (default)
    python validate_env.py

    # 5x5 network
    python validate_env.py --config configs/mappo_config_5x5.yaml

    # more steps, e.g. inspect at peak traffic
    python validate_env.py --steps 20 --start-step 120

Options:
    --config        Path to YAML config  (default: configs/mappo_config_v2.yaml)
    --steps         Number of steps to print  (default: 5)
    --start-step    Skip this many steps silently first (default: 0)
    --seed          Random seed  (default: 42)
"""

import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path

# -- allow importing from RP-5/ directly -------------------------------------
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

os.environ['LIBSUMO_AS_TRACI'] = '0'   # always use standard TraCI here

import traci
from marl_env.sumo_env import SUMOTrafficEnv


# ============================================================================
# Helpers
# ============================================================================

SEP  = '-' * 72
SEP2 = '=' * 72

def _hdr(text):
    pad = max(0, 70 - len(text))
    return f'\n{SEP}\n  {text}{" " * pad}\n{SEP}'

def _sub(text):
    return f'  -- {text}'


# ============================================================================
# Raw SUMO queries (mirrors internal env methods but returns verbose dicts)
# ============================================================================

def get_queue_details(agent_id, detector_mappings):
    """Return {det_id: jam_length_vehicles} for every detector of this agent."""
    out = {}
    for det_id in detector_mappings[agent_id]:
        try:
            q = traci.lanearea.getJamLengthVehicle(det_id)
        except Exception:
            q = 0
        out[det_id] = q
    return out


def get_waiting_time_details(agent_id, detector_mappings):
    """Return total accumulated waiting time (s) across all vehicles in detectors."""
    total = 0.0
    veh_count = 0
    for det_id in detector_mappings[agent_id]:
        try:
            veh_ids = traci.lanearea.getLastStepVehicleIDs(det_id)
            for v in veh_ids:
                w = traci.vehicle.getAccumulatedWaitingTime(v)
                total += w
                veh_count += 1
        except Exception:
            pass
    return total, veh_count


def get_edge_counts(edge_list):
    """Return {edge_id: vehicle_count} for a list of edge IDs."""
    out = {}
    for eid in edge_list:
        count = 0
        try:
            nlanes = traci.edge.getLaneNumber(eid)
            for i in range(nlanes):
                try:
                    count += traci.lane.getLastStepVehicleNumber(f'{eid}_{i}')
                except Exception:
                    pass
        except Exception:
            pass
        out[eid] = count
    return out


def get_tl_state(agent_id):
    """Return (current_phase_index, elapsed_seconds, phase_state_string)."""
    try:
        phase   = traci.trafficlight.getPhase(agent_id)
        elapsed = traci.trafficlight.getPhaseDuration(agent_id) \
                  - traci.trafficlight.getNextSwitch(agent_id) \
                  + traci.simulation.getTime()
        # getSpentDuration is cleaner if available
        try:
            elapsed = traci.trafficlight.getSpentDuration(agent_id)
        except Exception:
            pass
        state_str = traci.trafficlight.getRedYellowGreenState(agent_id)
        return phase, round(elapsed, 1), state_str
    except Exception:
        return 0, 0.0, '?'


# ============================================================================
# Observation breakdown
# ============================================================================

OBS_LABELS_LOCAL = (
    # Queue lengths (12)
    [f'queue_{i}' for i in range(12)] +
    # Direction one-hot (2)
    ['dir_NS', 'dir_EW'] +
    # Elapsed phase time (1)
    ['elapsed_time'] +
    # Movement pressures NS (3)
    ['press_NS_right', 'press_NS_straight', 'press_NS_left'] +
    # Movement pressures EW (3)
    ['press_EW_right', 'press_EW_straight', 'press_EW_left'] +
    # Pressure derivatives (6)
    ['dpress_NS_right', 'dpress_NS_straight', 'dpress_NS_left',
     'dpress_EW_right', 'dpress_EW_straight', 'dpress_EW_left'] +
    # Min-green flag (1)
    ['min_green_ok']
)

def neighbor_obs_labels(n):
    """Labels for neighbor n (0-indexed), 21 values each."""
    pfx = f'nbr{n+1}_'
    return [
        pfx+'queue_right', pfx+'queue_straight', pfx+'queue_left',
        pfx+'dir_NS',      pfx+'dir_EW',
        pfx+'press_right', pfx+'press_straight', pfx+'press_left',
        pfx+'total_press',
        pfx+'out_q_right', pfx+'out_q_straight', pfx+'out_q_left',
        pfx+'out_w_right', pfx+'out_w_straight', pfx+'out_w_left',
        pfx+'in_q_right',  pfx+'in_q_straight',  pfx+'in_q_left',
        pfx+'in_sp_right', pfx+'in_sp_straight',  pfx+'in_sp_left',
    ]


def print_observation(agent_id, obs, topology):
    """Pretty-print the full 70-dim observation with labels."""
    num_neighbors = len(topology.get(agent_id, []))
    neighbor_labels = []
    for n in range(num_neighbors):
        neighbor_labels += neighbor_obs_labels(n)
    all_labels = OBS_LABELS_LOCAL + neighbor_labels

    print(_sub('Observation vector (70 dims, normalized):'))
    # Local block
    print('    [Local features -- 28 dims]')
    local_block = list(zip(all_labels[:28], obs[:28]))
    for i, (lbl, val) in enumerate(local_block):
        marker = ' <' if abs(val) > 0.05 else ''
        print(f'      [{i:2d}] {lbl:<28s} = {val:+.4f}{marker}')
    # Neighbor blocks
    offset = 28
    for n in range(num_neighbors):
        nbr_id = topology.get(agent_id, [])[n] if n < len(topology.get(agent_id, [])) else f'nbr{n}'
        print(f'    [Neighbor {n+1}: {nbr_id} -- 21 dims  (offset {offset})]')
        for i in range(21):
            lbl = all_labels[offset + i] if (offset + i) < len(all_labels) else f'nbr{n}_{i}'
            val = obs[offset + i]
            marker = ' <' if abs(val) > 0.05 else ''
            print(f'      [{offset+i:2d}] {lbl:<28s} = {val:+.4f}{marker}')
        offset += 21


# ============================================================================
# Reward breakdown
# ============================================================================

def compute_reward_breakdown(agent_id, env, reward_cfg, norm_cfg):
    """
    Re-compute every reward component with verbose intermediate values.
    Returns a dict of named intermediate values + final reward.
    """
    rf   = env.reward_function
    det  = env.detector_mappings[agent_id]
    topo = env.network_topology
    edge_map = rf.edge_mappings.get(agent_id, {})

    result = {}

    # -- 1. Queue -------------------------------------------------------------
    total_q = 0
    per_det = {}
    for d in det:
        try:
            q = traci.lanearea.getJamLengthVehicle(d)
        except Exception:
            q = 0
        per_det[d] = q
        total_q += q
    max_q = len(det) * norm_cfg['queue_max']
    norm_q = total_q / max_q if max_q else 0.0
    result['queue_raw']        = total_q
    result['queue_max']        = max_q
    result['queue_norm']       = norm_q
    result['queue_per_det']    = per_det
    result['queue_weighted']   = norm_q * reward_cfg['queue_weight']

    # -- 2. Waiting time -------------------------------------------------------
    total_w, n_vehs = get_waiting_time_details(agent_id, env.detector_mappings)
    max_w = len(det) * norm_cfg['waiting_time_max']
    norm_w = total_w / max_w if max_w else 0.0
    result['wait_raw_s']       = round(total_w, 2)
    result['wait_vehicles']    = n_vehs
    result['wait_norm']        = norm_w
    result['wait_weighted']    = norm_w * reward_cfg['waiting_time_weight']

    # -- 3. Throughput ---------------------------------------------------------
    try:
        departed = traci.simulation.getDepartedNumber()
    except Exception:
        departed = 0
    norm_t = departed / (len(env.agent_ids) * 5.0)
    result['throughput_departed']  = departed
    result['throughput_norm']      = norm_t
    result['throughput_weighted']  = norm_t * reward_cfg['throughput_weight']

    # -- 4. Pressure -----------------------------------------------------------
    inc_edges = edge_map.get('incoming', [])
    out_edges = edge_map.get('outgoing', [])
    inc_counts = get_edge_counts(inc_edges)
    out_counts = get_edge_counts(out_edges)
    total_inc  = sum(inc_counts.values())
    total_out  = sum(out_counts.values())
    pressure   = total_inc - total_out
    pos_pres   = max(0, pressure)
    max_pres   = len(inc_edges) * 3 * norm_cfg['pressure_max']
    norm_pres  = pos_pres / max_pres if max_pres else 0.0
    result['pressure_incoming']  = inc_counts
    result['pressure_outgoing']  = out_counts
    result['pressure_raw']       = pressure
    result['pressure_pos']       = pos_pres
    result['pressure_norm']      = norm_pres
    result['pressure_weighted']  = norm_pres * reward_cfg['pressure_weight']

    # -- 5. Neighbor pressure --------------------------------------------------
    neighbors = topo.get(agent_id, [])
    nbr_details = {}
    total_nbr_pres = 0.0
    for nbr in neighbors:
        nbr_map    = rf.edge_mappings.get(nbr, {})
        nbr_inc    = sum(get_edge_counts(nbr_map.get('incoming', [])).values())
        nbr_out    = sum(get_edge_counts(nbr_map.get('outgoing', [])).values())
        nbr_pres   = max(0, nbr_inc - nbr_out)
        discounted = nbr_pres * norm_cfg['spatial_discount']
        nbr_details[nbr] = {
            'incoming': nbr_inc, 'outgoing': nbr_out,
            'pressure': nbr_inc - nbr_out,
            'pos_pressure': nbr_pres, 'discounted': discounted
        }
        total_nbr_pres += discounted
    max_nbr = len(neighbors) * 4 * 3 * norm_cfg['pressure_max'] if neighbors else 1
    norm_nbr = total_nbr_pres / max_nbr if max_nbr else 0.0
    result['neighbor_details']   = nbr_details
    result['neighbor_norm']      = norm_nbr
    result['neighbor_weighted']  = norm_nbr * reward_cfg['neighbor_pressure_weight']

    # -- Total -----------------------------------------------------------------
    components = {
        'queue':     result['queue_weighted'],
        'waiting':   result['wait_weighted'],
        'throughput':result['throughput_weighted'],
        'pressure':  result['pressure_weighted'],
        'neighbor':  result['neighbor_weighted'],
    }
    total   = sum(components.values())
    clipped = float(np.clip(total, reward_cfg['clip_min'], reward_cfg['clip_max']))
    result['components'] = components
    result['total_raw']  = total
    result['total_clipped'] = clipped
    return result


# ============================================================================
# Per-agent printer
# ============================================================================

def print_agent_step(agent_id, obs, env, reward_cfg, norm_cfg, topology, step):
    """Print full diagnostic for one agent at one step."""
    print(f'\n  +- Agent {agent_id} -----------------------------------------------')

    # TL state
    phase, elapsed, state_str = get_tl_state(agent_id)
    direction = 'NS' if phase in (0, 6) else 'EW'
    print(f'  |  TL: phase={phase}  elapsed={elapsed}s  direction={direction}')
    print(f'  |  signal state: {state_str}')

    # -- Reward breakdown -----------------------------------------------------
    rb = compute_reward_breakdown(agent_id, env, reward_cfg, norm_cfg)
    print(f'  |')
    print(f'  |  [Raw SUMO values]')

    # Queues per detector
    q_total = rb['queue_raw']
    print(f'  |    Queue (halted vehicles):  total={q_total}  '
          f'norm={rb["queue_norm"]:.4f}  (max={rb["queue_max"]})')
    for det_id, q in rb['queue_per_det'].items():
        if q > 0:
            print(f'  |      {det_id:<42s} = {q}')

    # Waiting time
    print(f'  |    Waiting time:  total={rb["wait_raw_s"]:.1f}s  '
          f'vehicles={rb["wait_vehicles"]}  norm={rb["wait_norm"]:.4f}')

    # Throughput
    print(f'  |    Throughput:  departed={rb["throughput_departed"]}  '
          f'norm={rb["throughput_norm"]:.4f}')

    # Pressure
    print(f'  |    Pressure (incoming - outgoing):')
    for eid, cnt in rb['pressure_incoming'].items():
        out_cnt = rb['pressure_outgoing'].get(
            eid[1:] if eid.startswith('-') else f'-{eid}', 0)
        print(f'  |      inc {eid:<20s} = {cnt:3d}')
    for eid, cnt in rb['pressure_outgoing'].items():
        print(f'  |      out {eid:<20s} = {cnt:3d}')
    print(f'  |      raw={rb["pressure_raw"]:+.0f}  '
          f'positive={rb["pressure_pos"]:.0f}  '
          f'norm={rb["pressure_norm"]:.4f}')

    # Neighbor pressure
    print(f'  |    Neighbor pressure (spatial_discount={norm_cfg["spatial_discount"]}):')
    for nbr, nd in rb['neighbor_details'].items():
        print(f'  |      {nbr}: inc={nd["incoming"]}  out={nd["outgoing"]}  '
              f'pressure={nd["pressure"]:+.0f}  '
              f'discounted={nd["discounted"]:.4f}')
    print(f'  |      norm={rb["neighbor_norm"]:.4f}')

    # -- Reward formula -------------------------------------------------------
    print(f'  |')
    print(f'  |  [Reward calculation]')
    comp = rb['components']
    w = reward_cfg
    print(f'  |    queue     : {rb["queue_norm"]:.4f} x ({w["queue_weight"]})         '
          f'= {comp["queue"]:+.5f}')
    print(f'  |    waiting   : {rb["wait_norm"]:.4f} x ({w["waiting_time_weight"]})         '
          f'= {comp["waiting"]:+.5f}')
    print(f'  |    throughput: {rb["throughput_norm"]:.4f} x (+{w["throughput_weight"]})         '
          f'= {comp["throughput"]:+.5f}')
    print(f'  |    pressure  : {rb["pressure_norm"]:.4f} x ({w["pressure_weight"]})         '
          f'= {comp["pressure"]:+.5f}')
    print(f'  |    neighbor  : {rb["neighbor_norm"]:.4f} x ({w["neighbor_pressure_weight"]})         '
          f'= {comp["neighbor"]:+.5f}')
    print(f'  |    ---------------------------------------------------------')
    print(f'  |    total (unclipped) = {rb["total_raw"]:+.5f}')
    clip_note = '  <- CLIPPED' if abs(rb["total_clipped"] - rb["total_raw"]) > 1e-6 else ''
    print(f'  |    total (clipped)   = {rb["total_clipped"]:+.5f}'
          f'  [clip_min={w["clip_min"]}, clip_max={w["clip_max"]}]{clip_note}')

    # -- Observation ----------------------------------------------------------
    print(f'  |')
    print_observation(agent_id, obs, topology)

    print(f'  +-----------------------------------------------------------------')


# ============================================================================
# Main
# ============================================================================

def load_env_config(yaml_path):
    """Build the env_config dict the same way train_mappo.py does."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    env_cfg = config['env_config']
    env_config = {
        'agents':            config['agents'],
        'network_topology':  config['network_topology'],
        'detectors':         config['detectors'],
        'normalization':     config['normalization'],
        'reward_config':     config['reward_config'],
        'edge_connectivity': config.get('edge_connectivity', {}),
        'network_file':  str(project_root / env_cfg['network_file']),
        'route_file':    str(project_root / env_cfg['route_file']),
        'use_gui':       env_cfg.get('use_gui', False),
        'num_seconds':   env_cfg.get('num_seconds', 3600),
        'delta_time':    env_cfg.get('delta_time', 5),
        'yellow_time':   env_cfg.get('yellow_time', 3),
        'min_green':     env_cfg.get('min_green', 10),
        'max_green':     env_cfg.get('max_green', 50),
        'sumo_seed':     env_cfg.get('sumo_seed', 42),
        'enforce_min_green': env_cfg.get('enforce_min_green', False),
        'tl_program_id': env_cfg.get('tl_program_id', '0'),
    }
    return env_config, config['reward_config'], config['normalization'], config['network_topology']


def main():
    parser = argparse.ArgumentParser(description='Validate MAPPO env -- one episode')
    parser.add_argument('--config',      default='configs/mappo_config_v2.yaml')
    parser.add_argument('--steps',       type=int, default=5,
                        help='Number of steps to print (default: 5)')
    parser.add_argument('--start-step',  type=int, default=0,
                        help='Silent warm-up steps before printing (default: 0)')
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    config_path = project_root / args.config
    print(SEP2)
    print(f'  MAPPO Environment Validator')
    print(f'  Config  : {config_path}')
    print(f'  Steps   : {args.start_step} warm-up + {args.steps} printed')
    print(SEP2)

    env_config, reward_cfg, norm_cfg, topology = load_env_config(config_path)
    agent_ids = env_config['agents']
    print(f'  Agents  : {agent_ids}')
    print(f'  Network : {Path(env_config["network_file"]).name}')
    print(SEP2)

    # -- Create and reset environment -----------------------------------------
    env = SUMOTrafficEnv(env_config)
    obs_dict, _ = env.reset(seed=args.seed)

    # -- Warm-up steps (silent) ------------------------------------------------
    if args.start_step > 0:
        print(f'\n  Running {args.start_step} silent warm-up steps ...')
        actions = {aid: 0 for aid in agent_ids}
        for _ in range(args.start_step):
            obs_dict, _, term, _, _ = env.step(actions)
            if term.get('__all__', False):
                print('  Episode ended during warm-up.')
                env.close()
                return
        print(f'  Warm-up complete.  Sim time = '
              f'{traci.simulation.getTime():.0f}s')

    # -- Diagnostic steps ------------------------------------------------------
    actions = {aid: 0 for aid in agent_ids}   # always stay on phase 0

    for step_idx in range(args.steps):
        sim_time = traci.simulation.getTime()
        n_vehicles = traci.simulation.getMinExpectedNumber()

        print(_hdr(f'Step {args.start_step + step_idx + 1}   '
                   f'sim_time={sim_time:.0f}s   '
                   f'vehicles_in_network={n_vehicles}'))

        for aid in agent_ids:
            obs = obs_dict[aid]
            print_agent_step(aid, obs, env, reward_cfg, norm_cfg, topology, step_idx)

        # Network summary
        net_reward = sum(
            compute_reward_breakdown(aid, env, reward_cfg, norm_cfg)['total_clipped']
            for aid in agent_ids
        )
        print(f'\n  Network mean reward this step: {net_reward / len(agent_ids):+.4f}')

        # Take one step
        obs_dict, reward_dict, term, trunc, info = env.step(actions)
        print(f'  Env-returned rewards: '
              + '  '.join(f'{k}={v:+.4f}' for k, v in reward_dict.items()))

        if term.get('__all__', False):
            print('\n  Episode ended.')
            break

    print(f'\n{SEP2}')
    print('  Validation complete.')
    print(SEP2)
    env.close()


if __name__ == '__main__':
    main()
