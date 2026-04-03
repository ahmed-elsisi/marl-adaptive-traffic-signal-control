"""
5×5 SUMO Network Generator for MAPPO Traffic Signal Control

Generates all SUMO XML files for a 5×5 grid of 25 signalized junctions and
prints the YAML config blocks (detectors, network_topology, edge_connectivity)
needed for mappo_config_5x5.yaml.

Grid layout (row 0 = top, row 4 = bottom):
  J1  J2  J3  J4  J5    row 0  y=1500  x=300–1500
  J6  J7  J8  J9  J10   row 1  y=1200
  J11 J12 J13 J14 J15   row 2  y=900
  J16 J17 J18 J19 J20   row 3  y=600
  J21 J22 J23 J24 J25   row 4  y=300

Junction ID:  J{5*row + col + 1}  (row/col 0-indexed)
Spacing:      300m between junctions; 200m to boundary nodes

Edge naming:
  H_{r}_{c}  — J(r,c) → J(r,c+1), east   (r=0..4, c=0..3)
 -H_{r}_{c}  — J(r,c+1) → J(r,c), west   (reverse of above)
  V_{r}_{c}  — J(r,c) → J(r+1,c), south  (r=0..3, c=0..4)
 -V_{r}_{c}  — J(r+1,c) → J(r,c), north  (reverse of above)
  BN_{c}     — boundary_N_{c} → J(0,c),   south entry
 -BN_{c}     — J(0,c) → boundary_N_{c},   north exit
  BS_{c}     — J(4,c) → boundary_S_{c},   south exit
 -BS_{c}     — boundary_S_{c} → J(4,c),   north entry
  BW_{r}     — boundary_W_{r} → J(r,0),   east entry
 -BW_{r}     — J(r,0) → boundary_W_{r},   west exit
  BE_{r}     — J(r,4) → boundary_E_{r},   east exit
 -BE_{r}     — boundary_E_{r} → J(r,4),   west entry

Usage:
  cd RP-5
  python generate_5x5_network.py

Outputs (all in sumo_network_5x5/):
  5x5-grid.nod.xml, 5x5-grid.edg.xml, 5x5-grid.con.xml
  5x5-grid.net.xml  (via netconvert)
  5x5-grid.ttl.xml, 5x5-grid.add.xml, 5x5-grid.rou.xml
  5x5-grid.sumocfg
"""

import os
import subprocess
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROWS = 5
COLS = 5
SPACING = 300           # metres between junctions
BOUNDARY_OFFSET = 200   # metres from edge junctions to boundary nodes
X0 = 300                # x-coord of col 0
Y0 = 1500               # y-coord of row 0 (top)

LANES = 3
SPEED = 13.89           # m/s ≈ 50 km/h

# 8-phase TL program (same structure as 2×2 network)
# Phases 0, 2, 4, 6 are the 4 green phases used as actions.
# Each state string has 12 chars = 4 incoming edges × 3 lanes.
TL_PHASES = [
    ('25', 'GGrgrrGGrgrr'),   # Phase 0: NS through + right  (action 0)
    ('3',  'GrrGyrGrrGyr'),   # Phase 1: yellow transition
    ('25', 'GrrGgrGrrGGr'),   # Phase 2: EW through + right  (action 2)
    ('3',  'GrrGryGrrGry'),   # Phase 3: yellow transition
    ('8',  'GrrGrGGrrGrG'),   # Phase 4: EW left turns        (action 3)
    ('3',  'GryGrrGryGrr'),   # Phase 5: yellow transition
    ('8',  'GrGGrrGrGGrr'),   # Phase 6: NS left turns        (action 1)
    ('3',  'GyrgrrGyrgrr'),   # Phase 7: yellow transition
]


# ---------------------------------------------------------------------------
# Helpers: junction / edge IDs
# ---------------------------------------------------------------------------

def jid(r: int, c: int) -> str:
    """Junction ID for grid position (row r, col c), both 0-indexed."""
    return f"J{r * COLS + c + 1}"


def bn(c):  return f"BN_{c}"
def bs(c):  return f"BS_{c}"
def bw(r):  return f"BW_{r}"
def be(r):  return f"BE_{r}"

def neg(edge: str) -> str:
    """SUMO negation: '-edge' ↔ 'edge'."""
    return edge[1:] if edge.startswith('-') else f"-{edge}"


def h_edge(r, c):   return f"H_{r}_{c}"   # east  J(r,c)→J(r,c+1)
def v_edge(r, c):   return f"V_{r}_{c}"   # south J(r,c)→J(r+1,c)


def node_x(c): return X0 + c * SPACING
def node_y(r): return Y0 - r * SPACING


# ---------------------------------------------------------------------------
# Per-junction helpers
# ---------------------------------------------------------------------------

def incoming_edges(r: int, c: int) -> list:
    """
    Return the 4 incoming edge IDs for junction J(r,c) in the fixed order:
      [from_N, from_S, from_W, from_E]
    This order matches the TL state string positions (3 chars each, 12 total).
    """
    from_n = bn(c)          if r == 0     else v_edge(r - 1, c)
    from_s = neg(bs(c))     if r == ROWS - 1 else neg(v_edge(r, c))
    from_w = bw(r)          if c == 0     else h_edge(r, c - 1)
    from_e = neg(be(r))     if c == COLS - 1 else neg(h_edge(r, c))
    return [from_n, from_s, from_w, from_e]


def outgoing_edges_for_dir(r: int, c: int) -> dict:
    """
    Return the outgoing edge for each cardinal direction from J(r,c).
    """
    return {
        'N': neg(bn(c))       if r == 0         else neg(v_edge(r - 1, c)),
        'S': bs(c)            if r == ROWS - 1  else v_edge(r, c),
        'W': neg(bw(r))       if c == 0         else neg(h_edge(r, c - 1)),
        'E': be(r)            if c == COLS - 1  else h_edge(r, c),
    }


# For each incoming direction, map lane → outgoing direction
# Lane 0 = right turn, Lane 1 = straight, Lane 2 = left turn
TURN_MAP = {
    'N': {0: 'W', 1: 'S', 2: 'E'},   # incoming from north, traveling south
    'S': {0: 'E', 1: 'N', 2: 'W'},   # incoming from south, traveling north
    'W': {0: 'N', 1: 'E', 2: 'S'},   # incoming from west, traveling east
    'E': {0: 'S', 1: 'W', 2: 'N'},   # incoming from east, traveling west
}

INCOMING_DIRS = ['N', 'S', 'W', 'E']


# ---------------------------------------------------------------------------
# Neighbor assignment (2 neighbors per junction, keep obs at 70 dims)
# ---------------------------------------------------------------------------

def get_neighbors(r: int, c: int) -> list:
    """
    Primary = East neighbor (col+1); rightmost col uses West (col-1).
    Secondary = South neighbor (row+1); bottom row uses North (row-1).
    Returns list of (neighbor_r, neighbor_c) tuples.
    """
    if c < COLS - 1:
        n1 = (r, c + 1)
    else:
        n1 = (r, c - 1)

    if r < ROWS - 1:
        n2 = (r + 1, c)
    else:
        n2 = (r - 1, c)

    return [n1, n2]


def edge_between(r1, c1, r2, c2):
    """Return (j1→j2 edge, j2→j1 edge) for adjacent junctions."""
    if r1 == r2:
        # Same row — horizontal
        if c2 == c1 + 1:
            return h_edge(r1, c1), neg(h_edge(r1, c1))
        else:
            return neg(h_edge(r1, c2)), h_edge(r1, c2)
    else:
        # Same col — vertical
        if r2 == r1 + 1:
            return v_edge(r1, c1), neg(v_edge(r1, c1))
        else:
            return neg(v_edge(r2, c1)), v_edge(r2, c1)


# ---------------------------------------------------------------------------
# File generators
# ---------------------------------------------------------------------------

def write_nodes(out_dir: Path):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
             ' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">',
             '']

    # 25 junction nodes
    lines.append('    <!-- Junction nodes -->')
    for r in range(ROWS):
        for c in range(COLS):
            x = node_x(c)
            y = node_y(r)
            lines.append(f'    <node id="{jid(r, c)}" x="{x}" y="{y}" type="traffic_light"/>')
    lines.append('')

    # 20 boundary nodes (type priority, no TL)
    lines.append('    <!-- Boundary nodes -->')
    for c in range(COLS):
        lines.append(f'    <node id="BN_node_{c}" x="{node_x(c)}" y="{Y0 + BOUNDARY_OFFSET}" type="priority"/>')
    for c in range(COLS):
        lines.append(f'    <node id="BS_node_{c}" x="{node_x(c)}" y="{node_y(ROWS-1) - BOUNDARY_OFFSET}" type="priority"/>')
    for r in range(ROWS):
        lines.append(f'    <node id="BW_node_{r}" x="{X0 - BOUNDARY_OFFSET}" y="{node_y(r)}" type="priority"/>')
    for r in range(ROWS):
        lines.append(f'    <node id="BE_node_{r}" x="{node_x(COLS-1) + BOUNDARY_OFFSET}" y="{node_y(r)}" type="priority"/>')

    lines += ['', '</nodes>', '']
    (out_dir / '5x5-grid.nod.xml').write_text('\n'.join(lines), encoding='utf-8')
    print('  [OK] 5x5-grid.nod.xml')


def write_edges(out_dir: Path):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
             ' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">',
             '']

    def edge_tag(eid, frm, to, nl=LANES, spd=SPEED, pri=1):
        return (f'    <edge id="{eid}" from="{frm}" to="{to}"'
                f' numLanes="{nl}" speed="{spd:.2f}" priority="{pri}"/>')

    # Internal horizontal edges (bidirectional)
    lines.append('    <!-- Internal horizontal edges -->')
    for r in range(ROWS):
        for c in range(COLS - 1):
            j_from = jid(r, c)
            j_to   = jid(r, c + 1)
            eid    = h_edge(r, c)
            lines.append(edge_tag(eid,       j_from, j_to))
            lines.append(edge_tag(neg(eid),  j_to,   j_from))
        lines.append('')

    # Internal vertical edges (bidirectional)
    lines.append('    <!-- Internal vertical edges -->')
    for r in range(ROWS - 1):
        for c in range(COLS):
            j_from = jid(r, c)
            j_to   = jid(r + 1, c)
            eid    = v_edge(r, c)
            lines.append(edge_tag(eid,       j_from, j_to))
            lines.append(edge_tag(neg(eid),  j_to,   j_from))
        lines.append('')

    # Boundary edges (bidirectional: entry + exit)
    lines.append('    <!-- Boundary edges (top — north) -->')
    for c in range(COLS):
        j = jid(0, c)
        lines.append(edge_tag(bn(c),       f"BN_node_{c}", j))   # entry from north
        lines.append(edge_tag(neg(bn(c)),  j, f"BN_node_{c}"))   # exit  to   north

    lines.append('')
    lines.append('    <!-- Boundary edges (bottom — south) -->')
    for c in range(COLS):
        j = jid(ROWS - 1, c)
        lines.append(edge_tag(bs(c),       j, f"BS_node_{c}"))   # exit  to   south
        lines.append(edge_tag(neg(bs(c)),  f"BS_node_{c}", j))   # entry from south

    lines.append('')
    lines.append('    <!-- Boundary edges (left — west) -->')
    for r in range(ROWS):
        j = jid(r, 0)
        lines.append(edge_tag(bw(r),       f"BW_node_{r}", j))   # entry from west
        lines.append(edge_tag(neg(bw(r)),  j, f"BW_node_{r}"))   # exit  to   west

    lines.append('')
    lines.append('    <!-- Boundary edges (right — east) -->')
    for r in range(ROWS):
        j = jid(r, COLS - 1)
        lines.append(edge_tag(be(r),       j, f"BE_node_{r}"))   # exit  to   east
        lines.append(edge_tag(neg(be(r)),  f"BE_node_{r}", j))   # entry from east

    lines += ['', '</edges>', '']
    (out_dir / '5x5-grid.edg.xml').write_text('\n'.join(lines), encoding='utf-8')
    print('  [OK] 5x5-grid.edg.xml')


def write_connections(out_dir: Path):
    """
    12 connections per junction (4 directions × 3 lanes).
    Order: North incoming (lanes 0,1,2), South (0,1,2), West (0,1,2), East (0,1,2).
    This consistent ordering lets us use the same TL state strings everywhere.
    """
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<connections xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
             ' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/connections_file.xsd">',
             '']

    for r in range(ROWS):
        for c in range(COLS):
            lines.append(f'    <!-- {jid(r, c)} -->')
            inc = incoming_edges(r, c)                      # [N, S, W, E]
            out = outgoing_edges_for_dir(r, c)              # {N, S, W, E}
            dirs = INCOMING_DIRS                             # ['N','S','W','E']

            for idx, direction in enumerate(dirs):
                from_edge = inc[idx]
                for lane in range(LANES):
                    to_dir = TURN_MAP[direction][lane]
                    to_edge = out[to_dir]
                    lines.append(
                        f'    <connection from="{from_edge}" to="{to_edge}"'
                        f' fromLane="{lane}" toLane="{lane}"/>'
                    )
            lines.append('')

    lines += ['</connections>', '']
    (out_dir / '5x5-grid.con.xml').write_text('\n'.join(lines), encoding='utf-8')
    print('  [OK] 5x5-grid.con.xml')


def run_netconvert(out_dir: Path):
    sumo_home = os.environ.get('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo')
    netconvert = os.path.join(sumo_home, 'bin', 'netconvert.exe')
    if not os.path.exists(netconvert):
        netconvert = 'netconvert'   # fallback: hope it's on PATH

    cmd = [
        netconvert,
        '--node-files',       str(out_dir / '5x5-grid.nod.xml'),
        '--edge-files',       str(out_dir / '5x5-grid.edg.xml'),
        '--connection-files', str(out_dir / '5x5-grid.con.xml'),
        '--output-file',      str(out_dir / '5x5-grid.net.xml'),
        '--no-turnarounds',   'true',
        '--tls.default-type', 'static',
        '--tls.layout',       'opposites',
    ]
    print('  Running netconvert...')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print('  [ERROR] netconvert failed:')
        print(result.stderr[-2000:])
        raise RuntimeError('netconvert failed')
    # NOTE: we keep the netconvert-generated <tlLogic programID="0"> in net.xml.
    # Those entries also define the tl= link-index assignments in <connection>
    # elements that SUMO requires at net-load time.
    # Our custom 8-phase programs use programID="1" in ttl.xml to avoid conflict.
    print('  [OK] 5x5-grid.net.xml')


def write_tl_logic(out_dir: Path):
    lines = ['<additionals>']
    for r in range(ROWS):
        for c in range(COLS):
            nxt = [str((i + 1) % len(TL_PHASES)) for i in range(len(TL_PHASES))]
            lines.append(f'    <tlLogic id="{jid(r, c)}" type="static" programID="1" offset="0">')
            for i, (dur, state) in enumerate(TL_PHASES):
                lines.append(f'        <phase duration="{dur}"  state="{state}" next="{nxt[i]}"/>')
            lines.append('    </tlLogic>')
    lines += ['', '</additionals>', '']
    (out_dir / '5x5-grid.ttl.xml').write_text('\n'.join(lines), encoding='utf-8')
    print('  [OK] 5x5-grid.ttl.xml')


def write_detectors(out_dir: Path):
    """
    One laneAreaDetector per incoming lane (4 edges × 3 lanes × 25 junctions = 300).

    pos="-100" places each detector in the LAST 100m of the lane, right before
    the junction (SUMO negative pos counts from the end of the lane).
    friendlyPos="true" clamps the detector to valid positions for short edges.
    """
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
             ' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">',
             '']

    for r in range(ROWS):
        for c in range(COLS):
            lines.append(f'    <!-- {jid(r, c)} -->')
            for edge in incoming_edges(r, c):
                for lane in range(LANES):
                    det_id  = f"det_{edge}_{lane}_stop"
                    lane_id = f"{edge}_{lane}"
                    lines.append(
                        f'    <laneAreaDetector id="{det_id}" lane="{lane_id}"'
                        f' pos="-100.00" length="100.00" friendlyPos="true" file="NUL"/>'
                    )
            lines.append('')

    lines += ['</additional>', '']
    (out_dir / '5x5-grid.add.xml').write_text('\n'.join(lines), encoding='utf-8')
    print('  [OK] 5x5-grid.add.xml')


def write_routes(out_dir: Path):
    """
    Vehicle demand scaled for 5×5 grid (5× more boundary entry points).
    3 temporal phases:
      Low  (    0– 600s): 10 N-S flows + 10 W-E flows, period=40s
      Peak ( 600–2400s):  Same axis flows at period=12s + 10 diagonal cross-flows
      Decline (2400–3600s): 10 main axis flows, period=25s
    """
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
             ' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
             '',
             '    <vType id="car" length="5.00" minGap="2.50" maxSpeed="13.90"'
             ' carFollowModel="IDM" accel="3.0" decel="6.0"/>',
             '',
             '    <!-- -- Low demand (0–600s) ----------------------------------- -->',
             ]

    # N-S flows: BN_{c} → BS_{c}
    for c in range(COLS):
        lines.append(
            f'    <flow id="low_NS_{c}" type="car" begin="0" end="600"'
            f' from="{bn(c)}" to="{bs(c)}" period="40"/>'
        )
    # W-E flows: BW_{r} → BE_{r}
    for r in range(ROWS):
        lines.append(
            f'    <flow id="low_WE_{r}" type="car" begin="0" end="600"'
            f' from="{bw(r)}" to="{be(r)}" period="40"/>'
        )

    lines += ['',
              '    <!-- -- Peak demand (600–2400s) -------------------------------- -->']

    # N-S and W-E main-axis at high frequency
    for c in range(COLS):
        lines.append(
            f'    <flow id="peak_NS_{c}" type="car" begin="600" end="2400"'
            f' from="{bn(c)}" to="{bs(c)}" period="10"/>'
        )
        lines.append(
            f'    <flow id="peak_SN_{c}" type="car" begin="600" end="2400"'
            f' from="{neg(bs(c))}" to="{neg(bn(c))}" period="12"/>'
        )
    for r in range(ROWS):
        lines.append(
            f'    <flow id="peak_WE_{r}" type="car" begin="600" end="2400"'
            f' from="{bw(r)}" to="{be(r)}" period="10"/>'
        )
        lines.append(
            f'    <flow id="peak_EW_{r}" type="car" begin="600" end="2400"'
            f' from="{neg(be(r))}" to="{neg(bw(r))}" period="12"/>'
        )

    # 10 diagonal cross-flows (N-boundary col → E/W-boundary, and W-boundary row → S-boundary)
    for i in range(5):
        c_src = i
        r_dst = (i + 2) % ROWS
        lines.append(
            f'    <flow id="peak_diag_NE_{i}" type="car" begin="600" end="2400"'
            f' from="{bn(c_src)}" to="{be(r_dst)}" period="15"/>'
        )
        r_src = i
        c_dst = (i + 2) % COLS
        lines.append(
            f'    <flow id="peak_diag_WS_{i}" type="car" begin="600" end="2400"'
            f' from="{bw(r_src)}" to="{bs(c_dst)}" period="15"/>'
        )

    lines += ['',
              '    <!-- -- Decline (2400–3600s) ----------------------------------- -->']

    for c in range(COLS):
        lines.append(
            f'    <flow id="decline_NS_{c}" type="car" begin="2400" end="3600"'
            f' from="{bn(c)}" to="{bs(c)}" period="25"/>'
        )
    for r in range(ROWS):
        lines.append(
            f'    <flow id="decline_WE_{r}" type="car" begin="2400" end="3600"'
            f' from="{bw(r)}" to="{be(r)}" period="25"/>'
        )

    lines += ['', '</routes>', '']
    (out_dir / '5x5-grid.rou.xml').write_text('\n'.join(lines), encoding='utf-8')
    print('  [OK] 5x5-grid.rou.xml')


def write_sumocfg(out_dir: Path):
    content = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <sumoConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
            <input>
                <net-file value="5x5-grid.net.xml"/>
                <route-files value="5x5-grid.rou.xml"/>
                <additional-files value="5x5-grid.add.xml,5x5-grid.ttl.xml"/>
            </input>
        </sumoConfiguration>
    """)
    (out_dir / '5x5-grid.sumocfg').write_text(content, encoding='utf-8')
    print('  [OK] 5x5-grid.sumocfg')


# ---------------------------------------------------------------------------
# YAML section printers
# ---------------------------------------------------------------------------

def print_yaml_agents():
    print("\n# -- agents ----------------------------------------------------------")
    print("agents:")
    for r in range(ROWS):
        for c in range(COLS):
            print(f'  - "{jid(r, c)}"')


def print_yaml_network_topology():
    print("\n# -- network_topology ------------------------------------------------")
    print("network_topology:")
    for r in range(ROWS):
        for c in range(COLS):
            neighbors = get_neighbors(r, c)
            neighbor_ids = [f'"{jid(nr, nc)}"' for nr, nc in neighbors]
            print(f'  {jid(r, c)}: [{", ".join(neighbor_ids)}]')


def print_yaml_detectors():
    print("\n# -- detectors -------------------------------------------------------")
    print("detectors:")
    for r in range(ROWS):
        for c in range(COLS):
            inc = incoming_edges(r, c)   # [from_N, from_S, from_W, from_E]
            ns_edges  = [inc[0], inc[1]]  # N and S incoming
            ew_edges  = [inc[2], inc[3]]  # W and E incoming
            inc_str   = ', '.join(f'"{e}"' for e in inc)
            ns_str    = ', '.join(f'"{e}"' for e in ns_edges)
            ew_str    = ', '.join(f'"{e}"' for e in ew_edges)
            print(f'  {jid(r, c)}:')
            print(f'    incoming_edges: [{inc_str}]')
            print(f'    movements: [0, 1, 2]')
            print(f'    ns_edges: [{ns_str}]')
            print(f'    ew_edges: [{ew_str}]')


def print_yaml_edge_connectivity():
    """
    edge_connectivity structure (matches obs_builder.py convention):
      outgoing: edges FROM neighbor TO this junction
      ingoing:  edges FROM this junction TO neighbor
    """
    print("\n# -- edge_connectivity -----------------------------------------------")
    print("edge_connectivity:")
    for r in range(ROWS):
        for c in range(COLS):
            jname = jid(r, c)
            neighbors = get_neighbors(r, c)
            print(f'  {jname}:')
            for nr, nc in neighbors:
                nname = jid(nr, nc)
                # j_to_n: edge from J(r,c) → J(nr,nc)
                # n_to_j: edge from J(nr,nc) → J(r,c)
                j_to_n, n_to_j = edge_between(r, c, nr, nc)
                print(f'    {nname}:')
                print(f'      outgoing: ["{n_to_j}"]')   # neighbor→this
                print(f'      ingoing:  ["{j_to_n}"]')   # this→neighbor


def print_yaml_model_config():
    print("\n# -- model_config (custom_model_config excerpt) ----------------------")
    print("model_config:")
    print("  custom_model: \"mappo_centralized\"")
    print("  custom_model_config:")
    print(f"    num_agents: {ROWS * COLS}")
    agent_ids = [f'"{jid(r, c)}"' for r in range(ROWS) for c in range(COLS)]
    print(f"    agent_ids: [{', '.join(agent_ids)}]")
    print("    actor_hiddens: [128, 64]")
    print("    actor_activation: \"tanh\"")
    print("    critic_hiddens: [512, 256, 128]   # scaled for 1750-dim global state")
    print("    critic_activation: \"relu\"")
    print("    use_orthogonal_init: true")
    print("    orthogonal_gain: 0.01")
    print("    use_value_normalization: true")
    print("    use_lstm: false")
    print("    lstm_cell_size: 64")
    print("    max_seq_len: 20")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = Path(__file__).parent
    out_dir = script_dir / 'sumo_network_5x5'
    out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("5×5 SUMO Network Generator")
    print(f"Output directory: {out_dir}")
    print("=" * 70)

    print("\n[1/8] Writing node file...")
    write_nodes(out_dir)

    print("\n[2/8] Writing edge file...")
    write_edges(out_dir)

    print("\n[3/8] Writing connection file...")
    write_connections(out_dir)

    print("\n[4/8] Running netconvert...")
    try:
        run_netconvert(out_dir)
    except RuntimeError:
        print("\n  *** netconvert failed. Fix errors above, then re-run. ***")
        print("  Continuing with remaining file generation...")

    print("\n[5/8] Writing TL logic (phase programs)...")
    write_tl_logic(out_dir)

    print("\n[6/8] Writing detectors...")
    write_detectors(out_dir)

    print("\n[7/8] Writing route file...")
    write_routes(out_dir)

    print("\n[8/8] Writing sumocfg...")
    write_sumocfg(out_dir)

    print("\n" + "=" * 70)
    print("File generation complete.")
    print("=" * 70)

    print("\n\n" + "=" * 70)
    print("YAML CONFIG BLOCKS")
    print("(Copy these into configs/mappo_config_5x5.yaml)")
    print("=" * 70)

    print_yaml_agents()
    print_yaml_network_topology()
    print_yaml_detectors()
    print_yaml_edge_connectivity()
    print_yaml_model_config()

    print("\n\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Validate visually:")
    print(f'   sumo-gui.exe -c "{out_dir}/5x5-grid.sumocfg"')
    print("2. Check each junction has exactly 12 TL-controlled links in net.xml")
    print("3. Copy YAML blocks above into configs/mappo_config_5x5.yaml")
    print("4. Run smoke test: python train_mappo.py --config configs/mappo_config_5x5.yaml --iterations 1")
    print()


if __name__ == '__main__':
    main()
