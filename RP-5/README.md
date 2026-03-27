# \# SUMO Files Directory

# 

# This directory should contain your SUMO simulation files for the 2×2 grid network.

# 

# \## Required Files

# 

# 1\. \*\*2x2grid.net.xml\*\* - SUMO network file (junctions, edges, lanes, traffic lights)

# 2\. \*\*2x2grid.rou.xml\*\* - Route/vehicle file (traffic demand)

# 3\. \*\*2x2grid.det.xml\*\* - Detector file (loop detectors at stop lines)

# 4\. \*\*2x2grid.sumocfg\*\* - SUMO configuration file (references all above files)

# 

# \## Quick Setup

# 

# \### Option 1: Using netgenerate (Recommended for testing)

# 

# ```bash

# \# Generate 2x2 grid network

# netgenerate --grid \\

# &nbsp;   --grid.number 2 \\

# &nbsp;   --grid.length 200 \\

# &nbsp;   --default.lanenumber 3 \\

# &nbsp;   --default-junction-type traffic\_light \\

# &nbsp;   --tls.guess true \\

# &nbsp;   --junctions.join \\

# &nbsp;   --output-file 2x2grid.net.xml

# 

# \# Generate random traffic

# python $SUMO\_HOME/tools/randomTrips.py \\

# &nbsp;   -n 2x2grid.net.xml \\

# &nbsp;   -r 2x2grid.rou.xml \\

# &nbsp;   -e 3600 \\

# &nbsp;   -p 2.0 \\

# &nbsp;   --fringe-factor 10 \\

# &nbsp;   --min-distance 200 \\

# &nbsp;   --validate

# 

# \# Generate detectors at stop lines

# python $SUMO\_HOME/tools/generateTLSE2Detectors.py \\

# &nbsp;   -n 2x2grid.net.xml \\

# &nbsp;   -o 2x2grid.det.xml \\

# &nbsp;   -d 250 \\

# &nbsp;   -f 60

# 

# \# Test the simulation

# sumo -c 2x2grid.sumocfg --start --quit-on-end

# ```

# 

# \### Option 2: Using NETEDIT (GUI-based)

# 

# ```bash

# \# Open NETEDIT

# netedit

# 

# \# Create network graphically:

# \# 1. Add 4 junctions (J1, J2, J3, J4) in a grid

# \# 2. Set junction types to "traffic\_light"

# \# 3. Connect with 3-lane edges

# \# 4. Save as 2x2grid.net.xml

# 

# \# Then generate routes and detectors as above

# ```

# 

# \## Network Requirements

# 

# \*\*Critical\*\*: Your network must have these junction IDs:

# \- `J1` - Top-left intersection

# \- `J2` - Top-right intersection

# \- `J3` - Bottom-left intersection

# \- `J4` - Bottom-right intersection

# 

# \*\*Topology\*\* (must match config):

# ```

# J1 ─── J2

# │      │

# J3 ─── J4

# ```

# 

# \*\*Detectors\*\*: Each junction needs 12 lane area detectors:

# \- 4 incoming directions × 3 lanes (right, straight, left)

# \- Naming convention: `det\_<edge>\_<lane>\_stop`

# \- Example: `det\_-E6\_0\_stop` for J1's west approach, right lane

# 

# \## Verification

# 

# After generating files, verify they work:

# 

# ```bash

# \# Check network file is valid

# netconvert -s 2x2grid.net.xml --plain-output-prefix test\_

# 

# \# Check routes are valid

# duarouter -n 2x2grid.net.xml -r 2x2grid.rou.xml \\

# &nbsp;   --ignore-errors --no-warnings

# 

# \# Run quick simulation

# sumo -c 2x2grid.sumocfg --duration-log.statistics \\

# &nbsp;   --start --quit-on-end

# 

# \# List traffic light IDs (should show J1, J2, J3, J4)

# sumo -c 2x2grid.sumocfg --tls.all-off \\

# &nbsp;   --duration-log.statistics | grep "traffic light"

# ```

# 

# \## Detector Configuration

# 

# The detector file (2x2grid.det.xml) should have entries like:

# 

# ```xml

# <additional>

# &nbsp;   <laneAreaDetector id="det\_-E6\_0\_stop" lane="-E6\_0" pos="0" endPos="250" freq="60"/>

# &nbsp;   <laneAreaDetector id="det\_-E6\_1\_stop" lane="-E6\_1" pos="0" endPos="250" freq="60"/>

# &nbsp;   <!-- ... more detectors ... -->

# </additional>

# ```

# 

# \*\*Important Parameters\*\*:

# \- `pos="0"` - Start at lane beginning

# \- `endPos="250"` - Detection zone length (meters)

# \- `freq="60"` - Update frequency (seconds) - can be any value, we read on-demand

# 

# \## Traffic Demand

# 

# For testing, use these parameters for `randomTrips.py`:

# 

# ```bash

# \# Light traffic (for initial testing)

# -p 5.0   # Period between vehicles: 5 seconds

# 

# \# Medium traffic (for training)

# -p 2.0   # Period: 2 seconds

# 

# \# Heavy traffic (for evaluation)

# -p 1.0   # Period: 1 second

# ```

# 

# \## Troubleshooting

# 

# \*\*Error\*\*: "Junction J1 not found"

# \- Make sure junction IDs in network match: J1, J2, J3, J4

# \- Check with: `grep 'junction id=' 2x2grid.net.xml`

# 

# \*\*Error\*\*: "Detector not found"

# \- Regenerate detectors with correct naming

# \- Check detector IDs match the config in `configs/mappo\_config.yaml`

# 

# \*\*Traffic not flowing\*\*:

# \- Check routes are valid: `duarouter -n 2x2grid.net.xml -r 2x2grid.rou.xml`

# \- Increase traffic volume: reduce `-p` value in randomTrips

# \- Check for network connectivity issues in NETEDIT

# 

# \*\*Vehicles teleporting\*\*:

# \- This is normal when traffic is very heavy

# \- Can disable with `--time-to-teleport -1` in sumocfg

# 

# \## Example: Complete Setup Script

# 

# Save this as `setup\_sumo\_files.sh`:

# 

# ```bash

# \#!/bin/bash

# set -e

# 

# echo "Generating SUMO network files..."

# 

# \# Generate network

# netgenerate --grid \\

# &nbsp;   --grid.number 2 \\

# &nbsp;   --grid.length 200 \\

# &nbsp;   --default.lanenumber 3 \\

# &nbsp;   --default-junction-type traffic\_light \\

# &nbsp;   --tls.guess true \\

# &nbsp;   --output-file 2x2grid.net.xml

# 

# echo "✓ Network generated"

# 

# \# Generate routes

# python $SUMO\_HOME/tools/randomTrips.py \\

# &nbsp;   -n 2x2grid.net.xml \\

# &nbsp;   -r 2x2grid.rou.xml \\

# &nbsp;   -e 3600 \\

# &nbsp;   -p 2.0 \\

# &nbsp;   --fringe-factor 10 \\

# &nbsp;   --validate

# 

# echo "✓ Routes generated"

# 

# \# Generate detectors

# python $SUMO\_HOME/tools/generateTLSE2Detectors.py \\

# &nbsp;   -n 2x2grid.net.xml \\

# &nbsp;   -o 2x2grid.det.xml \\

# &nbsp;   -d 250 \\

# &nbsp;   -f 60

# 

# echo "✓ Detectors generated"

# 

# \# Test

# echo "Testing simulation..."

# sumo -c 2x2grid.sumocfg --duration-log.statistics --start --quit-on-end

# 

# echo "✓ All SUMO files ready!"

# ```

# 

# Run it:

# ```bash

# chmod +x setup\_sumo\_files.sh

# ./setup\_sumo\_files.sh

# ```

# 

# \## Resources

# 

# \- SUMO Documentation: https://sumo.dlr.de/docs/

# \- netgenerate: https://sumo.dlr.de/docs/netgenerate.html

# \- randomTrips: https://sumo.dlr.de/docs/Tools/Trip.html

# \- Detectors: https://sumo.dlr.de/docs/Simulation/Output/Lanearea\_Detectors\_(E2).html

# 

# ---

# 

# \*\*Note\*\*: The placeholder files in this directory need to be replaced with actual SUMO-generated files before running the training.

