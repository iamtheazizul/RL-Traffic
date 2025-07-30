import subprocess
import platform

# Step 01: Generate Route Whitelist
route_whitelist = ("python3 randomTrips.py "
                   "--net-file network/cleaned_network.net.xml "
                   "--route-file config/demand/heavy/heavy_routes_2.rou.xml "
                   "--fringe-factor 300 "
                   "--max-distance 5000 "
                   "--min-distance 200 "
                   "--speed-exponent 4.0 "
                   "--period 0.1 "
                   "--validate")
subprocess.run(route_whitelist, shell=True)