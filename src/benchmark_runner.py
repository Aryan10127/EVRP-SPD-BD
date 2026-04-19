import os
import glob
import math
import sys
import re

import heuristic
import heuristic2

# Setup basic parameters
DATASET_DIRS = [
    r"Dataset\Dataset\five",
    r"Dataset\Dataset\dataset_50",
    r"Dataset\Dataset\dataset_70"
]

def format_cell(text, width=15):
    return str(text).ljust(width)

def euclidean_distance(n1, n2):
    return math.sqrt((n1['x'] - n2['x']) ** 2 + (n1['y'] - n2['y']) ** 2)

def build_distance_dict(nodes):
    distances = {}
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if i != j:
                distances[(n1['NodeID'], n2['NodeID'])] = euclidean_distance(n1, n2)
    return distances

def parse_dataset(filename):
    nodes, hub_node_id, charging_stations, customers = [], None, [], []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('Q ', 'C ', 'r ', 'g ', 'v ', 'StringID')):
                continue
            parts = line.split()
            if len(parts) >= 9:
                node = {
                    'NodeID':          parts[0],
                    'Type':            parts[1],
                    'x':               float(parts[2]),
                    'y':               float(parts[3]),
                    'delivery_weight': float(parts[4]),
                    'pickup_weight':   float(parts[8]),
                    'customer_id':     parts[0],
                }
                nodes.append(node)
                if parts[1] == 'd':
                    hub_node_id = parts[0]
                elif parts[1] == 'f':
                    node['charging_station']    = True
                    node['charging_station_id'] = parts[0]
                    charging_stations.append(node)
                elif parts[1] == 'c':
                    customers.append(node)
    return nodes, hub_node_id, charging_stations, customers

def parse_params(filename):
    with open(filename, 'r') as f:
        content = f.read()
    params = {}
    q = re.search(r'Q\s+Vehicle fuel tank capacity\s*/([0-9.]+)/', content)
    c = re.search(r'C\s+Vehicle load capacity\s*/([0-9.]+)/', content)
    r = re.search(r'r\s+fuel consumption rate\s*/([0-9.]+)/', content)
    g = re.search(r'g\s+inverse refueling rate\s*/([0-9.]+)/', content)
    v = re.search(r'v\s+average Velocity\s*/([0-9.]+)/', content)
    if q: params['battery_capacity'] = float(q.group(1)) * 1000
    if c: params['max_capacity']     = float(c.group(1))
    if r: params['consumption_rate'] = float(r.group(1))
    if g: params['charging_rate']    = float(g.group(1))
    if v: params['speed']            = float(v.group(1))
    return params

print(f"| {format_cell('Dataset', 20)} | {format_cell('Algorithm')} | {format_cell('Avg Distance')} | {format_cell('Fleet Health', 15)} | {format_cell('Avg Cycles')} | {format_cell('Mean DoD')} |")
print(f"| {'-'*20} | {'-'*15} | {'-'*15} | {'-'*15} | {'-'*15} | {'-'*15} |")

for dataset_dir in DATASET_DIRS:
    files = glob.glob(os.path.join(dataset_dir, "*.txt"))
    if not files:
        print(f"No files in {dataset_dir}")
        continue
        
    for file in files[:1]:  # Pick first 1 file from each directory
        params = parse_params(file)
        if not params:
            continue
        
        battery_capacity = params['battery_capacity'] * 0.25 # tweak
        max_capacity = params['max_capacity']
        beta = params['consumption_rate']
        min_thresh = 0.3 * battery_capacity
        
        nodes, hub_node_id, charging_stations, customers = parse_dataset(file)
        num_evs = max(2, len(customers) // 3)
        node_distances = build_distance_dict(nodes)
        
        dataset_name = os.path.basename(file)
        print(f"Testing {dataset_name}...", file=sys.stderr)
        
        # RUN BASELINE
        heuristic.set_params(hub_node_id, charging_stations, battery_capacity, 100, beta, min_thresh)
        sols1 = heuristic.greedy_cvrpspd_solutions(customers, num_evs, max_capacity, node_distances, num_solutions=2)
        
        s1_dist, s1_fh, s1_cyc, s1_dod = 0.0, 0.0, 0.0, 0.0
        s1_count = len(sols1)
        if s1_count > 0:
            for sol in sols1:
                s1_dist += sol.get('total_distance', 0)
                s1_fh += sol.get('fleet_health', 0)
                t_cyc, t_dod = 0, 0
                for vs in sol.get('vehicle_stats', []):
                    t_cyc += vs['num_cycles']
                    t_dod += vs['avg_dod']
                if len(sol.get('vehicle_stats', [])) > 0:
                    vs_len = len(sol['vehicle_stats'])
                    s1_cyc += (t_cyc / vs_len)
                    s1_dod += (t_dod / vs_len)
            
            s1_dist, s1_fh, s1_cyc, s1_dod = s1_dist/s1_count, s1_fh/s1_count, s1_cyc/s1_count, s1_dod/s1_count
            print(f"| {format_cell(dataset_name, 20)} | {format_cell('Baseline (TA)')} | {format_cell(round(s1_dist, 1))} | {format_cell(round(s1_fh, 4), 15)} | {format_cell(round(s1_cyc, 2))} | {format_cell(round(s1_dod, 4))} |")

        # RUN HEURISTIC2
        heuristic2.set_params(hub_node_id, charging_stations, battery_capacity, 100, beta, min_thresh)
        sols2 = heuristic2.greedy_cvrpspd_solutions(customers, num_evs, max_capacity, node_distances, num_solutions=2)
        
        s2_dist, s2_fh, s2_cyc, s2_dod = 0.0, 0.0, 0.0, 0.0
        s2_count = len(sols2)
        if s2_count > 0:
            for sol in sols2:
                s2_dist += sol.get('total_distance', 0)
                s2_fh += sol.get('fleet_health', 0)
                t_cyc, t_dod = 0, 0
                for vs in sol.get('vehicle_stats', []):
                    t_cyc += vs['num_cycles']
                    t_dod += vs['avg_dod']
                if len(sol.get('vehicle_stats', [])) > 0:
                    vs_len = len(sol['vehicle_stats'])
                    s2_cyc += (t_cyc / vs_len)
                    s2_dod += (t_dod / vs_len)
                    
            s2_dist, s2_fh, s2_cyc, s2_dod = s2_dist/s2_count, s2_fh/s2_count, s2_cyc/s2_count, s2_dod/s2_count
            print(f"| {format_cell('--', 20)} | {format_cell('Heuristic 2')} | {format_cell(round(s2_dist, 1))} | {format_cell(round(s2_fh, 4), 15)} | {format_cell(round(s2_cyc, 2))} | {format_cell(round(s2_dod, 4))} |")
            
        print(f"| {'-'*20} | {'-'*15} | {'-'*15} | {'-'*15} | {'-'*15} | {'-'*15} |")

