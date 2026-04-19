import os
import glob
import math
import sys
import re
from fpdf import FPDF

import heuristic
import heuristic2
import heuristic3
import heuristic4

DATASET_DIRS = [
    r"Dataset\Dataset\five",
    r"Dataset\Dataset\dataset_50",
    r"Dataset\Dataset\dataset_70",
    r"Dataset\Dataset\hundred"
]

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

def run_algo(algo_module, func_name, customers, num_evs, max_capacity, node_distances, num_solutions):
    func = getattr(algo_module, func_name)
    sols = func(customers, num_evs, max_capacity, node_distances, num_solutions=num_solutions)
    if not sols:
        return 0, 0, 0, 0
    
    avg_dist, avg_fh, avg_cyc, avg_dod = 0.0, 0.0, 0.0, 0.0
    for sol in sols:
        avg_dist += sol.get('total_distance', 0)
        avg_fh += sol.get('fleet_health', 0)
        
        t_cyc, t_dod = 0, 0
        vstats = sol.get('vehicle_stats', [])
        if vstats:
            for vs in vstats:
                t_cyc += vs['num_cycles']
                t_dod += vs['avg_dod']
            avg_cyc += t_cyc / len(vstats)
            avg_dod += t_dod / len(vstats)
            
    n = len(sols)
    return avg_dist/n, avg_fh/n, avg_cyc/n, avg_dod/n

results = []

for dataset_dir in DATASET_DIRS:
    files = glob.glob(os.path.join(dataset_dir, "*.txt"))
    if not files:
        print(f"No files in {dataset_dir}")
        continue
        
    for file in files[:1]: 
        params = parse_params(file)
        if not params: continue
        
        battery_capacity = params['battery_capacity'] * 0.25
        max_capacity = params['max_capacity']
        beta = params['consumption_rate']
        min_thresh = 0.3 * battery_capacity
        
        nodes, hub_node_id, charging_stations, customers = parse_dataset(file)
        num_evs = max(2, len(customers) // 3)
        node_distances = build_distance_dict(nodes)
        
        dataset_name = os.path.basename(file)
        print(f"Running {dataset_name}...")
        
        # H1
        heuristic.set_params(hub_node_id, charging_stations, battery_capacity, 100, beta, min_thresh)
        d1, fh1, c1, dod1 = run_algo(heuristic, 'greedy_cvrpspd_solutions', customers, num_evs, max_capacity, node_distances, 2)
        results.append([dataset_name, 'H1 (Baseline)', round(d1,1), round(fh1,4), round(c1,2), round(dod1,4)])
        
        # H2
        heuristic2.set_params(hub_node_id, charging_stations, battery_capacity, 100, beta, min_thresh)
        d2, fh2, c2, dod2 = run_algo(heuristic2, 'greedy_cvrpspd_solutions', customers, num_evs, max_capacity, node_distances, 2)
        results.append(['', 'H2 (Greedy Battery)', round(d2,1), round(fh2,4), round(c2,2), round(dod2,4)])
        
        # H3
        heuristic3.set_params(hub_node_id, charging_stations, battery_capacity, 100, beta, min_thresh)
        d3, fh3, c3, dod3 = run_algo(heuristic3, 'regret_cvrpspd_solutions', customers, num_evs, max_capacity, node_distances, 2)
        results.append(['', 'H3 (Regret-Based)', round(d3,1), round(fh3,4), round(c3,2), round(dod3,4)])
        
        # H4
        heuristic4.set_params(hub_node_id, charging_stations, battery_capacity, 100, beta, min_thresh)
        d4, fh4, c4, dod4 = run_algo(heuristic4, 'grasp_cvrpspd_solutions', customers, num_evs, max_capacity, node_distances, 2)
        results.append(['', 'H4 (GRASP)', round(d4,1), round(fh4,4), round(c4,2), round(dod4,4)])


###################################
# PDF GENERATION
###################################
class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, 'EVRP-SPD-BD: Multi-Heuristic Comparative Analysis', align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(5)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(44, 62, 80)
        self.cell(0, 8, title, new_x='LMARGIN', new_y='NEXT')
        self.ln(2)

    def body_text(self, text, indent=0):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        if indent > 0:
            self.set_x(self.get_x() + indent)
        self.multi_cell(0, 5, text)
        self.ln(3)

    def add_table(self, headers, rows, col_widths):
        self.set_font('Helvetica', 'B', 10)
        self.set_fill_color(44, 62, 80)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, h, border=1, fill=True, align='C')
        self.ln()
        
        self.set_font('Helvetica', '', 9)
        self.set_text_color(0, 0, 0)
        
        for row_idx, row in enumerate(rows):
            if row[0] != '':
                self.set_fill_color(220, 230, 241) # Light blue header
            else:
                if row_idx % 2 == 0:
                    self.set_fill_color(255, 255, 255)
                else:
                    self.set_fill_color(245, 245, 245)
                    
            for i, val in enumerate(row):
                self.cell(col_widths[i], 8, str(val), border=1, fill=True, align='C')
            self.ln()
        self.ln(5)

pdf = PDFReport()
pdf.add_page()

pdf.section_title("1. Evaluated Heuristics")

pdf.set_font('Helvetica', 'B', 10)
pdf.cell(0, 6, "H1 (Baseline): Pure Energy-Optimized Greedy", new_x='LMARGIN', new_y='NEXT')
pdf.body_text("This heuristic blindly minimizes total travel distance using standard energy calculations. It evaluates combinations based purely on Euclidean travel costs, resulting in minimal distances but frequently causing deep battery depth-of-discharges (DoD) as it disregards battery health.", indent=5)

pdf.set_font('Helvetica', 'B', 10)
pdf.cell(0, 6, "H2 (Greedy Battery): Accelerated Degradation-Aware", new_x='LMARGIN', new_y='NEXT')
pdf.body_text("Introduces a non-linear battery degradation model into the insertion score. It adds heavy penalties for route insertions that result in deep discharging and forces vehicles to recharge adaptively. By prioritizing healthier batteries over raw distance, it functions as a natural load balancer.", indent=5)

pdf.set_font('Helvetica', 'B', 10)
pdf.cell(0, 6, "H3 (Regret-Based): Degradation-Aware Regret-2", new_x='LMARGIN', new_y='NEXT')
pdf.body_text("Uses standard regret-insertion principles to evaluate the delta between a customer's absolute best insertion vs their second-best placement. While regret often excels in distance-only problems, in the context of EVs with highly non-linear degradation curves, it forces lengthy initial detours that severely impact fleet health.", indent=5)

pdf.set_font('Helvetica', 'B', 10)
pdf.cell(0, 6, "H4 (GRASP): Greedy Randomized Adaptive Search Procedure", new_x='LMARGIN', new_y='NEXT')
pdf.body_text("An advanced metaheuristic approach building upon H2. Instead of selecting the absolute mathematical best candidate at every step, it generates a 'Restricted Candidate List' (RCL) of the Top-N insertions and picks randomly. This strategic randomness prevents the algorithm from being trapped in local optimums, outperforming deterministic approaches natively.", indent=5)

pdf.ln(5)
pdf.add_page()
pdf.section_title("2. Benchmark Results")
pdf.body_text("The following table evaluates algorithm performance across four dataset scales: Small (5 nodes), Medium (50 nodes), Large (70 nodes), and Extra Large (100 nodes).")

headers = ['Dataset', 'Algorithm', 'Avg Dist.', 'Fleet Health', 'Avg Cyc.', 'Mean DoD']
col_widths = [26, 48, 25, 25, 21, 23]

pdf.add_table(headers, results, col_widths)

pdf.output("Heuristics_Comparison.pdf")
print("Saved to Heuristics_Comparison.pdf")
