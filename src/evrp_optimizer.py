import copy
import random
import glob
import math
import os
import re


HUB_NODE_ID           = None
CHARGING_STATIONS     = []
BATTERY_CAPACITY      = None
VEHICLE_MASS          = None
BETA                  = None
MIN_BATTERY_THRESHOLD = None

# ==================== NEW: Battery Degradation Global Parameters ====================
W1        = 1.0    # Weight for travel detour cost in CS scoring
W2        = 10.0   # Weight for degradation penalty in CS scoring
ALPHA_SOC = 0.3    # Tunable hyperparameter: weight of SoC-deviation penalty
                   # relative to DoD in the degradation cost function.
                   # 0.0 = only DoD matters, 1.0 = SoC penalty weighted equally to DoD.
                   # Selected via sensitivity analysis across benchmark instances.
# ==================== END NEW ====================

# ==================== NEW: Idea 1 + Idea 2 — Combined Insertion Score Weights ====================
# LAMBDA: controls how much degradation (DoD/SoC) influences the insertion score.
#         Higher → algorithm actively avoids battery-damaging assignments.
#         Lower  → algorithm behaves closer to original energy-only greedy.
# MU:     controls how much the current cycle count of a vehicle penalises it
#         from receiving more customers. Higher → cycles spread evenly across fleet.
LAMBDA = 40_000      # degradation term weight (Scaled down since new cost function produces larger values)
MU     = 1000        # cycle count penalty weight
# ==================== END NEW ====================


def set_params(hub_node_id, charging_stations, battery_capacity,
               vehicle_mass, beta, min_battery_threshold):
    global HUB_NODE_ID, CHARGING_STATIONS, BATTERY_CAPACITY
    global VEHICLE_MASS, BETA, MIN_BATTERY_THRESHOLD

    HUB_NODE_ID           = hub_node_id
    CHARGING_STATIONS     = charging_stations
    BATTERY_CAPACITY      = battery_capacity
    VEHICLE_MASS          = vehicle_mass
    BETA                  = beta
    MIN_BATTERY_THRESHOLD = min_battery_threshold


def get_distance(node_distances, node1_id, node2_id, default=None):
    key = (node1_id, node2_id)
    if key in node_distances:
        return node_distances[key]
    rev = (node2_id, node1_id)
    if rev in node_distances:
        return node_distances[rev]
    return default


def energy_usage(distance, weight):
    return BETA * (weight + VEHICLE_MASS) * distance


# ==================== NEW: Adaptive Charging Threshold ====================
def adaptive_threshold(current_battery):
    """
    Instead of a fixed MIN_BATTERY_THRESHOLD, dynamically raise the
    threshold when the battery is already dangerously low, so the
    vehicle charges earlier and avoids deep discharge cycles (high DoD).

    Logic:
      - If SoC < 35%  → charge sooner (threshold = 40% of capacity)
        This prevents the battery from going below ~35%, reducing DoD.
      - If SoC > 85%  → keep threshold low (30%), no urgency to charge.
      - Otherwise     → use the standard MIN_BATTERY_THRESHOLD.
    """
    soc = current_battery / BATTERY_CAPACITY
    if soc < 0.35:
        return 0.40 * BATTERY_CAPACITY   # Charge sooner — already deep
    elif soc > 0.85:
        return 0.30 * BATTERY_CAPACITY   # Normal threshold — SoC is healthy
    else:
        return MIN_BATTERY_THRESHOLD
# ==================== END NEW ====================


# ==================== NEW: Pre-charge Degradation Estimator ====================
def estimate_charge_degradation(battery_before_charge):
    """
    Estimates degradation cost of one charge event BEFORE it happens.
    Used during CS selection to score stations by degradation penalty.

    Parameters:
        battery_before_charge : battery level (absolute) just before charging

    How it works (Simple Cost Function):
        - DoD  = how deeply discharged the battery is right now
                 (1.0 means fully empty, 0.0 means still full)
        - mean_soc = average of SoC before and after charging
                     (after = 1.0 since we always fully recharge)
        - soc_penalty = absolute distance of mean_soc from ideal 50%
        - Returns dod + 0.5 * soc_penalty
    """
    soc_before = battery_before_charge / BATTERY_CAPACITY
    dod        = 1.0 - soc_before               # deeper discharge = higher DoD
    mean_soc   = (soc_before + 1.0) / 2.0       # average across this cycle
    soc_penalty = abs(mean_soc - 0.5)
    return dod + ALPHA_SOC * soc_penalty
# ==================== END NEW ====================


# ==================== NEW: SoC Time-Series Builder ====================
def build_soc_series(route, node_distances):
    """
    Walks through a completed route and records the battery SoC (0.0–1.0)
    at every node. This time-series is the input for Rainflow cycle counting.

    Key detail: at CS nodes, SoC is recorded BEFORE charging so that
    each discharge depth (DoD) is captured accurately.
    """
    soc_series     = []
    battery        = BATTERY_CAPACITY
    current_weight = sum(
        n.get('delivery_weight', 0)
        for n in route
        if not n.get('charging_station', False)
        and n.get('customer_id') not in ('hub_start', 'hub_end')
    )
    prev_id = route[0]['NodeID']
    soc_series.append(battery / BATTERY_CAPACITY)  # Start at 100%

    for node in route[1:]:
        curr_id  = node['NodeID']
        dist     = get_distance(node_distances, prev_id, curr_id, 0)
        battery -= energy_usage(dist, current_weight)
        battery  = max(battery, 0)

        if node.get('charging_station', False):
            soc_series.append(battery / BATTERY_CAPACITY)  # SoC BEFORE charge
            battery = BATTERY_CAPACITY                      # Full recharge
        else:
            current_weight -= node.get('delivery_weight', 0)
            current_weight += node.get('pickup_weight', 0)
            current_weight  = max(current_weight, 0)

        soc_series.append(battery / BATTERY_CAPACITY)
        prev_id = curr_id

    return soc_series
# ==================== END NEW ====================


# ==================== NEW: Rainflow Cycle Counter ====================
def rainflow_count(soc_series):
    """
    Simplified Rainflow algorithm applied to a SoC time-series.
    Standard method in fatigue analysis — extracts individual
    charge-discharge cycles from a complex waveform.

    Returns a list of (DoD, mean_SoC) tuples, one per extracted cycle.

    How it works:
        Uses a stack to scan SoC values in sequence.
        Whenever three consecutive points show that the middle segment
        (s0→s1) is smaller than the following segment (s1→s2), the
        middle segment forms a complete cycle and is extracted.
        Remaining segments in the stack become half-cycles.
    """
    cycles = []
    stack  = []

    for soc in soc_series:
        stack.append(soc)
        while len(stack) >= 3:
            s0, s1, s2   = stack[-3], stack[-2], stack[-1]
            range_01     = abs(s1 - s0)
            range_12     = abs(s2 - s1)

            if range_12 >= range_01:
                dod      = range_01
                mean_soc = (s0 + s1) / 2.0
                cycles.append((dod, mean_soc))
                stack.pop(-2)
                stack.pop(-2)
            else:
                break

    # Remaining stack segments treated as half-cycles
    for i in range(len(stack) - 1):
        dod      = abs(stack[i + 1] - stack[i])
        mean_soc = (stack[i] + stack[i + 1]) / 2.0
        if dod > 0:
            cycles.append((dod, mean_soc))

    return cycles
# ==================== END NEW ====================


# ==================== NEW: Degradation Model ====================
def compute_degradation(soc_series):
    """
    Battery degradation cost function based on two factors:
      1. Depth of Discharge (DoD) — deeper cycles cost more
      2. Average State of Charge (SoC) — operating far from
         50% SoC costs more (too high or too low both harmful)

    Formula per cycle:
        cost(c) = DoD_c + ALPHA_SOC × |mean_SoC_c - 0.5|

    ALPHA_SOC is a tunable hyperparameter (default 0.3) that controls
    how much SoC deviation contributes relative to DoD.

    Returns avg_cycle_cost (normalized per cycle) so the metric
    stays bounded in a meaningful [0, ~1] range.
    """
    cycles = rainflow_count(soc_series)
    if not cycles:
        return 0.0, 0, 0.0, 0.0

    total_deg = 0.0
    for dod, mean_soc in cycles:
        soc_penalty = abs(mean_soc - 0.5)   # 0 at healthy 50%, rises toward extremes
        total_deg  += dod + ALPHA_SOC * soc_penalty

    num_cycles     = len(cycles)
    avg_cycle_cost = total_deg / num_cycles   # normalize per cycle
    avg_dod        = sum(d for d, _ in cycles) / num_cycles
    avg_soc        = sum(s for _, s in cycles) / num_cycles

    return avg_cycle_cost, num_cycles, round(avg_dod, 4), round(avg_soc, 4)
# ==================== END NEW ====================


# ==================== NEW: Idea 1 — Route Simulation with Degradation ====================
def simulate_route_degradation(sequence, node_distances):
    """
    Simulates driving a sequence of customer nodes from the hub and back,
    tracking a full SoC time-series along the way.

    Called during the greedy insertion evaluation loop (Phase 1) to estimate
    the degradation cost of assigning a particular customer to a vehicle.

    Unlike build_soc_series() which works on a finished route dict,
    this works directly on a raw customer sequence (list of node dicts)
    before the route is committed — so it can be used for scoring
    candidate insertions before any route is actually built.

    Returns:
        total_energy  : float — total energy consumed (same as before)
        degradation   : float — estimated degradation from this route's cycles
        feasible      : bool  — False if no CS reachable mid-route
    """
    temp_load    = sum(c.get('delivery_weight', 0) for c in sequence)
    temp_battery = BATTERY_CAPACITY
    temp_pos     = HUB_NODE_ID
    total_energy = 0.0
    feasible     = True

    # SoC series starts at full battery at depot
    soc_series   = [temp_battery / BATTERY_CAPACITY]

    for cust in sequence:
        next_id = cust['NodeID']
        dist    = get_distance(node_distances, temp_pos, next_id, 10000)
        needed  = energy_usage(dist, temp_load)

        while temp_battery - needed < adaptive_threshold(temp_battery):
            candidates = find_reachable_cs(temp_pos, temp_battery, temp_load, node_distances)
            nearest_cs = random.choice(candidates) if candidates else None
            if nearest_cs is None:
                feasible = False
                break
            d_cs          = get_distance(node_distances, temp_pos, nearest_cs, 10000)
            total_energy += energy_usage(d_cs, temp_load)
            temp_battery -= energy_usage(d_cs, temp_load)
            # Record SoC BEFORE charging (captures the DoD of this cycle)
            soc_series.append(temp_battery / BATTERY_CAPACITY)
            temp_battery  = BATTERY_CAPACITY
            soc_series.append(temp_battery / BATTERY_CAPACITY)
            temp_pos      = nearest_cs
            dist          = get_distance(node_distances, temp_pos, next_id, 10000)
            needed        = energy_usage(dist, temp_load)

        if not feasible:
            break

        temp_battery -= needed
        total_energy += needed
        temp_pos      = next_id
        temp_load     = temp_load - cust.get('delivery_weight', 0) + cust.get('pickup_weight', 0)
        temp_load     = max(temp_load, 0)
        soc_series.append(temp_battery / BATTERY_CAPACITY)

    if not feasible:
        return total_energy, 0.0, False

    # Simulate return to hub
    d_hub = get_distance(node_distances, temp_pos, HUB_NODE_ID, 10000)
    e_hub = energy_usage(d_hub, temp_load)

    if temp_battery - e_hub < 0:
        candidates = find_reachable_cs(temp_pos, temp_battery, temp_load, node_distances)
        nearest_cs = random.choice(candidates) if candidates else None
        if nearest_cs is None:
            return total_energy, 0.0, False
        d_cs          = get_distance(node_distances, temp_pos, nearest_cs, 10000)
        total_energy += energy_usage(d_cs, temp_load)
        temp_battery -= energy_usage(d_cs, temp_load)
        soc_series.append(temp_battery / BATTERY_CAPACITY)
        temp_battery  = BATTERY_CAPACITY
        soc_series.append(temp_battery / BATTERY_CAPACITY)
        temp_pos      = nearest_cs
        d_hub         = get_distance(node_distances, temp_pos, HUB_NODE_ID, 10000)
        e_hub         = energy_usage(d_hub, temp_load)

    if temp_battery - e_hub < 0:
        return total_energy, 0.0, False

    total_energy += e_hub
    temp_battery -= e_hub
    soc_series.append(temp_battery / BATTERY_CAPACITY)

    # Compute degradation from the simulated SoC series
    degradation, _, _, _ = compute_degradation(soc_series)

    return total_energy, degradation, True
# ==================== END NEW ====================


def find_reachable_cs(position, battery, load, node_distances):
    reachable = []
    for cs in CHARGING_STATIONS:
        cs_id = cs['NodeID']
        dist  = get_distance(node_distances, position, cs_id, 10000)
        cost  = energy_usage(dist, load)
        # ==================== MODIFIED: use adaptive_threshold instead of fixed MIN_BATTERY_THRESHOLD ====================
        if battery - cost >= adaptive_threshold(battery):
        # ==================== END MODIFIED ====================
            reachable.append((cs_id, cost))
    reachable.sort(key=lambda x: x[1])
    top3 = [cs_id for cs_id, _ in reachable[:3]]
    return top3 if top3 else None


def add_charging_stations(route, node_distances):
    if not route or len(route) <= 1:
        return None

    all_customers  = [n for n in route[1:-1] if not n.get('charging_station', False)]
    current_weight = sum(n.get('delivery_weight', 0) for n in all_customers)

    new_route     = [route[0]]
    battery_level = BATTERY_CAPACITY
    prev_id       = route[0]['NodeID']

    for node in route[1:-1]:
        current_id = node['NodeID']

        for _ in range(5):
            dist = get_distance(node_distances, prev_id, current_id, float('inf'))
            if dist == float('inf'):
                return None
            if battery_level - energy_usage(dist, current_weight) >= 0:
                break

            best_cs    = None
            best_score = float('inf')
            for cs in CHARGING_STATIONS:
                cs_id = cs['NodeID']
                if cs_id in (prev_id, current_id):
                    continue
                d1 = get_distance(node_distances, prev_id, cs_id, float('inf'))
                if d1 == float('inf') or battery_level - energy_usage(d1, current_weight) < 0:
                    continue
                d2 = get_distance(node_distances, cs_id, current_id, float('inf'))
                if d2 == float('inf') or BATTERY_CAPACITY - energy_usage(d2, current_weight) < 0:
                    continue

                # ==================== MODIFIED: CS scored by detour + degradation penalty ====================
                battery_at_cs  = battery_level - energy_usage(d1, current_weight)
                deg_penalty    = estimate_charge_degradation(battery_at_cs)
                combined_score = W1 * (d1 + d2) + W2 * deg_penalty
                if combined_score < best_score:
                    best_score = combined_score
                    best_cs    = cs
                # ==================== END MODIFIED ====================

            if best_cs is None:
                return None

            cs_id         = best_cs['NodeID']
            d_cs          = get_distance(node_distances, prev_id, cs_id, 0)
            battery_level -= energy_usage(d_cs, current_weight)
            if battery_level < 0:
                return None

            new_route.append({
                'NodeID':           cs_id,
                'charging_station': True,
                'delivery_weight':  0,
                'pickup_weight':    0,
                'customer_id':      f"CS_{cs_id}",
            })
            battery_level = BATTERY_CAPACITY
            prev_id       = cs_id
        else:
            return None

        dist           = get_distance(node_distances, prev_id, current_id, float('inf'))
        battery_level -= energy_usage(dist, current_weight)
        if battery_level < 0:
            return None

        new_route.append(node.copy())
        current_weight -= node.get('delivery_weight', 0)
        current_weight += node.get('pickup_weight', 0)
        current_weight  = max(0, current_weight)
        prev_id         = current_id

    # Return to hub
    for _ in range(5):
        d_hub = get_distance(node_distances, prev_id, HUB_NODE_ID, float('inf'))
        if d_hub == float('inf'):
            return None
        if battery_level - energy_usage(d_hub, current_weight) >= 0:
            break

        best_cs    = None
        best_score = float('inf')
        for cs in CHARGING_STATIONS:
            cs_id = cs['NodeID']
            if cs_id == prev_id:
                continue
            d1 = get_distance(node_distances, prev_id, cs_id, float('inf'))
            if d1 == float('inf') or battery_level - energy_usage(d1, current_weight) < 0:
                continue
            d2 = get_distance(node_distances, cs_id, HUB_NODE_ID, float('inf'))
            if d2 == float('inf') or BATTERY_CAPACITY - energy_usage(d2, current_weight) < 0:
                continue

            # ==================== MODIFIED: CS scored by detour + degradation penalty ====================
            battery_at_cs  = battery_level - energy_usage(d1, current_weight)
            deg_penalty    = estimate_charge_degradation(battery_at_cs)
            combined_score = W1 * (d1 + d2) + W2 * deg_penalty
            if combined_score < best_score:
                best_score = combined_score
                best_cs    = cs
            # ==================== END MODIFIED ====================

        if best_cs is None:
            return None

        cs_id         = best_cs['NodeID']
        d_cs          = get_distance(node_distances, prev_id, cs_id, 0)
        battery_level -= energy_usage(d_cs, current_weight)
        if battery_level < 0:
            return None

        new_route.append({
            'NodeID':           cs_id,
            'charging_station': True,
            'delivery_weight':  0,
            'pickup_weight':    0,
            'customer_id':      f"CS_{cs_id}",
        })
        battery_level = BATTERY_CAPACITY
        prev_id       = cs_id
    else:
        return None

    new_route.append({
        'NodeID':           HUB_NODE_ID,
        'charging_station': False,
        'delivery_weight':  0,
        'pickup_weight':    0,
        'customer_id':      'hub_end',
    })
    return new_route


def greedy_cvrpspd_solutions(dataset, num_evs, max_capacity, node_distances, num_solutions=2000):

    solutions = []
    print(f"Generating {num_solutions} greedy solutions...")

    for sol_idx in range(num_solutions):

        routes = []
        for k in range(num_evs):
            routes.append([{
                'NodeID':           HUB_NODE_ID,
                'charging_station': False,
                'delivery_weight':  0,
                'pickup_weight':    0,
                'customer_id':      'hub_start',
            }])

        unassigned = copy.deepcopy(list(dataset))
        random.shuffle(unassigned)

        while unassigned:
            best_insertion = None
            # ==================== MODIFIED: Idea 1+2 — insertion scored by energy + degradation + cycle penalty ====================
            # Previously: best_energy = float('inf') and winner = lowest total_energy
            # Now:        best_score  = float('inf') and winner = lowest combined score where:
            #             combined_score = total_energy
            #                           + LAMBDA * route_degradation   (Idea 1: DoD/SoC aware)
            #                           + MU     * current_cycle_count (Idea 2: cycle spread)
            best_score = float('inf')
            # ==================== END MODIFIED ====================

            for k in range(num_evs):
                for customer in unassigned:

                    sequence = [
                        s for s in routes[k][1:]
                        if not s.get('charging_station', False)
                        and s.get('customer_id') != 'hub_end'
                    ]
                    sequence.append(customer)

                    initial_load = sum(c.get('delivery_weight', 0) for c in sequence)
                    if initial_load > max_capacity:
                        continue

                    # ==================== MODIFIED: Idea 1 — use simulate_route_degradation instead of manual simulation ====================
                    # Previously: manual simulation loop tracking only total_energy,
                    #             then: if total_energy < best_energy → best_insertion
                    # Now: simulate_route_degradation returns energy + degradation + feasibility
                    #      in one call, then both feed into combined_score
                    total_energy, route_degradation, feasible = simulate_route_degradation(
                        sequence, node_distances
                    )

                    if not feasible:
                        continue
                    # ==================== END MODIFIED ====================

                    # ==================== MODIFIED: Idea 2 — count existing CS stops on this vehicle as cycle penalty ====================
                    # current_cycles = how many times this vehicle has already charged
                    # A vehicle with many cycles already is penalised, steering new
                    # customers toward fresher vehicles with fewer charge events so far.
                    current_cycles = sum(
                        1 for node in routes[k]
                        if node.get('charging_station', False)
                    )
                    cycle_penalty = MU * current_cycles
                    # ==================== END MODIFIED ====================

                    # ==================== MODIFIED: Idea 1+2 — final combined insertion score ====================
                    combined_score = total_energy + LAMBDA * route_degradation + cycle_penalty
                    combined_score += random.uniform(0, total_energy * 0.05)

                    if combined_score < best_score:
                        best_score     = combined_score
                        best_insertion = {'vehicle': k, 'customer': customer}
                    # ==================== END MODIFIED ====================

            if best_insertion is None:
                customer = unassigned[0]
                k = min(range(num_evs),
                        key=lambda x: len([s for s in routes[x] if not s.get('charging_station', False)]))
                routes[k].append(copy.deepcopy(customer))
                unassigned.remove(customer)
                continue

            k        = best_insertion['vehicle']
            customer = best_insertion['customer']

            new_route = [{'NodeID': HUB_NODE_ID, 'charging_station': False,
                          'delivery_weight': 0, 'pickup_weight': 0, 'customer_id': 'hub_start'}]

            sequence = [
                s for s in routes[k][1:]
                if not s.get('charging_station', False) and s.get('customer_id') != 'hub_end'
            ]
            sequence.append(customer)

            temp_load    = sum(c.get('delivery_weight', 0) for c in sequence)
            temp_pos     = HUB_NODE_ID
            temp_battery = BATTERY_CAPACITY

            for cust in sequence:
                next_id = cust['NodeID']
                dist    = get_distance(node_distances, temp_pos, next_id, 10000)
                needed  = energy_usage(dist, temp_load)

                # ==================== MODIFIED: adaptive_threshold replaces fixed MIN_BATTERY_THRESHOLD ====================
                if temp_battery - needed < adaptive_threshold(temp_battery):
                # ==================== END MODIFIED ====================
                    candidates = find_reachable_cs(temp_pos, temp_battery, temp_load, node_distances)
                    nearest_cs = random.choice(candidates) if candidates else None
                    if nearest_cs:
                        new_route.append({
                            'NodeID':           nearest_cs,
                            'charging_station': True,
                            'delivery_weight':  0,
                            'pickup_weight':    0,
                            'customer_id':      f"CS_{nearest_cs}",
                        })
                        d_cs         = get_distance(node_distances, temp_pos, nearest_cs, 10000)
                        temp_battery = BATTERY_CAPACITY
                        temp_pos     = nearest_cs
                        dist         = get_distance(node_distances, temp_pos, next_id, 10000)
                        needed       = energy_usage(dist, temp_load)

                temp_battery -= needed
                temp_pos      = next_id
                temp_load     = temp_load - cust.get('delivery_weight', 0) + cust.get('pickup_weight', 0)
                new_route.append(copy.deepcopy(cust))

            routes[k] = new_route
            unassigned.remove(customer)

        # Phase 2: close all routes with hub_end
        for k in range(num_evs):
            if len(routes[k]) <= 1:
                continue

            # ==================== FIX: simulate actual battery level through the route ====================
            temp_battery = BATTERY_CAPACITY
            temp_load    = sum(
                n.get('delivery_weight', 0) for n in routes[k]
                if not n.get('charging_station', False)
                and n.get('customer_id') not in ('hub_start', 'hub_end')
            )
            prev_id = routes[k][0]['NodeID']
            for stop in routes[k][1:]:
                curr_id = stop['NodeID']
                dist = get_distance(node_distances, prev_id, curr_id, 0)
                temp_battery -= energy_usage(dist, temp_load)
                if stop.get('charging_station', False):
                    temp_battery = BATTERY_CAPACITY
                else:
                    temp_load -= stop.get('delivery_weight', 0)
                    temp_load += stop.get('pickup_weight', 0)
                    temp_load  = max(0, temp_load)
                prev_id = curr_id
            temp_pos = routes[k][-1]['NodeID']
            # ==================== END FIX ====================

            d_hub = get_distance(node_distances, temp_pos, HUB_NODE_ID, 10000)
            e_hub = energy_usage(d_hub, temp_load)

            if temp_battery - e_hub < 0:
                candidates = find_reachable_cs(temp_pos, temp_battery, temp_load, node_distances)
                nearest_cs = random.choice(candidates) if candidates else None
                if nearest_cs:
                    routes[k].append({
                        'NodeID':           nearest_cs,
                        'charging_station': True,
                        'delivery_weight':  0,
                        'pickup_weight':    0,
                        'customer_id':      f"CS_{nearest_cs}",
                    })
                    temp_pos     = nearest_cs
                    temp_battery = BATTERY_CAPACITY

            routes[k].append({
                'NodeID':           HUB_NODE_ID,
                'charging_station': False,
                'delivery_weight':  0,
                'pickup_weight':    0,
                'customer_id':      'hub_end',
            })

        # ==================== MODIFIED: chromosome now stores degradation metrics alongside routes ====================
        chromosome = [route for route in routes if len(route) > 2]
        if chromosome:
            vehicle_degradations = []
            vehicle_stats        = []
            total_distance       = 0.0

            for route in chromosome:
                # Add up distances
                prev_id = route[0]['NodeID']
                for node in route[1:]:
                    curr_id = node['NodeID']
                    total_distance += get_distance(node_distances, prev_id, curr_id, 0)
                    prev_id = curr_id
                    
                soc_series                    = build_soc_series(route, node_distances)
                deg, n_cyc, a_dod, a_soc     = compute_degradation(soc_series)
                vehicle_degradations.append(deg)
                vehicle_stats.append({
                    'num_cycles':  n_cyc,
                    'avg_dod':     a_dod,
                    'avg_soc':     a_soc,
                    'degradation': round(deg, 6),
                })

            fleet_health = 1.0 - (sum(vehicle_degradations) / len(vehicle_degradations))
            fleet_health = max(0.0, round(fleet_health, 6))

            solutions.append({
                'chromosome':    chromosome,
                'fleet_health':  fleet_health,
                'vehicle_stats': vehicle_stats,
                'total_distance': total_distance
            })
        # ==================== END MODIFIED ====================

        if (sol_idx + 1) % 100 == 0:
            print(f"  {sol_idx + 1}/{num_solutions} done...")

    print(f"Generated {len(solutions)} valid chromosomes.")
    return solutions


if __name__ == "__main__":

    DATASET_FOLDER   = r"C:\Users\Aryan Phad\OneDrive\Desktop\capstone\Dataset\Dataset\hundred"
    NUM_EVS          = 20
    NUM_SOLUTIONS    = 3
    SERVICE_TIME     = 5
    VEHICLE_MASS_VAL = 100

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

    dataset_files = sorted(glob.glob(os.path.join(DATASET_FOLDER, "*.txt")))[:3]
    if not dataset_files:
        print(f"No .txt files found in {DATASET_FOLDER}")
        exit(1)

    print(f"Found {len(dataset_files)} dataset file(s)\n")

    for dataset_file in dataset_files:
        dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*60}")

        params = parse_params(dataset_file)
        if not params:
            print("Could not parse parameters. Skipping.")
            continue

        battery_capacity      = params['battery_capacity']
        battery_capacity      = battery_capacity * 0.25  # reduce to 25% to force CS stops
        max_capacity          = params['max_capacity']
        beta                  = params['consumption_rate']
        min_battery_threshold = 0.3 * battery_capacity

        nodes, hub_node_id, charging_stations, customers = parse_dataset(dataset_file)
        NUM_EVS        = max(2, len(customers) // 3)
        node_distances = build_distance_dict(nodes)

        print(f"Customers: {len(customers)},  CS: {len(charging_stations)},  Hub: {hub_node_id}")
        print(f"Battery capacity: {battery_capacity},  Max capacity: {max_capacity},  Beta: {beta}")

        set_params(
            hub_node_id           = hub_node_id,
            charging_stations     = charging_stations,
            battery_capacity      = battery_capacity,
            vehicle_mass          = VEHICLE_MASS_VAL,
            beta                  = beta,
            min_battery_threshold = min_battery_threshold,
        )

        solutions = greedy_cvrpspd_solutions(
            dataset        = customers,
            num_evs        = NUM_EVS,
            max_capacity   = max_capacity,
            node_distances = node_distances,
            num_solutions  = NUM_SOLUTIONS,
        )

        print(f"Final chromosome count: {len(solutions)}")

        # ==================== MODIFIED: print loop updated to show BD metrics ====================
        for chrom_idx, sol in enumerate(solutions):
            chromosome    = sol['chromosome']
            fleet_health  = sol['fleet_health']
            vehicle_stats = sol['vehicle_stats']

            print(f"\n  Chromosome {chrom_idx + 1}:  Fleet Battery Health = {fleet_health}")
            for route_idx, route in enumerate(chromosome):
                stats = vehicle_stats[route_idx]
                print(f"    Route {route_idx + 1}:  "
                      f"cycles={stats['num_cycles']}  "
                      f"avg_dod={stats['avg_dod']}  "
                      f"avg_soc={stats['avg_soc']}  "
                      f"degradation={stats['degradation']}")
                for node in route:
                    node_type = "CS" if node.get('charging_station', False) \
                                else node.get('customer_id', node['NodeID'])
                    print(f"      -> {node['NodeID']}  ({node_type})  "
                          f"delivery={node.get('delivery_weight', 0)}  "
                          f"pickup={node.get('pickup_weight', 0)}")
        # ==================== END MODIFIED ====================