import pandas as pd
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import copy
from collections import defaultdict
from matplotlib.patches import FancyArrowPatch


# -----------------------------------------------------------------------------
# Chiplet specs with TOPS (in Tera‑ops/s) and energy_per_mac (in J)
# -----------------------------------------------------------------------------
chiplet_specs = {
    "Standard":    {"base": (128, 128), "rowKnob": 87.5, "colKnob": 12.5, "tops": 30.0, "energy_per_mac": 0.87e-12},
    "Shared":      {"base": (764, 764), "rowKnob": 68.0, "colKnob": 1.3,  "tops": 27.0, "energy_per_mac": 0.30e-12},
    "Adder":       {"base": (64,  64),  "rowKnob": 81.0, "colKnob": 0.4,  "tops": 11.0, "energy_per_mac": 0.18e-12},
    "Accumulator": {"base": (256, 256),"rowKnob": 55.0, "colKnob": 51.0, "tops": 35.0, "energy_per_mac": 0.22e-12},
    "ADC_Less":    {"base": (128, 128),"rowKnob": 64.4, "colKnob": 20.0, "tops": 3.8,  "energy_per_mac": 0.27e-12},
}

# -----------------------------------------------------------------------------
# Chip specs & capacity helper 
# -----------------------------------------------------------------------------
chipletTypesDict = {
    "Standard":    {"Size": 16384,  "Bits/cell": 2, "TOPS": 30e12,  "Energy/MAC": 0.87e-12},
    "Shared":      {"Size": 583696, "Bits/cell": 1, "TOPS": 27e12,  "Energy/MAC": 0.30e-12},
    "Adder":       {"Size": 4096,   "Bits/cell": 1, "TOPS": 11e12,  "Energy/MAC": 0.18e-12},
    "Accumulator": {"Size": 65536,  "Bits/cell": 2, "TOPS": 35e12,  "Energy/MAC": 0.22e-12},
    "ADC_Less":    {"Size": 163840,  "Bits/cell": 1, "TOPS": 3.8e12, "Energy/MAC": 0.27e-12},
}

XBARS_PER_TILE = 96
TILES_PER_CHIPLET = 16
MAX_CHIPLET_POWER = 8.0  # Watts

def get_chip_capacity_bits(chip_type, tiles=TILES_PER_CHIPLET, xbars=XBARS_PER_TILE):
    info = chipletTypesDict[chip_type]
    return info["Size"] * info["Bits/cell"] * xbars * tiles

def calculate_chiplets_needed(weight_bits, chiplet_type):
    info = chipletTypesDict[chiplet_type]
    xbar_capacity = info["Size"] * info["Bits/cell"]
    xbars_needed = math.ceil(weight_bits / xbar_capacity)
    return max(1, math.ceil(xbars_needed / (TILES_PER_CHIPLET * XBARS_PER_TILE)))
# -----------------------------------------------------------------------------
# getOUSize: finds best crossbar dims under 8 W instantaneous peak power
# -----------------------------------------------------------------------------
def getOUSize(xbar_sparsity, num_xbars, chiplet_type, weight_bits, activation_sparsity, _=None):
    try:
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.termination import get_termination
        from pymoo.optimize import minimize
    except ImportError:
        return _getOUSize_manual(xbar_sparsity, num_xbars, chiplet_type, weight_bits, activation_sparsity)

    spec = chiplet_specs[chiplet_type]
    base_r, base_c = spec["base"]
    base_tops = spec["tops"]
    base_energy = spec["energy_per_mac"]
    min_rows = int(base_r * activation_sparsity)
    scales = np.array([0.0675, 0.125, 0.25, 0.5, 0.75, 0.9, 1.0])

    class ProblemDef(Problem):
        def __init__(self):
            super().__init__(
                n_var=2, n_obj=2, n_constr=1,
                xl=np.array([0, 0]), xu=np.array([len(scales)-1, len(scales)-1])
            )
        def _evaluate(self, X, out, *args, **kwargs):
            idx = np.round(X).astype(int)
            idx = np.clip(idx, 0, len(scales)-1)
            row_scales = scales[idx[:,0]]
            col_scales = scales[idx[:,1]]

            F1 = np.zeros(len(X))
            F2 = np.zeros(len(X))
            G1 = np.zeros(len(X))

            for i in range(len(X)):
                r = int(base_r * row_scales[i])
                c = int(base_c * col_scales[i])
                if r < min_rows:
                    # invalid: too few rows
                    F1[i] = F2[i] = 1e6
                    G1[i] = 1e6
                    continue

                # scaled energy per MAC
                rowE = spec["rowKnob"] * row_scales[i]
                colE = spec["colKnob"] * col_scales[i]
                e_per_mac = base_energy * ((rowE + colE) / 100)

                # instantaneous TOPS → ops/s
                tops_ops = base_tops * row_scales[i] * col_scales[i] * 1e12

                # instantaneous peak power
                peak_power = e_per_mac * tops_ops

                # normalized EDP and power objectives
                edp = e_per_mac / tops_ops
                F1[i] = edp / (base_energy / (base_tops * 1e12))
                F2[i] = peak_power / (base_energy * base_tops * 1e12)

                # constraint: ≤ 8 W
                G1[i] = peak_power - MAX_CHIPLET_POWER

            out["F"] = np.column_stack([F1, F2])
            out["G"] = G1.reshape(-1,1)

    algorithm = NSGA2(
        pop_size=50,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", 150)
    res = minimize(ProblemDef(), algorithm, termination, seed=1, verbose=False)

    # choose best by 0.3*EDP + 0.7*Power
    weights = np.array([0.3, 0.7])
    best_idx = np.argmin((res.F * weights).sum(axis=1))
    i0, i1 = np.round(res.X[best_idx]).astype(int)
    i0, i1 = np.clip([i0, i1], 0, len(scales)-1)

    r = int(base_r * scales[i0])
    c = int(base_c * scales[i1])
    rowE = spec["rowKnob"] * scales[i0]
    colE = spec["colKnob"] * scales[i1]
    ## FIXed from previous version: add the remaining 100% - rowE - colE
    e_per_mac = base_energy * ((rowE + colE) / 100)
    scaled_tops = base_tops * scales[i0] * scales[i1]

    return r, c, scaled_tops, e_per_mac

def _getOUSize_manual(xbar_sparsity, num_xbars, chiplet_type, weight_bits, activation_sparsity, _=None):
    spec = chiplet_specs[chiplet_type]
    base_r, base_c = spec["base"]
    base_tops = spec["tops"]
    base_energy = spec["energy_per_mac"]
    min_rows = int(base_r * activation_sparsity)
    scales = [1.0, 0.9, 0.75, 0.5, 0.25, 0.125, 0.0675]
    best = None
    best_metric = float('inf')
    for rs in scales:
        for cs in scales:
            r, c = int(base_r*rs), int(base_c*cs)
            if r < min_rows:
                continue

            rowE = spec["rowKnob"]*rs + spec["colKnob"]*cs
            e_per_mac = base_energy * (rowE/100)
            tops_ops = base_tops * rs * cs * 1e12
            peak_power = e_per_mac * tops_ops
            if peak_power > MAX_CHIPLET_POWER:
                continue

            edp = e_per_mac / tops_ops
            norm_edp   = edp   / (base_energy/(base_tops*1e12))
            norm_pow   = peak_power/(base_energy*base_tops*1e12)
            metric = 0.3*norm_edp + 0.7*norm_pow

            if metric < best_metric:
                best_metric = metric
                best = (r, c, base_tops*rs*cs, e_per_mac)

    return best or (base_r, base_c, spec["tops"], spec["energy_per_mac"])

# -----------------------------------------------------------------------------
# Layer compute helper
# -----------------------------------------------------------------------------
def compute_layer_time_energy(allocation_list, total_macs):
    total_bits = sum(a["allocated_bits"] for a in allocation_list)
    times, energies = [], []
    for a in allocation_list:
        frac = a["allocated_bits"] / total_bits
        macs_i = total_macs * frac
        tops_i = a["optimized_tops"] * 1e12
        epm   = a["optimized_energy_per_mac"]
        t_i = macs_i / tops_i
        e_i = macs_i * epm
        times.append(t_i)
        energies.append(e_i)
    layer_time   = max(times)
    layer_energy = sum(energies)
    layer_power  = layer_energy / layer_time if layer_time>0 else 0
    layer_edp    = layer_energy * layer_time
    max_cpwr     = max((e/t if t>0 else 0) for e,t in zip(energies, times))
    return layer_time, layer_energy, layer_power, layer_edp, max_cpwr

# -----------------------------------------------------------------------------
# Scheduler + detailed per-layer & global summary
# -----------------------------------------------------------------------------
def scheduler(csv_path, chip_distribution):
    df = pd.read_csv(csv_path)
    df["Adjusted_Weights_bits"] = df["Weights(KB)"]*(1-df["Weight_Sparsity(0-1)"])*1024*8

    # build chip inventory
    inv = []
    types = list(chipletTypesDict.keys())
    for ct, cnt in zip(types, chip_distribution):
        for i in range(cnt):
            inv.append({"id":f"{ct}_{i}", "type":ct,
                        "capacity_left":get_chip_capacity_bits(ct)})
    layers = []
    for _, row in df.iterrows():
        layer_id     = int(row["Layer #"])
        rem_bits     = row["Adjusted_Weights_bits"]
        total_macs   = row["MACs"]
        allocs       = []
        total_bits   = rem_bits
        for chip in inv:
            if rem_bits <= 0: break
            if chip["capacity_left"] <= 0: continue
            alloc = min(rem_bits, chip["capacity_left"])
            # weightReq    = row["Weights(KB)"]*1024*8
            # cap          = chipletTypesDict[chip["type"]]["Size"]*chipletTypesDict[chip["type"]]["Bits/cell"]
            # XbarReqCeil  = math.ceil(weightReq / cap)
            # inherentS    = (XbarReqCeil - weightReq/cap)/XbarReqCeil
            # WS           = row["Weight_Sparsity(0-1)"]/XbarReqCeil
            # XbarSparsity = inherentS + WS
            AS           = row["Activation_Sparsity(0-1)"]
            weight_nonzero_bits = alloc

            # how many bits per crossbar
            cap = chipletTypesDict[chip["type"]]["Size"] * chipletTypesDict[chip["type"]]["Bits/cell"]

            # number of crossbars you really need to hold those non‑zero bits
            xbars_req = math.ceil(weight_nonzero_bits / cap)

            # if you spread the non‑zeros evenly across them:
            per_xbar_nonzeros = weight_nonzero_bits / xbars_req

            # fraction of each xbar that’s empty
            xbar_sparsity = (cap - per_xbar_nonzeros) / cap

            r, c, tops, epm = getOUSize(
                    xbar_sparsity,
                    xbars_req,
                    chip["type"],
                    weight_nonzero_bits,
                    row["Activation_Sparsity(0-1)"]
                )
            

            # base values if optimization method is not wanted
            # r = chiplet_specs[chip["type"]]["base"][0]
            # c = chiplet_specs[chip["type"]]["base"][1]
            # tops = chiplet_specs[chip["type"]]["tops"]
            # epm = chiplet_specs[chip["type"]]["energy_per_mac"]
            print(f"Rows: {r}, Cols: {c}, TOPS: {tops}, E/MAC: {epm:.5e} J")

            frac        = alloc / total_bits
            macs_assigned = total_macs * frac
            util        = alloc / chip["capacity_left"]

            chip["capacity_left"] -= alloc
            rem_bits    -= alloc

            allocs.append({
                "chip_id": chip["id"],
                "chip_type": chip["type"],
                "allocated_bits": int(alloc),
                "MACs_assigned": int(macs_assigned),
                "Chiplets_reqd": math.ceil(xbars_req/(TILES_PER_CHIPLET*XBARS_PER_TILE)),
                "Crossbars_used": xbars_req,
                "Crossbar_sparsity": xbar_sparsity,
                "weight sparsity":row["Weight_Sparsity(0-1)"],
                "Activation Sparsity": AS,
                "optimal_ou_row": r,
                "optimal_ou_col": c,
                "optimized_tops": tops,
                "optimized_energy_per_mac": epm,
                "chiplet_resource_taken": util*100
            })

        if rem_bits > 0:
            raise RuntimeError(f"Layer {layer_id} not fully allocated: {rem_bits:.0f} bits remain")

        t, e, p, edp, maxp = compute_layer_time_energy(allocs, total_macs)
        layers.append({
            "layer": layer_id,
            "allocations": allocs,
            "time_s": t,
            "energy_J": e,
            "avg_power_W": p,
            "edp": edp,
            "max_chiplet_power_W": maxp
        })

    return layers

def extract_layer_summary(csv_path, chip_distribution):
    """
    Runs scheduler(csv_path, chip_distribution) and returns:
      • df_summary: DataFrame with per‑layer: Layer, Chips_Required, Time_s, EDP_Js,
        Activations_KB and Avg_Power_per_Chip_W
      • slowest_layer: layer ID with the maximum Time_s
      • workload_edp: total_energy * max_latency over all layers
      • num_chips_used: count of distinct chip_ids allocated across all layers
    """
    # 1) run your existing scheduler
    layers = scheduler(csv_path, chip_distribution)

    # 2) read activations from the CSV
    df_csv = pd.read_csv(csv_path)

    # 3) build per‑layer summary
    summary = []
    used_chips = set()
    for layer in layers:
        lid   = layer["layer"]
        time  = layer["time_s"]
        edp_l = layer["edp"]
        # look up Activations(KB) by layer #
        act_kb = float(df_csv.loc[df_csv["Layer #"] == lid, "Activations(KB)"].iloc[0])
        # count how many chips were used in this layer
        n_chips = len(layer["allocations"])

        # collect each allocation's resource_taken (percent)
        utils = [a["chiplet_resource_taken"] for a in layer["allocations"]]
        # track distinct chips globally
        used_chips.update(a["chip_id"] for a in layer["allocations"])
        # average power per chip
        avg_p_chip = layer["avg_power_W"] / n_chips if n_chips else 0.0
        ###### Can use average or the last chips utilization
        # average % utilization across chips, rounded
        chip_util  = round(sum(utils) / n_chips, 4) if n_chips else 0.0
        # NEW: take the last chip's utilization instead of the average
        #chip_util  = utils[-1]  if utils else 0.0

        summary.append({
            "Layer":                lid,
            "Chips_Required":       n_chips,
            "Chip_Utilization%":     chip_util,
            "Time_s":               time,
            "EDP_Js":               edp_l,
            "Activations_KB":       act_kb,
            "Avg_Power_per_Chip_W": avg_p_chip
        })

    # 4) dataframe + slowest layer
    df_summary    = pd.DataFrame(summary).sort_values("Layer")
    slowest_layer = int(df_summary.loc[df_summary["Time_s"].idxmax(), "Layer"])

    # 5) total workload EDP
    max_latency  = max(l["time_s"]   for l in layers)
    total_energy = sum(l["energy_J"] for l in layers)
    workload_edp = total_energy * max_latency

    # 6) how many distinct chips got used
    num_chips_used = len(used_chips)

    for lr in layers:
        print(f"\nLayer {lr['layer']}:")
        for a in lr["allocations"]:
            print(" ", a)
        print(f"  → Time: {lr['time_s']:.3e}s, Energy: {lr['energy_J']:.3e}J, "
              f"Power: {lr['avg_power_W']:.3e}W, MaxP: {lr['max_chiplet_power_W']:.3e}W, EDP: {lr['edp']:.3e}")

    # tab-separated so Excel will parse into columns
    df_summary.to_clipboard(sep="\t", index=False)
    print(f"[+] Per-layer summary copied to clipboard (TSV).")
    

    return df_summary, slowest_layer, workload_edp, num_chips_used, total_energy, max_latency

def getLayerStats(chip_dist):
# list all of your model CSVs here
    workloads = [
        "workloads/vgg16_stats.csv" # Only work with vgg16 for now
        # "workloads/vgg19_stats.csv",
        # "workloads/resnet18_stats.csv",
        # "workloads/resnet50_stats.csv",
        # "workloads/resnet101_stats.csv",
    ]

    for wl in workloads:
        df_sum, slow, w_edp, n_chips, total_energy, max_latency = extract_layer_summary(wl, chip_dist)
        print(f"\n=== {wl} ===")
        print(df_sum.to_string(index=False))
        print(f"Distinct chips used: {n_chips}")
        print(f"Layer taking longest: {slow}")
        print(f"Final compute latency: {max_latency:.3e} s")
        print(f"Final compute energy : {total_energy:.3e} J")
        print(f"Final compute EDP: {w_edp:.3e} J·s")

# --- UTILITIES ---

def factor_pairs(n):
    """Return all pairs (r, c) such that r * c == n."""
    pairs = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            pairs.append((i, n // i))
    return pairs

def choose_dims(n):
    """Choose pair (rows, cols) with minimal |rows - cols| from factor pairs of n."""
    pairs = factor_pairs(n)
    if not pairs:
        return (1, n)
    # Include both orientations
    all_pairs = pairs + [(c, r) for r, c in pairs if r != c]
    # Pick the pair with the smallest difference
    return min(all_pairs, key=lambda x: abs(x[0] - x[1]))

# --- GRAPH BUILDERS ---

def buildMeshGraph(rows, cols):
    G = nx.Graph()
            
    for i in range(rows):
        for j in range(cols):
            G.add_node((i,j))
            if i + 1 < rows:
                G.add_edge((i, j), (i + 1, j))
            if j + 1 < cols:
                G.add_edge((i, j), (i, j + 1))

    return G

def buildKiteGraph(rows, cols):
    G = nx.Graph()

    for i in range(rows):
        for j in range(cols):
            G.add_node((i, j))
            if i + 2 < rows:
                G.add_edge((i, j), (i + 2, j), weight=2)
            if j + 2 < cols:
                G.add_edge((i, j), (i, j + 2), weight=2)

    for i in range(rows):
        for j in range(cols):
            node = (i,j)
            current_degree = G.degree(node)
            if current_degree == 4:
                continue
            # Define neighbor directions to try (right, down, left, up)
            neighbors = [(i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j)]
            for nbr in neighbors:
                if nbr in G.nodes and G.degree(nbr) < 4 and not G.has_edge(node, nbr):
                    G.add_edge(node, nbr)
                    current_degree += 1
                    if current_degree == 4:
                        break

    return G

def buildHexaMeshGraph(rows, cols):
    G = nx.Graph()

    for i in range(rows):
        for j in range(cols):
            G.add_node((i, j))
            even_row = (i % 2 == 0)

            # Right neighbor
            if j + 1 < cols:
                G.add_edge((i, j), (i, j + 1))
            # Bottom-right neighbor
            if i + 1 < rows and (j + (0 if even_row else 1)) < cols:
                G.add_edge((i, j), (i + 1, j + (0 if even_row else 1)), weight=2)
            # Bottom-left neighbor
            if i + 1 < rows and (j - (1 if even_row else 0)) >= 0:
                G.add_edge((i, j), (i + 1, j - (1 if even_row else 0)), weight=2)

    return G

# --- MAIN BUILDER ---

def build_chiplet_mesh(results, topology="mesh"):
    if topology == "floret":
        G, chiplet_name_map, label_map, lam, psi = Floret.build_floret_graph_from_results(results)
        return G, chiplet_name_map, label_map, lam, psi
    
    chiplet_to_layers = defaultdict(list)

    # Step 1: Map each chiplet to all layers that used it
    for lr in results:
        layer_id = lr["layer"]
        for alloc in lr["allocations"]:
            chip_id = alloc["chip_id"]
            chiplet_to_layers[chip_id].append(layer_id)

    # Step 2: Determine layout shape
    chiplet_ids = list(chiplet_to_layers.keys())
    total_chiplets = len(chiplet_ids)
    rows, cols = choose_dims(total_chiplets)

    # Step 3: Build appropriate topology
    if topology == "mesh":
        G = buildMeshGraph(rows, cols)
        pos_func = get_grid_positions
    elif topology == "kite":
        G = buildKiteGraph(rows, cols)
        pos_func = get_grid_positions
    elif topology == "hexa":
        G = buildHexaMeshGraph(rows, cols)
        pos_func = getHexPositions
    elif topology == "sfc":
        G = topologyComparison.build_SFC_graph(rows, cols)
        pos_func = get_grid_positions
    else:
        raise ValueError(f"Unsupported topology type: {topology}")
    
    # Step 4: Map (x, y) grid coords → chiplet names
    mapping = {}
    chiplet_name_map = {}
    i = 0
    for x in range(rows):
        for y in range(cols):
            if i < total_chiplets:
                node_name = chiplet_ids[i]
                mapping[(x, y)] = node_name
                chiplet_name_map[node_name] = pos_func(rows, cols).get((x, y), (y, -x))  # fallback
                i += 1

    G = nx.relabel_nodes(G, mapping)
    
    # Step 5: Format layer labels for each chiplet
    label_map = {}
    for chiplet, layers in chiplet_to_layers.items():
        label_map[chiplet] = f"{chiplet}\nL{','.join(map(str, layers))}"

    return G, chiplet_name_map, label_map

# --- POSITIONING ---

def get_grid_positions(rows, cols):
    return {(i, j): (j, -i) for i in range(rows) for j in range(cols)}

def getHexPositions(rows, cols, spacing=1.0):
    pos = {}
    for i in range(rows):
        for j in range(cols):
            x = j * spacing + (spacing / 2 if i % 2 else 0)
            y = -i * (spacing * 0.866)  # sin(60°) ≈ 0.866 for hex height
            pos[(i, j)] = (x, y)
    return pos

# --- PLOTTING ---

def plot_chiplet_mesh(G, pos, labels, topology="mesh", lam=None, psi=None):
    output_dir = "WorkloadLayerMappingImages"
    os.makedirs(output_dir, exist_ok=True)

    chiplet_id = next(iter(labels.values())).split("\n")[0]
    base_name = chiplet_id.split("_0")[0] if "_0" in chiplet_id else chiplet_id
    filename = f"{output_dir}/{base_name}_{topology}.png"

    if topology == "kite":
        fig, ax = draw_curved_graph(G, pos, labels, rad=0.2)
        fig.savefig(filename, bbox_inches='tight')
    elif topology == "floret":
        if lam is None or psi is None:
            raise ValueError("lam and psi must be provided for floret topology.")
        Floret.display_floret_graph(G, pos, labels, lam, psi, show=False) 
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.figure(figsize=(10, 7))
        nx.draw(G, pos, with_labels=True, labels=labels,
                node_size=1000, node_color='lightblue', font_size=8)
        plt.title(f"{topology.capitalize()} Layout with Layer Assignments")
        plt.savefig(filename, bbox_inches='tight')

    print(f"✅ Mesh plot saved as: {filename}")
    #plt.show()


def draw_curved_graph(G, pos, labels, rad=0.2, title: str = None):
    fig, ax = plt.subplots(figsize=(10, 7))

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1000, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)

    # Curved edges
    for u, v in G.edges():
        if u == v: continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                connectionstyle=f"arc3,rad={rad}",
                                color='gray',
                                arrowstyle='-',
                                linewidth=1)
        ax.add_patch(arrow)

    # Use the caller’s title, or no title at all
    if title is not None:
        ax.set_title(title)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    return fig, ax


def simulate_packet_transfer_graph(G, src, dst, params):
    """
    Simulates packet transfer over a single-level graph topology (e.g. mesh, kite, hexamesh).
    Assumes every hop is an interposer-level (NoI) hop.
    """
    path = nx.shortest_path(G, source=src, target=dst, weight="weight")
    raw_hops = len(path) - 1
    weighted_hops = sum(G[path[i]][path[i+1]].get("weight", 1) for i in range(len(path) - 1))

    packet_bits = params["packet_size_bytes"] * 8

    latency_per_hop_ns = params["noi_hops_latency_ns"]
    energy_per_bit = params["e_cross"]
    bandwidth_bps = params["noi_bandwidth_bits_per_sec"]
    frequency = params["freq_inter_hz"]

    total_latency_ns = weighted_hops * latency_per_hop_ns
    total_energy = weighted_hops * packet_bits * energy_per_bit
    total_cycles = int(total_latency_ns * frequency * 1e-9)
    edp = total_energy * total_latency_ns

    time_per_packet_s = (packet_bits * weighted_hops) / bandwidth_bps

    print(f"Path: {path}")
    print(f"Total Hops: {raw_hops}")
    print(f"Weighted Hops: {weighted_hops}")
    print(f"Latency: {total_latency_ns} ns ({total_cycles} cycles)")
    print(f"Energy: {total_energy:.2e} J")
    print(f"EDP: {edp:.2e} J·ns")
    print(f"Time per packet (based on bandwidth): {time_per_packet_s:.2e} s")

    return {
        "path": path,
        "hops": raw_hops,
        "weighted_hops": weighted_hops,
        "latency_ns": total_latency_ns,
        "energy_joules": total_energy,
        "cycles": total_cycles,
        "edp": edp,
        "time_per_packet_s": time_per_packet_s
    }

def simulate_activations_between_layers(workload, results, G, params):
    """
    Simulates activation transfer between layers in a CNN across chiplets.

    Handles:
    - Layers split across multiple chiplets
    - Chiplets with multiple layers
    - Intra-chiplet activation flow (NoC)

    Args:
        workload: List of dicts with 'layer' and 'activations_kb'
        results: List of dicts with 'layer' and 'allocations' (each has 'chip_id')
        G: networkx graph of chiplet-level connectivity
        params: dictionary of energy, latency, freq, and bandwidth

    Returns:
        List of simulation records (one per inter-chiplet transfer)
    """
    # Map: layer → list of chiplets it's on
    layer_to_chiplets = defaultdict(list)
    for r in results:
        layer = r['layer']
        for alloc in r['allocations']:
            layer_to_chiplets[layer].append(alloc['chip_id'])

    simulation_log = []

    for i in range(len(workload) - 1):
        src_layer = i
        dst_layer = i + 1
        activations_kb = workload[i]["activations_kb"]
        packet_bits = activations_kb * 8192  # 1 KB = 8192 bits

        # Get src and dst chiplets to iterate over later
        src_chiplets = layer_to_chiplets.get(src_layer, [])
        dst_chiplets = layer_to_chiplets.get(dst_layer, [])

        # Check for intra-chiplet communication (NoC)
        # common_chiplets = set(src_chiplets) & set(dst_chiplets)
        # for chip in common_chiplets:
        #     print(f"🧠 NoC Transfer on {chip}: L{src_layer} → L{dst_layer}")

        # Inter-chiplet traffic: all combinations of src_chiplet → dst_chiplet
        for sc in src_chiplets:
            for dc in dst_chiplets:
                if sc == dc:
                    continue  # skip NoC (already logged)

                if sc not in G.nodes or dc not in G.nodes:
                    print(f"⚠️ Chiplet {sc} or {dc} not in graph")
                    continue

                try:
                    path = nx.shortest_path(G, source=sc, target=dc, weight="weight")
                except nx.NetworkXNoPath:
                    print(f"❌ No path from {sc} to {dc}, skipping.")
                    continue

                # both diagonal and direct single hop links exist. Recalculate the number of hops  considering the diagonal link
                raw_hops = len(path) - 1 # 
                weighted_hops = sum(G[path[i]][path[i+1]].get("weight",1) for i in range(len(path)-1))
                # in case of weighted hops, we need to multiply it by the number of packets going over the link.
                # maintain an array of path. Some path is 1 cycle, some is 2 cycle and so on
                #if ("Mesh, Kite", k = 32; "Floret", k = 64; "HexaMesh", k = 24)
                # as the layer is divided on k chiplets; Li+1 is on m chiplets, the packet_bits is also going to scale down to capture that we are now
                # we are now reducing the number of packet_bits per chiplet where it is divided by k. Hence, packate_bits_new = pscket_bits/k
                packet_bits_per_chip = packet_bits / len(src_chiplets)
                
                # print(f"\n--- Simulation for Layer {src_layer} → {dst_layer} ---")
                # print(f"Packet bits (per chiplet): {packet_bits_per_chip:.2f} bits")
                # print(f"Weighted hops: {weighted_hops}")
                # print(f"Bus width: {params['bus_width']} bits")

                hop_latency_ns = params["noi_hops_latency_ns"]
                latency_ns = (weighted_hops * hop_latency_ns *(packet_bits_per_chip/(params["bus_width"])))
                latency_s = latency_ns * 1e-9
                # Latency based -> cyles till first bit arrives at destination
                cycles_latency = int(latency_s * params["freq_inter_hz"]) 

                # print(f"Latency-based:")
                # print(f"  latency_ns = {latency_ns:.4f} ns")
                # print(f"  latency_s = {latency_s:.6e} s")
                # print(f"  cycles_latency = {cycles_latency}")

                bandwidth_bps = params["noi_bandwidth_bits_per_sec"]
                time_throughput_s = (packet_bits_per_chip * weighted_hops) / bandwidth_bps
                #throughput based -> how many clock cycles it takes to push ALL the bits through the channel.
                cycles_throughput = int(time_throughput_s * params["freq_inter_hz"]) 
                # print(f"Throughput-based:")
                # print(f"  time_throughput_s = {time_throughput_s:.6e} s")
                # print(f"  cycles_throughput = {cycles_throughput}")

                cycles = cycles_latency # latency for now, how soon destination is recieving bits
                
                energy_j = weighted_hops * packet_bits_per_chip * params["e_cross"]/(params["bus_width"])
                edp = energy_j * latency_s # J•s
                # print(f"Energy: {energy_j:.4e} J")
                # print(f"EDP: {edp:.4e} J·s")

                simulation_log.append({
                    "src_layer": src_layer,
                    "dst_layer": dst_layer,
                    "src_chiplet": sc,
                    "dst_chiplet": dc,
                    "activations_kb": activations_kb,
                    "hops": raw_hops,
                    "weighted_hops": weighted_hops,
                    "latency_s": latency_s,
                    "energy_joules": energy_j,
                    "cycles": cycles,
                    "edp": edp,
                    "path": path
                })

    return simulation_log


if __name__ == "__main__":
    workload_csv = "workloads/vgg16_stats.csv"
    df = pd.read_csv(workload_csv)
    workload = [
        { "layer": int(row["Layer #"]), "activations_kb": float(row["Activations(KB)"]) }
        for _, row in df.iterrows()
    ]
    chip_dist    = [100, 0, 0, 0, 0]# hetOU
    results      = scheduler(workload_csv, chip_dist)

    # per-layer details
    for lr in results:
        print(f"\nLayer {lr['layer']}:")
        for a in lr["allocations"]:
            print(" ", a)
        print(f"  → Time: {lr['time_s']:.3e}s, Energy: {lr['energy_J']:.3e}J, "
              f"Power: {lr['avg_power_W']:.3e}W, MaxP: {lr['max_chiplet_power_W']:.3e}W, EDP: {lr['edp']:.3e}")

    # global summary for computation costs
    compute_energy = sum(l["energy_J"] for l in results)
    compute_latency  = max(l["time_s"]   for l in results)
    compute_workload_edp = compute_energy * compute_latency
    print("\nWorkload summary:")
    print(f"  Final Compute latency: {compute_latency:.3e} s")
    print(f"  Final Compute energy : {compute_energy:.3e} J")
    print(f"  Final Compute EDP    : {compute_workload_edp:.3e} J·s")

    # check violations
    bad = [l["layer"] for l in results if l["max_chiplet_power_W"] > MAX_CHIPLET_POWER]
    if bad:
        print(f"\nWarning: layers {bad} exceed the {MAX_CHIPLET_POWER}W peak‑power cap.")
    
    print("\nGettig layer stats\n")
    # getLayerStats(chip_dist)
    topologies = [
        "kite",
        "mesh",
        "hexa",
        "floret"
        #"sfc"
    ]

    params = {
    "noc_hops_latency_ns": 1,
    "noi_hops_latency_ns": 0.8694,  # 1.449 mm × 0.6 ns/mm - 1 hop = this time
    "packet_size_bytes": 64,
    "freq_intra_hz": 2e9,
    "freq_inter_hz": 1.15e9, # consistent / cannot change
    "e_intra": 10e-12,
    "e_cross": 50e-12,
    "noc_bandwidth_bits_per_sec": 32e9,
    "noi_bandwidth_bits_per_sec": 4.6e9, # Bus width (32) * Frequency / 8 bits
    "bus_width" : 32
    }

    bandwidth_scale = {
    "kite":   1.0,
    "mesh":   1.0,
    "hexa":   1/3,
    "floret": 2.0
    #"sfc":    2.0
    }

    buswidth_scale = {
    "kite":   1.0,
    "mesh":   1.0,
    "hexa":   3/4,
    "floret": 2.0
    #"sfc":    2.0
    }

    ## FOCUSING ON COMPUTE TIME FOR NOW, BELOW IS COMMUNICATION COSTS
    # for t in topologies:
    #     result = build_chiplet_mesh(results, topology=t)
    #     if t == "floret":
    #         G, pos, labels, lam, psi = result
    #         # plot_chiplet_mesh(G, pos, labels, topology=t, lam=lam, psi=psi) UNCOMMENT TO PLOT
    #     else:
    #         G, pos, labels = result
    #         # plot_chiplet_mesh(G, pos, labels, topology=t) UNCOMMENT TO PLOT
        
    #     # Initialize totals
    #     total_hops = 0
    #     total_weighted_hops = 0
    #     communicate_latency = 0
    #     longest_layer = 0
    #     total_cycles = 0
    #     communicate_energy = 0.0

    #     # Track total latency per layer transfer
    #     layer_pair_latency = defaultdict(float)

    #      # Scale interconnect bandwidth before simulating
    #     scaled_params = params.copy()
    #     scaled_params["noi_bandwidth_bits_per_sec"] = (params["noi_bandwidth_bits_per_sec"] * bandwidth_scale.get(t, 1.0))
    #     scaled_params["bus_width"] = (params["bus_width"] * buswidth_scale.get(t, 1.0))

    #     print(f"Topology: {t}")
    #     logs = simulate_activations_between_layers(workload, results, G, scaled_params)
    #     for log in logs:
    #         # print(f"Layer {log['src_layer']} → {log['dst_layer']}, Latency: {log['latency_s']:.2e} s, Energy: {log['energy_joules']:.2e} J, Path: {log['path']}")
    #         # print(f"L{log['src_layer']} → L{log['dst_layer']}, {log['src_chiplet']} → {log['dst_chiplet']}, {log['activations_kb']} KBs,{log['hops']} hops, {log['energy_joules']:.2e} J")
    #         layer_pair = (log["src_layer"], log["dst_layer"])
    #         layer_pair_latency[layer_pair] += log["latency_s"]
    #         # Accumulate totals 
    #         total_hops       += log["hops"]
    #         total_weighted_hops += log["weighted_hops"]
    #         total_cycles     += log["cycles"]
    #         communicate_energy += log["energy_joules"]

    #     # Determine the longest layer transfer time
    #     longest_layer_pair = max(layer_pair_latency, key=layer_pair_latency.get)
    #     communicate_latency = layer_pair_latency[longest_layer_pair]
    #     longest_layer = longest_layer_pair[0]  

    #     # Communcation costs for each topology
    #     print(f"\n📊 Communication Cost Summary for {t.upper()}:")
    #     print(f"Longest layer: {longest_layer} -> {longest_layer+1}")
    #     print(f"Latency: {communicate_latency:.2e} s ({total_cycles} cycles)")
    #     print(f"Energy: {communicate_energy:.2e} J")
    #     print(f"EDP: {(communicate_energy * communicate_latency):.2e} J•s")

        # total_energy_combined = communicate_energy + compute_energy
        # total_latency_combined = max(communicate_latency, compute_latency)
        # total_edp_combined = total_energy_combined * total_latency_combined
        
        # print(f"\n📊 Combined Compute and Communication Cost Summary for {t.upper()}:")
        # print(f"Latency: {total_latency_combined:.2e} s")
        # print(f"Energy: {total_energy_combined:.2e} J")
        # print(f"EDP: {(total_energy_combined * total_latency_combined):.2e} J•s")

        #results_copy = copy.deepcopy(results)
