import os
import numpy as np
import pandas as pd
import math
import hashlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter, LogLocator
from itertools import product
import mapperV3
import subprocess


XBARS_PER_TILE = 96
TILES_PER_CHIPLET = 16

chipletSpecs = {
    "Standard":    {"base": (128, 128), "rowKnob": 87.5, "colKnob": 12.5, "tops": 30.0, "energy_per_mac": 0.87e-12},
    "Shared":      {"base": (764, 764), "rowKnob": 68.0, "colKnob": 1.3,  "tops": 27.0, "energy_per_mac": 0.30e-12},
    "Adder":       {"base": (64,  64),  "rowKnob": 81.0, "colKnob": 0.4,  "tops": 11.0, "energy_per_mac": 0.18e-12},
    "Accumulator": {"base": (256, 256),"rowKnob": 50.5, "colKnob": 49.5, "tops": 35.0, "energy_per_mac": 0.22e-12},
    "ADC_Less":    {"base": (128, 128),"rowKnob": 64.4, "colKnob": 20.0, "tops": 3.8,  "energy_per_mac": 0.27e-12},
}
# Tops should be multiplied by 10e12 

chipletTypesDict = {
    "Standard":    {"Size": 16384,  "Bits/cell": 2, "TOPS": 30e12,  "Energy/MAC": 0.87e-12},
    "Shared":      {"Size": 583696, "Bits/cell": 1, "TOPS": 27e12,  "Energy/MAC": 0.30e-12},
    "Adder":       {"Size": 4096,   "Bits/cell": 1, "TOPS": 11e12,  "Energy/MAC": 0.18e-12},
    "Accumulator": {"Size": 65536,  "Bits/cell": 2, "TOPS": 35e12,  "Energy/MAC": 0.22e-12},
    "ADC_Less":    {"Size": 163840,  "Bits/cell": 1, "TOPS": 3.8e12, "Energy/MAC": 0.27e-12},
}

## Readjusts the Energy and Tops of a chiplet based on the new row and col (for OU) as well as giving power density
def customizeOU(row: int, col: int, chipletName: str):
    baseRow, baseCol = chipletSpecs[chipletName]["base"]
    epm = chipletSpecs[chipletName]["energy_per_mac"]
    rowKnob = chipletSpecs[chipletName]["rowKnob"]
    rowKnobPercent = rowKnob / 100.0  # Convert to fraction
    colKnob = chipletSpecs[chipletName]["colKnob"]
    colKnobPercent = colKnob / 100.0  # Convert to fraction

    EnergyRow = epm *  rowKnobPercent * (row / baseRow)
    EnergyCol = epm *  colKnobPercent * (col / baseCol)

    # Energy per mac is Energy Total
    EnergyTotal = EnergyRow + EnergyCol # J / operation
    tops = chipletSpecs[chipletName]["tops"] * (row / baseRow) * (col / baseCol) # Adjust TOPS

    powerDensity = EnergyTotal * tops * 1e12  # Scaled by 1e12 
    # Power (Watts) = J/operation * Operation/second = J/s (W)
    return EnergyTotal, tops, powerDensity
    

## Finds factors of n that are foldable with a input step size and constrained to min_row, max_row, and max_col, returns all possible options
# if n = 1147 it rounds to step of 16 -> 1152 and checks factors of increments of 16 so it checks 1168, 1184, 1200 etc until nice factors
def get_approx_foldable_factors(n, min_row, max_row, max_col, step=16):
    """
    Finds (row, col) pairs such that:
    - row ≥ min_row and ≤ max_row
    - col ≤ max_col
    - row and col are multiples of `step`
    - row * col ≥ n
    """
    factors = []

    # Round n up to multiple of step
    if n % step != 0:
        n = ((n // step) + 1) * step

    # Round min_row up to multiple of step
    if min_row % step != 0:
        min_row = ((min_row // step) + 1) * step

    # Iterate possible rows
    for row in range(min_row, max_row + 1, step):
        raw_col = math.ceil(n / row)

        # Round col up to next multiple of step
        if raw_col % step != 0:
            col = ((raw_col // step) + 1) * step
        else:
            col = raw_col

        # If col > max_col but row ≤ max_col, try swapping
        if col > max_col and row <= max_col and row >= min_row:
            if row <= max_col and col <= max_row:  # swap and check constraints
                factors.append((col, row))

        # Normal check (no swap needed)
        if col <= max_col:
            factors.append((row, col))

    return factors

## Input is a list of factors, and it selects the config with the smallest row value.
def select_lowest_row_config(factor_list, min_R):
    """
    Selects the config with the smallest row value >= min_R.
    If multiple have the same row, returns the one with the smallest column.
    """
    if not factor_list:
        return None

    # Filter factors to only those meeting row >= min_R
    valid_factors = [f for f in factor_list if f[0] >= min_R]

    # If no valid factors remain, return None
    if not valid_factors:
        return None

    # Select lowest row, then lowest column among valid
    return min(valid_factors, key=lambda x: (x[0], x[1]))


## Pareto-based rank seletion function selects the config with the lowest EDP
def rank_based_selection(configs):
    if not configs:
        print("No valid OU configurations found for ranking.")
        return None  # gracefully handle empty input
    
    best_config = min(configs, key=lambda x: x["edp"])
    return best_config

## finds the configurations that are powers of 2 and have the minimum area based on weight sparsity, and returns smallest area
def get_power2_crossbar_dims(weight_sparsity, chiplet_dim):
    # Get required MACS based on weight sparsity
    required_macs = chiplet_dim * chiplet_dim * (1 - weight_sparsity)

    # Possible powers of 2 for dimensions up to chiplet size
    powers = [2 ** i for i in range(3, int(math.log2(chiplet_dim)) + 1)]

    # Store valid (row, col) pairs
    valid_dims = []
    for r, c in product(powers, repeat=2):
        if r * c >= required_macs:
            valid_dims.append((r, c))

    if not valid_dims:
        # fallback: largest possible power-of-two square
        max_power = 2 ** int(math.log2(chiplet_dim))
        return (max_power, max_power)

    # Return the one with minimal area (r * c)
    best_dim = min(valid_dims, key=lambda x: x[0] * x[1])
    return best_dim

# Helper function to find best OU for a chiplet based on the layer characteristics
def best_OU_for_chip(chipletName: str, idealCrossbarDim: tuple, crossbarReq: int):
    
    if chipletName == "Accumulator":
        idealCrossbarDim = (min(idealCrossbarDim[0], 32), min(idealCrossbarDim[1], 32))

    if chipletName == "Shared":
        idealCrossbarDim = (idealCrossbarDim[0] * 2 , idealCrossbarDim[1] * 2)
    colLimit = idealCrossbarDim[1] # set colum limit to the required dimension

    if chipletName == "Standard":
        colLimit = 32 # FOR IR LIMITATION, only use for Standard

    accumulatorBufferSize = 32 # Set accumulator buffer size to 32
    step = 4
    max_col_limit = min(colLimit, idealCrossbarDim[1]) # choose the lowest value as limit for column search space
    possibleOUConfigs = []
    for ou_row in range(step, max(idealCrossbarDim[0], step) + 1, step):
        for ou_col in range(step, max_col_limit + 1, step):
            # How many OUs to cover the "ideal area" for this layer
            OUrequired = math.ceil((idealCrossbarDim[0] * idealCrossbarDim[1]) / (ou_row * ou_col))

            # Latency/Energy model (matching your chip-type equations)
            if chipletName == "Standard":
                latency = (ou_row * 4) * OUrequired
                energy  = crossbarReq * ou_row * ou_col * latency
            elif chipletName in ("Shared", "Adder"):
                latency = 4 * ou_row * ou_col * math.log2(max(2, ou_row)) * OUrequired
                energy  = crossbarReq * ou_row * latency
            elif chipletName == "Accumulator":
                latency = 4 * ou_row * OUrequired * 2 * (OUrequired / accumulatorBufferSize)
                energy  = crossbarReq * ou_row * ou_col * latency
            else:  # ADCLESS or unknown (your code set zeros)
                latency = 0
                energy  = 0

            edp = latency * energy

            # Your per-OU customizations (EPM, TOPS, power density)
            epm, tops, power_density = customizeOU(ou_row, ou_col, chipletName)
            if power_density > 8:
                continue  # respect your power density limit

            possibleOUConfigs.append({
                "chip": chipletName,
                "ou_row": ou_row,
                "ou_col": ou_col,
                "latency": latency,
                "energy": energy,
                "edp": edp,
                "power_density": power_density,
                "epm": epm,
                "tops": tops,
                "OUrequired": OUrequired,
            })
    # Your ranker decides the Pareto/best tradeoff
    best = rank_based_selection(possibleOUConfigs)
    return best


# MAIN FUNCTION
########################################################################################################
# Parameters are chiplet distribution, chiplet Name, a given workload from workloads/name_stats.csv, and optional manual mode 
# where you modify the ou row and ou col within the function (inside the if manual is True block)
# Main function to compute crossbar metrics for a given chiplet and workload as it chooses optimal OU config
def computeCrossbarMetrics(chip_distribution, chipletName: str, workloadStatsCSV: str,  manualOU: bool = False, manual_ou_row: int = None, manual_ou_col: int = None):
    df = pd.read_csv(workloadStatsCSV)

    # build chip inventory
    inv = []
    types = list(chipletTypesDict.keys())
    for ct, cnt in zip(types, chip_distribution):
        for i in range(cnt):
            inv.append({"id":f"{ct}_{i}", "type":ct,
                        "capacity_left": mapperV3.get_chip_capacity_bits(ct)})


    results = []
    layers = []
    layer = 0
    # Iterate through the workload
    with open("workload_OU_config_results.txt", "w") as f:
        for _, row in df.iterrows():
            layer += 1
            rem_bits = row["Weights(KB)"] * (1 - row["Weight_Sparsity(0-1)"]) * 1024 * 8
            total_macs   = row["MACs"]
            allocs       = []
            total_bits   = rem_bits

            # Get Layer characteristics
            weightsKB = row["Weights(KB)"]
            weightSparsity = row["Weight_Sparsity(0-1)"]
            activationSparsity = row["Activation_Sparsity(0-1)"]
            activationsKB = row["Activations(KB)"]



            # ** For Computing EPD based on the selected OU and its other characteristics ** #
            for chip in inv:
                if rem_bits <= 0: break
                if chip["capacity_left"] <= 0: continue
                alloc = min(rem_bits, chip["capacity_left"])
                AS           = row["Activation_Sparsity(0-1)"]
                weight_nonzero_bits = alloc

                baseCrossbarRow = chipletSpecs[chip["type"]]["base"][0]  # Base row for the chiplet
                baseCrossbarCol = chipletSpecs[chip["type"]]["base"][1]  # Base col for the chiplet
                ## For now set the crossbar size limit to 128 x 128
                baseCrossbarRow = 128
                baseCrossbarCol = 128


                non_zero_bits = math.ceil((weightsKB * (1 - weightSparsity)) * 1024 * 8)
                # ** For Selecting the optimal OU and its other characteristics ** #
                # Metrics for each layer, crossbars required, min reequired crossbars(in square form), Macs per crossbar, activations per crossbar
                crossbarsReq = math.ceil(non_zero_bits / (chipletTypesDict[chip["type"]]["Size"] * chipletTypesDict[chip["type"]]["Bits/cell"]))

                # Gets minimum required Crossbar, Macs per cross bar and activations per crossbar // old metrics used
                minRequiredCrossbars = math.ceil(math.sqrt(chipletTypesDict[chip["type"]]["Size"]* (1 - weightSparsity)))
                MACSperCrossbar = math.ceil(total_macs / crossbarsReq)
                activationsPerCrossbar = (activationsKB * 1024 * 8 * (1 - activationSparsity)) / crossbarsReq


                # Step 1, get required row by accounting for activation sparsity
                rowReq = math.ceil(baseCrossbarRow * (1 - activationSparsity))
                # Step 2, find required OU dimensions
                adjustedOUDimensionReq = math.ceil(baseCrossbarCol * baseCrossbarRow * (1 - weightSparsity))
                # Step 3, get array of foldable factors to select from
                factors = get_approx_foldable_factors(adjustedOUDimensionReq, rowReq, baseCrossbarRow, baseCrossbarCol)
                # Step 4, select config with lowest rows
                idealCrossbarDim = select_lowest_row_config(factors, rowReq)
                # Fallback if no factors found
                if idealCrossbarDim is None:
                    idealCrossbarDim = (baseCrossbarRow, baseCrossbarCol)
                # Get col required based on row required and the number of MACS ## this is for finding the Minimum Crossbar Dimesnions
                colReq = math.ceil(adjustedOUDimensionReq / rowReq)

                # how many bits per crossbar
                cap = chipletTypesDict[chip["type"]]["Size"] * chipletTypesDict[chip["type"]]["Bits/cell"]

                # number of crossbars you really need to hold those non‑zero bits
                xbars_req = math.ceil(weight_nonzero_bits / cap)

                # if you spread the non‑zeros evenly across them:
                per_xbar_nonzeros = weight_nonzero_bits / xbars_req

                # fraction of each xbar that’s empty
                xbar_sparsity = (cap - per_xbar_nonzeros) / cap

                frac        = alloc / total_bits
                macs_assigned = total_macs * frac
                util        = alloc / chip["capacity_left"]

                chip["capacity_left"] -= alloc
                rem_bits    -= alloc

                            # For properly computing allocations

                best_config = best_OU_for_chip(chip["type"], idealCrossbarDim, crossbarsReq)
                #print(f"[DEBUG] Layer {layer} best config:", best_config)
                bestChipName = best_config["chip"]
                bestOUrow = best_config["ou_row"]
                bestOUcol = best_config["ou_col"]
                bestEPM = best_config["epm"]
                bestTOPS = best_config["tops"]
                bestLatency = best_config["latency"]
                bestEnergy = best_config["energy"]
                bestOUReq = best_config["OUrequired"]

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
                    "optimal_ou_row": bestOUrow,
                    "optimal_ou_col": bestOUcol,
                    "optimized_tops": bestTOPS,
                    "optimized_energy_per_mac": bestEPM,
                    "chiplet_resource_taken": util*100
                })

            if rem_bits > 0:
                raise RuntimeError(f"Layer {layer} not fully allocated: {rem_bits:.0f} bits remain")
            adjustedActivationsKB = ((activationsKB * idealCrossbarDim[1]) / baseCrossbarCol)
            
            

            t, e, p, edp, maxp = mapperV3.compute_layer_time_energy(allocs, total_macs)
            layers.append({
                "layer": layer,
                "allocations": allocs,
                "time_s": t,
                "energy_J": e,
                "avg_power_W": p,
                "edp": edp,
                "max_chiplet_power_W": maxp
            })
            # ** For Computing EPD based on the selected OU and its other characteristics ** #

            results.append({
                "layer": layer,
                "chip": bestChipName,
                "Weights(KB)": weightsKB,
                "Weight_Sparsity": weightSparsity,
                "Non-Zero Bits": non_zero_bits,
                "crossbars_required": crossbarsReq ,
                "Activations (KB)": activationsKB,
                "Adjusted Activations (KB)": adjustedActivationsKB,
                "min_required_crossbars": f"{minRequiredCrossbars} x {minRequiredCrossbars}",
                "MACs_per_crossbar": MACSperCrossbar,
                "activations_per_crossbar": activationsPerCrossbar,
                "Minimum_row_Req": rowReq,
                "Minimum OU Dimension Required": adjustedOUDimensionReq,
                "OU required": bestOUReq,
                "Latency": bestLatency,
                "Energy": bestEnergy,
                "ou_row": bestOUrow,
                "ou_col": bestOUcol,
                "EPM": bestEPM,
                "Tops": bestTOPS,
                "chiplet_name": chipletName,
                "Factors": factors,
                "Minimum Crossbar Dimensions Based on Activations": f"{rowReq} x {colReq}",
                "Ideal Crossbar Dimensions": idealCrossbarDim,
                "Rank Based Pareto Config": best_config,
            })

    return results, layers


def plotLayerSparsityWithBestOU(workloadStatsCSV: str, chipletName: str = None, configs: list = None, res_list: list = None, save_folder: str = "workload layers WS vs OU"):
    # --- mode detection ---
    multi_mode = res_list is not None and len(res_list) > 0
    
    # Ensure save folder exists
    os.makedirs(save_folder, exist_ok=True)
    
    # Load the workload CSV
    df = pd.read_csv(workloadStatsCSV)

    # Extract sparsity percentage from CSV
    sparsity = df["Weight_Sparsity(0-1)"].tolist()
    sparsity_percent = [s * 100 for s in sparsity]
    layers = list(range(1, len(sparsity) + 1))  # 1..N

    # Get workload name from the CSV filename for title
    base_filename = os.path.basename(workloadStatsCSV)  # "vgg16_stats.csv"
    workload_name = base_filename.split("_stats")[0].upper()  # "VGG16
    # Check if filename includes 'sparse'
    if "pruned" in base_filename.lower():
        workload_name += " (Sparse)"

    # Figure
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plot for sparsity %
    ax1.bar(layers, sparsity_percent, color='skyblue', label='sparsity%',edgecolor='black', linewidth=1)
    ax1.text(0.5, -0.25, "DNN layers",transform=ax1.transAxes,ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.set_ylabel("Sparsity %", color='black', fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_xticks(layers)
    ax1.set_xticklabels([str(i) for i in layers], rotation=0, fontsize=8)

    # Secondary axis for OU size curves
    ax2 = ax1.twinx()
    ax2.set_ylabel("OU_size (row × col)", color="black", fontweight="bold")

    legend_handles, legend_labels = ax1.get_legend_handles_labels()

    # --- Single-chiplet mode (backward compatible) ---
    if not multi_mode:
        if configs is None:
            raise ValueError("Provide `configs` for single mode or `res_list` for multi mode.")
        
        # Extract OU rows/cols per layer
        chipNames = [r["chip"] for r in configs]
        ou_rows = [r["Rank Based Pareto Config"]["ou_row"] for r in configs]
        ou_cols = [r["Rank Based Pareto Config"]["ou_col"] for r in configs]
        ou_sizes = [r * c for r, c in zip(ou_rows, ou_cols)]

        # Plot single OU-size line
        line2, = ax2.plot(layers, ou_sizes, marker="o", color="orange", linewidth=2, label=f"OU ({chipletName})")
        legend_handles += [line2]
        legend_labels  += [f"OU ({chipletName})"]

        # Scaling for OU axis
        ax2.set_ylim(0, max(ou_sizes) * 1.2 if len(ou_sizes) else 1)
        
        # Draw rotated OU configs below each tick
        for x, r, c, n in zip(layers, ou_rows, ou_cols, chipNames):
            ax1.text(x, -0.06, f"{n}: {r}x{c}", fontsize=8, rotation=90, ha='center', va='top', transform=ax1.get_xaxis_transform())

        title = (f"Sparsity vs. OU Size per Layer for {workload_name} on {chipletName} Chiplet")
        plt.title(title)
        ax1.legend(legend_handles, legend_labels, loc="upper right")

        save_path = os.path.join(save_folder, f"{workload_name}_{chipletName}.png")
        # Export single-chiplet table (3 columns; no numeric OU size) to csv for viewing in table
        export_layer_data_to_csv(
            layers=layers,
            sparsity_percent=sparsity_percent,
            ou_rows=ou_rows,
            ou_cols=ou_cols,
            save_folder=save_folder,
            workload_name=workload_name,
            chipletName=chipletName,
        )
    else:
        # Plot each chiplet's OU curve in a different color (matplotlib cycles automatically)
        max_ou_for_scale = 0
        for entry in res_list:
            chip = entry["chiplet"]
            cfgs = entry["configs"]
            # (layers list in entry is not strictly needed if per-layer counts match CSV)
            ou_rows = [r["Rank Based Pareto Config"]["ou_row"] for r in cfgs]
            ou_cols = [r["Rank Based Pareto Config"]["ou_col"] for r in cfgs]
            ou_sizes = [r * c for r, c in zip(ou_rows, ou_cols)]

            # If configs shorter than CSV layers, truncate to min length
            n = min(len(layers), len(ou_sizes))
            xs = layers[:n]
            ys = ou_sizes[:n]

            line, = ax2.plot(xs, ys, marker="o", linewidth=2, label=f"OU ({chip})")
            legend_handles.append(line)
            legend_labels.append(f"OU ({chip})")

            if ys:
                max_ou_for_scale = max(max_ou_for_scale, max(ys))

        ax2.set_ylim(0, max_ou_for_scale * 1.2 if max_ou_for_scale > 0 else 1)

        # Title without chiplet name (shared chart)
        title = f"Sparsity vs. OU Size per Layer for {workload_name} (Multiple Chiplets)"
        plt.title(title)

        # Legend at bottom middle for multi chiplets
        ax1.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=len(legend_labels), fontsize = 12)

        # Save with MULTI suffix
        save_path = os.path.join(save_folder, f"{workload_name}_MULTI.png")

        # Optional: write a combined CSV with a Chiplet column (no numeric OU size)
        # Build rows as [Layer, Sparsity%, Chiplet, OU Config]
        combined_rows = []
        for entry in res_list:
            chip = entry["chiplet"]
            cfgs = entry["configs"]
            ou_rows = [r["Rank Based Pareto Config"]["ou_row"] for r in cfgs]
            ou_cols = [r["Rank Based Pareto Config"]["ou_col"] for r in cfgs]
            n = min(len(layers), len(ou_rows), len(ou_cols))
            for i in range(n):
                combined_rows.append([
                    layers[i],
                    sparsity_percent[i],
                    chip,
                    f"{ou_rows[i]} x {ou_cols[i]}",
                ])
        if combined_rows:
            df_out = pd.DataFrame(
                combined_rows,
                columns=["Layer #", "Weight Sparsity (%)", "Chiplet", "OU Config"]
            )
            csv_path = os.path.join(save_folder, f"{workload_name}_MULTI_layer_data_table.csv")
            df_out.to_csv(csv_path, index=False, sep="\t")
            print(f"Combined layer data saved to: {csv_path}")

    # Legend & save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to: {save_path}")
    # plt.show()

  
# used in the above function to save to a table
def export_layer_data_to_csv(layers, sparsity_percent, ou_rows, ou_cols, save_folder, workload_name, chipletName):
    """
    Save per-layer data (sparsity, OU config, OU size) to CSV.

    Args:
        layers (list): Layer numbers.
        sparsity_percent (list): Weight sparsity percentages.
        ou_rows (list): OU row dimensions.
        ou_cols (list): OU column dimensions.
        save_folder (str): Folder to save the CSV.
        workload_name (str): Name of the workload (e.g., VGG16 or VGG16 (Sparse)).
        chipletName (str): Name of the chiplet.

    Returns:
        str: Path to the saved CSV file.
    """
    # Ensure save folder exists
    os.makedirs(save_folder, exist_ok=True)

    # Build table
    data = []
    for i, layer in enumerate(layers):
        ou_config = f"{ou_rows[i]} x {ou_cols[i]}"
        data.append([layer, sparsity_percent[i] * .01, ou_config])

    # Create dataframe
    df = pd.DataFrame(data, columns=["Layer #", "Weight Sparsity (%)", "OU Config"])

    # Save path
    csv_path = os.path.join(save_folder, f"{workload_name}_{chipletName}_layer_data_table.csv")
    df.to_csv(csv_path, index=False, sep='\t')

    print(f"Layer data saved to: {csv_path}")
    return csv_path

## Used to get the compute time, energy, edp for each layer
def print_layer_table_for_workload_computation(res, layers, chiplet_name):
    table_rows = []
    seen_chiplets = set()
    for r, lr in zip(res, layers):
        layer_num = r["layer"]
        layer_activations = r['Activations (KB)']
        layer_activations_adjusted = r["Adjusted Activations (KB)"]
        
        dims = r['Ideal Crossbar Dimensions']
        Rreq = dims[0]
        Creq = dims[1]

        # Get best config safely
        best_cfg = r.get('Rank Based Pareto Config')
        if not isinstance(best_cfg, dict):
            # Fallback if no valid config
            r = c =  0
        else:
            r = int(best_cfg['ou_row'])
            c = int(best_cfg['ou_col'])
            latency = best_cfg['latency']
            energy = best_cfg['energy']
            power_density = best_cfg['power_density']
            edp = latency * energy

        Xbars_used = 0
        chips_used = 0
        # Add all chip_ids used in this layer to the set and get Xbars used from allocations in that layer
        for alloc in lr["allocations"]:
            chips_used += 1
            seen_chiplets.add(alloc["chip_id"])
            Xbars_used += alloc["Crossbars_used"]
            weight_sparsity = alloc["weight sparsity"]
            activation_sparsity = alloc["Activation Sparsity"]

        # Tiles per layer used.
        tiles_used = Xbars_used / XBARS_PER_TILE
        # Chiplet # is count of unique chiplets seen so far
        chiplet_count = len(seen_chiplets)

        # Append row
        table_rows.append([
            layer_num,
            latency,
            energy,
            edp,
            weight_sparsity,
            activation_sparsity,
            Xbars_used,
            tiles_used,
            Rreq,
            Creq,
            r,
            c,
            power_density,
            layer_activations,
            layer_activations_adjusted,
            chiplet_count,
            chips_used
        ])

    # Fix column list (comma between Energy and EDP)
    df = pd.DataFrame(
        table_rows,
        columns=[
            "Layer #",
            "Latency",
            "Energy",
            "EDP",
            "Weight Sparsity",
            "Activation Sparsity",
            "Crossbars used",
            "Tiles Used",
            "R Required",
            "C Required",
            "r",
            "c",
            "Power Density",
            "Original Layer Activations (KB)",
            "Activations adjusted by Creq (KB)",
            "Chiplet #",
            "Chips Used"
        ]
    )

    # Print TSV to terminal and copy to clipboard
    tsv_data = df.to_csv(sep='\t', index=False)
    print(tsv_data)
    subprocess.run("pbcopy", text=True, input=tsv_data)
    print("→ Data copied to clipboard (tab-delimited). Just paste into Excel.")

    # Save to CSV
    df.to_csv("layer_table.csv", index=False)

def print_layer_OU_info(res):
    for r in res:
        print(f"Layer: {r['layer']}")
        print(f"Original Activations (KB): {r['Activations (KB)']}")
        print(f"Adjusted Activations (KB): {r['Adjusted Activations (KB)']}")
        print(f"Weights (KB): {r['Weights(KB)']}")
        print(f"Weight Sparsity: {r['Weight_Sparsity']}")
        print(f"Non-Zero Bits: {r['Non-Zero Bits']}")
        print(f"Minimum Row Requirement: {r['Minimum_row_Req']}")
        print(f"Crossbars Required: {r['crossbars_required']}")
        print(f"Minimum OU Dimension Required: {r['Minimum OU Dimension Required']}")
        print(f"Factors of OU Dimension: {r['Factors']}")
        print(f"Minimum Crossbar Dimensions Based on Activations: {r['Minimum Crossbar Dimensions Based on Activations']}")
        print(f"Ideal Crossbar Dimensions: {r['Ideal Crossbar Dimensions']}")
        print(f"Rank Based Pareto Config: row:{r['Rank Based Pareto Config']['ou_row']}, col:{r['Rank Based Pareto Config']['ou_col']}, latency: {r['Rank Based Pareto Config']['latency']}, energy: {r['Rank Based Pareto Config']['energy']}, power_density: {r['Rank Based Pareto Config']['power_density']:.2f} W")
        print("-" * 40)

def print_layer_compute_metrics(layers):
    for lr in layers:
        print(f"\nLayer {lr['layer']}:")
        for a in lr["allocations"]:
            print(" ", a)
        ## Not using these current metrics but thought it may be useful to look at trends
        print(f"  → Time: {lr['time_s']:.3e}s, Energy: {lr['energy_J']:.3e}J, "
            f"Power: {lr['avg_power_W']:.3e}W, MaxP: {lr['max_chiplet_power_W']:.3e}W, EDP: {lr['edp']:.3e}")


def run_workloads_across_chiplets_WS_vs_OU( workloads, chiplets, chip_count=1000, save_folder="workload layers WS vs OU",):
    for workload_csv in workloads:
            print(f"WORKLOAD --- {workload_csv}")
            res_list = []
            for i, chip in enumerate(chiplets):
                print(f"CHIP --- {chip}")
                chipDist = [0] * len(chiplets)
                chipDist[i] = chip_count

                res, layers = computeCrossbarMetrics(chipDist, chipletName=chip, workloadStatsCSV=workload_csv)
                res_list.append({"chiplet": chip, "configs": res, "layers": layers})

            plotLayerSparsityWithBestOU(workloadStatsCSV=workload_csv, res_list=res_list, save_folder=save_folder)

def sweep_workloads_for_each_chiplet_WS_vs_OU( workloads, chiplets, chip_count = 1000, save_folder="workload layers WS vs OU"):
    for workload_csv in workloads:
        print(f"WORKLOAD --- {workload_csv}")
        for i, chip in enumerate(chiplets):
            print(f"CHIP --- {chip}")
            chipDist = [0] * len(chiplets)
            chipDist[i] = chip_count

            res, layers = computeCrossbarMetrics(chipDist, chipletName=chip, workloadStatsCSV=workload_csv)
            plotLayerSparsityWithBestOU(workloadStatsCSV=workload_csv, chipletName=chip, configs=res)


#def plotLayersWithBestEDPandOU(workload, chiplets):


if __name__ == "__main__":
    # --- For iterating over all combinations use these
    workloads = ["workloads/resnet18_stats.csv", "workloads/resnet18_stats_pruned.csv", "workloads/vgg16_stats.csv",
                 "workloads/vgg16_stats_pruned.csv", "workloads/vgg11_stats.csv", "workloads/vgg11_stats_pruned.csv"]
    chiplets = ["Standard", "Shared", "Adder", "Accumulator"] # Accumulator as well, but not used in this script as it stretches WS vs OU graph

    # --- For working with one chiplet and one workload at a time use these
    workload_csv = "workloads/vgg11_stats_pruned.csv"
    chiplet = "HetOU"  # Choose from "Standard", "Shared", "Adder", "Accumulator"
    chipDist = [10, 8, 0, 8, 0]
    chipCount = 10000

    # -- FOR SINGULAR WORKLOAD AND CHIPLET USE ONLY -- #
    res, layers = computeCrossbarMetrics(chipDist, chipletName=chiplet, workloadStatsCSV=workload_csv)
    # -- FOR MANUAL OU SELECTION USE
    #res, layers = computeCrossbarMetrics(chipDist, chipletName=chiplet, workloadStatsCSV=workload_csv, manualOU=True, manual_ou_row=32, manual_ou_col= 32)
    print_layer_table_for_workload_computation(res, layers, chiplet)
    # -- Optional print statement to show layer characteristics regarding OU -- #
    print_layer_OU_info(res)
    # -- Optional print statement to show the chiplet allocations and OLD energy/latency metrics from mapperV3.py -- #
    print_layer_compute_metrics(layers)

    # -- FOR SINGULAR WORKLOAD AND  HETEROGENIUS CHIPLET SET USE ONLY -- #


    # --- To plot list of chiplets OU across list of workloads, saves in workload layers WS vs OU folder --- #
    # --- plots images and writes csv files to 'workload layers WS vs OU' folder
    # run_workloads_across_chiplets_WS_vs_OU(workloads, chiplets, chipCount)
    ## --- To Run only one chiplet and get individual image run the below function
    plotLayerSparsityWithBestOU(workloadStatsCSV=workload_csv, chipletName=chiplet, configs=res)

    ## --- Sweep all workloads and plot each chip INDIVIDUALLY, rather than comparitvely --- ##
    # sweep_workloads_for_each_chiplet_WS_vs_OU(workloads, chiplets, chipCount)

    


    