# Hetero-OU
Consists of two main py files: Model_Stats.py for gathering a worklaods characteristics and understandingOU.py to run the simulations to gather data and plot it. mapperV3.py is used for helper functions to calculate the bit computations for each layer, tracking the number of chiplets used, how many crossbars were used, and other metrics regarding each chiplets usage in a given layer. 

## Generating Model Characteristics

Before running any chiplet or OU size simulations, we first need to extract per-layer characteristics of the target DNN model (weights, MACs, sparsity, activation sizes, etc.).  
This is done using **`Model_Stats.py`**.

### How it Works
- Loads the model from `torchvision.models` (or any compatible model function).
- Runs a forward pass with a dummy input while collecting:
  - **Weight size (KB)** – assuming `float64` precision (8 bytes).
  - **Weight sparsity** – fraction of near-zero weights.
  - **Activation size (KB)** – memory usage of activations.
  - **Activation sparsity** – fraction of near-zero activation values.
  - **MAC count** – multiply–accumulate operations for each layer.
- Saves the results as a `.csv` in the `workloads/` folder for later use.

### Usage
1. Open `Model_Stats.py`.
2. Add your desired models to the `models_to_run` dictionary, for example:
   ```python
   models_to_run = {
       "VGG11": models.vgg11_bn,
       "ResNet50": models.resnet50
   }
3. Run 'python Model_Stats.py'
4. The generated CSV (e.g., workloads/vgg11_stats.csv) will contain one row per layer with all relevant stats.

## Running OU Simulations

Once you have workload CSVs, you can run **OU optimization** and **chiplet simulations** using `understandingOU.py`.

### Key Function — `computeCrossbarMetrics`
Computes the optimal OU (Operational Unit) configuration for each layer of a given workload.

##### workload_OU_config_results.txt
Whenevery this function is run, it will put all the ou configs stats in a large txt file for manually checking the metrics.
Such as ouRow, ouCol, energy, latency, edp, OU required, Power density, TOPS, EPM

**Parameters:**
- `chip_distribution` — list of chiplet counts (aligned with `chipletTypesDict` order).
- `chipletName` — string name of the chiplet (`"Standard"`, `"Shared"`, `"Adder"`, etc.).
- `workloadStatsCSV` — path to the workload CSV file.
- `manualOU` — set to `True` to manually override OU dimensions.
- `manual_ou_row` / `manual_ou_col` — dimensions used when `manualOU=True`.

**Returns:**
- Layer-by-layer OU configuration, latency, energy, and EDP metrics. (function -> print_layer_OU_info())
- Allocation and chip usage details. (function -> print_layer_compute_metrics())

### Example — Single Chiplet Run
```python
workload_csv = "workloads/vgg11_stats.csv"
chiplet = "Shared"
chipDist = [0, 10, 0, 0, 0]  # Only Shared chiplets

res, layers = computeCrossbarMetrics(
    chipDist,
    chipletName=chiplet,
    workloadStatsCSV=workload_csv
)

plotLayerSparsityWithBestOU(
    workloadStatsCSV=workload_csv,
    chipletName=chiplet,
    configs=res
)
```


### Example - Manual OU Overide
```python
res, layers = computeCrossbarMetrics(
    chipDist,
    chipletName="Shared",
    workloadStatsCSV="workloads/vgg11_stats.csv",
    manualOU=True,
    manual_ou_row=16,
    manual_ou_col=16
)
```


### Example — Multiple Chiplets on One Workload
```python
workloads = [
    "workloads/vgg11_stats.csv",
    "workloads/vgg11_stats_pruned.csv"
]
chiplets = ["Standard", "Shared", "Adder"]

run_workloads_across_chiplets_WS_vs_OU(workloads, chiplets, chip_count=1000)
```

### Output structure
Plots:
    •    Saved in workload layers WS vs OU/.
    •    Single chiplet: <WORKLOAD>_<CHIPLET>.png
    •    Multiple chiplets: <WORKLOAD>_MULTI.png

CSV Tables:
    •    <WORKLOAD>_<CHIPLET>_layer_data_table.csv — single run.
    •    <WORKLOAD>_MULTI_layer_data_table.csv — multiple chiplet run.

#### Power Density Constraint
    •    Any OU configuration with power density > 8.0 W is automatically skipped.
    •    In manual mode, exceeding this limit will raise an error.
