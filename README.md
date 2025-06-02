# GNN Bench

A scalable benchmarking suite for Graph Neural Networks (GNNs), supporting full-graph training on CPU, single-GPU, or multi-GPU (DDP). Includes logging of key metrics (loss, accuracy, throughput, timing) into a SQLite database, and automatic plotting of results into Markdown and PNG files.

---

## Features

- **Full-graph training** on Planetoid (Cora, Citeseer, Pubmed) or OGB (ogbn-arxiv, ogbn-products, etc.).
- **Models**: 2-layer GCN or 2-layer GAT (configurable hidden dim, number of heads, dropout).
- **Distributed**: PyTorch DDP for multi-GPU on one node or across multiple nodes.
- **Logging**: Per-run parameters and metrics (final losses, accuracies, average throughput/time, total time) saved to `results/results.db`.
- **Plotting**: Automatically generate:
  - **`results/plots/<timestamp>_<experiment>_acc_vs_gpus.png`**  
  - **`results/plots/<timestamp>_<experiment>_throughput_vs_gpus.png`**  
  - **`results/results.md`** with embedded images, summary table, and metadata.
- **Config-driven**: YAML sweep files let you specify multiple experiments, including parameter sweeps (batch_size, learning rate, world_size, etc.).
- **Single-line summaries**: Prints a concise “Averages (excl. 1st epoch)” line at the end of each run.  
- **Easy install/clean**: `setup.sh` creates or activates a virtual environment, installs dependencies (CPU or CUDA), and cleans artifacts.

---

## Repository Structure

```
.
├── README.md
├── setup.sh
├── config
│   ├── default.yaml
│   └── gat_ogbn_arxiv.yaml
├── src
│   └── gnn_bench
│       ├── __init__.py
│       ├── cli.py
│       ├── db_logger.py
│       ├── ddp_utils.py
│       ├── plot.py
│       └── train.py
└── results
    ├── results.db         # SQLite database (created after first run)
    ├── plots
    │   └── <timestamp>_…  # Auto-generated PNGs
    └── results.md         # Auto-generated Markdown report
```

---

## Installation

1. **Clone the repository** (if you haven’t already):
   ```bash
   git clone https://github.com/yourusername/gnn-bench.git
   cd gnn-bench
   ```

2. **Run `setup.sh` to create or activate a Python 3.10 venv and install dependencies**. For CPU-only:
   ```bash
   source setup.sh install cpu
   ```
   For CUDA 12.1 (PyTorch + PyG CUDA wheels):
   ```bash
   source setup.sh install cuda121
   ```
   For editable install (so changes in `src/` reflect immediately):
   ```bash
   source setup.sh install -e
   ```
   Or editable + CPU extras:
   ```bash
   source setup.sh install cpu -e
   ```

3. **Verify** you have `gnn_bench_run` and `gnn_bench_plot` in your path:
   ```bash
   which gnn_bench_run
   which gnn_bench_plot
   ```

---

## Configuration

Put your YAML files under `config/`. Two examples are provided:

- **`config/default.yaml`**:  
  A simple example (Cora + GCN, no DDP).
- **`config/gat_ogbn_arxiv.yaml`**:  
  A more advanced GAT on OGBN-Arxiv, sweeping batch_size.

### Example `config/gat_ogbn_arxiv.yaml`

```yaml
experiments:
  - experiment_name: "gat_ogbn_arxiv"
    dataset:     "ogbn-arxiv"         # OGB’s ArXiv citation graph
    model:       "gat"                # Graph Attention Network
    epochs:      5                    # Number of epochs
    batch_size:  [256, 512, 1024]     # Sweep over these batch sizes
    lr:          0.005                # Learning rate
    hidden_dim:  128                  # Hidden dimension size
    num_heads:   4                    # GAT attention heads
    dropout:     0.6                  # Dropout probability
    seed:        42
    world_size:  1                    # Full-graph, single-GPU/CPU
```

- **`batch_size`** is currently *not used* in full-graph training (all nodes in one pass). It influences *only* throughput reporting.  
- To switch to true mini-batch neighbor sampling (so `batch_size` actually changes accuracy), you would need to replace the full-graph loop with a `NeighborLoader`. But if your graph fits in memory, full-graph is simpler and perfectly acceptable.

---

## Usage

### Running Experiments

```bash
# Run all experiments defined in a YAML config, then plot:
gnn_bench_run --config config/gat_ogbn_arxiv.yaml --plots
```

- **`--config`** (or `-c`) points to your YAML file. If omitted, defaults to `config/default.yaml`.
- **`--plots`** tells the script to run the plotting step after all experiments finish.
- **`--sort-by`** controls sorting in `results.md` (default = `date`). Options: `date`, `acc`, `throughput`.

Each experiment in the YAML file expands into one (or more) runs. For each run you’ll see:

1. A one-line parameter summary:  
   ```
   ▶ Experiment parameters: dataset=ogbn-arxiv  model=gat  epochs=5  batch_size=256  lr=0.0050  hidden_dim=128  num_heads=4  dropout=0.60  seed=42  world_size=1
   ```

2. Per-epoch metrics (for 5 epochs by default):  
   ```
   [E001] TLoss=3.9240 TAcc=0.0165 VLoss=3.2270 VAcc=0.0764 Time=6.62s Thr=25587.26
   [E002] …
   …
   ```

3. An end-of-run summary (excluding the first epoch):  
   ```
   === Averages (excl. 1st epoch) ===
   TLoss=3.4067 TAcc=0.1717 VLoss=2.8599 VAcc=0.2763 Time=4.82s Thr=35241.55 smpls/s
   ```

4. If you swept `batch_size`, you’ll see this three times (once per value 256, 512, 1024).

5. After all runs, plots and a Markdown report are generated under `results/`.

---

### Plotting Only

If you already ran experiments and only want to regenerate plots (e.g., after changing `--sort-by`):

```bash
gnn_bench_plot --db results/results.db --output-dir results --sort-by throughput
```

- **`--db <path>`**: path to the SQLite DB (default = `results/results.db`).
- **`--output-dir <dir>`**: directory to save `plots/` and `results.md`.  
- **`--sort-by`**: `date` (default), `acc`, or `throughput`.

---

## Results Directory

After a successful run:

```
results/
├── results.db
├── plots/
│   ├── 2025-06-01_20-08-04_gat_ogbn_arxiv_acc_vs_gpus.png
│   └── 2025-06-01_20-08-04_gat_ogbn_arxiv_throughput_vs_gpus.png
└── results.md
```

- **`results.db`**: SQLite database containing tables:  
  - `runs` (one row per run, with columns: experiment_name, dataset, model, batch_size, lr, hidden_dim, seed, world_size, final_train_loss, final_val_loss, final_val_acc, avg_time, avg_throughput, total_train_time, timestamp, etc.)  
- **`results/plots/*.png`**:  
  - `acc_vs_gpus.png`: average validation accuracy vs. world_size (GPUs).  
  - `throughput_vs_gpus.png`: average throughput vs. world_size.  
- **`results/results.md`**:  
  A Markdown report that includes:  
  1. A summary table of all runs (parameters, final metrics, timestamp).  
  2. Embedded plots.  
  3. Detailed metadata per experiment (all hyperparameters and metrics).

---

## Why Loss & Accuracy Are Identical Across “batch_size” When Using Full-Graph

- **Full-graph training** means each epoch processes *all* train nodes in one forward/backward pass:
  ```python
  out = model(data.x, data.edge_index)
  loss = criterion(out[train_mask], data.y[train_mask])
  ```
- In this mode, `batch_size` is **not used** to split the data. Thus, changing `batch_size` does nothing to the parameter updates—every run follows the same exact gradients (given identical seed).  
- The only metric that changes slightly is **throughput**, calculated as `(num_nodes)/(epoch_time)`, because minor timing fluctuations occur on repeated runs.

If you want a benchmark where `batch_size` *does* affect training (and model convergence), switch to a **mini-batch neighbor sampler** (e.g., `torch_geometric.loader.NeighborLoader`). That way, each iteration processes only a subset of nodes (and their sampled neighbors), and `batch_size` truly controls the number of nodes per update.

---

## Supported Datasets

- **Planetoid**:  
  - `Cora` (2,708 nodes, 5,429 edges)  
  - `Citeseer` (3,327 nodes, 4,732 edges)  
  - `Pubmed` (19,717 nodes, 44,338 edges)  
  ```python
  from torch_geometric.datasets import Planetoid
  ds = Planetoid("data/Cora", "Cora")
  ```

- **OGB Node Property**:  
  - `ogbn-arxiv` (≈170 k nodes, 1.2 M edges)  
  - `ogbn-products` (≈2.4 M nodes, 61 M edges)  
  - `ogbn-proteins` (≈132 k nodes, 39 M edges)  
  ```python
  from ogb.nodeproppred import PygNodePropPredDataset
  ds = PygNodePropPredDataset(name="ogbn-arxiv", root="data/ogbn-arxiv")
  ```

- **PyG Built-ins** (for quick tests):  
  - `Reddit` (≈232 k nodes, 11.6 M edges)  
  - `Flickr` (≈89 k nodes, 6.6 M edges)  
  ```python
  from torch_geometric.datasets import Reddit, Flickr
  ds1 = Reddit("data/Reddit")
  ds2 = Flickr("data/Flickr")
  ```

---

## Multi-GPU / Distributed Setup

To run a multi-GPU experiment (e.g., 4 GPUs on one node), add DDP fields to your YAML:

```yaml
experiments:
  - experiment_name: "gat_ogbn_arxiv_ddp"
    dataset:        "ogbn-arxiv"
    model:          "gat"
    epochs:         5
    batch_size:     [512]           # still full-graph (batch size ignored)
    lr:             0.005
    hidden_dim:     128
    num_heads:      4
    dropout:        0.6
    seed:           42
    world_size:     4               # spawn 4 processes
    nnodes:         1
    nproc_per_node: 4
    master_addr:    "127.0.0.1"
    master_port:    "29500"
```

- `world_size = nnodes * nproc_per_node`.  
- Each process binds to a separate GPU via `torchrun`.  
- In full-graph mode, each GPU still processes the entire graph—DDP averages gradients across GPUs.  
- Throughput (samples/sec) should roughly scale linearly when you go from 1→2→4 GPUs, since each GPU computes a replica of the forward/backward and synchronizes gradients.

---

## Troubleshooting & Tips

- **`pkg_resources` deprecation warning**:  
  You might see:
  ```
  UserWarning: pkg_resources is deprecated as an API. …
    from pkg_resources import parse_version
  ```
  This comes from the `outdated` package (used by some dependencies). It’s harmless. To silence it, add at the top of `src/gnn_bench/cli.py`:
  ```python
  import warnings
  warnings.filterwarnings(
      "ignore",
      category=UserWarning,
      message="pkg_resources is deprecated as an API.*"
  )
  ```

- **Missing `throughput` in DB**:  
  If you customize logging and forget to write `"throughput"` into `metrics`, the plotter will raise:
  ```
  RuntimeError: Plotting error: database 'results.db' missing column 'throughput'…
  ```
  Just ensure your `train.py` logs both `"throughput": avg_throughput` and `"avg_throughput": avg_throughput`.

- **Batch size has no effect** under full-graph training. Only throughput changes. See **Why Loss & Accuracy Are Identical** above.

---

## Example Workflow

1. **Install (CPU)**  
   ```bash
   source setup.sh install cpu
   ```

2. **Run GAT on OGBN-Arxiv**  
   ```bash
   gnn_bench_run --config config/gat_ogbn_arxiv.yaml --plots
   ```

3. **Inspect Results**  
   - Open `results/results.md` in a Markdown viewer or browser.  
   - Examine `results/plots/…acc_vs_gpus.png` and `…throughput_vs_gpus.png`.  

4. **Plot Only (no new runs)**  
   ```bash
   gnn_bench_plot --db results/results.db --output-dir results --sort-by acc
   ```

---

## License

MIT License. Feel free to adapt and extend for your own GNN benchmarking needs.

---

## Contact

Dr. Mares Barekzai  
mares.barekzai@lrz.de  
