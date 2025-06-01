# src/gnn_bench/cli.py

import argparse
import yaml
import itertools
import os
import subprocess
import sys
from pathlib import Path

from .train import main as train_main
from .plot import main as plot_main

def _run_entry():
    """
    Core logic from run_entry(), renamed so we can wrap it safely.
    """
    parser = argparse.ArgumentParser(
        description="GNN Benchmark CLI (CPU/GPU/Distributed)."
    )
    parser.add_argument(
        "--config", "-c", type=str,
        help="Path to YAML config file defining experiments. Default: config/default.yaml"
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="After experiments complete, run the plotting step."
    )
    parser.add_argument(
        "--sort-by", choices=["date", "acc", "throughput"], default="date",
        help="How to sort runs in results.md. 'date'=timestamp, 'acc'=final_val_acc, 'throughput'=throughput."
    )
    args = parser.parse_args()

    # Determine config path
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(os.getcwd(), "config", "default.yaml")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"YAML config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "experiments" not in cfg or not isinstance(cfg["experiments"], list):
        raise ValueError("YAML must contain a top-level 'experiments: [ ... ]' list.")

    last_results_db = None

    # Iterate through each experiment block in YAML
    for exp in cfg["experiments"]:
        # Expand list-valued fields into combinations
        keys = list(exp.keys())
        values = [exp[k] if isinstance(exp[k], list) else [exp[k]] for k in keys]

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))

            # 1) Determine experiment_name
            exp_name = params.get("experiment_name")
            if not exp_name:
                exp_name = "exp_" + "_".join(
                    f"{k}{str(v).replace('.', '')}" for k, v in params.items()
                )

            # 2) Required parameters (with defaults)
            dataset    = params["dataset"]
            model      = params["model"]
            epochs     = int(params.get("epochs", 10))
            batch_size = int(params.get("batch_size", 32))
            lr         = float(params.get("lr", 0.001))
            hidden_dim = int(params.get("hidden_dim", 64))
            seed       = int(params.get("seed", 42))
            world_size = int(params.get("world_size", 1))

            # 3) Optional GAT-specific parameters
            num_heads = int(params.get("num_heads", 1))    # default 1 for GCN or if missing
            dropout   = float(params.get("dropout", 0.0))  # default 0.0 if missing

            # 4) Build results_db path
            results_db = params.get("results_db", "results/results.db")
            last_results_db = results_db
            Path(os.path.dirname(results_db)).mkdir(parents=True, exist_ok=True)

            if world_size > 1:
                # Distributed run: gather DDP fields
                nnodes         = int(params.get("nnodes", 1))
                nproc_per_node = int(params.get("nproc_per_node", world_size // nnodes))
                node_rank      = int(params.get("node_rank", 0))
                master_addr    = params.get("master_addr", "127.0.0.1")
                master_port    = str(params.get("master_port", "29500"))

                # Build the `torchrun` command
                cmd = [
                    "torchrun",
                    f"--nproc_per_node={nproc_per_node}",
                    f"--nnodes={nnodes}",
                    f"--node_rank={node_rank}",
                    f"--rdzv_id={exp_name}",
                    f"--rdzv_backend=c10d",
                    f"--rdzv_endpoint={master_addr}:{master_port}",
                    "gnn_bench_run",  # invokes the same entry point
                    f"--dataset={dataset}",
                    f"--model={model}",
                    f"--epochs={epochs}",
                    f"--batch-size={batch_size}",
                    f"--lr={lr}",
                    f"--hidden-dim={hidden_dim}",
                    f"--seed={seed}",
                    f"--world-size={world_size}",
                    f"--experiment-name={exp_name}",
                    f"--results-db={results_db}",
                    f"--num-heads={num_heads}",
                    f"--dropout={dropout}"
                ]
                print(f"\n▶ Running distributed experiment: {exp_name}")
                print("  Command:", " ".join(cmd))
                subprocess.run(cmd, check=True)

            else:
                # Single-process run (CPU or single-GPU)
                from argparse import Namespace
                single_args = Namespace(
                    dataset=dataset,
                    model=model,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    hidden_dim=hidden_dim,
                    seed=seed,
                    world_size=1,
                    rank=0,
                    local_rank=0,
                    distributed_backend="nccl",
                    master_addr="127.0.0.1",
                    master_port="29500",
                    experiment_name=exp_name,
                    results_db=results_db,
                    num_heads=num_heads,
                    dropout=dropout
                )
                print(f"\n▶ Running single‐process experiment: {exp_name}")
                train_main(single_args)

    # After all experiments, optionally call plotting
    if args.plots and last_results_db is not None:
        print(f"\n▶ Generating plots for: {last_results_db}")
        plot_main(
            db_path=last_results_db,
            output_dir=os.path.dirname(last_results_db),
            overwrite=True,
            sort_by=args.sort_by
        )

def run_entry():
    """
    Safe wrapper around _run_entry(): catches any exception,
    prints it, and returns without exiting the shell abruptly.
    """
    try:
        _run_entry()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return
