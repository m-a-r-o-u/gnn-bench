# src/gnn_bench/cli.py

import argparse
import yaml
import itertools
import os
import subprocess
import sys
import traceback
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
    parser.add_argument(
        "--no-cuda", action="store_true",
        help="Disable CUDA and run on CPU."
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=1,
        help="Number of parallel independent experiments (world_size=1) to run concurrently."
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

    # Prepare lists for distributed and single-process tasks
    ddp_tasks = []       # (cmd_list, exp_name)
    sp_tasks = []        # Namespace args for train_main

    # Collect experiment runs from YAML config
    for exp in cfg["experiments"]:
        keys = list(exp.keys())
        values = [exp[k] if isinstance(exp[k], list) else [exp[k]] for k in keys]
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))

            # Determine experiment_name
            exp_name = params.get("experiment_name") or (
                "exp_" + "_".join(f"{k}{str(v).replace('.', '')}" for k, v in params.items())
            )

            # Required parameters with defaults
            dataset    = params["dataset"]
            model      = params["model"]
            epochs     = int(params.get("epochs", 10))
            batch_size = int(params.get("batch_size", 32))
            lr         = float(params.get("lr", 0.001))
            hidden_dim = int(params.get("hidden_dim", 64))
            seed       = int(params.get("seed", 42))
            world_size = int(params.get("world_size", 1))

            # Optional GAT-specific parameters
            num_heads = int(params.get("num_heads", 1))
            dropout   = float(params.get("dropout", 0.0))
            no_cuda   = bool(params.get("no_cuda", False))

            # Prepare results_db path
            results_db = params.get("results_db", "results/results.db")
            last_results_db = results_db
            Path(os.path.dirname(results_db)).mkdir(parents=True, exist_ok=True)

            if world_size > 1:
                # Build torchrun command for DDP (distributed multi-GPU/multi-node)
                nnodes         = int(params.get("nnodes", 1))
                nproc_per_node = int(params.get("nproc_per_node", world_size // nnodes))
                node_rank      = int(params.get("node_rank", 0))
                master_addr    = params.get("master_addr", "127.0.0.1")
                master_port    = str(params.get("master_port", "29500"))

                cmd = [
                    "torchrun",
                    f"--nproc_per_node={nproc_per_node}",
                    f"--nnodes={nnodes}",
                    f"--node_rank={node_rank}",
                    f"--rdzv_id={exp_name}",
                    f"--rdzv_backend=c10d",
                    f"--rdzv_endpoint={master_addr}:{master_port}",
                    "gnn_bench_run",
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
                    f"--dropout={dropout}",
                ]
                if no_cuda:
                    cmd.append("--no-cuda")
                ddp_tasks.append((cmd, exp_name))
            else:
                # Single-process (CPU or single-GPU) run args for direct invocation
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
                    dropout=dropout,
                    no_cuda=no_cuda,
                )
                sp_tasks.append(single_args)
    # Execute distributed tasks (DDP) sequentially
    for cmd, exp_name in ddp_tasks:
        print(f"\n▶ Running distributed experiment: {exp_name}")
        print("  Command:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    # Execute single-process tasks, possibly in parallel
    if sp_tasks:
        import concurrent.futures

        def _run_single(args_ns):
            print(f"\n▶ Running single-process experiment: {args_ns.experiment_name}")
            train_main(args_ns)

        if args.jobs > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
                futures = {executor.submit(_run_single, ns): ns.experiment_name for ns in sp_tasks}
                for fut in concurrent.futures.as_completed(futures):
                    # propagate exceptions
                    fut.result()
        else:
            for ns in sp_tasks:
                _run_single(ns)

    # After all experiments, optionally call plotting
    if args.plots and last_results_db is not None:
        print(f"\n▶ Generating plots for: {last_results_db}")
        try:
            plot_main(
                db_path=last_results_db,
                output_dir=os.path.dirname(last_results_db),
                overwrite=True,
                sort_by=args.sort_by
            )
        except Exception:
            traceback.print_exc()
            print(
                "Plotting failed, possibly due to missing display or GUI backend."
                " We force the non-interactive 'Agg' backend in plot.py to avoid crashes."
            )

def run_entry():
    """
    Wrapper around _run_entry(): catches exceptions, prints full traceback,
    and exits with a non-zero status code.
    """
    try:
        _run_entry()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
