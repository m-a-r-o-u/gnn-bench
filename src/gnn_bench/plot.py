# src/gnn_bench/plot.py

import os
import sqlite3
import matplotlib
# Use non-interactive backend to avoid crashes on headless servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def main(db_path: str, output_dir: str, overwrite: bool = False, sort_by: str = "date"):
    """
    Reads the `runs` table from db_path, detects if a single parameter was swept,
    and plots metrics (val accuracy and throughput) against that parameter
    (e.g., batch_size or world_size). Generates:
      - PNG plots in output_dir/plots
      - A Markdown report output_dir/results.md with embedded images and metadata.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # Determine ORDER BY clause
        if sort_by == "date":
            order_clause = "ORDER BY timestamp DESC"
        elif sort_by == "acc":
            order_clause = "ORDER BY final_val_acc DESC"
        elif sort_by == "throughput":
            order_clause = "ORDER BY throughput DESC"
        else:
            order_clause = "ORDER BY timestamp DESC"

        # Fetch relevant columns
        query = f"""
            SELECT
                experiment_name,
                dataset,
                model,
                batch_size,
                world_size,
                final_val_acc,
                throughput,
                timestamp
            FROM runs
            {order_clause}
        """
        c.execute(query)
        rows = c.fetchall()
        conn.close()

    except sqlite3.OperationalError as e:
        msg = str(e)
        if "no such column" in msg.lower():
            raise RuntimeError(
                f"Plotting error: database '{db_path}' is missing required columns.\n"
                "Ensure 'batch_size', 'world_size', 'final_val_acc', and 'throughput' are logged."
            )
        else:
            raise

    if not rows:
        print(f"No runs found in {db_path}. Nothing to plot.")
        return

    # Extract columns into numpy arrays
    experiment_names = [row[0] for row in rows]
    datasets = [row[1] for row in rows]
    models = [row[2] for row in rows]
    batch_sizes = np.array([row[3] for row in rows], dtype=int)
    world_sizes = np.array([row[4] for row in rows], dtype=int)
    val_accs = np.array([row[5] for row in rows], dtype=float)
    throughputs = np.array([row[6] for row in rows], dtype=float)
    timestamps = [row[7] for row in rows]

    # Determine primary experiment info (first row)
    primary_exp = experiment_names[0]
    primary_dataset = datasets[0]
    primary_model = models[0]
    timestamp_for_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Identify swept parameter
    unique_bs = sorted(set(batch_sizes.tolist()))
    unique_ws = sorted(set(world_sizes.tolist()))

    if len(unique_bs) > 1 and len(unique_ws) == 1:
        # Sweep over batch_size
        x_values = unique_bs
        x_label = "Batch Size"
        avg_acc = [val_accs[batch_sizes == bs].mean() for bs in x_values]
        avg_thr = [throughputs[batch_sizes == bs].mean() for bs in x_values]
    elif len(unique_ws) > 1 and len(unique_bs) == 1:
        # Sweep over world_size
        x_values = unique_ws
        x_label = "Number of GPUs"
        avg_acc = [val_accs[world_sizes == ws].mean() for ws in x_values]
        avg_thr = [throughputs[world_sizes == ws].mean() for ws in x_values]
    else:
        # Default to world_size if no clear single sweep
        x_values = unique_ws
        x_label = "Number of GPUs"
        avg_acc = [val_accs[world_sizes == ws].mean() for ws in x_values]
        avg_thr = [throughputs[world_sizes == ws].mean() for ws in x_values]

    # Plot: Validation Accuracy vs. swept parameter
    plt.figure()
    plt.plot(x_values, avg_acc, marker="o")
    plt.xlabel(x_label)
    plt.ylabel("Average Final Validation Accuracy")
    plt.title(f"{primary_exp}: {primary_model} on {primary_dataset}")
    plt.grid(True)
    acc_filename = f"{timestamp_for_filename}_{primary_exp}_acc_vs_{x_label.replace(' ', '_').lower()}.png"
    acc_path = os.path.join(plots_dir, acc_filename)
    if not os.path.exists(acc_path) or overwrite:
        plt.savefig(acc_path)
        print(f"Saved figure: {acc_path}")
    plt.close()

    # Plot: Throughput vs. swept parameter
    plt.figure()
    plt.plot(x_values, avg_thr, marker="o")
    plt.xlabel(x_label)
    plt.ylabel("Average Throughput (samples/sec)")
    plt.title(f"{primary_exp}: {primary_model} on {primary_dataset}")
    plt.grid(True)
    thr_filename = f"{timestamp_for_filename}_{primary_exp}_throughput_vs_{x_label.replace(' ', '_').lower()}.png"
    thr_path = os.path.join(plots_dir, thr_filename)
    if not os.path.exists(thr_path) or overwrite:
        plt.savefig(thr_path)
        print(f"Saved figure: {thr_path}")
    plt.close()

    # Generate results.md
    md_path = os.path.join(output_dir, "results.md")
    if os.path.exists(md_path) and not overwrite:
        print(f"{md_path} already exists. Use overwrite=True to re-generate.")
        return

    with open(md_path, "w") as f:
        f.write(f"# GNN Benchmark Results: {primary_exp}\n\n")
        f.write(f"**Dataset:** {primary_dataset}  \n")
        f.write(f"**Model:** {primary_model}  \n")
        f.write(f"**Sweep parameter:** {x_label}  \n\n")

        # Summary table
        f.write("## Summary of Runs\n\n")
        f.write(f"| {x_label} | Val Acc | Throughput |\n")
        f.write(f"|:{'-' * len(x_label)}:|:-------:|:---------:|\n")
        for xv, acc_val, thr_val in zip(x_values, avg_acc, avg_thr):
            f.write(f"| {xv:^8d} | {acc_val:.4f} | {thr_val:.2f} |\n")
        f.write("\n")

        # Embed plots
        f.write("## Plots\n\n")
        f.write(f"![Val Acc vs {x_label}](plots/{acc_filename})\n\n")
        f.write(f"![Throughput vs {x_label}](plots/{thr_filename})\n\n")

        # Detailed metadata per run
        f.write("## Detailed Experiment Metadata\n\n")
        f.write("| Experiment | Dataset | Model | Batch Size | GPUs | Val Acc | Throughput | Timestamp |\n")
        f.write("|:----------|:-------|:------|:----------:|:----:|:-------:|:---------:|:---------:|\n")
        for exp_name, ds, mdl, bs, ws, acc_val, thr_val, ts in rows:
            f.write(
                f"| {exp_name} | {ds} | {mdl} | {bs:^10d} | {ws:^4d} | "
                f"{acc_val:.4f} | {thr_val:.2f} | {ts} |\n"
            )
        f.write("\n")

    print(f"Generated Markdown report: {md_path}")
