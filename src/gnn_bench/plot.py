# src/gnn_bench/plot.py

import os
import sqlite3
import subprocess
import platform
import matplotlib.pyplot as plt
import numpy as np

def _open_file(path):
    """
    Cross-platform open: on macOS uses 'open', on Linux 'xdg-open', on Windows 'start'.
    """
    plat = platform.system()
    if plat == "Darwin":       # macOS
        subprocess.run(["open", path])
    elif plat == "Windows":    # Windows
        os.startfile(path)    # type: ignore
    else:                      # assume Linux/Unix
        subprocess.run(["xdg-open", path])

def main(db_path: str, output_dir: str, overwrite: bool = False,
         sort_by: str = "date", auto_open: bool = False):
    """
    Reads the `runs` table from db_path, sorts runs by `sort_by`,
    produces:
      - Timestamped PNGs for accuracy and throughput
      - results.md summary with embedded images and metadata tables
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # Build ORDER BY clause
        if sort_by == "date":
            order_clause = "ORDER BY timestamp DESC"
        elif sort_by == "acc":
            order_clause = "ORDER BY final_val_acc DESC"
        elif sort_by == "throughput":
            order_clause = "ORDER BY throughput DESC"
        else:
            order_clause = "ORDER BY timestamp DESC"

        # Fetch fields, including 'throughput' (required by legacy code)
        query = f"""
            SELECT
                experiment_name,
                dataset,
                model,
                epochs,
                batch_size,
                lr,
                hidden_dim,
                seed,
                world_size,
                rank,
                final_train_loss,
                final_val_loss,
                final_val_acc,
                total_train_time,
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
        if "no such column: throughput" in msg.lower():
            raise RuntimeError(
                f"Plotting error: database '{db_path}' missing column 'throughput'.\n"
                "  This usually means your training code did not write a 'throughput' field.\n"
                "  Ensure 'metrics' in train.py includes 'throughput'."
            )
        else:
            raise

    if len(rows) == 0:
        print(f"No runs found in {db_path}. Nothing to plot.")
        return

    # Determine the “primary” experiment for naming
    primary_exp = rows[0][0]  # experiment_name of first row
    now = np.datetime64('now')  # placeholder; we'll override with Python datetime
    from datetime import datetime
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Build data structures
    data_acc = {}         # data_acc[model][world_size] = list of final_val_acc
    data_throughput = {}  # data_throughput[model][world_size] = list of throughput

    for (
        experiment_name,
        dataset, model, epochs, batch_size, lr, hidden_dim,
        seed, world_size, rank,
        final_train_loss, final_val_loss, final_val_acc,
        total_train_time, throughput, timestamp
    ) in rows:
        data_acc.setdefault(model, {}).setdefault(world_size, []).append(final_val_acc)
        data_throughput.setdefault(model, {}).setdefault(world_size, []).append(throughput)

    # 1) Plot: Validation Accuracy vs. #GPUs
    plt.figure()
    for model, ws_dict in data_acc.items():
        ws_list = sorted(ws_dict.keys())
        avg_acc = [np.mean(ws_dict[ws]) for ws in ws_list]
        plt.plot(ws_list, avg_acc, marker="o", label=model)
    plt.xlabel("Number of GPUs (world_size)")
    plt.ylabel("Average Final Validation Accuracy")
    plt.title("GNN Benchmark: Val Accuracy vs. #GPUs")
    plt.legend()
    plt.grid(True)

    acc_filename = f"{now_str}_{primary_exp}_acc_vs_gpus.png"
    acc_fig = os.path.join(plots_dir, acc_filename)
    if not os.path.exists(acc_fig) or overwrite:
        plt.savefig(acc_fig)
        print(f"Saved figure: {acc_fig}")
    plt.close()

    # 2) Plot: Throughput vs. #GPUs
    plt.figure()
    for model, ws_dict in data_throughput.items():
        ws_list = sorted(ws_dict.keys())
        avg_thr = [np.mean(ws_dict[ws]) for ws in ws_list]
        plt.plot(ws_list, avg_thr, marker="o", label=model)
    plt.xlabel("Number of GPUs (world_size)")
    plt.ylabel("Average Throughput (samples/sec)")
    plt.title("GNN Benchmark: Throughput vs. #GPUs")
    plt.legend()
    plt.grid(True)

    thr_filename = f"{now_str}_{primary_exp}_throughput_vs_gpus.png"
    thr_fig = os.path.join(plots_dir, thr_filename)
    if not os.path.exists(thr_fig) or overwrite:
        plt.savefig(thr_fig)
        print(f"Saved figure: {thr_fig}")
    plt.close()

    # 3) Generate results.md
    md_path = os.path.join(output_dir, "results.md")
    if os.path.exists(md_path) and not overwrite:
        print(f"{md_path} already exists. Use overwrite=True to re-generate.")
        return

    with open(md_path, "w") as f:
        f.write("# GNN Benchmark Results\n\n")

        # Summary table
        f.write("## Summary of Runs\n\n")
        f.write("| Experiment | Dataset | Model | GPUs | Batch | LR     | Hidden | VAcc     | Thr       | Time    | Timestamp |\n")
        f.write("|:----------|:-------|:------|:----:|:-----:|:------|:------:|:--------:|:---------:|:-------:|:---------:|\n")
        for (
            experiment_name,
            dataset, model, epochs, batch_size, lr, hidden_dim,
            seed, world_size, rank,
            final_train_loss, final_val_loss, final_val_acc,
            total_train_time, throughput, timestamp
        ) in rows:
            time_str = f"{total_train_time:.2f}s"
            f.write(
                f"| {experiment_name} | {dataset} | {model} | {world_size:^3d} | {batch_size:^3d} | "
                f"{lr:.4f} | {hidden_dim:^3d} | {final_val_acc:.4f} | {throughput:.2f} | {time_str} | {timestamp} |\n"
            )
        f.write("\n")

        # Embed plots
        f.write("## Plots\n\n")
        f.write(f"![Val Acc vs GPUs](plots/{acc_filename})\n\n")
        f.write("**Fig 1:** Final validation accuracy vs. #GPUs.\n\n")
        f.write(f"![Throughput vs GPUs](plots/{thr_filename})\n\n")
        f.write("**Fig 2:** Throughput (samples/sec) vs. #GPUs.\n\n")

        # Detailed metadata
        f.write("## Detailed Experiment Metadata\n\n")
        for (
            experiment_name,
            dataset, model, epochs, batch_size, lr, hidden_dim,
            seed, world_size, rank,
            final_train_loss, final_val_loss, final_val_acc,
            total_train_time, throughput, timestamp
        ) in rows:
            f.write(f"### Experiment: `{experiment_name}`\n\n")
            f.write("| Parameter         | Value               |\n")
            f.write("|:------------------|:--------------------|\n")
            f.write(f"| Dataset           | {dataset}           |\n")
            f.write(f"| Model             | {model}             |\n")
            f.write(f"| Epochs            | {epochs}            |\n")
            f.write(f"| Batch Size        | {batch_size}        |\n")
            f.write(f"| Learning Rate     | {lr:.4f}           |\n")
            f.write(f"| Hidden Dim        | {hidden_dim}        |\n")
            f.write(f"| Seed              | {seed}              |\n")
            f.write(f"| World Size (GPUs) | {world_size}        |\n")
            f.write(f"| Rank              | {rank}              |\n")
            f.write(f"| Final Train Loss  | {final_train_loss:.4f} |\n")
            f.write(f"| Final Val Loss    | {final_val_loss:.4f} |\n")
            f.write(f"| Final Val Acc     | {final_val_acc:.4f} |\n")
            f.write(f"| Total Time        | {total_train_time:.2f}s |\n")
            f.write(f"| Throughput        | {throughput:.2f}    |\n")
            f.write(f"| Timestamp         | {timestamp}         |\n\n")

    print(f"Generated Markdown report: {md_path}")

    if auto_open:
        _open_file(md_path)
