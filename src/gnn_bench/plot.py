# src/gnn_bench/plot.py

import os
import sqlite3
import matplotlib
# Use non-interactive backend to avoid crashes on headless servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timezone
import yaml


def main(db_path: str, output_dir: str, overwrite: bool = False, sort_by: str = "date", config_path: str | None = None):
    """
    Reads the `runs` table from db_path, detects if a single parameter was swept,
    and plots metrics (val accuracy and throughput) against that parameter
    (e.g., batch_size or world_size). Generates:
      - PNG plots in output_dir/plots
      - A Markdown report output_dir/<timestamp>_<experiment>_results.md with
        embedded images and metadata.
    If `config_path` is provided, the YAML config is parsed so that the report
    can include a high level overview and experiment differences.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    config_text = ""
    exp_cfg = {}
    diff_params = {}
    common_cfg = {}
    sweep_param = {}
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r") as fcfg:
            config_text = fcfg.read()
            cfg = yaml.safe_load(config_text) or {}
            experiments_cfg = cfg.get("experiments", [])
            # Determine value sets per key
            value_sets: dict[str, set[str]] = {}
            for exp in experiments_cfg:
                for k, v in exp.items():
                    value_sets.setdefault(k, set()).add(str(v))
            varying_keys = {k for k, vals in value_sets.items() if len(vals) > 1}
            common_cfg = {
                k: next(iter(vals)) for k, vals in value_sets.items() if len(vals) == 1
            }
            for exp in experiments_cfg:
                name = exp.get("experiment_name", "exp")
                exp_cfg[name] = exp
                diff_params[name] = {k: exp[k] for k in varying_keys if k in exp}
                for k, v in exp.items():
                    if isinstance(v, list):
                        sweep_param[name] = k
                        break

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

        if exp_cfg:
            rows = [r for r in rows if r[0] in exp_cfg]

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

    timestamp_for_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Group rows by experiment
    grouped = {}
    first_ts = {}
    for row in rows:
        exp = row[0]
        grouped.setdefault(exp, []).append(row)
        ts = datetime.fromisoformat(row[7])
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            ts = ts.replace(tzinfo=timezone.utc)
        prev_ts = first_ts.get(exp, ts)
        if prev_ts.tzinfo is None or prev_ts.tzinfo.utcoffset(prev_ts) is None:
            prev_ts = prev_ts.replace(tzinfo=timezone.utc)
        first_ts[exp] = min(prev_ts, ts)

    # Sort experiments by earliest timestamp
    sorted_exps = sorted(grouped.keys(), key=lambda x: first_ts[x])

    # Generate timestamped results.md so successive runs don't overwrite
    md_filename = f"{timestamp_for_filename}_results.md"
    md_path = os.path.join(output_dir, md_filename)
    if os.path.exists(md_path) and not overwrite:
        print(f"{md_path} already exists. Use overwrite=True to re-generate.")
        return

    with open(md_path, "w") as f:
        f.write("# GNN Benchmark Results\n\n")
        if common_cfg:
            f.write("## Common Settings\n\n")
            f.write("```yaml\n")
            f.write(yaml.safe_dump(common_cfg, sort_keys=False))
            f.write("```\n\n")

        for exp in sorted_exps:
            exp_rows = grouped[exp]
            dataset = exp_rows[0][1]
            model = exp_rows[0][2]
            param = sweep_param.get(exp)
            unique_bs = sorted({r[3] for r in exp_rows})
            unique_ws = sorted({r[4] for r in exp_rows})
            if not param:
                if len(unique_bs) > 1 and len(unique_ws) == 1:
                    param = "batch_size"
                else:
                    param = "world_size"
            idx = 3 if param == "batch_size" else 4
            x_label = "Batch Size" if param == "batch_size" else "Number of GPUs"
            x_values = sorted({r[idx] for r in exp_rows})
            avg_acc = [np.mean([r[5] for r in exp_rows if r[idx] == xv]) for xv in x_values]
            avg_thr = [np.mean([r[6] for r in exp_rows if r[idx] == xv]) for xv in x_values]

            # Plots per experiment
            plt.figure()
            plt.plot(x_values, avg_acc, marker="o")
            plt.xlabel(x_label)
            plt.ylabel("Average Final Validation Accuracy")
            plt.title(f"{exp}: {model} on {dataset}")
            plt.grid(True)
            acc_filename = f"{timestamp_for_filename}_{exp}_acc_vs_{x_label.replace(' ', '_').lower()}.png"
            acc_path = os.path.join(plots_dir, acc_filename)
            if not os.path.exists(acc_path) or overwrite:
                plt.savefig(acc_path)
                print(f"Saved figure: {acc_path}")
            plt.close()

            plt.figure()
            plt.plot(x_values, avg_thr, marker="o")
            plt.xlabel(x_label)
            plt.ylabel("Average Throughput (samples/sec)")
            plt.title(f"{exp}: {model} on {dataset}")
            plt.grid(True)
            thr_filename = f"{timestamp_for_filename}_{exp}_throughput_vs_{x_label.replace(' ', '_').lower()}.png"
            thr_path = os.path.join(plots_dir, thr_filename)
            if not os.path.exists(thr_path) or overwrite:
                plt.savefig(thr_path)
                print(f"Saved figure: {thr_path}")
            plt.close()

            f.write(f"## {exp}\n\n")
            diff_cfg = diff_params.get(exp)
            if diff_cfg:
                f.write("**Different config:**\n")
                f.write("```yaml\n")
                f.write(yaml.safe_dump(diff_cfg, sort_keys=False))
                f.write("```\n\n")

            f.write("| {} | Val Acc | Throughput |\n".format(x_label))
            f.write("|:{}:|:-------:|:---------:|\n".format('-' * len(x_label)))
            for xv, acc_val, thr_val in zip(x_values, avg_acc, avg_thr):
                f.write(f"| {xv:^8} | {acc_val:.4f} | {thr_val:.2f} |\n")
            f.write("\n")
            f.write(f"![Val Acc vs {x_label}](plots/{acc_filename})\n\n")
            f.write(f"![Throughput vs {x_label}](plots/{thr_filename})\n\n")

        # Experiment metadata table
        f.write("## Experiment Metadata from the Database\n\n")
        f.write("| Experiment | Dataset | Model | Batch Size | GPUs | Val Acc | Throughput | Timestamp |\n")
        f.write("|:----------|:-------|:------|:----------:|:----:|:-------:|:---------:|:---------:|\n")

        rows_meta = sorted(
            rows,
            key=lambda r: datetime.fromisoformat(r[7]),
            reverse=True,
        )

        for r in rows_meta:
            ts = datetime.fromisoformat(r[7])
            ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S")
            f.write(
                f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:^10d} | {r[4]:^4d} | {r[5]:.4f} | {r[6]:.2f} | {ts_str} |\n"
            )
        f.write("\n")

    print(f"Generated Markdown report: {md_path}")
