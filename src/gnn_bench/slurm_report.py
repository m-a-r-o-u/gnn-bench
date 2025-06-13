#!/usr/bin/env python3
"""Summarize SLURM usage for one or more users."""

from __future__ import annotations

import argparse
import csv
import subprocess
from typing import Dict, List

import pandas as pd


def _parse_elapsed(elapsed: str) -> int:
    """Return elapsed time in seconds."""
    days = 0
    if "-" in elapsed:
        day_part, time_part = elapsed.split("-", 1)
        days = int(day_part)
    else:
        time_part = elapsed
    parts = time_part.split(":")
    parts = [0] * (3 - len(parts)) + [int(p) for p in parts]
    hours, minutes, seconds = parts
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _parse_tres(tres: str) -> Dict[str, str]:
    res: Dict[str, str] = {}
    for item in tres.split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            res[k] = v
    return res


def _parse_mem(mem: str) -> float:
    """Return memory in GB."""
    mem = mem.strip().upper()
    if mem.endswith("G"):
        return float(mem[:-1])
    if mem.endswith("M"):
        return float(mem[:-1]) / 1024.0
    if mem.endswith("T"):
        return float(mem[:-1]) * 1024.0
    try:
        return float(mem)
    except ValueError:
        return 0.0


def _fetch_sacct(users: List[str], start: str, end: str) -> pd.DataFrame:
    cmd = [
        "sacct",
        "-n",
        "-P",
        "-X",
        "-S",
        start,
        "-E",
        end,
        "--format=User,Partition,Elapsed,CPUTimeRAW,AllocTRES",
    ]
    if users:
        cmd.extend(["-u", ",".join(users)])
    output = subprocess.check_output(cmd, text=True)
    records = []
    reader = csv.reader(output.splitlines(), delimiter="|")
    for user, part, elapsed, cpu_raw, tres in reader:
        if not user:
            continue
        elapsed_sec = _parse_elapsed(elapsed)
        tres_dict = _parse_tres(tres)
        cpu_hours = float(cpu_raw) / 3600.0
        gpu_count = int(tres_dict.get("gres/gpu", tres_dict.get("gpu", 0)))
        mem_gb = _parse_mem(tres_dict.get("mem", "0"))
        gpu_hours = elapsed_sec / 3600.0 * gpu_count
        ram_hours = elapsed_sec / 3600.0 * mem_gb
        records.append(
            {
                "UserID": user.strip(),
                "Partition": part.strip(),
                "CPU_Hours": cpu_hours,
                "GPU_Hours": gpu_hours,
                "RAM_Hours": ram_hours,
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate aggregated SLURM usage report"
    )
    parser.add_argument(
        "--user", action="append", dest="users", help="User(s) to query", required=False
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--partition",
        default="lrz-dgx-a100-80x8",
        help="Partition to show breakdown for",
    )
    args = parser.parse_args()

    df = _fetch_sacct(args.users or [], args.start, args.end)
    if df.empty:
        print("No jobs found for the given query")
        return

    totals = (
        df.groupby("UserID")[["CPU_Hours", "GPU_Hours", "RAM_Hours"]]
        .sum()
        .reset_index()
    )
    part = (
        df[df["Partition"] == args.partition]
        .groupby("UserID")[["CPU_Hours", "GPU_Hours", "RAM_Hours"]]
        .sum()
        .add_prefix(f"{args.partition}_")
        .reset_index()
    )
    merged = totals.merge(part, on="UserID", how="left").fillna(0)

    overall = merged.drop(columns=["UserID"]).sum()
    overall["UserID"] = "ALL_USERS"
    merged = pd.concat([pd.DataFrame([overall]), merged], ignore_index=True)

    print(merged.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
