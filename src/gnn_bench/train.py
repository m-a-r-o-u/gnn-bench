# src/gnn_bench/train.py

import time
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

from .ddp_utils import setup_ddp, cleanup_ddp
from .db_logger import DBLogger


class GCN(torch.nn.Module):
    """2-layer GCN for node classification."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    """2-layer GAT for node classification."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, dropout):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = nn.functional.elu(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(name: str):
    """
    Returns (data, in_channels, num_classes, train_mask, val_mask).
    Supports Planetoid (e.g. Cora) and OGB (e.g. ogbn-arxiv).
    """
    if name in ["Cora", "Citeseer", "Pubmed"]:
        path = os.path.join(os.getcwd(), "data", name)
        dataset = Planetoid(path, name)
        data = dataset[0]
        train_mask = data.train_mask
        val_mask = data.val_mask
        in_channels = dataset.num_node_features
        num_classes = dataset.num_classes
        return data, in_channels, num_classes, train_mask, val_mask

    elif name.startswith("ogbn-"):
        # Allowlist PyG classes before torch.load (PyTorch 2.6+)
        torch.serialization.add_safe_globals([
            DataEdgeAttr,
            DataTensorAttr,
            GlobalStorage
        ])

        dataset = PygNodePropPredDataset(
            name=name,
            root=os.path.join(os.getcwd(), "data", name)
        )
        split_idx = dataset.get_idx_split()
        data = dataset[0]
        data.y = data.y.squeeze(1)

        train_idx = split_idx["train"]
        val_idx = split_idx["valid"]
        num_nodes = data.num_nodes

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[val_idx] = True

        in_channels = dataset.num_node_features
        num_classes = dataset.num_classes
        return data, in_channels, num_classes, train_mask, val_mask

    else:
        raise ValueError(f"Unsupported dataset: {name}")


def train_epoch(model, optimizer, criterion, data, train_mask, device):
    """
    Runs one training epoch, returns (loss, accuracy).
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask].to(device))
    loss.backward()
    optimizer.step()

    preds = out[train_mask].argmax(dim=1)
    acc = (preds == data.y[train_mask].to(device)).float().mean().item()
    return loss.item(), acc


def evaluate(model, criterion, data, mask, device):
    """
    Evaluates on the given mask (validation/test), returns (loss, accuracy).
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        loss = criterion(out[mask], data.y[mask].to(device))
        preds = out[mask].argmax(dim=1)
        acc = (preds == data.y[mask].to(device)).float().mean().item()
    return loss.item(), acc


def main(args):
    """
    Single run of GNN training & evaluation (node-level).
    - Supports Planetoid (Cora/Citeseer/Pubmed) and OGB (e.g. ogbn-arxiv).
    - Prints a one-line summary of parameters used at start of run.
    - Prints per-epoch metrics: TLoss, TAcc, VLoss, VAcc, Time, Thr.
    - At end prints a concise summary: batch_size, Avg VAcc, Avg Thr.
    - Logs to SQLite via DBLogger.
    """
    set_seed(args.seed)

    is_ddp = args.world_size > 1
    if is_ddp:
        setup_ddp(args)
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print parameter summary (only rank 0/non-DDP)
    if (not is_ddp) or (is_ddp and args.rank == 0):
        summary = [
            f"dataset={args.dataset}",
            f"model={args.model}",
            f"epochs={args.epochs}",
            f"batch_size={args.batch_size}",
            f"lr={args.lr:.4f}",
            f"hidden_dim={args.hidden_dim}",
        ]
        if args.model.lower() == "gat":
            summary += [f"num_heads={args.num_heads}", f"dropout={args.dropout:.2f}"]
        summary += [f"seed={args.seed}", f"world_size={args.world_size}"]
        print("\n▶ Ex. Params: " + "  ".join(summary) + "\n")

    data, in_channels, num_classes, train_mask, val_mask = load_dataset(args.dataset)
    data = data.to(device)

    if args.model.lower() == "gcn":
        model = GCN(in_channels, args.hidden_dim, num_classes).to(device)
    elif args.model.lower() == "gat":
        model = GAT(in_channels, args.hidden_dim, num_classes, args.num_heads, args.dropout).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    if is_ddp:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Metrics (excluding first epoch)
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    throughputs = []
    epoch_times = []

    # All epoch times (including first), for total_train_time
    epoch_times_all = []

    final_train_loss = None
    final_val_loss = None
    final_val_acc = None

    num_nodes = data.num_nodes

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        loss_train, acc_train = train_epoch(model, optimizer, criterion, data, train_mask, device)
        loss_val, acc_val = evaluate(model, criterion, data, val_mask, device)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        epoch_throughput = num_nodes / epoch_time

        epoch_times_all.append(epoch_time)

        if (not is_ddp) or (is_ddp and args.rank == 0):
            print(
                f"[E{epoch:03d}] "
                f"TLoss={loss_train:.4f} "
                f"TAcc={acc_train:.4f} "
                f"VLoss={loss_val:.4f} "
                f"VAcc={acc_val:.4f} "
                f"Time={epoch_time:.2f}s "
                f"Thr={epoch_throughput:.2f}"
            )

        final_train_loss = loss_train
        final_val_loss = loss_val
        final_val_acc = acc_val

        if epoch > 1:
            train_losses.append(loss_train)
            train_accuracies.append(acc_train)
            val_losses.append(loss_val)
            val_accuracies.append(acc_val)
            throughputs.append(epoch_throughput)
            epoch_times.append(epoch_time)

    if train_losses:
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_acc = sum(train_accuracies) / len(train_accuracies)
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_acc = sum(val_accuracies) / len(val_accuracies)
        avg_throughput = sum(throughputs) / len(throughputs)
        avg_time = sum(epoch_times) / len(epoch_times)
    else:
        avg_train_loss = final_train_loss
        avg_train_acc = 0.0
        avg_val_loss = final_val_loss
        avg_val_acc = final_val_acc
        avg_throughput = num_nodes / (epoch_end - epoch_start)
        avg_time = (epoch_end - epoch_start)

    # End‐of‐run summary (only rank 0/non-DDP)
    if (not is_ddp) or (is_ddp and args.rank == 0):
        print("\n=== Averages (excl. 1st epoch) ===")
        print(
            f"TLoss={avg_train_loss:.4f} "
            f"TAcc={avg_train_acc:.4f} "
            f"VLoss={avg_val_loss:.4f} "
            f"VAcc={avg_val_acc:.4f} "
            f"Time={avg_time:.2f}s "
            f"Thr={avg_throughput:.2f} smpls/s\n"
        )

    # Log to database (only rank 0/non-DDP)
    if (not is_ddp) or (is_ddp and args.rank == 0):
        logger = DBLogger(args.results_db)
        params = {
            "experiment_name": args.experiment_name,
            "dataset": args.dataset,
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "seed": args.seed,
            "world_size": args.world_size,
            "rank": args.rank,
        }
        metrics = {
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "final_val_acc": final_val_acc,
            "throughput": avg_throughput,            # for backward compatibility
            "avg_train_loss": avg_train_loss,
            "avg_train_acc": avg_train_acc,
            "avg_val_loss": avg_val_loss,
            "avg_val_acc": avg_val_acc,
            "avg_time": avg_time,
            "avg_throughput": avg_throughput,
            "total_train_time": sum(epoch_times_all),
        }
        logger.log_run(params, metrics)

    if is_ddp:
        cleanup_ddp()



























