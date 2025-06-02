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
from torch_geometric.loader import NeighborLoader

from ogb.nodeproppred import PygNodePropPredDataset

from .db_logger import DBLogger
from .ddp_utils import setup_ddp, cleanup_ddp


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
    Supports Planetoid (e.g. Cora/Citeseer/Pubmed) and OGB (e.g. ogbn-arxiv).
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
        dataset = PygNodePropPredDataset(name=name, root=os.path.join(os.getcwd(), "data"))
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]
        val_idx = split_idx["valid"]

        # Create boolean masks
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


def train_epoch(model, optimizer, criterion, train_loader, device):
    """
    Runs one training epoch using mini-batches from train_loader.
    Returns (average loss, average accuracy).
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index)
        out_root = out[: batch.batch_size]
        y_root = batch.y[: batch.batch_size].to(device)

        loss = criterion(out_root, y_root)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * out_root.size(0)
        total_correct += int((out_root.argmax(dim=1) == y_root).sum())
        total_examples += out_root.size(0)

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc


def evaluate(model, criterion, val_loader, device):
    """
    Evaluates on the validation set using mini-batches from val_loader.
    Returns (average loss, average accuracy).
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            out_root = out[: batch.batch_size]
            y_root = batch.y[: batch.batch_size].to(device)

            loss = criterion(out_root, y_root)
            total_loss += float(loss) * out_root.size(0)
            total_correct += int((out_root.argmax(dim=1) == y_root).sum())
            total_examples += out_root.size(0)

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc


def main(args):
    """
    Single run of GNN training & evaluation (node-level) with mini-batch neighbor sampling.
    """
    set_seed(args.seed)

    is_ddp = args.world_size > 1
    if is_ddp:
        setup_ddp(args)
        if torch.cuda.is_available() and not getattr(args, "no_cuda", False):
            device = torch.device(f"cuda:{args.local_rank}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and not getattr(args, "no_cuda", False) else "cpu")

    # Print parameter summary (only rank 0/non-DDP)
    if (not is_ddp) or (is_ddp and args.rank == 0):
        summary = [
            f"dataset={args.dataset}",
            f"model={args.model}",
            f"epochs={args.epochs}",
            f"batch_size={args.batch_size}",
            f"lr={args.lr}",
            f"hidden_dim={args.hidden_dim}",
            f"seed={args.seed}",
            f"world_size={args.world_size}",
            f"rank={args.rank}",
        ]
        print("▶ Ex. Params:", " | ".join(summary))

    # Load data
    data, in_channels, num_classes, train_mask, val_mask = load_dataset(args.dataset)

    # Build model and move to device
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

    # Create NeighborLoader for training
    train_loader = NeighborLoader(
        data,
        input_nodes=train_mask,
        num_neighbors=[10, 10],
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Create NeighborLoader for validation (no shuffling)
    val_loader = NeighborLoader(
        data,
        input_nodes=val_mask,
        num_neighbors=[10, 10],
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Prepare metrics storage
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    throughputs = []
    epoch_times = []
    epoch_times_all = []

    num_train_nodes = int(train_mask.sum().item())

    # Training loop
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        loss_train, acc_train = train_epoch(model, optimizer, criterion, train_loader, device)
        loss_val, acc_val = evaluate(model, criterion, val_loader, device)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        # Throughput: number of training nodes processed per second
        epoch_throughput = num_train_nodes / epoch_time
        epoch_times_all.append(epoch_time)

        if (not is_ddp) or (is_ddp and args.rank == 0):
            print(
                f"[E{epoch:03d}] "
                f"TLoss={loss_train:.4f} "
                f"TAcc={acc_train:.4f} "
                f"VLoss={loss_val:.4f} "
                f"VAcc={acc_val:.4f} "
                f"Time={epoch_time:.2f}s "
                f"Thr={epoch_throughput:.2f} nodes/s"
            )

        if epoch > 1:
            train_losses.append(loss_train)
            train_accuracies.append(acc_train)
            val_losses.append(loss_val)
            val_accuracies.append(acc_val)
            throughputs.append(epoch_throughput)
            epoch_times.append(epoch_time)

    # Compute averages (excluding first epoch)
    if train_losses:
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_acc = sum(train_accuracies) / len(train_accuracies)
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_acc = sum(val_accuracies) / len(val_accuracies)
        avg_throughput = sum(throughputs) / len(throughputs)
        avg_time = sum(epoch_times) / len(epoch_times)

        # Print summary
        if (not is_ddp) or (is_ddp and args.rank == 0):
            print()
            print("=== Averages (excl. 1st epoch) ===")
            print(
                f"TLoss={avg_train_loss:.4f} "
                f"TAcc={avg_train_acc:.4f} "
                f"VLoss={avg_val_loss:.4f} "
                f"VAcc={avg_val_acc:.4f} "
                f"Time={avg_time:.2f}s "
                f"Thr={avg_throughput:.2f} nodes/s\n"
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
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "final_val_acc": val_accuracies[-1] if val_accuracies else None,
            "throughput": avg_throughput if train_losses else None,
            "avg_train_loss": avg_train_loss if train_losses else None,
            "avg_train_acc": avg_train_acc if train_losses else None,
            "avg_val_loss": avg_val_loss if train_losses else None,
            "avg_val_acc": avg_val_acc if train_losses else None,
            "avg_time": avg_time if train_losses else None,
            "avg_throughput": avg_throughput if train_losses else None,
            "total_train_time": sum(epoch_times_all),
        }
        logger.log_run(params, metrics)

    if is_ddp:
        cleanup_ddp()
