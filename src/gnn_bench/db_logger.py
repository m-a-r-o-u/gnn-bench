# src/gnn_bench/db_logger.py

import sqlite3
import threading
from typing import Dict, Any

_LOCK = threading.Lock()

class DBLogger:
    """
    Thread‐safe SQLite logger. Each run writes one row to `runs`:
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      experiment_name TEXT,
      dataset TEXT,
      model TEXT,
      epochs INTEGER,
      batch_size INTEGER,
      lr REAL,
      hidden_dim INTEGER,
      seed INTEGER,
      world_size INTEGER,
      rank INTEGER,
      final_train_loss REAL,
      final_val_loss REAL,
      final_val_acc REAL,
      total_train_time REAL,
      throughput REAL,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with _LOCK:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT,
                    dataset TEXT,
                    model TEXT,
                    epochs INTEGER,
                    batch_size INTEGER,
                    lr REAL,
                    hidden_dim INTEGER,
                    seed INTEGER,
                    world_size INTEGER,
                    rank INTEGER,
                    final_train_loss REAL,
                    final_val_loss REAL,
                    final_val_acc REAL,
                    total_train_time REAL,
                    throughput REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            conn.close()

    def log_run(self, params: Dict[str, Any], metrics: Dict[str, float]):
        """
        params should contain:
          experiment_name, dataset, model, epochs, batch_size, lr,
          hidden_dim, seed, world_size, rank
        metrics should contain:
          final_train_loss, final_val_loss, final_val_acc,
          total_train_time, throughput
        """
        with _LOCK:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("""
                INSERT INTO runs (
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
                    throughput
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                params["experiment_name"],
                params["dataset"],
                params["model"],
                params["epochs"],
                params["batch_size"],
                params["lr"],
                params["hidden_dim"],
                params["seed"],
                params["world_size"],
                params["rank"],
                metrics["final_train_loss"],
                metrics["final_val_loss"],
                metrics["final_val_acc"],
                metrics["total_train_time"],
                metrics["throughput"],
            ))
            conn.commit()
            conn.close()
