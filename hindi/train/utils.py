"""
utils.py
========
Shared training utilities: random seeding and logging.
"""

from __future__ import annotations

import os
import random
import csv

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Lock random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrainingLogger:
    """
    Structured logger that writes training and validation events to a CSV file.
    """
    def __init__(self, log_dir: str, log_file: str = "training_log.csv") -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)
        self.fields = [
            "step",
            "tokens_seen",
            "event_type",   # "train" or "val"
            "lr",
            "train_loss",
            "val_loss",
            "val_perplexity"
        ]
        
        # Write header if file doesn't exist or is empty
        write_header = not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0
        if write_header:
            with open(self.log_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def log_train(self, step: int, tokens_seen: int, lr: float, train_loss: float) -> None:
        """Log a training step event."""
        with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow({
                "step": step,
                "tokens_seen": tokens_seen,
                "event_type": "train",
                "lr": f"{lr:.2e}",
                "train_loss": f"{train_loss:.4f}",
                "val_loss": "",
                "val_perplexity": ""
            })

    def log_val(self, step: int, tokens_seen: int, val_loss: float, val_perplexity: float) -> None:
        """Log a validation step event."""
        with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow({
                "step": step,
                "tokens_seen": tokens_seen,
                "event_type": "val",
                "lr": "",
                "train_loss": "",
                "val_loss": f"{val_loss:.4f}",
                "val_perplexity": f"{val_perplexity:.2f}"
            })
