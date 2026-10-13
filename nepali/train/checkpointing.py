"""
checkpointing.py
================
Robust saving and loading of model checkpoints, including exact training state.
"""

from __future__ import annotations

import os
import glob
from dataclasses import asdict
from typing import Tuple, Any, Dict

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler

from nepali.train.dataset import StreamingPackedDataset
from nepali.train.config import TrainingConfig
from nepali.train.scheduler import WarmupCosineScheduler


def save_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: WarmupCosineScheduler,
    training_step: int,
    tokens_seen: int,
    best_val_loss: float,
    dataset: StreamingPackedDataset,
    config: TrainingConfig,
    is_best: bool = False,
    scaler: GradScaler | None = None,
) -> None:
    """
    Save a full training checkpoint atomically.

    Parameters
    ----------
    checkpoint_dir : str
        Directory to save checkpoints.
    model : nn.Module
        The Transformer model.
    optimizer : Optimizer
        The AdamW optimizer.
    scheduler : WarmupCosineScheduler
        The learning rate scheduler.
    training_step : int
        Current optimizer update step.
    tokens_seen : int
        Total tokens processed so far.
    best_val_loss : float
        The best validation loss seen so far.
    dataset : StreamingPackedDataset
        The dataset to extract the resume state_dict from.
    config : TrainingConfig
        The training configuration.
    is_best : bool, optional
        Whether to also save this checkpoint as the `best_model.pt`.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Build checkpoint payload once
    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "training_step": training_step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "dataset_state": dataset.state_dict(),
        "config": asdict(config),
    }

    if scaler is not None:
        checkpoint_dict["scaler_state_dict"] = scaler.state_dict()

    # 2. Save regular checkpoint atomically
    ckpt_name = f"checkpoint_step_{training_step}.pt"
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    tmp_path = ckpt_path + ".tmp"
    
    torch.save(checkpoint_dict, tmp_path)
    os.replace(tmp_path, ckpt_path)

    # 3. If best, save best_model.pt atomically
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        best_tmp = best_path + ".tmp"
        torch.save(checkpoint_dict, best_tmp)
        os.replace(best_tmp, best_path)

    # 4. Cleanup old regular checkpoints (keep config.keep_last_checkpoints)
    # Match pattern: checkpoint_step_*.pt
    pattern = os.path.join(checkpoint_dir, "checkpoint_step_*.pt")
    files = glob.glob(pattern)

    # Parse steps and sort numerically
    def extract_step(filepath: str) -> int:
        filename = os.path.basename(filepath)
        # expected format: checkpoint_step_{step}.pt
        try:
            return int(filename.replace("checkpoint_step_", "").replace(".pt", ""))
        except ValueError:
            return -1

    files_with_steps = [(extract_step(f), f) for f in files]
    # Filter out any files that didn't parse correctly (e.g., returns -1)
    files_with_steps = [fws for fws in files_with_steps if fws[0] >= 0]
    
    # Sort by step ascending
    files_with_steps.sort(key=lambda x: x[0])

    # Keep only the last N
    keep_n = config.keep_last_checkpoints
    if len(files_with_steps) > keep_n:
        to_delete = files_with_steps[:-keep_n]
        for _, filepath in to_delete:
            try:
                os.remove(filepath)
            except OSError:
                pass


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: WarmupCosineScheduler,
    dataset: StreamingPackedDataset,
    device: torch.device,
    scaler: GradScaler | None = None,
) -> Tuple[int, int, float, Dict[str, Any]]:
    """
    Load a checkpoint, restoring all model, optimizer, scheduler, and dataset states.

    Parameters
    ----------
    checkpoint_path : str
        Path to the `.pt` checkpoint file.
    model : nn.Module
        Model to load weights into.
    optimizer : Optimizer
        Optimizer to load state into.
    scheduler : WarmupCosineScheduler
        Scheduler to load state into.
    dataset : StreamingPackedDataset
        Dataset to restore offsets/buffer state into.
    device : torch.device
        Device to map tensors to safely.

    Returns
    -------
    Tuple[int, int, float, Dict[str, Any]]
        A tuple of (training_step, tokens_seen, best_val_loss, config_dict).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Safely load to target device
    checkpoint = torch.load(checkpoint_path, map_location=device)

    required_keys = {
        "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "training_step", "tokens_seen", "best_val_loss", "dataset_state", "config"
    }
    missing = required_keys - set(checkpoint.keys())
    if missing:
        raise RuntimeError(f"Checkpoint missing required keys: {missing}")

    # Restore states
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    dataset.load_state_dict(checkpoint["dataset_state"])

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    training_step = checkpoint["training_step"]
    tokens_seen = checkpoint["tokens_seen"]
    best_val_loss = checkpoint["best_val_loss"]
    config_dict = checkpoint["config"]

    return training_step, tokens_seen, best_val_loss, config_dict
