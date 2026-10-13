"""
evaluate.py
===========
Provides the evaluation loop for the decoder-only Transformer.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    eval_batches: int,
    device: torch.device
) -> tuple[float, float]:
    """
    Evaluate the model on the validation dataloader.

    Parameters
    ----------
    model : nn.Module
        The decoder-only Transformer model.
    dataloader : DataLoader
        The validation dataloader yielding {"input_ids": ..., "labels": ...}.
    eval_batches : int
        Maximum number of batches to evaluate on.
    device : torch.device
        Device to move data to.

    Returns
    -------
    tuple[float, float]
        Average validation loss and perplexity.
    """
    # 1. Capture the original mode
    was_training = model.training

    # 2. Put model in evaluation mode
    model.eval()

    total_loss = 0.0
    batches_run = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= eval_batches:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Forward pass
            logits = model(input_ids)  # (B, T, vocab_size)

            # Compute cross entropy loss.
            # The dataset already shifts labels (input_ids=window[:ctx], labels=window[1:ctx+1]).
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            batches_run += 1

    if batches_run == 0:
        val_loss = 0.0
        val_perplexity = 0.0
    else:
        val_loss = total_loss / batches_run
        # Prevent math overflow for extremely large losses
        try:
            val_perplexity = math.exp(val_loss)
        except OverflowError:
            val_perplexity = float("inf")

    # 3. Restore original training mode
    model.train(was_training)

    return val_loss, val_perplexity
