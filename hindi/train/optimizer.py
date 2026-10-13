"""
optimizer.py
============
Creates the AdamW optimizer with exact parameter grouping for weight decay.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import AdamW

from .config import TrainingConfig
from hindi.model.transformer import DecoderOnlyTransformer


def create_optimizer(model: DecoderOnlyTransformer, config: TrainingConfig) -> AdamW:
    """
    Create an AdamW optimizer for the model with strict parameter grouping.

    Rules for weight decay:
    - Apply weight decay: all weight matrices (except token embeddings).
    - No weight decay: all biases, all LayerNorm parameters, token embeddings.
    
    Tied weights (output_proj.weight == token_embedding.weight) must appear
    exactly once in the optimizer, placed in the no-weight-decay group.
    """
    decay = set()
    no_decay = set()

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue

            param_id = id(p)
            
            # If we've already grouped this parameter (e.g. tied weights), skip.
            if param_id in decay or param_id in no_decay:
                continue
                
            if isinstance(m, nn.Embedding):
                # Token embeddings get no weight decay.
                no_decay.add(param_id)
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm weights and biases get no weight decay.
                no_decay.add(param_id)
            elif isinstance(m, nn.Linear):
                # Biases get no weight decay. Linear weights get weight decay.
                if pn == "weight":
                    decay.add(param_id)
                elif pn == "bias":
                    no_decay.add(param_id)
            else:
                raise ValueError(f"Unexpected module type {type(m)} with parameter {pn}")

    # Build the actual parameter lists by finding the params that match these IDs
    decay_params = []
    no_decay_params = []
    
    seen_ids = set()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
            
        param_id = id(p)
        if param_id in seen_ids:
            continue
            
        seen_ids.add(param_id)
        
        if param_id in decay:
            decay_params.append(p)
        elif param_id in no_decay:
            no_decay_params.append(p)
        else:
            raise ValueError(f"Parameter {n} was not placed in any group!")

    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )

    return optimizer
