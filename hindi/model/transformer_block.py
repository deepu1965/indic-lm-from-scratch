"""
transformer_block.py
====================
A single Pre-LayerNorm Transformer block for the decoder-only model.

Architecture (Pre-LN)
----------------------
Each block applies layer normalisation **before** the sub-layer (not after),
which improves training stability compared to the original Post-LN paper.

    x = x + residual_dropout( attention( layer_norm_1(x) ) )
    x = x + residual_dropout( ffn( layer_norm_2(x) ) )

This is often called GPT-style or Pre-LN Transformer.

Sub-layers
----------
1. ``CausalSelfAttention``   — manually implemented attention with RoPE and
                               causal masking (see attention.py).

2. Feed-Forward Network (FFN)
       Linear(d_model → ffn_dim) → GELU → Linear(ffn_dim → d_model)
   Applied independently and identically to every token position.

Dropout locations in this block
---------------------------------
- ``residual_dropout`` is applied to the output of *each* sub-layer before the
  residual addition.
- Attention-internal dropout (applied to attention probabilities) is handled
  inside ``CausalSelfAttention``; it is not duplicated here.

Tensor shapes
-------------
  input  : (B, T, d_model)
  output : (B, T, d_model)       — same shape, residual connections preserve it
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention
from .config import ModelConfig


class TransformerBlock(nn.Module):
    """
    One Pre-LayerNorm Transformer block (attention + FFN with residuals).

    Parameters
    ----------
    config : ModelConfig
        Validated model configuration loaded from YAML.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # ── Layer normalisation ───────────────────────────────────────────────
        # Applied BEFORE each sub-layer (Pre-LN convention).
        # elementwise_affine=True (default): learns scale (γ) and shift (β).
        self.ln1 = nn.LayerNorm(config.d_model)   # before attention
        self.ln2 = nn.LayerNorm(config.d_model)   # before FFN

        # ── Causal self-attention ─────────────────────────────────────────────
        self.attention = CausalSelfAttention(config)

        # ── Feed-forward network ──────────────────────────────────────────────
        # Position-wise: applies the same two-layer MLP to every token
        # independently.  Shape flow:
        #   (B, T, d_model) → (B, T, ffn_dim) → GELU → (B, T, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim, bias=True),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.d_model, bias=True),
        )

        # ── Residual dropout ──────────────────────────────────────────────────
        # Applied to sub-layer outputs *before* the residual addition.
        # Both sub-layers share the same dropout module (same rate, independent
        # application).
        self.resid_dropout = nn.Dropout(p=config.residual_dropout)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformer block forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, T, d_model)
            Input hidden states.

        Returns
        -------
        torch.Tensor, shape (B, T, d_model)
            Output hidden states after attention and FFN sub-layers.
        """
        # ── Attention sub-layer (Pre-LN) ──────────────────────────────────────
        #
        #  1. Apply LayerNorm to x.                  (B, T, d_model)
        #  2. Pass through causal self-attention.    (B, T, d_model)
        #  3. Apply residual dropout to the output.  (B, T, d_model)
        #  4. Add residual (x unchanged).            (B, T, d_model)
        #
        x = x + self.resid_dropout(
            self.attention(self.ln1(x))
        )   # (B, T, d_model)

        # ── FFN sub-layer (Pre-LN) ────────────────────────────────────────────
        #
        #  1. Apply LayerNorm to the updated x.      (B, T, d_model)
        #  2. Pass through the FFN:                  (B, T, d_model)
        #       Linear → GELU → Linear
        #  3. Apply residual dropout to the output.  (B, T, d_model)
        #  4. Add residual (x from after attention). (B, T, d_model)
        #
        x = x + self.resid_dropout(
            self.ffn(self.ln2(x))
        )   # (B, T, d_model)

        return x   # (B, T, d_model)
