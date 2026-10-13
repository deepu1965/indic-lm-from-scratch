"""
attention.py
============
Manually implemented multi-head causal self-attention with RoPE.

Architecture choices
--------------------
- Three **separate** linear projections for Q, K, V (Option A from the spec).
  This is the clearest choice for understanding and explaining each projection
  independently during evaluation.

- The **causal mask** is precomputed once at init as an upper-triangular bool
  matrix of size (context_length × context_length) and registered as a buffer.
  During the forward pass, we simply slice [:T, :T] — no per-step recomputation.

- **RoPE** is applied to Q and K only (not V).  The ``RotaryEmbedding`` object
  is instantiated inside this module.

- Attention is computed **manually** (no ``F.scaled_dot_product_attention``):

      scores = (Q @ Kᵀ) / sqrt(head_dim)        — (B, H, T, T)
      scores = scores.masked_fill(mask, -inf)     — causal masking
      probs  = softmax(scores, dim=-1)            — (B, H, T, T)
      probs  = dropout(probs)
      out    = probs @ V                          — (B, H, T, head_dim)
      out    = concat(heads) then output proj     — (B, T, d_model)

Forward pass data flow (tensor shapes)
---------------------------------------
  input  X       : (B, T, d_model)
  Q = X @ W_q    : (B, T, d_model)  → reshaped → (B, H, T, head_dim)
  K = X @ W_k    : (B, T, d_model)  → reshaped → (B, H, T, head_dim)
  V = X @ W_v    : (B, T, d_model)  → reshaped → (B, H, T, head_dim)
  Q (RoPE)       : (B, H, T, head_dim)
  K (RoPE)       : (B, H, T, head_dim)
  scores         : (B, H, T, T)
  probs          : (B, H, T, T)
  context        : (B, H, T, head_dim)
  context concat : (B, T, d_model)
  output         : (B, T, d_model)  ← W_o projection
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention implemented from scratch.

    No ``nn.MultiheadAttention`` or ``F.scaled_dot_product_attention`` is used.

    Parameters
    ----------
    config : ModelConfig
        Validated model configuration loaded from YAML.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim      # d_model // num_heads = 64
        self.context_length = config.context_length

        # ── Q, K, V projections (Option A: three separate Linear layers) ──────
        # Each maps (B, T, d_model) → (B, T, d_model).
        # The output is later reshaped into (B, H, T, head_dim).
        #
        # Biases are included by default (bias=True).
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=True)  # query
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=True)  # key
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=True)  # value

        # ── Output projection ─────────────────────────────────────────────────
        # Merges all heads: (B, T, d_model) → (B, T, d_model).
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=True)

        # ── Dropout ───────────────────────────────────────────────────────────
        # Applied to attention probability distribution after softmax.
        self.attn_dropout = nn.Dropout(p=config.attention_dropout)

        # ── Rotary Positional Embedding ───────────────────────────────────────
        # Precomputes cos/sin tables of shape (context_length, head_dim // 2).
        # Stored as buffers — moves to the correct device with the model.
        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            context_length=self.context_length,
        )

        # ── Causal mask ───────────────────────────────────────────────────────
        # Upper-triangular bool matrix: True = future position (to be masked).
        # Shape: (context_length, context_length) = (512, 512).
        #
        # Example for T=4:
        #   [[F, T, T, T],
        #    [F, F, T, T],
        #    [F, F, F, T],
        #    [F, F, F, F]]
        #
        # diagonal=1 means the diagonal itself (current position) is NOT masked,
        # i.e. token t can see itself and all earlier tokens.
        causal_mask = torch.triu(
            torch.ones(
                self.context_length, self.context_length, dtype=torch.bool
            ),
            diagonal=1,
        )   # shape: (context_length, context_length) = (512, 512)

        # Register as a non-learnable, persistent buffer.
        # persistent=True: included in state_dict (saves/loads correctly).
        self.register_buffer("causal_mask", causal_mask, persistent=True)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-head causal self-attention forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, T, d_model)
            Input hidden states from the previous layer or embedding.

        Returns
        -------
        torch.Tensor, shape (B, T, d_model)
            Output hidden states after attention and output projection.
        """
        B, T, C = x.shape   # C == d_model for clarity below
        H = self.num_heads
        D = self.head_dim   # head_dim = C // H

        # ── Step 1: Q, K, V projections ───────────────────────────────────────
        # Each Linear operates on the last dimension (C = d_model).
        #
        # Q: (B, T, d_model)
        # K: (B, T, d_model)
        # V: (B, T, d_model)
        Q = self.W_q(x)   # (B, T, d_model)
        K = self.W_k(x)   # (B, T, d_model)
        V = self.W_v(x)   # (B, T, d_model)

        # ── Step 2: Reshape into heads ────────────────────────────────────────
        # (B, T, d_model) → (B, T, H, D) → (B, H, T, D)
        #
        # .view() reshapes without copying data (requires contiguous memory).
        # .transpose() swaps the T and H axes for attention computation.
        Q = Q.view(B, T, H, D).transpose(1, 2)   # (B, H, T, D)
        K = K.view(B, T, H, D).transpose(1, 2)   # (B, H, T, D)
        V = V.view(B, T, H, D).transpose(1, 2)   # (B, H, T, D)

        # ── Step 3: Apply RoPE to Q and K (NOT to V) ─────────────────────────
        # cos/sin: (T, D//2) = (T, 32)
        cos, sin = self.rope.get_cos_sin(seq_len=T, device=x.device)

        Q = RotaryEmbedding.apply_rope(Q, cos, sin)   # (B, H, T, D) — new tensor
        K = RotaryEmbedding.apply_rope(K, cos, sin)   # (B, H, T, D) — new tensor
        # V is left unchanged — positional encoding does not apply to values.

        # ── Step 4: Scaled dot-product attention scores ───────────────────────
        # Q @ Kᵀ: each head computes T×T pairwise dot products.
        #
        # K.transpose(-2, -1): (B, H, D, T)
        # scores:              (B, H, T, T)
        scale = math.sqrt(D)   # sqrt(head_dim) = sqrt(64) = 8.0
        scores = (Q @ K.transpose(-2, -1)) / scale   # (B, H, T, T)

        # ── Step 5: Apply causal mask ─────────────────────────────────────────
        # Slice the precomputed full-size mask to the current sequence length.
        # mask: (T, T)  — True at future positions (j > i)
        mask = self.causal_mask[:T, :T]   # (T, T)

        # Replace future positions with -inf so softmax drives them to ~0.
        # masked_fill broadcasts (T, T) against (B, H, T, T).
        scores = scores.masked_fill(mask, float("-inf"))   # (B, H, T, T)

        # ── Step 6: Softmax → attention probabilities ─────────────────────────
        # softmax over the last dimension (key axis).
        # Positions filled with -inf become 0 after softmax.
        attn_probs = F.softmax(scores, dim=-1)   # (B, H, T, T)

        # ── Step 7: Attention dropout ─────────────────────────────────────────
        # Dropout is applied to the probability matrix, randomly zeroing some
        # attention weights.  Only active during training (model.train()).
        attn_probs = self.attn_dropout(attn_probs)   # (B, H, T, T)

        # ── Step 8: Weighted sum of values ────────────────────────────────────
        # context[b, h, t, :] = sum over keys of attn_prob * v
        context = attn_probs @ V   # (B, H, T, D)

        # ── Step 9: Concatenate heads ─────────────────────────────────────────
        # (B, H, T, D) → (B, T, H, D) → (B, T, H*D = d_model)
        #
        # .contiguous() ensures memory is laid out correctly for .view().
        context = context.transpose(1, 2).contiguous()   # (B, T, H, D)
        context = context.view(B, T, H * D)              # (B, T, d_model)

        # ── Step 10: Output projection ────────────────────────────────────────
        # Projects the concatenated multi-head context back to d_model.
        out = self.W_o(context)   # (B, T, d_model)

        return out   # (B, T, d_model)
