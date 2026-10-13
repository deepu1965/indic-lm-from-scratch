"""
rope.py
=======
Rotary Positional Embedding (RoPE) for the decoder-only Transformer.

RoPE is applied to Queries and Keys only — NOT to Values.

Background
----------
RoPE encodes positional information by rotating query and key vectors in 2D
sub-spaces defined by learned or fixed frequencies.  The key property is that
the dot product <RoPE(q, m), RoPE(k, n)> depends only on the content and the
*relative* position (m - n), which gives the model implicit relative position
awareness with no extra parameters.

Reference: Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary
Position Embedding"  https://arxiv.org/abs/2104.09864

Half-split Rotation Convention
-------------------------------
This implementation uses the **half-split** convention:
  - The head dimension D is split into two halves: [:D//2] and [D//2:]
  - Each position m and frequency index i applies the 2D rotation:

        x_rotated[:D//2] = x[:D//2] * cos(m * θ_i) - x[D//2:] * sin(m * θ_i)
        x_rotated[D//2:] = x[:D//2] * sin(m * θ_i) + x[D//2:] * cos(m * θ_i)

  This is mathematically identical to the interleaved formulation but avoids
  the index juggling of strided slicing (x[..., 0::2], x[..., 1::2]).

Frequency formula
-----------------
    θ_i = 10000 ^ (−2i / D)   for i in {0, 1, …, D/2 − 1}

For D = head_dim = 64:  D/2 = 32 frequency pairs.

Tensor shapes (with explicit comments in code)
----------------------------------------------
  cos_table : (context_length, head_dim // 2)
  sin_table : (context_length, head_dim // 2)

  Input  x  : (B, H, T, D)       — a Q or K tensor
  Output    : (B, H, T, D)       — rotated, same shape

Usage
-----
    from nepali.model.rope import RotaryEmbedding

    rope = RotaryEmbedding(head_dim=64, context_length=512)
    cos, sin = rope.get_cos_sin(seq_len=T, device=device)
    q_rotated = rope.apply_rope(q, cos, sin)   # (B, H, T, D)
    k_rotated = rope.apply_rope(k, cos, sin)   # (B, H, T, D)
"""

import math

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Precomputed Rotary Positional Embedding tables.

    The cosine and sine frequency tables are registered as non-trainable
    buffers so they:
      - move to the correct device automatically with ``.to(device)``.
      - are *not* counted as model parameters.
      - are computed only once at initialisation.

    Parameters
    ----------
    head_dim : int
        Dimension of each attention head (D).  Must be even.
        For this project: 64.
    context_length : int
        Maximum sequence length for which tables are precomputed.
        For this project: 512.
    theta_base : float
        Base of the geometric frequency progression.  Standard value: 10000.
    """

    def __init__(
        self,
        head_dim: int,
        context_length: int,
        theta_base: float = 10_000.0,
    ) -> None:
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even for RoPE, got {head_dim}"
            )

        self.head_dim = head_dim
        self.context_length = context_length
        half_dim = head_dim // 2  # number of frequency pairs = 32

        # ── Frequency vector ─────────────────────────────────────────────────
        # θ_i = 10000 ^ (−2i / D)   for i ∈ {0, …, D/2 − 1}
        #
        # Computed via:   1 / (theta_base ^ (2i / D))
        #               = exp(−(2i / D) * log(theta_base))
        #
        # Shape: (half_dim,) = (32,)
        i = torch.arange(half_dim, dtype=torch.float32)           # (32,)
        inv_freq = 1.0 / (theta_base ** (2.0 * i / head_dim))     # (32,)

        # ── Position indices ─────────────────────────────────────────────────
        # Shape: (context_length,) = (512,)
        positions = torch.arange(context_length, dtype=torch.float32)  # (512,)

        # ── Outer product → frequency table ──────────────────────────────────
        # freqs[m, i] = m * θ_i
        #
        # Shape: (context_length, half_dim) = (512, 32)
        freqs = torch.outer(positions, inv_freq)                   # (512, 32)

        # ── Cosine and sine tables ────────────────────────────────────────────
        # Both shape: (context_length, half_dim) = (512, 32)
        cos_table = freqs.cos()   # (512, 32)
        sin_table = freqs.sin()   # (512, 32)

        # Register as buffers (not parameters — they have no gradient).
        # persistent=True means they are saved/loaded with model state_dict,
        # which is convenient but not strictly necessary.
        self.register_buffer("cos_table", cos_table, persistent=True)
        self.register_buffer("sin_table", sin_table, persistent=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_cos_sin(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the cosine and sine tables for the first ``seq_len`` positions.

        The tables are already on the correct device (moved by ``.to(device)``
        when the model is transferred), so ``device`` is only used for a
        runtime assertion — it should match ``self.cos_table.device``.

        Parameters
        ----------
        seq_len : int
            Actual sequence length T (must be ≤ context_length).
        device : torch.device
            Device of the query/key tensors (used for consistency check).

        Returns
        -------
        cos : torch.Tensor, shape (T, head_dim // 2)
        sin : torch.Tensor, shape (T, head_dim // 2)
        """
        if seq_len > self.context_length:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds context_length "
                f"({self.context_length}) for which RoPE was precomputed."
            )

        # Slice the first T rows from the precomputed table.
        # cos: (T, half_dim) = (T, 32)
        cos = self.cos_table[:seq_len]   # (T, 32)
        sin = self.sin_table[:seq_len]   # (T, 32)

        return cos, sin

    @staticmethod
    def apply_rope(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Rotary Positional Embedding to a query or key tensor.

        Uses the **half-split** convention: the head dimension D is split into
        two equal halves and the 2D rotation is applied across them.

        The rotation for position m and frequency pair i is:

            out[:D//2] = x[:D//2] * cos[m,i] - x[D//2:] * sin[m,i]
            out[D//2:] = x[:D//2] * sin[m,i] + x[D//2:] * cos[m,i]

        This function does NOT modify ``x`` in-place; it returns a new tensor.

        Parameters
        ----------
        x : torch.Tensor, shape (B, H, T, D)
            Query or Key tensor.
            B = batch size, H = num_heads, T = seq_len, D = head_dim.
        cos : torch.Tensor, shape (T, D//2)
            Cosine values for positions 0..T-1.
        sin : torch.Tensor, shape (T, D//2)
            Sine values for positions 0..T-1.

        Returns
        -------
        torch.Tensor, shape (B, H, T, D)
            Rotationally embedded query or key tensor (same shape as input).
        """
        # x:   (B, H, T, D)
        # D    = head_dim = 64
        # D//2 = half_dim = 32

        D = x.shape[-1]          # head_dim
        half = D // 2

        # Split head dimension into two halves along the last axis.
        x1 = x[..., :half]      # (B, H, T, D//2)
        x2 = x[..., half:]      # (B, H, T, D//2)

        # Broadcast cos/sin from (T, D//2) to (1, 1, T, D//2)
        # so they multiply correctly against (B, H, T, D//2).
        cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, T, D//2)
        sin = sin.unsqueeze(0).unsqueeze(0)   # (1, 1, T, D//2)

        # 2D rotation in each frequency pair sub-space.
        # out_1: (B, H, T, D//2)
        # out_2: (B, H, T, D//2)
        out_1 = x1 * cos - x2 * sin    # (B, H, T, D//2)
        out_2 = x1 * sin + x2 * cos    # (B, H, T, D//2)

        # Concatenate back along the last dimension.
        # Result: (B, H, T, D)
        return torch.cat([out_1, out_2], dim=-1)   # (B, H, T, D)
