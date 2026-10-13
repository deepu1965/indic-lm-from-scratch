"""
transformer.py
==============
DecoderOnlyTransformer — the full GPT-style decoder-only language model.

Architecture overview
---------------------

    Token IDs  (B, T)
         ↓
    Token Embedding   nn.Embedding(vocab_size, d_model)
         ↓
    Embedding Dropout
         ↓
    12 × TransformerBlock   (each: Pre-LN attention + FFN with residuals)
         ↓
    Final LayerNorm   nn.LayerNorm(d_model)
         ↓
    Output Projection   nn.Linear(d_model → vocab_size)
         ↓
    Logits  (B, T, vocab_size)

Weight tying
------------
The token embedding matrix ``(vocab_size × d_model)`` and the output
projection weight matrix ``(vocab_size × d_model)`` share the **same**
underlying parameter tensor.

After both are created we do:

    self.output_proj.weight = self.token_embedding.weight

This means there is a single parameter tensor for both, not two.  When the
optimiser updates this tensor (via the embedding gradient or the output gradient,
both flow back to the same tensor), both layers are updated simultaneously.

The output projection bias remains a *separate*, independent parameter.

How to verify tying is real (not a copy):
    assert model.output_proj.weight is model.token_embedding.weight

Input/output shapes
-------------------
  input_ids   : (B, T)  — integer token IDs, T ≤ context_length
  logits      : (B, T, vocab_size)

Kaggle / Nepali reuse
---------------------
From any Python environment:

    import sys
    sys.path.insert(0, "/path/to/individual-project-charan-1230")

    from nepali.model import DecoderOnlyTransformer
    from nepali.model.config import ModelConfig
    cfg = ModelConfig.from_yaml("nepali/configs/model_config.yaml")
    model = DecoderOnlyTransformer(cfg)

For Nepali, copy nepali/model/ → nepali/model/ and point at
nepali/configs/model_config.yaml.  No code changes required.
"""

import torch
import torch.nn as nn

from .config import ModelConfig
from .transformer_block import TransformerBlock


class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only (GPT-style) Transformer language model.

    Implemented from scratch using PyTorch primitive layers only:
    ``nn.Embedding``, ``nn.Linear``, ``nn.LayerNorm``, ``nn.Dropout``.

    Parameters
    ----------
    config : ModelConfig
        Validated model configuration loaded from YAML.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        # ── Token embedding ───────────────────────────────────────────────────
        # Maps integer token IDs → dense vectors.
        # Shape: (vocab_size, d_model) = (10000, 384).
        #
        # There is NO separate positional embedding table because we use RoPE,
        # which injects positional information inside each attention layer.
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # ── Embedding dropout ─────────────────────────────────────────────────
        # Applied to token embeddings before the first Transformer block.
        self.emb_dropout = nn.Dropout(p=config.embedding_dropout)

        # ── Transformer blocks ────────────────────────────────────────────────
        # Stack of num_layers (12) Pre-LN Transformer blocks.
        # nn.ModuleList ensures all parameters are registered correctly.
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        # ── Final layer normalisation ─────────────────────────────────────────
        # Applied after all Transformer blocks, before the output projection.
        # Essential for training stability with Pre-LN architecture.
        self.ln_f = nn.LayerNorm(config.d_model)

        # ── Output projection ─────────────────────────────────────────────────
        # Maps from d_model → vocab_size to produce raw logits.
        # Shape: weight (vocab_size, d_model), bias (vocab_size,).
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=True)

        # ── Weight tying ──────────────────────────────────────────────────────
        # Share the token embedding weight with the output projection weight.
        #
        # This does NOT copy values — it makes both attributes point to the
        # *exact same* Python/PyTorch parameter object (same storage, same grad).
        #
        # Effect on parameter count:
        #   Without tying: 2 × (vocab_size × d_model) = 2 × 3 840 000
        #   With tying   : 1 × (vocab_size × d_model) = 1 × 3 840 000
        #
        # The bias of output_proj remains an independent parameter.
        self.output_proj.weight = self.token_embedding.weight

        # ── Parameter initialisation ──────────────────────────────────────────
        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """
        Initialise model parameters following GPT-2 conventions.

        - Embeddings: Normal(mean=0, std=0.02)
        - Linear weights: Normal(mean=0, std=0.02)
        - Linear and LayerNorm biases: zeros
        - LayerNorm weights (γ): ones
        """
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass of the decoder-only Transformer.

        Parameters
        ----------
        input_ids : torch.Tensor, shape (B, T)
            Integer token ID sequences.  Values must be in [0, vocab_size).
            T must satisfy T ≤ context_length (512); a ValueError is raised
            if this constraint is violated.

        Returns
        -------
        logits : torch.Tensor, shape (B, T, vocab_size)
            Raw (unnormalised) log-probabilities over the vocabulary for every
            token position.  Apply softmax to obtain a probability distribution
            or pass directly to CrossEntropyLoss during training.

        Raises
        ------
        ValueError
            If ``T > config.context_length``.
        """
        B, T = input_ids.shape

        # ── Context length guard ──────────────────────────────────────────────
        if T > self.config.context_length:
            raise ValueError(
                f"Sequence length T={T} exceeds the model's context_length="
                f"{self.config.context_length}.  "
                f"Truncate the input to at most {self.config.context_length} "
                f"tokens before calling the model."
            )

        # ── Token embeddings ──────────────────────────────────────────────────
        # input_ids : (B, T)          — integer indices
        # x         : (B, T, d_model) — dense embedding vectors
        x = self.token_embedding(input_ids)   # (B, T, d_model)

        # ── Embedding dropout ─────────────────────────────────────────────────
        # (B, T, d_model)
        x = self.emb_dropout(x)   # (B, T, d_model)

        # ── Transformer blocks ────────────────────────────────────────────────
        # Each block: (B, T, d_model) → (B, T, d_model).
        for block in self.blocks:
            x = block(x)   # (B, T, d_model)

        # ── Final LayerNorm ───────────────────────────────────────────────────
        x = self.ln_f(x)   # (B, T, d_model)

        # ── Output projection → logits ────────────────────────────────────────
        # (B, T, d_model) → (B, T, vocab_size)
        #
        # Weight is tied to token_embedding.weight, so no separate weight
        # matrix is stored here.
        logits = self.output_proj(x)   # (B, T, vocab_size)

        return logits   # (B, T, vocab_size)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """
        Return the total number of **trainable** parameters.

        Because of weight tying, the embedding weight and output projection
        weight are the *same* tensor — ``id(param)`` is used to avoid counting
        it twice.  Python's built-in ``id()`` returns the memory address of the
        tensor object, so two attributes referring to the same storage produce
        the same id.

        Returns
        -------
        int
            Total number of scalar trainable parameters.
        """
        seen: set[int] = set()
        total = 0
        for param in self.parameters():
            param_id = id(param)
            if param_id not in seen:
                seen.add(param_id)
                total += param.numel()
        return total
