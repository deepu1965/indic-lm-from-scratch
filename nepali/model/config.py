"""
config.py
=========
ModelConfig: loads and validates the model hyperparameters from a YAML file.

This is the ONLY file that needs to change when replicating the model for
a different language (e.g. Nepali).  Point it at the Nepali YAML and
everything else stays the same.

Usage
-----
    from nepali.model.config import ModelConfig
    cfg = ModelConfig.from_yaml("nepali/configs/model_config.yaml")
    print(cfg.head_dim)   # derived property — 384 // 6 = 64
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """
    All architecture hyperparameters for the decoder-only Transformer.

    Parameters are loaded from a YAML file via :meth:`from_yaml`.
    Do NOT instantiate this class directly with hardcoded values in model code;
    always load from YAML so that the same model files work for any language.

    Attributes
    ----------
    vocab_size : int
        Vocabulary size of the BPE tokenizer (e.g. 10000 for Nepali v10000).
    context_length : int
        Maximum sequence length.  RoPE frequencies and the causal mask are
        precomputed up to this length.
    d_model : int
        Embedding / hidden dimension.  Must be divisible by ``num_heads``.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of Transformer blocks stacked sequentially.
    ffn_dim : int
        Inner dimension of the feed-forward network (typically 4 × d_model).
    embedding_dropout : float
        Dropout probability applied to token embeddings before the first block.
    attention_dropout : float
        Dropout probability applied to attention probabilities after softmax.
    residual_dropout : float
        Dropout probability applied to each sub-layer output before the
        residual addition.
    """

    # Required fields (no defaults — must be present in YAML)
    vocab_size: int
    context_length: int
    d_model: int
    num_heads: int
    num_layers: int
    ffn_dim: int
    embedding_dropout: float
    attention_dropout: float
    residual_dropout: float

    # ── Derived property ─────────────────────────────────────────────────────

    @property
    def head_dim(self) -> int:
        """
        Dimension of each individual attention head.

        Derived from ``d_model // num_heads``; not stored in the YAML to
        prevent the config from holding a redundant, potentially inconsistent
        value.

        For the Nepali model: 384 // 6 = 64.
        """
        return self.d_model // self.num_heads

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        """
        Load and validate a ModelConfig from a YAML file.

        Parameters
        ----------
        yaml_path : str
            Path to the YAML configuration file (absolute or relative to the
            current working directory).

        Returns
        -------
        ModelConfig

        Raises
        ------
        FileNotFoundError
            If ``yaml_path`` does not exist.
        KeyError
            If a required key is missing from the YAML.
        ValueError
            If any hyperparameter value is invalid.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(
                f"Model config file not found: {yaml_path!r}\n"
                "Run from the repository root, e.g.:\n"
                "  python -m nepali.model.validate_model"
            )

        with open(yaml_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        # Extract required keys with helpful error messages
        required_keys = [
            "vocab_size", "context_length", "d_model", "num_heads",
            "num_layers", "ffn_dim",
            "embedding_dropout", "attention_dropout", "residual_dropout",
        ]
        for key in required_keys:
            if key not in raw:
                raise KeyError(
                    f"Required key {key!r} missing from config file: {yaml_path!r}"
                )

        cfg = cls(
            vocab_size=int(raw["vocab_size"]),
            context_length=int(raw["context_length"]),
            d_model=int(raw["d_model"]),
            num_heads=int(raw["num_heads"]),
            num_layers=int(raw["num_layers"]),
            ffn_dim=int(raw["ffn_dim"]),
            embedding_dropout=float(raw["embedding_dropout"]),
            attention_dropout=float(raw["attention_dropout"]),
            residual_dropout=float(raw["residual_dropout"]),
        )

        cfg._validate()
        return cfg

    # ── Validation ───────────────────────────────────────────────────────────

    def _validate(self) -> None:
        """
        Validate all hyperparameter values.  Raises ``ValueError`` on any
        inconsistency so that misconfigured models fail fast at load time.
        """
        errors: list[str] = []

        # Positive integers
        for name, val in [
            ("vocab_size", self.vocab_size),
            ("context_length", self.context_length),
            ("d_model", self.d_model),
            ("num_heads", self.num_heads),
            ("num_layers", self.num_layers),
            ("ffn_dim", self.ffn_dim),
        ]:
            if val <= 0:
                errors.append(f"  {name} must be > 0, got {val}")

        # Dropout in [0, 1)
        for name, val in [
            ("embedding_dropout", self.embedding_dropout),
            ("attention_dropout", self.attention_dropout),
            ("residual_dropout", self.residual_dropout),
        ]:
            if val < 0.0 or val >= 1.0:
                errors.append(
                    f"  {name} must be in [0, 1), got {val}"
                )

        # head_dim consistency
        if self.d_model % self.num_heads != 0:
            errors.append(
                f"  d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads}); "
                f"current head_dim would be {self.d_model / self.num_heads:.2f}"
            )

        if errors:
            raise ValueError(
                "ModelConfig validation failed:\n" + "\n".join(errors)
            )

    # ── Repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ModelConfig(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  context_length={self.context_length},\n"
            f"  d_model={self.d_model},\n"
            f"  num_heads={self.num_heads},\n"
            f"  head_dim={self.head_dim},  # derived\n"
            f"  num_layers={self.num_layers},\n"
            f"  ffn_dim={self.ffn_dim},\n"
            f"  embedding_dropout={self.embedding_dropout},\n"
            f"  attention_dropout={self.attention_dropout},\n"
            f"  residual_dropout={self.residual_dropout},\n"
            f")"
        )
