"""
config.py
=========
TrainingConfig: loads and validates the training hyperparameters from a YAML file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import yaml


@dataclass
class TrainingConfig:
    """
    All training hyperparameters for the decoder-only Transformer.

    Parameters are loaded from a YAML file via :meth:`from_yaml`.
    """

    seed: int

    # Batch configuration
    batch_size: int
    gradient_accumulation_steps: int
    context_length: int
    effective_tokens_per_update: int

    # Training duration
    total_steps: int

    # Optimizer
    learning_rate: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float

    # LR scheduler
    warmup_steps: int
    min_learning_rate: float

    # Stability
    max_grad_norm: float

    # Precision
    mixed_precision: str

    # Evaluation
    eval_interval: int
    eval_batches: int

    # Checkpointing
    checkpoint_interval: int
    keep_last_checkpoints: int

    # Logging
    log_interval: int

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """
        Load and validate a TrainingConfig from a YAML file.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(
                f"Training config file not found: {yaml_path!r}\n"
                "Run from the repository root."
            )

        with open(yaml_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        required_keys = [
            "seed", "batch_size", "gradient_accumulation_steps", "context_length",
            "effective_tokens_per_update", "total_steps", "learning_rate",
            "beta1", "beta2", "eps", "weight_decay", "warmup_steps",
            "min_learning_rate", "max_grad_norm", "mixed_precision",
            "eval_interval", "eval_batches", "checkpoint_interval",
            "keep_last_checkpoints", "log_interval"
        ]
        
        for key in required_keys:
            if key not in raw:
                raise KeyError(
                    f"Required key {key!r} missing from config file: {yaml_path!r}"
                )

        cfg = cls(
            seed=int(raw["seed"]),
            batch_size=int(raw["batch_size"]),
            gradient_accumulation_steps=int(raw["gradient_accumulation_steps"]),
            context_length=int(raw["context_length"]),
            effective_tokens_per_update=int(raw["effective_tokens_per_update"]),
            total_steps=int(raw["total_steps"]),
            learning_rate=float(raw["learning_rate"]),
            beta1=float(raw["beta1"]),
            beta2=float(raw["beta2"]),
            eps=float(raw["eps"]),
            weight_decay=float(raw["weight_decay"]),
            warmup_steps=int(raw["warmup_steps"]),
            min_learning_rate=float(raw["min_learning_rate"]),
            max_grad_norm=float(raw["max_grad_norm"]),
            mixed_precision=str(raw["mixed_precision"]),
            eval_interval=int(raw["eval_interval"]),
            eval_batches=int(raw["eval_batches"]),
            checkpoint_interval=int(raw["checkpoint_interval"]),
            keep_last_checkpoints=int(raw["keep_last_checkpoints"]),
            log_interval=int(raw["log_interval"]),
        )

        cfg._validate()
        return cfg

    def _validate(self) -> None:
        """
        Validate all hyperparameter values. Raises ``ValueError`` on any
        inconsistency.
        """
        errors: list[str] = []

        if self.batch_size <= 0:
            errors.append(f"batch_size must be > 0, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            errors.append(f"gradient_accumulation_steps must be > 0, got {self.gradient_accumulation_steps}")
        if self.context_length <= 0:
            errors.append(f"context_length must be > 0, got {self.context_length}")
        
        expected_tokens = self.batch_size * self.context_length * self.gradient_accumulation_steps
        if self.effective_tokens_per_update != expected_tokens:
            errors.append(
                f"effective_tokens_per_update ({self.effective_tokens_per_update}) "
                f"does not match batch_size * context_length * gradient_accumulation_steps ({expected_tokens})"
            )

        if self.total_steps <= self.warmup_steps:
            errors.append(
                f"total_steps ({self.total_steps}) must be > warmup_steps ({self.warmup_steps})"
            )

        if self.learning_rate <= self.min_learning_rate:
            errors.append(
                f"learning_rate ({self.learning_rate}) must be > min_learning_rate ({self.min_learning_rate})"
            )

        if errors:
            raise ValueError(
                "TrainingConfig validation failed:\n" + "\n".join(errors)
            )

    def __repr__(self) -> str:
        return (
            f"TrainingConfig(\n"
            f"  batch_size={self.batch_size},\n"
            f"  gradient_accumulation_steps={self.gradient_accumulation_steps},\n"
            f"  context_length={self.context_length},\n"
            f"  effective_tokens_per_update={self.effective_tokens_per_update},\n"
            f"  ...\n"
            f")"
        )
