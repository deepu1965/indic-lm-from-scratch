"""
scheduler.py
============
Implements a Warmup Cosine Decay learning rate scheduler that operates
on a per-optimizer-update basis.
"""

from __future__ import annotations

import math
from typing import Dict, Any

from torch.optim import Optimizer
from .config import TrainingConfig


class WarmupCosineScheduler:
    """
    Custom LR scheduler: Linear warmup followed by cosine decay.
    Advances exactly once per optimizer update (not per micro-batch).
    """
    def __init__(self, optimizer: Optimizer, config: TrainingConfig) -> None:
        self.optimizer = optimizer
        self.warmup_steps = config.warmup_steps
        self.total_steps = config.total_steps
        self.max_lr = config.learning_rate
        self.min_lr = config.min_learning_rate
        
        # internal state: counts the number of optimizer steps taken
        self.current_step = 0
        
        self._update_lr()

    def step(self) -> None:
        """
        Advance the scheduler by one step and update the optimizer's LR.
        Should be called immediately after optimizer.step().
        """
        self.current_step += 1
        self._update_lr()

    def _update_lr(self) -> None:
        """
        Compute and apply the learning rate for the current step.
        """
        if self.current_step < self.warmup_steps:
            # Linear warmup: starts near 0, reaches max_lr at warmup_steps
            # To avoid 0 LR at step 0, we can linearly interpolate from a very small value,
            # or just start from 0. Let's do exact linear interpolation: 
            # LR at step 0 = 0. LR at step warmup_steps = max_lr.
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        elif self.current_step > self.total_steps:
            # Beyond total steps, stick to min_lr
            lr = self.min_lr
        else:
            # Cosine decay: from max_lr down to min_lr
            decay_ratio = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.min_lr + coeff * (self.max_lr - self.min_lr)
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self) -> list[float]:
        """Return the current learning rate for each param group."""
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        """Save the exact step state for resumption."""
        return {
            "current_step": self.current_step
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the step state and recalculate the LR."""
        self.current_step = state["current_step"]
        self._update_lr()
