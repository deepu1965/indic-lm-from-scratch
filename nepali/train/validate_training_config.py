"""
validate_training_config.py
===========================
Self-contained validation for TrainingConfig, Optimizer grouping, and LR Scheduler.
"""

from __future__ import annotations

import os
import sys

_THIS_FILE = os.path.abspath(__file__)
_TRAIN_DIR = os.path.dirname(_THIS_FILE)
_HINDI_DIR = os.path.dirname(_TRAIN_DIR)
_REPO_ROOT = os.path.dirname(_HINDI_DIR)

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nepali.model.config import ModelConfig
from nepali.model.transformer import DecoderOnlyTransformer
from nepali.train.config import TrainingConfig
from nepali.train.optimizer import create_optimizer
from nepali.train.scheduler import WarmupCosineScheduler


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

def _pass(msg: str) -> None:
    print(f"  [PASS] {msg}")

def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def check_config() -> TrainingConfig:
    _section("1. Training Configuration Load & Validation")
    
    yaml_path = os.path.join(_HINDI_DIR, "configs", "training_config.yaml")
    cfg = TrainingConfig.from_yaml(yaml_path)
    
    # Check effective tokens logic manually to be extra sure
    expected_tokens = cfg.batch_size * cfg.context_length * cfg.gradient_accumulation_steps
    if cfg.effective_tokens_per_update != expected_tokens:
        _fail("Effective tokens calculation does not match!")
        
    _pass(f"Loaded config successfully from {yaml_path}")
    _pass(f"Effective tokens per update: {cfg.effective_tokens_per_update}")
    return cfg


def check_optimizer(train_cfg: TrainingConfig) -> None:
    _section("2. Optimizer Parameter Grouping & Deduplication")
    
    # We need a small mock model configuration to test instantiation quickly
    model_cfg = ModelConfig(
        vocab_size=1000, context_length=512, d_model=128, num_heads=4,
        num_layers=2, ffn_dim=512, embedding_dropout=0.0,
        attention_dropout=0.0, residual_dropout=0.0
    )
    model = DecoderOnlyTransformer(model_cfg)
    
    opt = create_optimizer(model, train_cfg)
    
    # Opt groups: 0 -> decay, 1 -> no decay
    decay_group = opt.param_groups[0]["params"]
    no_decay_group = opt.param_groups[1]["params"]
    
    # Total trainable params in the model by id
    all_param_ids = {id(p) for p in model.parameters() if p.requires_grad}
    
    decay_ids = {id(p) for p in decay_group}
    no_decay_ids = {id(p) for p in no_decay_group}
    
    # No param should be in both
    overlap = decay_ids.intersection(no_decay_ids)
    if overlap:
        _fail(f"Found {len(overlap)} parameters in both decay and no_decay groups!")
        
    # All params should be accounted for
    missing = all_param_ids - (decay_ids.union(no_decay_ids))
    if missing:
        _fail(f"Found {len(missing)} parameters not assigned to any group!")
        
    # Extra params?
    extra = (decay_ids.union(no_decay_ids)) - all_param_ids
    if extra:
        _fail("Optimizer has parameters that are not in the model!")
        
    # Specifically check the tied embedding/output weights
    emb_id = id(model.token_embedding.weight)
    out_id = id(model.output_proj.weight)
    if emb_id != out_id:
        _fail("Model weights are not tied!")
        
    if emb_id not in no_decay_ids:
        _fail("Tied token_embedding.weight was not found in the no-weight-decay group!")
    if emb_id in decay_ids:
        _fail("Tied token_embedding.weight was incorrectly put in the weight-decay group!")

    _pass(f"Found {len(all_param_ids)} unique parameter tensors.")
    _pass("Weight tying correctly deduplicated in optimizer groups.")
    _pass("All trainable parameters assigned to exactly one group.")
    
    return model, opt


def check_scheduler(model: DecoderOnlyTransformer, opt, train_cfg: TrainingConfig) -> None:
    _section("3. Learning Rate Scheduler Semantics")
    
    scheduler = WarmupCosineScheduler(opt, train_cfg)
    
    # Step 0: starts near 0
    lr_start = scheduler.get_last_lr()[0]
    if lr_start != 0.0:
        _fail(f"LR should start at 0.0, got {lr_start}")
    _pass(f"Step 0 LR: {lr_start}")
    
    # Warmup
    for _ in range(train_cfg.warmup_steps // 2):
        scheduler.step()
    lr_mid_warmup = scheduler.get_last_lr()[0]
    _pass(f"Mid-warmup LR: {lr_mid_warmup:.2e}")
    if not (0 < lr_mid_warmup < train_cfg.learning_rate):
        _fail("LR not increasing correctly during warmup.")
        
    for _ in range(train_cfg.warmup_steps - (train_cfg.warmup_steps // 2)):
        scheduler.step()
    
    # At warmup_steps
    # After exactly warmup_steps calls to step(), current_step is warmup_steps.
    lr_peak = scheduler.get_last_lr()[0]
    _pass(f"Peak LR at step {scheduler.current_step}: {lr_peak:.2e}")
    if abs(lr_peak - train_cfg.learning_rate) > 1e-9:
        _fail(f"Peak LR should be {train_cfg.learning_rate}, got {lr_peak}")
        
    # Mid-decay
    decay_steps = train_cfg.total_steps - train_cfg.warmup_steps
    for _ in range(decay_steps // 2):
        scheduler.step()
    lr_mid_decay = scheduler.get_last_lr()[0]
    _pass(f"Mid-decay LR at step {scheduler.current_step}: {lr_mid_decay:.2e}")
    if not (train_cfg.min_learning_rate < lr_mid_decay < train_cfg.learning_rate):
        _fail("LR not decaying correctly.")
        
    # State save and load check
    saved_state = scheduler.state_dict()
    new_scheduler = WarmupCosineScheduler(opt, train_cfg)
    new_scheduler.load_state_dict(saved_state)
    lr_resumed = new_scheduler.get_last_lr()[0]
    if abs(lr_resumed - lr_mid_decay) > 1e-9:
        _fail(f"State load failed to restore LR. Expected {lr_mid_decay}, got {lr_resumed}")
    _pass("Scheduler state_dict() / load_state_dict() exact restoration verified.")
    
    # Final step
    remaining_steps = train_cfg.total_steps - scheduler.current_step
    for _ in range(remaining_steps):
        scheduler.step()
        
    lr_final = scheduler.get_last_lr()[0]
    _pass(f"Final LR at step {scheduler.current_step}: {lr_final:.2e}")
    if abs(lr_final - train_cfg.min_learning_rate) > 1e-9:
        _fail(f"Final LR should be {train_cfg.min_learning_rate}, got {lr_final}")


def main():
    print("=" * 60)
    print("  Phase 3.2 — Training Config & Optimizer Validation")
    print("=" * 60)

    cfg = check_config()
    model, opt = check_optimizer(cfg)
    check_scheduler(model, opt, cfg)
    
    _section("Summary")
    print("  All validation checks PASSED successfully.")


if __name__ == "__main__":
    main()
