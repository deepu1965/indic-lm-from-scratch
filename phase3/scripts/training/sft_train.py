"""
sft_train.py
============
Supervised Fine-Tuning (SFT) using LoRA for Phase 3.2.
Supports resume, gradient accumulation, AMP, validation, and early stopping.

Usage:
    python -m phase3.scripts.training.sft_train --lang hi
    python -m phase3.scripts.training.sft_train --lang hi --smoke-test
    python -m phase3.scripts.training.sft_train --lang hi --resume phase3/outputs/checkpoints/hi/latest.pt
"""

import os
import sys
import time
import argparse
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# Ensure we can import from the root
sys.path.insert(0, os.getcwd())

from phase3.scripts.training.lora import apply_lora, create_lora_optimizer
from phase3.scripts.training.sft_dataset import SFTDataset, SFTCollator, masked_cross_entropy, _PAD_ID

from hindi.model.config import ModelConfig
from hindi.model.transformer import DecoderOnlyTransformer
from hindi.train.scheduler import WarmupCosineScheduler


@dataclass
class SFTConfig:
    # LoRA
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    
    # Training
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    max_epochs: int = 5
    
    # Early Stopping
    patience: int = 2
    min_delta: float = 0.001


def setup_logger(lang: str) -> logging.Logger:
    logger = logging.getLogger(f"sft_{lang}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Evaluates the model on the validation set using masked cross entropy."""
    model.eval()
    total_loss = 0.0
    total_steps = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            
            with autocast('cuda'):
                logits = model(input_ids)
                loss = masked_cross_entropy(logits, labels, loss_mask)
                
            total_loss += loss.item()
            total_steps += 1
            
    model.train()
    return total_loss / total_steps if total_steps > 0 else float("inf")


def main():
    parser = argparse.ArgumentParser(description="LoRA SFT Training")
    parser.add_argument("--lang", type=str, required=True, choices=["hi", "ne"])
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick forward/backward test without training loop")
    args = parser.parse_args()
    
    lang = args.lang
    logger = setup_logger(lang)
    logger.info(f"Starting SFT setup for {lang.upper()}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    sft_cfg = SFTConfig()
    
    # ── Paths ────────────────────────────────────────────────────────────────
    base_model_cfg_path = f"{'hindi' if lang == 'hi' else 'nepali'}/configs/model_config.yaml"
    base_ckpt_path = f"{'hindi' if lang == 'hi' else 'nepali'}/outputs/checkpoints/best_model.pt"
    jsonl_path = f"phase3/outputs/reasoning_{lang}.jsonl"
    tokenizer_path = f"{'hindi' if lang == 'hi' else 'nepali'}/tokenizer/models/bpe_v10000.json"
    
    out_dir = f"phase3/outputs/checkpoints/{lang}"
    os.makedirs(out_dir, exist_ok=True)
    
    # ── Model & LoRA Initialization ──────────────────────────────────────────
    model_cfg = ModelConfig.from_yaml(base_model_cfg_path)
    model = DecoderOnlyTransformer(model_cfg).to(device)
    
    # Always load the Phase 2 pretrained weights first
    logger.info(f"Loading pretrained Phase 2 checkpoint: {base_ckpt_path}")
    phase2_ckpt = torch.load(base_ckpt_path, map_location=device)
    model.load_state_dict(phase2_ckpt["model_state_dict"])
    
    # Apply LoRA structure
    logger.info(f"Applying LoRA (rank={sft_cfg.rank}, alpha={sft_cfg.alpha})")
    apply_lora(model, rank=sft_cfg.rank, alpha=sft_cfg.alpha, dropout=sft_cfg.dropout)
    model.to(device)
    
    # ── Optimizer & Dataloaders ──────────────────────────────────────────────
    optimizer = create_lora_optimizer(model, sft_cfg.learning_rate, sft_cfg.weight_decay)
    scaler = GradScaler('cuda')
    
    # Create DataLoaders
    train_ds = SFTDataset(jsonl_path, tokenizer_path, split="train", max_seq_len=model_cfg.context_length)
    val_ds = SFTDataset(jsonl_path, tokenizer_path, split="val", max_seq_len=model_cfg.context_length)
    collator = SFTCollator(pad_id=_PAD_ID)
    
    train_loader = DataLoader(train_ds, batch_size=sft_cfg.micro_batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=sft_cfg.micro_batch_size, shuffle=False, collate_fn=collator)
    
    total_steps = len(train_loader) // sft_cfg.gradient_accumulation_steps * sft_cfg.max_epochs
    warmup_steps = int(total_steps * sft_cfg.warmup_ratio)
    
    class DummyConfig:
        def __init__(self, warmup_steps, total_steps, learning_rate, min_learning_rate):
            self.warmup_steps = warmup_steps
            self.total_steps = total_steps
            self.learning_rate = learning_rate
            self.min_learning_rate = min_learning_rate

    sched_cfg = DummyConfig(
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        learning_rate=sft_cfg.learning_rate,
        min_learning_rate=sft_cfg.learning_rate * 0.1
    )
    
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        config=sched_cfg
    )
    
    # ── State Initialization / Resume ────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        # Load the complete state dict (includes frozen base + LoRA)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["training_step"]
        best_val_loss = ckpt["best_val_loss"]
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        logger.info(f"Resumed at epoch {start_epoch}, step {global_step}, best_val_loss={best_val_loss:.4f}, epochs_no_improve={epochs_no_improve}")
    else:
        # Advance scheduler by 1 step so that the first optimizer update uses a non-zero warmup learning rate
        scheduler.step()
        
    # ── Smoke Test ───────────────────────────────────────────────────────────
    if args.smoke_test:
        logger.info("Running SMOKE TEST")
        model.train()
        
        # Verify every trainable parameter belongs to a LoRA A/B matrix
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert "lora_A" in name or "lora_B" in name, f"Unexpected trainable param: {name}"

        # Snapshot all parameters
        frozen_snaps = {}
        trainable_snaps = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_snaps[name] = param.detach().clone()
            else:
                frozen_snaps[name] = param.detach().clone()
        
        # Run 1 step (since we pre-stepped the scheduler, LR is > 0)
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        
        optimizer.zero_grad()
        with autocast('cuda'):
            logits = model(input_ids)
            loss = masked_cross_entropy(logits, labels, loss_mask)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Verify changes
        base_changed = False
        for name, param in model.named_parameters():
            if not param.requires_grad:
                if not torch.equal(param, frozen_snaps[name]):
                    base_changed = True
                    break
        
        lora_max_diff = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad:
                diff = torch.abs(param - trainable_snaps[name]).max().item()
                if diff > lora_max_diff:
                    lora_max_diff = diff
        
        print("\n" + "="*50)
        print(lang.upper())
        print(f"LoRA/base output difference before training: ~0 (tested via lora.py)")
        print(f"Base parameters changed after step: {int(base_changed)}")
        print(f"LoRA parameters changed after step: {lora_max_diff}")
        print(f"Loss: {'finite' if torch.isfinite(loss) else 'infinite/NaN'} ({loss.item():.4f})")
        print(f"Backward: PASS")
        print(f"Optimizer step: PASS")
        print("="*50 + "\n")
        
        sys.exit(0)

    # ── Training Loop ────────────────────────────────────────────────────────
    logger.info("Starting full SFT training...")
    model.train()
    
    for epoch in range(start_epoch, sft_cfg.max_epochs):
        logger.info(f"Epoch {epoch+1}/{sft_cfg.max_epochs}")
        
        optimizer.zero_grad()
        accumulated_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            
            with autocast('cuda'):
                logits = model(input_ids)
                loss = masked_cross_entropy(logits, labels, loss_mask)
                loss = loss / sft_cfg.gradient_accumulation_steps
                
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()
            
            if (step + 1) % sft_cfg.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), sft_cfg.max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                
                if global_step % 100 == 0:
                    lr = optimizer.param_groups[0]['lr']
                    logger.info(f"  Step {global_step} | Loss: {accumulated_loss:.4f} | LR: {lr:.2e}")
                accumulated_loss = 0.0

        # End of epoch validation
        val_loss = evaluate(model, val_loader, device)
        lr = optimizer.param_groups[0]['lr']
        logger.info(f"End of Epoch {epoch+1} | Train Step {global_step} | Val Loss: {val_loss:.4f} | LR: {lr:.2e}")
        
        is_best = False
        if val_loss < best_val_loss - sft_cfg.min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            is_best = True
            logger.info(f"  New best validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            logger.info(f"  No improvement for {epochs_no_improve} epoch(s)")
            
        # Checkpointing
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch + 1,
            "training_step": global_step,
            "best_val_loss": best_val_loss,
            "epochs_no_improve": epochs_no_improve,
            "lang": lang,
            "lora_config": {
                "rank": sft_cfg.rank,
                "alpha": sft_cfg.alpha,
                "dropout": sft_cfg.dropout
            },
            "base_checkpoint_path": base_ckpt_path
        }
        
        # Save latest
        latest_path = os.path.join(out_dir, "latest.pt")
        # Temporary atomic save
        tmp_path = latest_path + ".tmp"
        torch.save(ckpt, tmp_path)
        os.rename(tmp_path, latest_path)
        
        if is_best:
            best_path = os.path.join(out_dir, "best_model.pt")
            tmp_path = best_path + ".tmp"
            torch.save(ckpt, tmp_path)
            os.rename(tmp_path, best_path)
            
        if epochs_no_improve >= sft_cfg.patience:
            logger.info(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
            break
            
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
