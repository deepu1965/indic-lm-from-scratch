"""
train.py
========
End-to-end training entry point for the Hindi language model.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time

import torch
import torch.nn.functional as F
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

from hindi.model.config import ModelConfig
from hindi.model.transformer import DecoderOnlyTransformer
from hindi.train.config import TrainingConfig
from hindi.train.dataset import StreamingPackedDataset, create_dataloader
from hindi.train.optimizer import create_optimizer
from hindi.train.scheduler import WarmupCosineScheduler
from hindi.train.evaluate import evaluate
from hindi.train.checkpointing import save_checkpoint, load_checkpoint
from hindi.train.utils import set_seed, TrainingLogger
from hindi.train.visualize import plot_loss_curve


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Hindi GPT-style Language Model.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint file to resume from.")
    parser.add_argument("--model-config", type=str, default="hindi/configs/model_config.yaml")
    parser.add_argument("--train-config", type=str, default="hindi/configs/training_config.yaml")
    parser.add_argument("--train-data", type=str, default="hindi/corpus/tokenised_splits/final_train_500M.jsonl")
    parser.add_argument("--val-data", type=str, default="hindi/corpus/tokenised_splits/val.jsonl")
    parser.add_argument("--out-dir", type=str, default="hindi/outputs")
    return parser.parse_args()


def setup_output_dir(out_dir: str, model_config_path: str, train_config_path: str, is_resume: bool) -> tuple[str, str, str]:
    """
    Sets up the output directory structure.
    If not resuming, safely creates directories and copies configs.
    """
    checkpoints_dir = os.path.join(out_dir, "checkpoints")
    logs_dir = os.path.join(out_dir, "logs")
    plots_dir = os.path.join(out_dir, "plots")

    if not is_resume:
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Copy configs for reproducibility
        shutil.copy(model_config_path, os.path.join(out_dir, "model_config.yaml"))
        shutil.copy(train_config_path, os.path.join(out_dir, "training_config.yaml"))
    else:
        if not os.path.exists(out_dir):
            raise FileNotFoundError(f"Cannot resume. Output directory {out_dir} does not exist.")
            
    return checkpoints_dir, logs_dir, plots_dir


def main() -> None:
    args = get_args()

    is_resume = args.resume is not None

    # Load configs
    model_cfg = ModelConfig.from_yaml(args.model_config)
    train_cfg = TrainingConfig.from_yaml(args.train_config)

    # Setup directories
    checkpoints_dir, logs_dir, plots_dir = setup_output_dir(
        args.out_dir, args.model_config, args.train_config, is_resume
    )
    
    log_csv = os.path.join(logs_dir, "training_log.csv")
    plot_png = os.path.join(plots_dir, "loss_curve.png")

    set_seed(train_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    model = DecoderOnlyTransformer(model_cfg).to(device)

    # Setup AMP
    use_amp = (device.type == "cuda" and train_cfg.mixed_precision in ("fp16", "bf16"))
    scaler = GradScaler() if use_amp else None

    # Build optimizer and scheduler
    optimizer = create_optimizer(model, train_cfg)
    scheduler = WarmupCosineScheduler(optimizer, train_cfg)

    # Build datasets
    train_ds = StreamingPackedDataset(args.train_data, context_length=train_cfg.context_length)
    val_ds = StreamingPackedDataset(args.val_data, context_length=train_cfg.context_length)

    # Resume state variables
    training_step = 0
    tokens_seen = 0
    best_val_loss = float("inf")

    if is_resume:
        print(f"Loading checkpoint from {args.resume}...")
        res_step, res_tokens, res_best_loss, _ = load_checkpoint(
            args.resume, model, optimizer, scheduler, train_ds, device, scaler=scaler
        )
        # Check compatibility? E.g. learning rate settings. We trust the loaded state for now.
        training_step = res_step
        tokens_seen = res_tokens
        best_val_loss = res_best_loss
        
        print(f"  Resuming from training step: {training_step}")
        print(f"  Tokens seen: {tokens_seen}")
        print(f"  Current learning rate: {scheduler.get_last_lr()[0]:.2e}")

    # Build dataloaders
    pin_mem = (device.type == "cuda")
    train_dl = create_dataloader(train_ds, batch_size=train_cfg.batch_size, pin_memory=pin_mem)
    val_dl = create_dataloader(val_ds, batch_size=train_cfg.batch_size, pin_memory=pin_mem)
    
    logger = TrainingLogger(logs_dir)

    print("\nStarting training loop...")
    model.train()
    
    # We use a micro-batch loop pulling from train_dl iterator manually
    # to perfectly control gradient accumulation steps and edge cases.
    train_iter = iter(train_dl)
    
    grad_acc_steps = train_cfg.gradient_accumulation_steps
    optimizer.zero_grad()
    
    # Keep track of accumulated loss for logging
    running_loss = 0.0
    accumulated_micro_loss = 0.0
    last_eval_is_best = False
    
    start_time = time.time()

    while training_step < train_cfg.total_steps:
        # ── Micro-batch Accumulation Loop ──
        micro_batches_collected = 0
        accumulated_micro_loss = 0.0
        
        for _ in range(grad_acc_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                # Dataset is exhausted.
                # In typical LM pretraining with packed streams, it shouldn't hit EOF prematurely,
                # but if it does, we break out. We intentionally DROP the incomplete accumulation group
                # to strictly respect the effective batch size configuration.
                print("\nReached End Of Dataset. Dropping final incomplete gradient accumulation group.")
                break
                
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            with autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16 if train_cfg.mixed_precision == "fp16" else torch.bfloat16):
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                scaled_loss = loss / grad_acc_steps
                
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
                
            accumulated_micro_loss += loss.item() / grad_acc_steps
            tokens_seen += (input_ids.numel())
            micro_batches_collected += 1

        if micro_batches_collected < grad_acc_steps:
            # We hit EOF and didn't get a full effective batch. We stop training.
            break

        # ── Optimizer Update ──
        if train_cfg.max_grad_norm > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            
        if scaler is not None:
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skip_step = (scale_before > scaler.get_scale())
        else:
            optimizer.step()
            skip_step = False
            
        optimizer.zero_grad()
        
        if not skip_step:
            scheduler.step()
            training_step += 1
        
        running_loss = 0.9 * running_loss + 0.1 * accumulated_micro_loss if training_step > 1 else accumulated_micro_loss
        
        # ── Logging ──
        if training_step % train_cfg.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            logger.log_train(training_step, tokens_seen, lr, accumulated_micro_loss)
            elapsed = time.time() - start_time
            print(f"Step {training_step}/{train_cfg.total_steps} | Loss: {accumulated_micro_loss:.4f} | LR: {lr:.2e} | Tokens: {tokens_seen} | Time: {elapsed:.1f}s")
            
        # ── Evaluation ──
        if training_step % train_cfg.eval_interval == 0:
            print(f"\n[Eval] Running validation at step {training_step}...")
            val_loss, val_ppl = evaluate(model, val_dl, train_cfg.eval_batches, device)
            logger.log_val(training_step, tokens_seen, val_loss, val_ppl)
            print(f"[Eval] Validation Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}\n")
            
            last_eval_is_best = (val_loss < best_val_loss)
            if last_eval_is_best:
                best_val_loss = val_loss
                
            # Plot
            try:
                plot_loss_curve(log_csv, plot_png)
            except Exception as e:
                print(f"Failed to plot loss curve: {e}")
                
            # If checkpointing happens at the exact same step, handle it below
            # We don't want to double-save best model if checkpoint_interval overlaps.
            # We will handle it by explicitly evaluating BEFORE checkpoint saving.

        # ── Checkpointing ──
        if training_step % train_cfg.checkpoint_interval == 0:
            # We only save as best_model if an evaluation JUST occurred and it was best
            is_best_for_ckpt = (training_step % train_cfg.eval_interval == 0) and last_eval_is_best
            
            save_checkpoint(
                checkpoints_dir, model, optimizer, scheduler,
                training_step, tokens_seen, best_val_loss, train_ds,
                train_cfg, is_best=is_best_for_ckpt, scaler=scaler
            )
            print(f"Checkpoint saved at step {training_step}")

    print("\nTraining loop complete! Finalizing state...")

    # ── Final Forced Evaluation ──
    final_is_best = False
    if training_step % train_cfg.eval_interval != 0:
        print(f"\n[Final Eval] Running validation at step {training_step}...")
        val_loss, val_ppl = evaluate(model, val_dl, train_cfg.eval_batches, device)
        logger.log_val(training_step, tokens_seen, val_loss, val_ppl)
        print(f"[Final Eval] Validation Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}\n")
        
        final_is_best = (val_loss < best_val_loss)
        if final_is_best:
            best_val_loss = val_loss
            
        try:
            plot_loss_curve(log_csv, plot_png)
        except Exception as e:
            print(f"Failed to plot loss curve: {e}")
    else:
        final_is_best = last_eval_is_best

    # ── Final Forced Checkpoint ──
    if training_step % train_cfg.checkpoint_interval != 0:
        save_checkpoint(
            checkpoints_dir, model, optimizer, scheduler,
            training_step, tokens_seen, best_val_loss, train_ds,
            train_cfg, is_best=final_is_best, scaler=scaler
        )
        print(f"Final checkpoint saved at step {training_step}")


if __name__ == "__main__":
    main()
