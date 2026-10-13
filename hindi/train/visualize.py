"""
visualize.py
============
Standalone module for plotting training and validation loss curves from CSV logs.
"""

from __future__ import annotations

import csv
import matplotlib.pyplot as plt

def plot_loss_curve(log_csv_path: str, output_image_path: str) -> None:
    """
    Parse the training_log.csv and generate a loss curve plot.
    
    Parameters
    ----------
    log_csv_path : str
        Path to the CSV log file.
    output_image_path : str
        Path to save the generated PNG plot.
    """
    train_steps = []
    train_losses = []
    
    val_steps = []
    val_losses = []
    
    with open(log_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            event = row["event_type"]
            
            if event == "train":
                train_steps.append(step)
                train_losses.append(float(row["train_loss"]))
            elif event == "val":
                val_steps.append(step)
                val_losses.append(float(row["val_loss"]))
                
    plt.figure(figsize=(10, 6))
    
    if train_steps:
        plt.plot(train_steps, train_losses, label="Training Loss", color="blue", alpha=0.6)
        
    if val_steps:
        # Plot validation with larger, distinct markers
        plt.plot(val_steps, val_losses, label="Validation Loss", color="red", marker="o", linestyle="dashed")
        
    plt.title("Training and Validation Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()
