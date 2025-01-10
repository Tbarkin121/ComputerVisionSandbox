# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:31:55 2025

@author: Plutonium
"""

import pandas as pd
import matplotlib.pyplot as plt

# Path to your local `results.csv` file
csv_path = "runs/classify/train5/results.csv"

# Load the CSV into a pandas DataFrame
try:
    results_df = pd.read_csv(csv_path)
    print("CSV loaded successfully.")

    # Extract relevant columns
    epochs = results_df['epoch']
    train_loss = results_df['train/loss']
    val_loss = results_df['val/loss']
    accuracy_top1 = results_df['metrics/accuracy_top1']
    accuracy_top5 = results_df['metrics/accuracy_top5']

    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Subplot 1: Train and Validation Loss
    axes[0].plot(epochs, train_loss, label="Train Loss", marker='o')
    axes[0].plot(epochs, val_loss, label="Validation Loss", marker='x')
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Subplot 2: Top-1 and Top-5 Accuracy
    axes[1].plot(epochs, accuracy_top1, label="Accuracy Top-1", marker='o')
    axes[1].plot(epochs, accuracy_top5, label="Accuracy Top-5", marker='x')
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"File not found: {csv_path}")
except Exception as e:
    print(f"An error occurred: {e}")