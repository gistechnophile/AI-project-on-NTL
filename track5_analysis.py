"""
Track 5 Results Analysis & Visualization
=========================================

Generate comparison figures for architecture deepening experiments:
  1. Architecture comparison bar chart (Test R, Test MAE, Params)
  2. Ensemble uncertainty vs prediction error
  3. Attention weight visualization (temporal attention maps)
  4. LaTeX table export

Usage:
    python track5_analysis.py \
        --results_json outputs/track5/architecture_comparison.json \
        --output_dir figures/track5
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from models.architectures import TemporalAttentionRegressor, TemporalPopulationRegressor, DeepEnsemble


def plot_architecture_comparison(results, output_dir):
    """Bar chart comparing architectures on Test R, Test MAE, and parameter count."""
    variants = list(results.keys())
    test_r = [results[v]["test_r"] for v in variants]
    test_mae = [results[v]["test_mae"] for v in variants]
    params = [results[v]["params"] / 1e6 for v in variants]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Test R
    ax = axes[0]
    bars = ax.bar(variants, test_r, color="steelblue", edgecolor="black")
    ax.set_ylabel("Test Pearson R", fontsize=11)
    ax.set_ylim(min(test_r) * 0.98, 1.0)
    ax.set_title("(a) Correlation", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, test_r):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=30)

    # Test MAE
    ax = axes[1]
    bars = ax.bar(variants, test_mae, color="coral", edgecolor="black")
    ax.set_ylabel("Test MAE (persons)", fontsize=11)
    ax.set_title("(b) Mean Absolute Error", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, test_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(test_mae)*0.02,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=30)

    # Params
    ax = axes[2]
    bars = ax.bar(variants, params, color="seagreen", edgecolor="black")
    ax.set_ylabel("Parameters (M)", fontsize=11)
    ax.set_title("(c) Model Size", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params)*0.02,
                f"{val:.1f}M", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Track 5: Architecture Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "architecture_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[Figure] Saved: {path}")
    plt.close()


def plot_ensemble_uncertainty(ensemble, dataloader, device, output_dir, n_batches=20):
    """Plot ensemble epistemic uncertainty vs prediction error."""
    ensemble.eval()
    all_mean, all_var, all_err = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            x = batch["image"].to(device)
            y = batch["target"].to(device)
            bu = batch.get("built_up_scalar")
            if bu is not None and getattr(ensemble.models[0], "use_built_up_scalar", False):
                bu = bu.to(device)

            mean, var = ensemble.predict_with_uncertainty(x, bu)
            err = (mean - y).abs()

            all_mean.extend(mean.cpu().numpy())
            all_var.extend(var.cpu().numpy())
            all_err.extend(err.cpu().numpy())

    all_mean = np.array(all_mean)
    all_var = np.array(all_var)
    all_err = np.array(all_err)
    std = np.sqrt(all_var)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Uncertainty vs Error
    ax = axes[0]
    ax.scatter(std, all_err, alpha=0.5, s=20, c="steelblue", edgecolor="none")
    ax.set_xlabel("Epistemic Uncertainty (std)", fontsize=11)
    ax.set_ylabel("Absolute Error (log scale)", fontsize=11)
    ax.set_title("(a) Uncertainty vs Error", fontsize=12, fontweight="bold")

    from scipy.stats import pearsonr
    if len(std) > 1 and std.std() > 0:
        r, p = pearsonr(std, all_err)
        ax.text(0.05, 0.95, f"R = {r:.3f}\np = {p:.2e}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Uncertainty distribution
    ax = axes[1]
    ax.hist(std, bins=30, color="coral", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Epistemic Uncertainty (std)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("(b) Uncertainty Distribution", fontsize=12, fontweight="bold")
    ax.axvline(std.mean(), color="darkred", linestyle="--", linewidth=2,
               label=f"Mean = {std.mean():.3f}")
    ax.legend()

    plt.suptitle("Deep Ensemble: Epistemic Uncertainty", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "ensemble_uncertainty.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[Figure] Saved: {path}")
    plt.close()


def visualize_attention_weights(model, sample_input, device, output_dir):
    """Extract and visualize temporal attention weights."""
    model.eval()
    model = model.to(device)
    B, T, C, H, W = sample_input.shape
    x = sample_input.to(device)
    x_flat = x.view(B * T, C, H, W)

    with torch.no_grad():
        feats = model.backbone(x_flat)
        feats = feats.view(B, T, model.feature_dim)
        feats = feats + model.pos_encoding[:, :T, :]

    attn_layer = model.attention.layers[0].self_attn
    attn_out, attn_weights = attn_layer(
        feats, feats, feats, need_weights=True, average_attn_weights=False
    )
    attn_weights = attn_weights.cpu().numpy()
    avg_attn = attn_weights.mean(axis=(0, 1))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(avg_attn, cmap="viridis", aspect="auto")
    ax.set_xlabel("Key month", fontsize=11)
    ax.set_ylabel("Query month", fontsize=11)
    ax.set_title("Temporal Self-Attention Weights (Averaged)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Attention weight")

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] * 6
    ticks = list(range(0, T, 6))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([months[i] for i in ticks], rotation=45, ha="right")
    ax.set_yticklabels([months[i] for i in ticks])

    plt.tight_layout()
    path = os.path.join(output_dir, "attention_weights.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[Figure] Saved: {path}")
    plt.close()


def print_latex_table(results):
    """Print a LaTeX-ready comparison table."""
    print("\n% LaTeX Table: Architecture Comparison")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Architecture & Params (M) & Test MAE & Test R & Best Val R \\\\")
    print("\\midrule")
    for name, res in results.items():
        print(f"{name} & {res['params']/1e6:.2f} & {res['test_mae']:.0f} & "
              f"{res['test_r']:.4f} & {res['best_val_r']:.4f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Architecture comparison on held-out test set.}")
    print("\\label{tab:architecture_comparison}")
    print("\\end{table}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_json", type=str, default=None)
    parser.add_argument("--ensemble_ckpt", type=str, default=None)
    parser.add_argument("--attention_ckpt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="figures/track5")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.results_json and os.path.exists(args.results_json):
        with open(args.results_json) as f:
            results = json.load(f)
        print(f"[Analysis] Loaded results for {len(results)} variants")
        if not args.no_plots:
            plot_architecture_comparison(results, args.output_dir)
        print_latex_table(results)
    else:
        print("[Warning] No results JSON found. Skipping comparison chart.")

    print(f"\n[Done] Figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
