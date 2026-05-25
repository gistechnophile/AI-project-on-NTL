"""
Uncertainty Quantification via Monte Carlo Dropout

Monte Carlo Dropout (Gal & Ghahramani, 2016) enables uncertainty estimation
in standard dropout-equipped networks without any architectural changes.

At inference time, dropout is kept ON. Running N forward passes produces
N predictions per sample. The mean is the prediction; the std is the
epistemic uncertainty.

This script:
  1. Loads a trained model
  2. Runs T forward passes with dropout enabled
  3. Computes mean, std, and confidence intervals per patch
  4. Maps uncertainty spatially
  5. Correlates uncertainty with prediction error (high uncertainty should
     correlate with high error — a sanity check)
"""

import os
import sys
import argparse
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.population_cnn import TemporalPopulationRegressor
from data_pipeline.dataset import TemporalPopulationRasterDataset as PopulationDataset


class MCDropoutPredictor:
    """
    Monte Carlo Dropout predictor.

    Usage:
        predictor = MCDropoutPredictor(model, device, n_passes=50)
        mean_pred, std_pred = predictor.predict(loader)
    """

    def __init__(self, model: nn.Module, device: torch.device, n_passes: int = 50):
        self.model = model
        self.device = device
        self.n_passes = n_passes

    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            mean_pred:  (N,) mean prediction across MC passes
            std_pred:   (N,) standard deviation (uncertainty)
            all_preds:  (N, n_passes) all individual predictions
        """
        all_batch_preds = []
        all_targets = []

        # Collect all data first
        all_x = []
        all_y = []
        all_bu = []

        for batch in dataloader:
            x = batch['image'].cpu().numpy()
            y = batch['target'].cpu().numpy()
            bu = batch.get('built_up_scalar')
            if bu is not None:
                all_bu.append(bu.cpu().numpy())
            all_x.append(x)
            all_y.append(y)

        all_x = np.concatenate(all_x, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        if len(all_bu) > 0:
            all_bu = np.concatenate(all_bu, axis=0)
        else:
            all_bu = None

        N = all_x.shape[0]
        all_preds = np.zeros((N, self.n_passes))

        print(f"[MC Dropout] Running {self.n_passes} stochastic forward passes...")

        self.model.train()  # KEEP DROPOUT ON!

        with torch.no_grad():
            for t in range(self.n_passes):
                if (t + 1) % 10 == 0:
                    print(f"  Pass {t+1}/{self.n_passes}")

                batch_preds = []
                batch_size = dataloader.batch_size or 8

                for i in range(0, N, batch_size):
                    x_batch = torch.from_numpy(all_x[i:i+batch_size]).float().to(self.device)

                    if all_bu is not None and getattr(self.model, 'use_built_up_scalar', False):
                        bu_batch = torch.from_numpy(all_bu[i:i+batch_size]).float().to(self.device)
                        pred = self.model(x_batch, bu_batch)
                    else:
                        pred = self.model(x_batch)

                    batch_preds.append(pred.cpu().numpy())

                all_preds[:, t] = np.concatenate(batch_preds)

        mean_pred = all_preds.mean(axis=1)
        std_pred = all_preds.std(axis=1)

        self.model.eval()
        return mean_pred, std_pred, all_preds, all_y


def plot_uncertainty_analysis(mean_pred: np.ndarray, std_pred: np.ndarray,
                              targets: np.ndarray, save_dir: str):
    """Generate publication-ready uncertainty analysis figures."""
    os.makedirs(save_dir, exist_ok=True)

    # Convert to original scale for interpretability
    mean_orig = np.expm1(mean_pred)
    std_orig = std_pred * mean_orig  # approximate: std(expm1(x)) ≈ std(x) * exp(x)
    target_orig = np.expm1(targets)
    error = np.abs(mean_orig - target_orig)

    # 1. Uncertainty vs Error scatter
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: Uncertainty vs Absolute Error
    ax = axes[0]
    ax.scatter(std_orig, error, alpha=0.4, s=20, c='#3498db', edgecolors='none')
    ax.set_xlabel('Predictive Uncertainty (std, people/pixel)', fontsize=11)
    ax.set_ylabel('Absolute Error (people/pixel)', fontsize=11)
    ax.set_title('Uncertainty vs Error', fontsize=12, fontweight='bold')

    # Correlation
    from scipy.stats import pearsonr
    r_ue, _ = pearsonr(std_orig, error)
    ax.text(0.05, 0.95, f'Pearson R = {r_ue:.3f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.grid(True, alpha=0.3)

    # Panel B: Prediction vs Target with error bars
    ax = axes[1]
    sample_idx = np.random.choice(len(mean_pred), size=min(200, len(mean_pred)), replace=False)
    ax.errorbar(target_orig[sample_idx], mean_orig[sample_idx],
                yerr=std_orig[sample_idx], fmt='o', alpha=0.5, c='#e74c3c',
                ecolor='#e74c3c', capsize=2, markersize=4)
    ax.plot([0, target_orig.max()], [0, target_orig.max()], 'k--', lw=1.5, label='1:1')
    ax.set_xlabel('Ground Truth (people/pixel)', fontsize=11)
    ax.set_ylabel('Predicted (people/pixel)', fontsize=11)
    ax.set_title('Predictions with Uncertainty Bars', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: Uncertainty distribution by density class
    ax = axes[2]
    rural_mask = target_orig < 20
    peri_mask = (target_orig >= 20) & (target_orig <= 100)
    urban_mask = target_orig > 100

    data_to_plot = [std_orig[rural_mask], std_orig[peri_mask], std_orig[urban_mask]]
    labels = ['Rural\n(<20)', 'Peri-urban\n(20-100)', 'Urban\n(>100)']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Predictive Uncertainty (std)', fontsize=11)
    ax.set_title('Uncertainty by Density Class', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Monte Carlo Dropout Uncertainty Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'uncertainty_analysis.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {os.path.join(save_dir, 'uncertainty_analysis.png')}")

    # 2. Calibration plot: uncertainty should correlate with error
    fig, ax = plt.subplots(figsize=(7, 5))

    # Bin by uncertainty and compute mean error per bin
    n_bins = 10
    bin_edges = np.percentile(std_orig, np.linspace(0, 100, n_bins + 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_errors = []
    std_errors = []

    for i in range(n_bins):
        mask = (std_orig >= bin_edges[i]) & (std_orig < bin_edges[i+1])
        if i == n_bins - 1:
            mask = (std_orig >= bin_edges[i]) & (std_orig <= bin_edges[i+1])
        if mask.sum() > 0:
            mean_errors.append(error[mask].mean())
            std_errors.append(error[mask].std())
        else:
            mean_errors.append(0)
            std_errors.append(0)

    ax.bar(bin_centers, mean_errors, width=np.diff(bin_edges)*0.8,
           color='#9b59b6', edgecolor='black', alpha=0.7, yerr=std_errors,
           capsize=4, error_kw={'linewidth': 1.5})
    ax.set_xlabel('Uncertainty Bin (std, people/pixel)', fontsize=11)
    ax.set_ylabel('Mean Absolute Error', fontsize=11)
    ax.set_title('Calibration: Higher Uncertainty → Higher Error?', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Trend line
    z = np.polyfit(bin_centers, mean_errors, 1)
    p = np.poly1d(z)
    ax.plot(bin_centers, p(bin_centers), 'r--', lw=2, label=f'Trend (slope={z[0]:.2f})')
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'uncertainty_calibration.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {os.path.join(save_dir, 'uncertainty_calibration.png')}")


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--ntl_dir', required=True)
    parser.add_argument('--pop', required=True)
    parser.add_argument('--border_mask', required=True)
    parser.add_argument('--built_up_path', default=None)
    parser.add_argument('--built_up_volume_path', default=None)
    parser.add_argument('--built_up_as_channel', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--n_passes', type=int, default=50)
    parser.add_argument('--output_dir', default='outputs/explainability')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    channel_names = ['NTL', 'POP_proxy']
    if args.built_up_path and args.built_up_as_channel:
        channel_names.append('BU_surface')
    if args.built_up_volume_path and args.built_up_as_channel:
        channel_names.append('BU_volume')

    # Load model
    model = TemporalPopulationRegressor(
        pretrained=args.pretrained,
        in_channels=len(channel_names),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))

    # Load dataset
    dataset = PopulationDataset(
        ntl_dir=args.ntl_dir,
        pop_path=args.pop,
        border_mask_path=args.border_mask,
        built_up_path=args.built_up_path,
        built_up_volume_path=args.built_up_volume_path,
        built_up_as_channel=args.built_up_as_channel,
    )

    from sklearn.model_selection import train_test_split
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_ds = Subset(dataset, val_idx)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Run MC Dropout
    print(f"\n[MC Dropout] Starting analysis with {args.n_passes} passes...")
    predictor = MCDropoutPredictor(model, device, n_passes=args.n_passes)
    mean_pred, std_pred, all_preds, targets = predictor.predict(val_loader)

    print(f"[MC Dropout] Results:")
    print(f"  Mean uncertainty (log scale): {std_pred.mean():.4f}")
    print(f"  Max uncertainty (log scale): {std_pred.max():.4f}")
    print(f"  Uncertainty vs Error correlation: {np.corrcoef(std_pred, np.abs(mean_pred - targets))[0,1]:.3f}")

    # Generate figures
    plot_uncertainty_analysis(mean_pred, std_pred, targets, args.output_dir)

    # Save data for later use
    np.savez(os.path.join(args.output_dir, 'mc_dropout_results.npz'),
             mean_pred=mean_pred, std_pred=std_pred, targets=targets,
             all_preds=all_preds)
    print(f"\n[MC Dropout] Data saved to {args.output_dir}/mc_dropout_results.npz")
