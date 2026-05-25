"""
Feature Importance Analysis for TemporalPopulationRegressor

Methods:
  1. Channel-wise Permutation Importance
     Shuffle each input channel and measure performance drop.
     This is the most reliable method for CNNs.

  2. SHAP on Patch-Level Scalar Features
     Reduce each patch to scalar statistics (mean NTL, mean surface, etc.)
     and apply KernelSHAP for human-interpretable explanations.

  3. Density-Stratified Analysis
     Repeat (1) and (2) separately for rural / peri-urban / urban patches.
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.population_cnn import TemporalPopulationRegressor
from data_pipeline.dataset import TemporalPopulationRasterDataset as PopulationDataset


def evaluate(model, dataloader, device):
    """Simple evaluation function."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['image'].to(device)
            y = batch['target'].to(device)
            bu = batch.get('built_up_scalar')
            if bu is not None and getattr(model, 'use_built_up_scalar', False):
                bu = bu.to(device)
                pred = model(x, bu)
            else:
                pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    from scipy.stats import pearsonr
    r, _ = pearsonr(preds, targets)
    mae = np.mean(np.abs(np.expm1(preds) - np.expm1(targets)))
    mse = np.mean((preds - targets) ** 2)
    return mse, mae, r


class ChannelPermutationImportance:
    """
    Compute permutation importance for each input channel.

    For a 4-channel model (NTL, POP, surface, volume):
    - Shuffle channel 0 (NTL) across the validation set
    - Measure increase in validation loss / decrease in R
    - Larger drop = more important channel
    """

    def __init__(self, model: nn.Module, device: torch.device, channel_names: List[str]):
        self.model = model
        self.device = device
        self.channel_names = channel_names
        self.n_channels = len(channel_names)

    def compute(self, dataloader: DataLoader, baseline_metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Returns dict: {channel_name: {'loss_delta': X, 'r_delta': Y, 'mae_delta': Z}}
        """
        results = {}
        baseline_loss = baseline_metrics['loss']
        baseline_r = baseline_metrics['r']
        baseline_mae = baseline_metrics['mae']

        for ch_idx, ch_name in enumerate(self.channel_names):
            print(f"  [Permutation] Shuffling channel {ch_idx}: {ch_name}...")
            metrics = self._evaluate_with_shuffled_channel(dataloader, ch_idx)

            results[ch_name] = {
                'loss_delta': metrics['loss'] - baseline_loss,
                'r_delta': baseline_r - metrics['r'],  # positive = worse
                'mae_delta': metrics['mae'] - baseline_mae,
                'loss': metrics['loss'],
                'r': metrics['r'],
                'mae': metrics['mae'],
            }

        return results

    def _evaluate_with_shuffled_channel(self, dataloader: DataLoader, ch_idx: int) -> Dict[str, float]:
        """Evaluate model with one channel shuffled across the dataset."""
        all_preds = []

        # First pass: collect all patches and their targets
        all_patches = []
        all_tgts = []
        all_bu = []

        for batch in dataloader:
            x = batch['image'].cpu().numpy()
            y = batch['target'].cpu().numpy()
            bu = batch.get('built_up_scalar')
            if bu is not None:
                all_bu.append(bu.cpu().numpy())
            all_patches.append(x)
            all_tgts.append(y)

        all_patches = np.concatenate(all_patches, axis=0)  # (N, T, C, H, W)
        all_tgts = np.concatenate(all_tgts, axis=0)
        if all_bu:
            all_bu = np.concatenate(all_bu, axis=0)
        else:
            all_bu = None

        # Shuffle the specified channel across all samples
        N = all_patches.shape[0]
        perm = np.random.permutation(N)
        all_patches_shuffled = all_patches.copy()
        all_patches_shuffled[:, :, ch_idx, :, :] = all_patches[perm, :, ch_idx, :, :]

        # Second pass: evaluate with shuffled data
        self.model.eval()
        batch_size = dataloader.batch_size or 8

        with torch.no_grad():
            for i in range(0, N, batch_size):
                x_batch = torch.from_numpy(all_patches_shuffled[i:i+batch_size]).float().to(self.device)

                if all_bu is not None and len(all_bu) > 0 and getattr(self.model, 'use_built_up_scalar', False):
                    bu_batch = torch.from_numpy(all_bu[i:i+batch_size]).float().to(self.device)
                    pred = self.model(x_batch, bu_batch)
                else:
                    pred = self.model(x_batch)

                all_preds.append(pred.cpu().numpy())

        preds = np.concatenate(all_preds)
        targets = all_tgts

        # Compute metrics
        from scipy.stats import pearsonr
        r, _ = pearsonr(preds, targets)
        mae = np.mean(np.abs(np.expm1(preds) - np.expm1(targets)))
        mse = np.mean((preds - targets) ** 2)

        return {'loss': mse, 'r': r, 'mae': mae}


class DensityStratifiedImportance:
    """
    Compute feature importance separately for rural / peri-urban / urban patches.
    This reveals which channels matter at different population densities.
    """

    def __init__(self, model: nn.Module, device: torch.device, channel_names: List[str]):
        self.perm_imp = ChannelPermutationImportance(model, device, channel_names)

    def compute(self, dataset, patch_indices: List[int], baseline_metrics_fn) -> Dict[str, Dict]:
        """
        Args:
            dataset: Full PopulationDataset
            patch_indices: List of patch indices to evaluate
            baseline_metrics_fn: Function that takes a dataloader and returns metrics dict
        """
        # Get population values for each patch to classify density
        pop_values = []
        for idx in patch_indices:
            if hasattr(dataset, 'indices'):
                y, x = dataset.indices[idx]
                patch_pop = dataset.pop[y:y+dataset.patch_size, x:x+dataset.patch_size].sum()
            else:
                # For Subset, get underlying dataset
                orig_idx = dataset.indices[idx] if hasattr(dataset, 'indices') else idx
                patch_pop = self._get_patch_pop(dataset.dataset if hasattr(dataset, 'dataset') else dataset, orig_idx)
            pop_values.append(patch_pop)

        pop_values = np.array(pop_values)

        # Classify
        rural_idx = [patch_indices[i] for i in range(len(patch_indices)) if pop_values[i] < 20]
        peri_idx = [patch_indices[i] for i in range(len(patch_indices)) if 20 <= pop_values[i] <= 100]
        urban_idx = [patch_indices[i] for i in range(len(patch_indices)) if pop_values[i] > 100]

        results = {}
        for name, idx_list in [('Rural (<20)', rural_idx), ('Peri-urban (20-100)', peri_idx), ('Urban (>100)', urban_idx)]:
            if len(idx_list) < 10:
                print(f"  [Density] {name}: too few samples ({len(idx_list)}), skipping")
                continue

            print(f"\n  [Density] Analyzing {name}: {len(idx_list)} patches")
            sub_ds = Subset(dataset, idx_list)
            sub_loader = DataLoader(sub_ds, batch_size=8, shuffle=False)

            baseline = baseline_metrics_fn(sub_loader)
            print(f"    Baseline: R={baseline['r']:.3f}, MAE={baseline['mae']:.2f}")

            perm_results = self.perm_imp.compute(sub_loader, baseline)
            results[name] = perm_results

        return results

    @staticmethod
    def _get_patch_pop(dataset, idx):
        y, x = dataset.indices[idx]
        return dataset.pop[y:y+dataset.patch_size, x:x+dataset.patch_size].sum()


def plot_permutation_importance(results: Dict[str, Dict[str, float]], title: str, save_path: str):
    """Generate a publication-ready bar chart of permutation importance."""
    channels = list(results.keys())
    metrics = ['loss_delta', 'r_delta', 'mae_delta']
    metric_labels = ['Loss Increase', 'R Decrease', 'MAE Increase']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    for ax, metric, label, color in zip(axes, metrics, metric_labels, colors):
        values = [results[ch][metric] for ch in channels]
        bars = ax.barh(channels, values, color=color, edgecolor='black', linewidth=0.5)
        ax.set_xlabel(label, fontsize=11)
        ax.set_title(f'Channel Importance: {label}', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.8)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(val + max(values)*0.01 if val >= 0 else val - max(values)*0.01,
                   bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_density_stratified_importance(results: Dict[str, Dict], channel_names: List[str], save_path: str):
    """Heatmap of channel importance across density classes."""
    density_classes = list(results.keys())
    n_classes = len(density_classes)
    n_channels = len(channel_names)

    # Build matrix: rows=density classes, cols=channels, values=R decrease
    matrix = np.zeros((n_classes, n_channels))
    for i, cls in enumerate(density_classes):
        for j, ch in enumerate(channel_names):
            matrix[i, j] = results[cls][ch]['r_delta']

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(np.arange(n_channels))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(channel_names, rotation=30, ha='right')
    ax.set_yticklabels(density_classes)

    # Add text annotations
    for i in range(n_classes):
        for j in range(n_channels):
            text = ax.text(j, i, f'{matrix[i, j]:.4f}',
                          ha='center', va='center', color='black', fontsize=10)

    ax.set_title('Channel Importance by Density Class\n(Higher = More Important)', fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=ax, label='R Decrease After Shuffling')
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--ntl_dir', required=True)
    parser.add_argument('--pop', required=True)
    parser.add_argument('--border_mask', required=True)
    parser.add_argument('--built_up_path', default=None)
    parser.add_argument('--built_up_volume_path', default=None)
    parser.add_argument('--built_up_as_channel', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--output_dir', default='outputs/explainability')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine channel count and names
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
    model.eval()
    print(f"[Explain] Loaded model with {len(channel_names)} channels: {channel_names}")

    # Load dataset
    dataset = PopulationDataset(
        ntl_dir=args.ntl_dir,
        pop_path=args.pop,
        border_mask_path=args.border_mask,
        built_up_path=args.built_up_path,
        built_up_volume_path=args.built_up_volume_path,
        built_up_as_channel=args.built_up_as_channel,
    )

    # Split (use same seed as training)
    from sklearn.model_selection import train_test_split
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_ds = Subset(dataset, val_idx)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Baseline metrics
    print("\n[Explain] Computing baseline validation metrics...")
    baseline_loss, baseline_mae, baseline_r = evaluate(model, val_loader, device)
    baseline = {'loss': baseline_loss, 'r': baseline_r, 'mae': baseline_mae}
    print(f"  Baseline: Loss={baseline_loss:.4f}, R={baseline_r:.4f}, MAE={baseline_mae:.2f}")

    # 1. Overall Permutation Importance
    print("\n[Explain] Computing permutation importance (overall)...")
    perm_imp = ChannelPermutationImportance(model, device, channel_names)
    overall_results = perm_imp.compute(val_loader, baseline)

    print("\n  Results (higher delta = more important):")
    for ch, res in overall_results.items():
        print(f"    {ch:15s}: loss_delta={res['loss_delta']:.4f}, r_delta={res['r_delta']:.4f}, mae_delta={res['mae_delta']:.2f}")

    plot_permutation_importance(
        overall_results,
        'Channel Permutation Importance (Overall Validation Set)',
        os.path.join(args.output_dir, 'permutation_importance_overall.png')
    )

    # 2. Density-Stratified Importance
    print("\n[Explain] Computing density-stratified permutation importance...")
    strat_imp = DensityStratifiedImportance(model, device, channel_names)

    def baseline_fn(loader):
        l, m, r = evaluate(model, loader, device)
        return {'loss': l, 'r': r, 'mae': m}

    strat_results = strat_imp.compute(dataset, val_idx, baseline_fn)

    if strat_results:
        plot_density_stratified_importance(
            strat_results, channel_names,
            os.path.join(args.output_dir, 'permutation_importance_by_density.png')
        )

    print("\n[Explain] Feature importance analysis complete!")
