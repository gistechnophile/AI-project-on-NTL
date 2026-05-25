#!/usr/bin/env python3
"""Generate figures for the scientific paper."""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import rasterio

# Paths
BASE_DIR = r"E:\Private\Lectures\AI\AI Application\paklight-pop"
NTL_DIR = os.path.join(BASE_DIR, "data", "aligned", "ntl_monthly_aligned")
POP_PATH = os.path.join(BASE_DIR, "data", "aligned", "pop_aligned", "pak_pop_2025_CN_100m_R2025A_v1_aligned.tif")
PRED_PATH = os.path.join(BASE_DIR, "outputs", "pred_population_scaled.tif")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150


def parse_date_from_folder(name):
    """Parse YYYY_MM_Mon from folder name like 'NTL_Pakistan_2020_01_Jan'"""
    parts = name.split("_")
    year = int(parts[2])
    month = int(parts[3])
    return datetime(year, month, 1)


def generate_temporal_trend():
    """Figure 1: Mean NTL radiance across Pakistan, 2020--2025."""
    folders = sorted(glob.glob(os.path.join(NTL_DIR, "NTL_Pakistan_*")))
    print(f"Found {len(folders)} monthly folders")

    dates = []
    means = []
    stds = []

    for folder in folders:
        # Find the .tif inside the folder
        tifs = glob.glob(os.path.join(folder, "*.tif"))
        if not tifs:
            continue
        tif = tifs[0]
        date = parse_date_from_folder(os.path.basename(folder))

        with rasterio.open(tif) as src:
            data = src.read(1)
            # Mask nodata and NaN
            valid = data[data > -9990]
            valid = valid[np.isfinite(valid)]
            valid = valid[valid >= 0]  # clip negative radiance
            if len(valid) > 0:
                means.append(valid.mean())
                stds.append(valid.std())
                dates.append(date)

    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot monthly means
    ax.plot(dates, means, color='#1f77b4', linewidth=1.5, marker='o', markersize=3, label='Monthly mean radiance')

    # 3-month rolling average
    window = 3
    if len(means) >= window:
        rolling = np.convolve(means, np.ones(window)/window, mode='valid')
        ax.plot(dates[window-1:], rolling, color='#ff7f0e', linewidth=2.5, label='3-month rolling mean')

    ax.fill_between(dates, means - stds, means + stds, alpha=0.15, color='#1f77b4', label='±1 std dev')

    # Annotate COVID dip and recovery
    covid_idx = dates.index(datetime(2020, 4, 1)) if datetime(2020, 4, 1) in dates else None
    if covid_idx is not None:
        ax.annotate('COVID-19 lockdown dip', xy=(dates[covid_idx], means[covid_idx]),
                    xytext=(dates[covid_idx + 6], means[covid_idx] + 2),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=9, color='red')

    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Mean NTL radiance (nW·cm⁻²·sr⁻¹)', fontsize=12)
    ax.set_title('Mean Nighttime Light Radiance Across Pakistan (VIIRS DNB, 2020–2025)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, 'fig_temporal_trend.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_spatial_error():
    """Figure 2: Spatial distribution of prediction errors (pred - GT)."""
    print("Loading prediction and ground truth rasters...")

    with rasterio.open(PRED_PATH) as src_pred:
        pred = src_pred.read(1)
        profile = src_pred.profile
        transform = src_pred.transform

    with rasterio.open(POP_PATH) as src_gt:
        gt = src_gt.read(1)

    # Mask invalid
    valid_mask = np.isfinite(pred) & np.isfinite(gt) & (gt >= 0) & (pred >= 0)

    # Compute signed error
    error = np.full_like(pred, np.nan)
    error[valid_mask] = pred[valid_mask] - gt[valid_mask]

    # Get extent
    height, width = pred.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    extent = [left, right, bottom, top]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Ground Truth
    ax = axes[0]
    gt_plot = np.where(valid_mask, gt, np.nan)
    im = ax.imshow(gt_plot, extent=extent, cmap='YlOrRd', vmin=0, vmax=100)
    ax.set_title('(a) Ground Truth Population', fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('People / 500 m pixel')

    # Panel B: Predicted
    ax = axes[1]
    pred_plot = np.where(valid_mask, pred, np.nan)
    im = ax.imshow(pred_plot, extent=extent, cmap='YlOrRd', vmin=0, vmax=100)
    ax.set_title('(b) Predicted Population', fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('People / 500 m pixel')

    # Panel C: Signed Error
    ax = axes[2]
    vmax_err = np.nanpercentile(np.abs(error), 99)
    im = ax.imshow(error, extent=extent, cmap='RdBu_r', vmin=-vmax_err, vmax=vmax_err)
    ax.set_title('(c) Prediction Error (Pred − GT)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('People / 500 m pixel')

    # Overall title
    fig.suptitle('Spatial Distribution of Population Predictions and Errors (Best Model)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, 'fig_spatial_error.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    generate_temporal_trend()
    generate_spatial_error()
    print("\nAll figures generated successfully!")
