"""
Visualization script for predicted vs. ground-truth population.
Generates a 4-panel figure: Predicted, Ground Truth, Absolute Error, Scatter plot.
"""
import argparse
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def visualize(args):
    # Load prediction
    with rasterio.open(args.pred) as src:
        pred = src.read(1).astype(np.float32)
        pred_nodata = src.nodata if src.nodata is not None else -9999.0

    # Load ground truth
    with rasterio.open(args.gt) as src:
        gt = src.read(1).astype(np.float32)
        gt_nodata = src.nodata if src.nodata is not None else -9999.0

    # Valid mask: where both are valid
    valid = (
        (pred != pred_nodata) & np.isfinite(pred) &
        (gt != gt_nodata) & np.isfinite(gt) &
        (gt >= 0)
    )

    pred_masked = np.where(valid, pred, np.nan)
    gt_masked = np.where(valid, gt, np.nan)
    err_masked = np.where(valid, np.abs(pred - gt), np.nan)

    # vmax for consistent color scaling (99th percentile of GT to avoid outliers skewing)
    vmax = np.nanpercentile(gt_masked, 99.5)
    if not (np.isfinite(vmax) and vmax > 0):
        vmax = np.nanmax(gt_masked)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Prediction
    ax = axes[0, 0]
    im1 = ax.imshow(pred_masked, cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_title("Predicted Population (model)")
    ax.axis("off")
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label="People / 500 m pixel")

    # Panel 2: Ground Truth
    ax = axes[0, 1]
    im2 = ax.imshow(gt_masked, cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_title("Ground Truth (WorldPop 2025)")
    ax.axis("off")
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label="People / 500 m pixel")

    # Panel 3: Absolute Error
    ax = axes[1, 0]
    err_vmax = np.nanpercentile(err_masked, 99.0)
    if not (np.isfinite(err_vmax) and err_vmax > 0):
        err_vmax = np.nanmax(err_masked)
    im3 = ax.imshow(err_masked, cmap="Reds", vmin=0, vmax=err_vmax)
    ax.set_title("Absolute Error |Pred − GT|")
    ax.axis("off")
    fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04, label="People / 500 m pixel")

    # Panel 4: Scatter plot (sampled for performance)
    ax = axes[1, 1]
    sample_idx = np.where(valid.ravel())[0]
    if len(sample_idx) > 50000:
        sample_idx = np.random.choice(sample_idx, 50000, replace=False)
    pred_sample = pred.ravel()[sample_idx]
    gt_sample = gt.ravel()[sample_idx]

    ax.hexbin(gt_sample, pred_sample, gridsize=80, cmap="Blues", bins="log", mincnt=1)
    ax.plot([0, vmax], [0, vmax], "r--", lw=1.5, label="1:1 line")
    ax.set_xlim(0, vmax)
    ax.set_ylim(0, vmax)
    ax.set_xlabel("Ground Truth (WorldPop)")
    ax.set_ylabel("Predicted (model)")
    ax.set_title("Predicted vs. Ground Truth (density plot)")
    ax.legend(loc="upper left")

    # Compute metrics for title
    mae = np.nanmean(err_masked)
    rmse = np.sqrt(np.nanmean((pred_masked - gt_masked) ** 2))
    r = np.corrcoef(pred_sample, gt_sample)[0, 1]
    total_pred = np.nansum(pred_masked)
    total_gt = np.nansum(gt_masked)

    fig.suptitle(
        f"Pakistan NTL → Population Prediction | "
        f"MAE={mae:.1f}  RMSE={rmse:.1f}  R={r:.3f}  "
        f"Total Pred={total_pred:,.0f}  Total GT={total_gt:,.0f}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = Path(args.output_dir) / "prediction_visualization.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[Visualization] Saved figure to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="outputs/pred_population.tif")
    parser.add_argument("--gt", default="data/aligned/pop_aligned/pak_pop_2025_CN_100m_R2025A_v1_aligned.tif")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()
    visualize(args)
