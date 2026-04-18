"""
Compare old 1km built-up vs new GHSL 100m built-up after alignment.
Prints statistics and saves a side-by-side visualization.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OLD_PATH = PROJECT_ROOT / "data" / "aligned" / "built_up_2020_aligned.tif"
NEW_PATH = PROJECT_ROOT / "data" / "aligned" / "built_up_2020_ghsl_100m_aligned.tif"
OUT_PATH = PROJECT_ROOT / "outputs" / "built_up_comparison.png"


def load_masked(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    nodata = -9999.0
    mask = (arr != nodata) & np.isfinite(arr) & (arr > -1000)
    return arr, mask


def main():
    if not OLD_PATH.exists():
        print(f"Old built-up not found: {OLD_PATH}")
        sys.exit(1)
    if not NEW_PATH.exists():
        print(f"New GHSL built-up not ready yet: {NEW_PATH}")
        sys.exit(0)

    old, old_mask = load_masked(OLD_PATH)
    new, new_mask = load_masked(NEW_PATH)

    print("=" * 60)
    print("Old 1km built-up stats")
    print(f"  valid pixels: {old_mask.sum():,}")
    print(f"  min: {old[old_mask].min():.4f}")
    print(f"  max: {old[old_mask].max():.4f}")
    print(f"  mean: {old[old_mask].mean():.4f}")
    print(f"  median: {np.median(old[old_mask]):.4f}")

    print("-" * 60)
    print("New GHSL 100m -> 500m built-up stats")
    print(f"  valid pixels: {new_mask.sum():,}")
    print(f"  min: {new[new_mask].min():.4f}")
    print(f"  max: {new[new_mask].max():.4f}")
    print(f"  mean: {new[new_mask].mean():.4f}")
    print(f"  median: {np.median(new[new_mask]):.4f}")

    # Scatter: old vs new pixel-wise (sample 50k points for speed)
    common_mask = old_mask & new_mask
    indices = np.where(common_mask.ravel())[0]
    if len(indices) > 50_000:
        rng = np.random.default_rng(42)
        indices = rng.choice(indices, size=50_000, replace=False)
    old_sample = old.ravel()[indices]
    new_sample = new.ravel()[indices]
    r = np.corrcoef(old_sample, new_sample)[0, 1]
    print(f"  pixel-wise Pearson R (old vs new): {r:.4f}")
    print("=" * 60)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    vmax = max(old[old_mask].max(), new[new_mask].max())
    im0 = axes[0, 0].imshow(old, vmin=0, vmax=vmax, cmap="hot")
    axes[0, 0].set_title("Old 1km built-up (aligned)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(new, vmin=0, vmax=vmax, cmap="hot")
    axes[0, 1].set_title("New GHSL 100m -> 500m built-up")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    diff = new - old
    im2 = axes[1, 0].imshow(diff, cmap="RdBu_r", vmin=-vmax * 0.5, vmax=vmax * 0.5)
    axes[1, 0].set_title("Difference (new - old)")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].scatter(old_sample, new_sample, alpha=0.3, s=5)
    axes[1, 1].plot([0, vmax], [0, vmax], "k--")
    axes[1, 1].set_xlabel("Old 1km built-up fraction")
    axes[1, 1].set_ylabel("New GHSL 100m built-up fraction")
    axes[1, 1].set_title(f"Pixel-wise correlation  R={r:.3f}")
    axes[1, 1].set_xlim(0, vmax)
    axes[1, 1].set_ylim(0, vmax)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"[Saved] Comparison plot -> {OUT_PATH}")


if __name__ == "__main__":
    main()
