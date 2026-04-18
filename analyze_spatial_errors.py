"""
Spatial error analysis for population predictions.
Breaks down MAE / RMSE / R / bias by urban/rural strata and population deciles,
and exports error heatmaps for GIS inspection.
"""
import argparse
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def load_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0
        profile = src.profile
        bounds = src.bounds
        crs = src.crs
    # compute pixel area in km^2
    transform = profile["transform"]
    if crs and crs.is_geographic:
        # degrees -> approximate km using central latitude
        lat = (bounds.top + bounds.bottom) / 2.0
        lat_rad = np.radians(lat)
        dx_km = abs(transform.a) * 111.32 * np.cos(lat_rad)
        dy_km = abs(transform.e) * 110.57
        pixel_area_km2 = dx_km * dy_km
    else:
        pixel_area_km2 = abs(transform.a * transform.e) / 1e6
    return arr, nodata, profile, pixel_area_km2


def density_class_label(dens_per_km2):
    if dens_per_km2 < 1:
        return "uninhabited"
    elif dens_per_km2 < 100:
        return "rural"
    elif dens_per_km2 < 1000:
        return "peri-urban"
    else:
        return "urban"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to prediction GeoTIFF")
    parser.add_argument("--gt", required=True, help="Path to ground-truth GeoTIFF")
    parser.add_argument("--border_mask", default=None, help="Optional border mask .tif")
    parser.add_argument("--out_dir", default="analysis")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Load] Reading rasters...")
    pred, pred_nodata, pred_profile, pixel_area_km2 = load_raster(args.pred)
    gt, gt_nodata, gt_profile, _ = load_raster(args.gt)

    # valid mask
    valid = np.isfinite(pred) & np.isfinite(gt) & (pred != pred_nodata) & (gt != gt_nodata)
    if args.border_mask:
        mask_arr, mask_nd, _, _ = load_raster(args.border_mask)
        valid &= (mask_arr > 0.5) & (mask_arr != mask_nd) & np.isfinite(mask_arr)

    if not valid.any():
        raise ValueError("No valid overlapping pixels found.")

    pred_v = pred[valid]
    gt_v = gt[valid]
    err = pred_v - gt_v
    abs_err = np.abs(err)
    density = gt_v / pixel_area_km2

    print(f"[Info] Valid pixels: {valid.sum():,} | Pixel area: {pixel_area_km2:.4f} km^2")
    print(f"[Overall] MAE={abs_err.mean():.2f} | RMSE={np.sqrt((err**2).mean()):.2f} | "
          f"Bias={err.mean():.2f} | Pearson R={np.corrcoef(pred_v, gt_v)[0,1]:.4f}")
    print(f"[Totals]  Pred={pred_v.sum():,.0f} | GT={gt_v.sum():,.0f} | "
          f"Diff={pred_v.sum()-gt_v.sum():,.0f} ({100*(pred_v.sum()/gt_v.sum()-1):+.1f}%)")

    # 1) Urban / Rural / Peri-urban / Uninhabited breakdown
    print("\n--- Error by Population Density Class ---")
    classes = [density_class_label(d) for d in density]
    uniq_classes = ["uninhabited", "rural", "peri-urban", "urban"]
    for cls in uniq_classes:
        idx = [c == cls for c in classes]
        idx = np.array(idx)
        if not idx.any():
            continue
        n = idx.sum()
        mae = abs_err[idx].mean()
        rmse = np.sqrt((err[idx]**2).mean())
        bias = err[idx].mean()
        r = np.corrcoef(pred_v[idx], gt_v[idx])[0, 1] if n > 2 else np.nan
        pct_pixels = 100 * n / valid.sum()
        pct_pop = 100 * gt_v[idx].sum() / gt_v.sum()
        print(f"{cls:12s} | n={n:>8,} ({pct_pixels:5.1f}%) | pop={pct_pop:5.1f}% | "
              f"MAE={mae:6.2f} | RMSE={rmse:7.2f} | Bias={bias:7.2f} | R={r:.4f}")

    # 2) Decile analysis
    print("\n--- Error by Ground-Truth Population Decile ---")
    deciles = np.percentile(gt_v, np.linspace(0, 100, 11))
    for i in range(10):
        lo = deciles[i]
        hi = deciles[i + 1]
        idx = (gt_v >= lo) & (gt_v < hi)
        if i == 9:
            idx = (gt_v >= lo) & (gt_v <= hi)  # include max in last bin
        n = idx.sum()
        if n == 0:
            continue
        mae = abs_err[idx].mean()
        rmse = np.sqrt((err[idx]**2).mean())
        bias = err[idx].mean()
        r = np.corrcoef(pred_v[idx], gt_v[idx])[0, 1] if n > 2 else np.nan
        print(f"Decile {i+1:2d}  | pop range [{lo:>10.1f}, {hi:>10.1f}) | n={n:>8,} | "
              f"MAE={mae:6.2f} | RMSE={rmse:7.2f} | Bias={bias:7.2f} | R={r:.4f}")

    # 3) Scatter plot (log scale for readability)
    print("\n[Plot] Generating scatter plot...")
    fig, ax = plt.subplots(figsize=(6, 6))
    # add small jitter for zeros if any
    x = gt_v + 1.0
    y = pred_v + 1.0
    ax.scatter(x, y, s=1, alpha=0.15, c="steelblue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("Ground-truth population + 1")
    ax.set_ylabel("Predicted population + 1")
    ax.set_title("Predicted vs Ground-Truth (log-log)")
    fig.tight_layout()
    scatter_path = out_dir / "scatter_loglog.png"
    fig.savefig(scatter_path, dpi=300)
    plt.close(fig)
    print(f"       Saved {scatter_path}")

    # 4) Error heatmaps
    print("\n[Export] Writing error heatmaps...")
    error_raster = np.full(pred.shape, pred_nodata, dtype=np.float32)
    error_raster[valid] = err.astype(np.float32)

    abserror_raster = np.full(pred.shape, pred_nodata, dtype=np.float32)
    abserror_raster[valid] = abs_err.astype(np.float32)

    profile = pred_profile.copy()
    profile.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=pred_nodata)

    err_path = out_dir / "error_map.tif"
    with rasterio.open(err_path, "w", **profile) as dst:
        dst.write(error_raster, 1)
    print(f"       Signed error -> {err_path}")

    abs_err_path = out_dir / "abserror_map.tif"
    with rasterio.open(abs_err_path, "w", **profile) as dst:
        dst.write(abserror_raster, 1)
    print(f"       Absolute error -> {abs_err_path}")

    # 5) Histogram of errors
    print("\n[Plot] Generating error histogram...")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err, bins=200, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction Error (pred - gt)")
    ax.set_ylabel("Pixel count")
    ax.set_title("Distribution of Per-Pixel Errors")
    # clip xlim to reasonable range based on percentiles
    p01, p99 = np.percentile(err, [1, 99])
    margin = max(abs(p01), abs(p99)) * 1.2
    ax.set_xlim(-margin, margin)
    fig.tight_layout()
    hist_path = out_dir / "error_histogram.png"
    fig.savefig(hist_path, dpi=300)
    plt.close(fig)
    print(f"       Saved {hist_path}")

    print("\n[Done] Analysis complete.")


if __name__ == "__main__":
    main()
