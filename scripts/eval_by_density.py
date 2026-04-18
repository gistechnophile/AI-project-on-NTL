"""
Stratified evaluation of population prediction by urban density class.
Bins (people / 500m pixel):
  - Rural:        < 100
  - Peri-urban:   100 – 1000
  - Urban core:   > 1000
Prints per-class and overall metrics.
"""
import argparse
import numpy as np
import rasterio


def load_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        nodata = src.nodata if src.nodata is not None else -9999.0
    mask = np.isfinite(arr) & (arr != nodata)
    return arr, mask


def compute_metrics(pred, gt, mask):
    if not mask.any():
        return {}
    p = pred[mask]
    g = gt[mask]
    mae = np.abs(p - g).mean()
    rmse = np.sqrt(((p - g) ** 2).mean())
    bias = (p - g).mean()
    r = np.corrcoef(p, g)[0, 1] if p.std() > 0 and g.std() > 0 else 0.0
    total_pred = p.sum()
    total_gt = g.sum()
    return {
        "n_pixels": int(mask.sum()),
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
        "r": float(r),
        "total_pred": float(total_pred),
        "total_gt": float(total_gt),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Predicted population raster")
    parser.add_argument("--gt", required=True, help="Ground-truth population raster")
    parser.add_argument("--scale", type=float, default=1.0, help="Optional scale factor to apply to pred")
    args = parser.parse_args()

    pred, pred_mask = load_raster(args.pred)
    gt, gt_mask = load_raster(args.gt)
    valid = pred_mask & gt_mask

    if args.scale != 1.0:
        pred = pred * args.scale

    # Density bins (people per 500m pixel) – tuned to actual GT distribution
    bins = {
        "Rural (<20)": (gt < 20) & valid,
        "Peri-urban (20-100)": (gt >= 20) & (gt <= 100) & valid,
        "Urban core (>100)": (gt > 100) & valid,
    }

    print("=" * 70)
    print(f"{'Class':<25} {'Pixels':>10} {'MAE':>10} {'RMSE':>10} {'Bias':>10} {'R':>8} {'PredTot':>12} {'GTTot':>12}")
    print("=" * 70)

    overall = compute_metrics(pred, gt, valid)
    print(
        f"{'Overall':<25} "
        f"{overall['n_pixels']:>10,} "
        f"{overall['mae']:>10.2f} "
        f"{overall['rmse']:>10.2f} "
        f"{overall['bias']:>10.2f} "
        f"{overall['r']:>8.4f} "
        f"{overall['total_pred']:>12,.0f} "
        f"{overall['total_gt']:>12,.0f}"
    )

    for name, bin_mask in bins.items():
        m = compute_metrics(pred, gt, bin_mask)
        if not m:
            print(f"{name:<25} {'—':>10} {'—':>10} {'—':>10} {'—':>10} {'—':>8} {'—':>12} {'—':>12}")
            continue
        print(
            f"{name:<25} "
            f"{m['n_pixels']:>10,} "
            f"{m['mae']:>10.2f} "
            f"{m['rmse']:>10.2f} "
            f"{m['bias']:>10.2f} "
            f"{m['r']:>8.4f} "
            f"{m['total_pred']:>12,.0f} "
            f"{m['total_gt']:>12,.0f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
