"""
Inference script for TemporalPopulationRegressor.
Reconstructs a full per-pixel population prediction raster.
The model predicts total population per patch; we disaggregate proportionally
using the POP proxy channel, then average overlapping predictions.
"""
import argparse
import numpy as np
import rasterio
import torch
from tqdm import tqdm
from pathlib import Path

from models.population_cnn import TemporalPopulationRegressor
from data_pipeline.dataset import TemporalPopulationRasterDataset


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load dataset (full aligned rasters in RAM)
    dataset = TemporalPopulationRasterDataset(
        ntl_dir=args.ntl_dir,
        pop_path=args.pop,
        patch_size=args.patch_size,
        stride=args.stride,
        border_mask_path=args.border_mask,
        ntl_cap=args.ntl_cap,
        valid_threshold=args.valid_threshold,
        built_up_path=args.built_up_path,
        built_up_volume_path=args.built_up_volume_path,
        built_up_as_channel=args.built_up_as_channel,
    )

    T = dataset.T
    h, w = dataset.h, dataset.w
    ps = args.patch_size

    # Load model
    in_channels = dataset.in_channels
    use_bu_scalar = (dataset.built_up is not None) and (not args.built_up_as_channel)
    model = TemporalPopulationRegressor(
        pretrained=args.pretrained,
        in_channels=in_channels,
        use_built_up_scalar=use_bu_scalar,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Model] Loaded checkpoint from epoch {ckpt.get('epoch', 'N/A')} | val_loss={ckpt.get('val_loss', 'N/A'):.4f}")

    # Accumulators for overlapping per-pixel predictions
    pred_accum = np.zeros((h, w), dtype=np.float64)
    count_accum = np.zeros((h, w), dtype=np.float32)

    # Run inference in batches
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            images = batch["image"].to(device)   # (B, T, 2, H, W)
            coords = batch["coords"].numpy()     # (B, 2) -> y, x
            bu = batch.get("built_up_scalar")
            if bu is not None:
                bu = bu.to(device)

            outputs = model(images, built_up_scalar=bu).cpu().numpy()  # (B,) -> log1p(pop)
            patch_totals = np.expm1(outputs)       # convert back to population count per patch

            for i in range(len(coords)):
                y, x = coords[i]
                # POP proxy weights for this patch (unnormalized, handle zeros)
                pop_proxy = dataset.pop[y:y+ps, x:x+ps].copy()
                pop_proxy[(pop_proxy == dataset.nodata) | ~np.isfinite(pop_proxy)] = 0.0
                proxy_sum = pop_proxy.sum()

                if proxy_sum > 0:
                    pixel_pred = (pop_proxy / proxy_sum) * patch_totals[i]
                else:
                    # uniform fallback if proxy is all zeros
                    pixel_pred = np.full_like(pop_proxy, patch_totals[i] / (ps * ps), dtype=np.float64)

                pred_accum[y:y+ps, x:x+ps] += pixel_pred
                count_accum[y:y+ps, x:x+ps] += 1.0

    # Average overlapping predictions
    mask = count_accum > 0
    pred_raster = np.full((h, w), -9999.0, dtype=np.float32)
    pred_raster[mask] = (pred_accum[mask] / count_accum[mask]).astype(np.float32)

    # Load georeference from first NTL file
    first_ntl = dataset.ntl_paths[0]
    with rasterio.open(first_ntl) as src:
        profile = src.profile.copy()

    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress="lzw",
        nodata=-9999.0,
    )

    output_path = Path(args.output_dir) / "pred_population.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(pred_raster, 1)

    print(f"[Output] Saved prediction raster to {output_path}")
    print(f"         Shape: {pred_raster.shape} | Valid pixels: {int(mask.sum())} / {mask.size}")
    valid_pop = pred_raster[mask].sum()
    print(f"         Predicted total population (valid area): {valid_pop:,.0f}")

    # Quick evaluation against aligned POP
    if args.evaluate and Path(args.pop).exists():
        with rasterio.open(args.pop) as src:
            pop_arr = src.read(1).astype(np.float32)
        eval_mask = mask & np.isfinite(pop_arr) & (pop_arr != (src.nodata if src.nodata is not None else -9999.0))
        if eval_mask.any():
            gt_total = pop_arr[eval_mask].sum()
            mae = np.abs(pred_raster[eval_mask] - pop_arr[eval_mask]).mean()
            rmse = np.sqrt(((pred_raster[eval_mask] - pop_arr[eval_mask]) ** 2).mean())
            r = np.corrcoef(pred_raster[eval_mask], pop_arr[eval_mask])[0, 1]
            print(f"[Eval] MAE={mae:.2f} | RMSE={rmse:.2f} | Pearson R={r:.4f}")
            print(f"[Eval] Ground-truth total pop (valid area): {gt_total:,.0f}")

            if args.scale_to_gt and valid_pop > 0:
                scale = gt_total / valid_pop
                pred_raster_scaled = pred_raster.copy()
                pred_raster_scaled[mask] = pred_raster_scaled[mask] * scale
                mae_s = np.abs(pred_raster_scaled[eval_mask] - pop_arr[eval_mask]).mean()
                rmse_s = np.sqrt(((pred_raster_scaled[eval_mask] - pop_arr[eval_mask]) ** 2).mean())
                r_s = np.corrcoef(pred_raster_scaled[eval_mask], pop_arr[eval_mask])[0, 1]
                print(f"[Scale] Applied post-hoc scale factor: {scale:.4f}")
                print(f"[Scaled Eval] MAE={mae_s:.2f} | RMSE={rmse_s:.2f} | Pearson R={r_s:.4f}")
                print(f"[Scaled Eval] Predicted total pop (valid area): {pred_raster_scaled[eval_mask].sum():,.0f}")

                scaled_path = Path(args.output_dir) / "pred_population_scaled.tif"
                with rasterio.open(scaled_path, "w", **profile) as dst:
                    dst.write(pred_raster_scaled, 1)
                print(f"[Scaled Output] Saved scaled raster to {scaled_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with TemporalPopulationRegressor")
    parser.add_argument("--ntl_dir", default="data/aligned/ntl_monthly_aligned")
    parser.add_argument("--pop", default="data/aligned/pop_aligned/pak_pop_2025_CN_100m_R2025A_v1_aligned.tif")
    parser.add_argument("--border_mask", default="data/aligned/border_mask.tif")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--ntl_cap", type=float, default=250.0)
    parser.add_argument("--valid_threshold", type=float, default=0.3)
    parser.add_argument("--evaluate", action="store_true", default=True)
    parser.add_argument("--scale_to_gt", action="store_true", default=False,
                        help="Apply post-hoc scaling so predicted total matches ground-truth total")
    parser.add_argument("--built_up_path", default=None,
                        help="Path to aligned built-up land cover .tif (optional 3rd channel)")
    parser.add_argument("--built_up_volume_path", default=None,
                        help="Path to aligned built-up volume .tif (optional extra channel)")
    parser.add_argument("--built_up_as_channel", action="store_true",
                        help="Use built-up raster as 3rd image channel (requires built_up_path or volume_path)")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use ImageNet-pretrained ResNet-18 encoder (must match training)")
    args = parser.parse_args()

    if args.built_up_as_channel and not (args.built_up_path or args.built_up_volume_path):
        raise ValueError("--built_up_as_channel requires --built_up_path or --built_up_volume_path")

    run_inference(args)
