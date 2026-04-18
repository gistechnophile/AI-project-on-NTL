"""
One-command script to align all monthly NTL rasters and the WorldPOP raster.
Also generates the India-Pakistan border mask to exclude floodlit artifacts.
"""
import argparse
import json
import os
from pathlib import Path

from data_pipeline.align_rasters import align_rasters
from data_pipeline.quality_audit import audit_raster_pair
from data_pipeline.monthly_utils import discover_monthly_files
from data_pipeline.border_mask import create_india_pakistan_border_mask, save_border_mask
import rasterio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntl_dir", required=True, help="Directory with raw monthly NTL GeoTIFFs")
    parser.add_argument("--pop", required=True, help="Path to raw WorldPOP GeoTIFF")
    parser.add_argument("--output", default="data/aligned", help="Output directory")
    parser.add_argument("--target_res", type=float, default=None, help="Target resolution (None = match NTL resolution)")
    parser.add_argument("--no_border_mask", action="store_true", help="Skip border mask generation")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    ntl_files = discover_monthly_files(args.ntl_dir)
    print(f"[Prepare] Found {len(ntl_files)} monthly NTL files to align.")

    aligned_ntl_dir = os.path.join(args.output, "ntl_monthly_aligned")
    aligned_pop_dir = os.path.join(args.output, "pop_aligned")
    os.makedirs(aligned_ntl_dir, exist_ok=True)
    os.makedirs(aligned_pop_dir, exist_ok=True)

    aligned_ntl_paths = []
    for ntl_path in ntl_files:
        name = Path(ntl_path).stem
        out_subdir = os.path.join(aligned_ntl_dir, name)
        ntl_aligned, pop_aligned = align_rasters(
            ntl_path=ntl_path,
            pop_path=args.pop,
            output_dir=out_subdir,
            target_res=args.target_res,
        )
        aligned_ntl_paths.append(ntl_aligned)

    # Keep one aligned POP reference (copy from the last alignment)
    final_pop_name = Path(args.pop).stem + "_aligned.tif"
    final_pop_path = os.path.join(aligned_pop_dir, final_pop_name)
    if not os.path.exists(final_pop_path):
        import shutil
        shutil.copy(pop_aligned, final_pop_path)

    # Generate border mask using the first aligned NTL as reference grid
    border_mask_path = os.path.join(args.output, "border_mask.tif")
    if not args.no_border_mask:
        with rasterio.open(aligned_ntl_paths[0]) as ref:
            mask = create_india_pakistan_border_mask(
                shape=(ref.height, ref.width),
                transform=ref.transform,
                crs=ref.crs,
                buffer_deg=0.35,
            )
        save_border_mask(mask, border_mask_path, ref.crs, ref.transform)
        print(f"[Prepare] Generated border mask: {border_mask_path}")

    # Run quality audit on the mean of all months vs pop
    with rasterio.open(aligned_ntl_paths[0]) as src:
        ntl_arr = src.read(1)
    for p in aligned_ntl_paths[1:]:
        with rasterio.open(p) as src:
            ntl_arr = ntl_arr + src.read(1)
    ntl_arr = ntl_arr / len(aligned_ntl_paths)

    with rasterio.open(final_pop_path) as src:
        pop_arr = src.read(1)

    scorecard = audit_raster_pair(ntl_arr, pop_arr)

    audit_path = os.path.join(args.output, "quality_audit.json")
    serializable = {}
    for k, v in scorecard.items():
        if isinstance(v, tuple):
            serializable[k] = {"value": float(v[0]), "status": v[1]}
        else:
            serializable[k] = float(v) if isinstance(v, (float, int)) else v
    with open(audit_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"[Prepare] Aligned {len(aligned_ntl_paths)} monthly files.")
    print(f"[Prepare] Aligned POP saved to {final_pop_path}")
    print(f"[Audit] Quality scorecard saved to {audit_path}")
    print(f"[Audit] Q_total = {scorecard['q_total']:.3f}")


if __name__ == "__main__":
    main()
