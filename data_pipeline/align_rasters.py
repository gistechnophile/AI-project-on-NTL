"""
Raster alignment pipeline for Pakistan NTL + WorldPOP.
Course concept: Data Quality & Preprocessing (Session 4)
"""
import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnum
from rasterio.mask import mask
from shapely.geometry import box


def align_rasters(
    ntl_path: str,
    pop_path: str,
    output_dir: str,
    target_crs: str = None,
    target_res: float = None,
    nodata_val: float = -9999.0,
):
    """
    Align NTL and WorldPOP rasters to the exact same grid.
    Uses the NTL raster as the master grid (CRS + resolution).
    The POP raster is downsampled with averaging and scaled by
    pixel-count ratio to preserve total population.
    Returns paths to aligned NTL and POP rasters.
    """
    os.makedirs(output_dir, exist_ok=True)
    ntl_path = Path(ntl_path)
    pop_path = Path(pop_path)

    # Read NTL metadata
    with rasterio.open(ntl_path) as ntl_src:
        ntl_crs = ntl_src.crs
        ntl_bounds = ntl_src.bounds
        ntl_nodata = ntl_src.nodata if ntl_src.nodata is not None else 0
        ntl_transform = ntl_src.transform
        ntl_width = ntl_src.width
        ntl_height = ntl_src.height
        ntl_count = ntl_src.count
        ntl_dtype = ntl_src.dtypes[0]
        if target_res is None:
            target_res = abs(ntl_transform.a)

    # Read POP metadata
    with rasterio.open(pop_path) as pop_src:
        pop_crs = pop_src.crs
        pop_nodata = pop_src.nodata if pop_src.nodata is not None else 0
        pop_count = pop_src.count
        pop_dtype = pop_src.dtypes[0]
        pop_transform = pop_src.transform
        pop_res = abs(pop_transform.a)
        if pop_src.crs is None:
            raise ValueError("WorldPOP raster has no CRS. Please assign EPSG:4326 or similar.")

    dst_crs = target_crs if target_crs is not None else ntl_crs

    aligned_ntl_path = os.path.join(output_dir, f"{ntl_path.stem}_aligned.tif")
    aligned_pop_path = os.path.join(output_dir, f"{pop_path.stem}_aligned.tif")

    # Determine common grid from NTL
    if dst_crs == ntl_crs:
        common_transform = ntl_transform
        common_width = ntl_width
        common_height = ntl_height
    else:
        common_transform, common_width, common_height = calculate_default_transform(
            ntl_crs, dst_crs, ntl_width, ntl_height, *ntl_bounds,
            resolution=(target_res, target_res)
        )

    # ---- Write aligned NTL ----
    with rasterio.open(ntl_path) as ntl_src:
        kwargs = {
            'driver': 'GTiff',
            'dtype': ntl_dtype,
            'count': ntl_count,
            'crs': dst_crs,
            'transform': common_transform,
            'width': common_width,
            'height': common_height,
            'nodata': nodata_val,
        }
        with rasterio.open(aligned_ntl_path, 'w', **kwargs) as dst:
            for i in range(1, ntl_count + 1):
                reproject(
                    source=rasterio.band(ntl_src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=ntl_transform,
                    src_crs=ntl_crs,
                    dst_transform=common_transform,
                    dst_crs=dst_crs,
                    resampling=ResamplingEnum.bilinear,
                    src_nodata=ntl_nodata,
                    dst_nodata=nodata_val,
                )

    # ---- Write aligned POP to SAME grid ----
    with rasterio.open(pop_path) as pop_src:
        kwargs = {
            'driver': 'GTiff',
            'dtype': pop_dtype,
            'count': pop_count,
            'crs': dst_crs,
            'transform': common_transform,
            'width': common_width,
            'height': common_height,
            'nodata': nodata_val,
        }
        with rasterio.open(aligned_pop_path, 'w', **kwargs) as dst:
            for i in range(1, pop_count + 1):
                reproject(
                    source=rasterio.band(pop_src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=pop_transform,
                    src_crs=pop_crs,
                    dst_transform=common_transform,
                    dst_crs=dst_crs,
                    resampling=ResamplingEnum.average,
                    src_nodata=pop_nodata,
                    dst_nodata=nodata_val,
                )

    # ---- Scale POP to preserve total population ----
    # average resampling gives mean per sub-pixel; multiply by area ratio
    ntl_res = abs(common_transform.a)
    if pop_res > 0 and ntl_res > 0:
        scale_factor = (ntl_res / pop_res) ** 2
        with rasterio.open(aligned_pop_path, 'r+') as dst:
            pop_data = dst.read(1)
            valid = (pop_data != nodata_val) & np.isfinite(pop_data)
            pop_data[valid] = pop_data[valid] * scale_factor
            dst.write(pop_data, 1)
        print(f"[Align] Scaled POP raster by {scale_factor:.2f}x to preserve total population.")

    print(f"[Align] Saved aligned rasters to {output_dir}")
    return aligned_ntl_path, aligned_pop_path


def extract_patch(
    raster_path: str,
    bounds: tuple,  # (minx, miny, maxx, maxy)
    output_path: str = None,
):
    """
    Extract a patch from a raster given bounds.
    """
    with rasterio.open(raster_path) as src:
        geom = box(*bounds)
        out_image, out_transform = mask(src, [geom], crop=True, nodata=src.nodata)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": src.nodata,
        })
        if output_path:
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
        return out_image[0], out_meta
