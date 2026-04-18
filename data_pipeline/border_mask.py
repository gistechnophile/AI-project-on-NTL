"""
Border mask generation for known NTL artifacts.
Specifically targets the India-Pakistan floodlit border, a well-documented
source of non-residential light contamination in DMSP/VIIRS data.
"""
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import LineString, mapping


def create_india_pakistan_border_mask(shape, transform, crs, buffer_deg=0.35):
    """
    Generates a binary mask (1 = valid, 0 = masked/border artifact)
    for the India-Pakistan floodlit border zone.
    """
    # Approximate India-Pakistan border waypoints (lat, lon)
    # From north (Karakoram) to south (Rann of Kutch)
    border_coords = [
        (37.0, 74.5),
        (35.5, 77.0),
        (34.0, 76.5),
        (32.5, 75.5),
        (32.0, 74.6),
        (31.0, 74.5),
        (30.0, 74.0),
        (29.0, 72.5),
        (28.0, 71.5),
        (27.0, 70.0),
        (26.0, 69.5),
        (25.0, 68.5),
        (24.0, 68.0),
        (23.5, 68.0),
    ]
    # Shapely uses (x, y) = (lon, lat)
    line = LineString([(lon, lat) for lat, lon in border_coords])
    buffered = line.buffer(buffer_deg)

    # Rasterize the buffer as 0 (masked), background as 1 (valid)
    mask = rasterize(
        [(mapping(buffered), 0)],
        out_shape=shape,
        transform=transform,
        fill=1,
        dtype=np.uint8,
        default_value=0,
    )
    return mask


def save_border_mask(mask, path, crs, transform):
    """Saves the border mask as a GeoTIFF."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=mask.dtype,
        crs=crs,
        transform=transform,
        nodata=None,
    ) as dst:
        dst.write(mask, 1)
