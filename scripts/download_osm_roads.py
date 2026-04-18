"""
Download OSM road network for Pakistan and rasterize to road density (km per 500m pixel).
"""
import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import rasterize
import osmnx as ox

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REF_PATH = PROJECT_ROOT / "data" / "aligned" / "border_mask.tif"
OUT_PATH = PROJECT_ROOT / "data" / "aligned" / "osm_road_density_km_per_500m_aligned.tif"


def main():
    if not REF_PATH.exists():
        print(f"ERROR: reference raster not found: {REF_PATH}")
        sys.exit(1)

    # Load reference grid
    with rasterio.open(REF_PATH) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = ref.shape
        ref_bounds = ref.bounds

    print(f"[OSM] Downloading road network for Pakistan bounds: {ref_bounds}")
    # OSMnx expects (north, south, east, west)
    north, south, east, west = ref_bounds.top, ref_bounds.bottom, ref_bounds.right, ref_bounds.left

    try:
        # Download drivable roads within bounding box
        G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type="drive", simplify=True)
        print(f"[OSM] Downloaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    except Exception as e:
        print(f"[OSM] Download failed: {e}")
        # Fallback: create empty raster
        road_density = np.zeros(ref_shape, dtype=np.float32)
        with rasterio.open(OUT_PATH, "w", driver="GTiff", height=ref_shape[0], width=ref_shape[1],
                           count=1, dtype="float32", crs=ref_crs, transform=ref_transform, compress="lzw") as dst:
            dst.write(road_density, 1)
        print(f"[OSM] Saved empty road density raster to {OUT_PATH}")
        return

    # Convert edges to GeoDataFrame with geometry and length
    gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
    gdf = gdf.to_crs(ref_crs)
    print(f"[OSM] Reprojected to {ref_crs}")

    # Rasterize: sum road length (km) per 500m pixel
    # Use the 'length' column if available, otherwise compute from geometry
    if "length" not in gdf.columns:
        gdf["length"] = gdf.geometry.length

    # Convert length to km
    gdf["length_km"] = gdf["length"] / 1000.0

    shapes = ((geom, length) for geom, length in zip(gdf.geometry, gdf["length_km"]))
    road_density = rasterize(
        shapes=shapes,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=0.0,
        dtype=np.float32,
        merge_alg=rasterio.enums.MergeAlg.add,  # sum overlapping roads
    )

    print(f"[OSM] Road density stats: min={road_density.min():.4f}, max={road_density.max():.4f}, mean={road_density.mean():.4f}")

    with rasterio.open(OUT_PATH, "w", driver="GTiff", height=ref_shape[0], width=ref_shape[1],
                       count=1, dtype="float32", crs=ref_crs, transform=ref_transform, compress="lzw", nodata=0.0) as dst:
        dst.write(road_density, 1)

    print(f"[OSM] Saved road density raster to {OUT_PATH}")


if __name__ == "__main__":
    main()
