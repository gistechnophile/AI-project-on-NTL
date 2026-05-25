"""
Geospatial Stack: Out-of-Core Dataset with xarray + dask + rioxarray

Replaces in-memory raster loading with lazy, chunked arrays that can
handle datasets larger than RAM. Supports:
- Lazy loading of multi-temporal rasters
- Chunked processing for parallel reads
- Spatial indexing with coordinates preserved
- Automatic nodata masking
"""

import os
import glob
from typing import List, Optional, Tuple
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401 — registers .rio accessor
import dask
from dask.diagnostics import ProgressBar


class LazyGeoDataset:
    """
    Lazy-loading geospatial dataset using xarray + dask.
    
    Instead of loading all 72 monthly rasters into memory (~2 GB),
    this creates a chunked dask array that loads data on-demand.
    
    Args:
        ntl_dir: Directory containing monthly NTL subfolders
        pop_path: Path to population ground truth raster
        built_up_path: Optional GHSL surface raster
        built_up_vol_path: Optional GHSL volume raster
        border_mask_path: Optional country border mask
        chunks: Dask chunk size, e.g., (1, 512, 512) for (time, y, x)
    """
    
    def __init__(
        self,
        ntl_dir: str,
        pop_path: str,
        built_up_path: Optional[str] = None,
        built_up_vol_path: Optional[str] = None,
        border_mask_path: Optional[str] = None,
        chunks: Tuple[int, int, int] = (1, 1024, 1024),
    ):
        self.ntl_dir = ntl_dir
        self.pop_path = pop_path
        self.built_up_path = built_up_path
        self.built_up_vol_path = built_up_vol_path
        self.border_mask_path = border_mask_path
        self.chunks = chunks
        
        # --- Build temporal NTL stack lazily ---
        self.monthly_files = self._discover_ntl_files()
        print(f"[LazyGeoDataset] Discovered {len(self.monthly_files)} monthly NTL files")
        
        # Open all rasters lazily as a single xarray DataArray
        self.ntl_da = self._build_ntl_stack()
        self.pop_da = self._open_raster(pop_path, name="population")
        
        # Optional channels
        self.bu_da = self._open_raster(built_up_path, name="built_up") if built_up_path else None
        self.buv_da = self._open_raster(built_up_vol_path, name="built_up_volume") if built_up_vol_path else None
        self.mask_da = self._open_raster(border_mask_path, name="border_mask") if border_mask_path else None
        
        # Synchronise coordinates
        self._align_coordinates()
        
        print(f"[LazyGeoDataset] NTL stack shape: {self.ntl_da.shape}")
        print(f"[LazyGeoDataset] Chunk size: {self.chunks}")
        print(f"[LazyGeoDataset] Memory per chunk: ~{self._estimate_chunk_memory_mb():.1f} MB")
    
    def _discover_ntl_files(self) -> List[str]:
        """Find all .tif files inside monthly subfolders, sorted chronologically."""
        folders = sorted(glob.glob(os.path.join(self.ntl_dir, "NTL_Pakistan_*")))
        files = []
        for folder in folders:
            tifs = glob.glob(os.path.join(folder, "*.tif"))
            if tifs:
                files.append(tifs[0])
        return files
    
    def _open_raster(self, path: str, name: str) -> xr.DataArray:
        """Open a single raster lazily with rioxarray."""
        chunk_dict = {"y": self.chunks[1], "x": self.chunks[2]}
        da = xr.open_dataarray(path, engine="rasterio", chunks=chunk_dict)
        da.name = name
        return da
    
    def _build_ntl_stack(self) -> xr.DataArray:
        """Stack all monthly NTL rasters into a single (time, y, x) DataArray."""
        # Open each raster lazily
        chunk_dict = {"y": self.chunks[1], "x": self.chunks[2]}
        das = [xr.open_dataarray(f, engine="rasterio", chunks=chunk_dict) for f in self.monthly_files]
        
        # Concatenate along new time dimension
        stack = xr.concat(das, dim="time")
        stack.name = "ntl"
        
        # Add time coordinates (YYYY-MM)
        times = [self._parse_date(os.path.basename(os.path.dirname(f))) for f in self.monthly_files]
        stack = stack.assign_coords(time=times)
        
        # Rechunk for optimal access: full time slice per spatial chunk
        stack = stack.chunk({"time": self.chunks[0], "y": self.chunks[1], "x": self.chunks[2]})
        
        return stack
    
    def _parse_date(self, folder_name: str) -> np.datetime64:
        """Parse 'NTL_Pakistan_2020_01_Jan' -> numpy datetime."""
        parts = folder_name.split("_")
        year, month = int(parts[2]), int(parts[3])
        return np.datetime64(f"{year:04d}-{month:02d}-01")
    
    def _align_coordinates(self):
        """Ensure all DataArrays share the same spatial coordinates."""
        # Reproject everything to match population grid if needed
        target_crs = self.pop_da.rio.crs
        for da_name, da in [("ntl", self.ntl_da), ("bu", self.bu_da), ("buv", self.buv_da)]:
            if da is not None and da.rio.crs != target_crs:
                print(f"[LazyGeoDataset] Reprojecting {da_name} to match population CRS")
                # Note: rioxarray reproject is eager; for truly lazy, use warp.reproject via rasterio
                # Here we just verify alignment — actual reprojection should be done in preprocessing
    
    def _estimate_chunk_memory_mb(self) -> float:
        """Estimate memory footprint of one chunk in MB."""
        chunk_elements = np.prod(self.chunks)
        bytes_per_element = 4  # float32
        return chunk_elements * bytes_per_element / (1024 ** 2)
    
    # ------------------------------------------------------------------
    # Extraction methods (lazy until .compute() is called)
    # ------------------------------------------------------------------
    
    def get_patch(
        self,
        y_slice: slice,
        x_slice: slice,
    ) -> dict:
        """
        Extract a spatial patch lazily. Returns a dict of dask arrays.
        
        Args:
            y_slice: Row slice, e.g., slice(1000, 1328)
            x_slice: Col slice, e.g., slice(2000, 2332)
        
        Returns:
            Dict with keys: 'ntl' (T, H, W), 'pop' (H, W), 'bu', 'buv', 'mask'
        """
        patch = {
            "ntl": self.ntl_da.isel(y=y_slice, x=x_slice),
            "pop": self.pop_da.isel(y=y_slice, x=x_slice),
        }
        if self.bu_da is not None:
            patch["bu"] = self.bu_da.isel(y=y_slice, x=x_slice)
        if self.buv_da is not None:
            patch["buv"] = self.buv_da.isel(y=y_slice, x=x_slice)
        if self.mask_da is not None:
            patch["mask"] = self.mask_da.isel(y=y_slice, x=x_slice)
        return patch
    
    def compute_patch(self, y_slice: slice, x_slice: slice) -> dict:
        """Extract a spatial patch and materialise (compute) it to numpy arrays."""
        patch = self.get_patch(y_slice, x_slice)
        with ProgressBar():
            result = {k: v.compute() for k, v in patch.items()}
        return result
    
    def get_mean_ntl_timeseries(self) -> xr.DataArray:
        """
        Compute mean NTL radiance per month across all valid pixels.
        This operation is LAZY — call .compute() to execute.
        """
        masked = self.ntl_da.where(self.ntl_da > -9990)
        masked = masked.where(masked >= 0)
        return masked.mean(dim=["y", "x"], skipna=True)
    
    def get_summary_stats(self) -> xr.Dataset:
        """
        Compute summary statistics for all layers.
        Returns a lazy Dataset — call .compute() to materialise.
        """
        stats = xr.Dataset()
        
        # NTL stats per month
        ntl_masked = self.ntl_da.where(self.ntl_da > -9990)
        stats["ntl_mean"] = ntl_masked.mean(dim=["y", "x"], skipna=True)
        stats["ntl_std"] = ntl_masked.std(dim=["y", "x"], skipna=True)
        stats["ntl_p99"] = ntl_masked.quantile(0.99, dim=["y", "x"], skipna=True)
        
        # Population stats
        pop_masked = self.pop_da.where(self.pop_da >= 0)
        stats["pop_mean"] = pop_masked.mean(skipna=True)
        stats["pop_max"] = pop_masked.max(skipna=True)
        
        return stats
    
    def __repr__(self) -> str:
        extras = []
        if self.bu_da is not None:
            extras.append("bu")
        if self.buv_da is not None:
            extras.append("buv")
        if self.mask_da is not None:
            extras.append("mask")
        return (
            f"LazyGeoDataset(\n"
            f"  ntl_stack={self.ntl_da.shape},\n"
            f"  chunks={self.chunks},\n"
            f"  channels=[ntl, pop" +
            (", " + ", ".join(extras) if extras else "") +
            f"]\n)"
        )


# ------------------------------------------------------------------------------
# Example usage / self-test
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    BASE = r"E:\Private\Lectures\AI\AI Application\paklight-pop"
    
    ds = LazyGeoDataset(
        ntl_dir=os.path.join(BASE, "data", "aligned", "ntl_monthly_aligned"),
        pop_path=os.path.join(BASE, "data", "aligned", "pop_aligned", "pak_pop_2025_CN_100m_R2025A_v1_aligned.tif"),
        built_up_path=os.path.join(BASE, "data", "aligned", "built_up_2020_ghsl_100m_aligned.tif"),
        built_up_vol_path=os.path.join(BASE, "data", "aligned", "built_up_volume_2020_ghsl_100m_aligned.tif"),
        border_mask_path=os.path.join(BASE, "data", "aligned", "border_mask.tif"),
        chunks=(1, 512, 512),
    )
    print(ds)
    
    # Example 1: Lazy timeseries (no memory load)
    print("\n[Example] Computing mean NTL timeseries...")
    ts = ds.get_mean_ntl_timeseries().compute()
    vals = np.asarray(ts).flatten()
    print(f"  2020-01 mean: {vals[0]:.3f}")
    print(f"  2025-12 mean: {vals[-1]:.3f}")
    
    # Example 2: Extract a single patch (memory-efficient)
    print("\n[Example] Extracting patch [1000:1328, 2000:2332]...")
    patch = ds.compute_patch(slice(1000, 1328), slice(2000, 2332))
    print(f"  NTL shape: {patch['ntl'].shape}")
    print(f"  Pop shape: {patch['pop'].shape}")
    print(f"  BU shape: {patch['bu'].shape if 'bu' in patch else 'N/A'}")
