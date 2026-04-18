"""
PyTorch Dataset for temporal raster patch extraction (monthly NTL + static POP).
Upgrades applied:
  1. Two-channel input [NTL, POP_proxy]
  2. NTL clipping: negatives -> 0, outliers capped at 250
  3. Border mask filtering
Session 3 + 4: Data-centric AI workflow.
"""
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from data_pipeline.monthly_utils import discover_monthly_files


class TemporalPopulationRasterDataset(Dataset):
    """
    Extracts patches from a stack of monthly NTL rasters and a single Population raster.
    Returns (T, 2, H, W) where channel 0 = NTL and channel 1 = static POP proxy.
    """
    def __init__(
        self,
        ntl_dir: str,
        pop_path: str,
        patch_size: int = 32,
        stride: int = 16,
        nodata: float = -9999.0,
        log_transform_pop: bool = True,
        border_mask_path: str = None,
        ntl_cap: float = 250.0,
        valid_threshold: float = 0.3,
        built_up_path: str = None,
        built_up_volume_path: str = None,
        built_up_as_channel: bool = False,
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.nodata = nodata
        self.log_transform_pop = log_transform_pop
        self.ntl_cap = ntl_cap
        self.valid_threshold = valid_threshold
        self.built_up_path = built_up_path
        self.built_up_volume_path = built_up_volume_path
        self.built_up_as_channel = built_up_as_channel

        # Discover monthly NTL files
        self.ntl_paths = discover_monthly_files(ntl_dir)
        if len(self.ntl_paths) == 0:
            raise ValueError(f"No monthly NTL files found in {ntl_dir}")
        print(f"[Dataset] Found {len(self.ntl_paths)} monthly NTL files.")

        # Load all monthly NTL arrays
        self.ntl_stack = []
        for p in self.ntl_paths:
            with rasterio.open(p) as src:
                self.ntl_stack.append(src.read(1).astype(np.float32))

        # Load static POP
        with rasterio.open(pop_path) as src:
            self.pop = src.read(1).astype(np.float32)

        # Validate shapes
        first_shape = self.ntl_stack[0].shape
        for i, arr in enumerate(self.ntl_stack):
            assert arr.shape == first_shape, (
                f"NTL file {self.ntl_paths[i]} has shape {arr.shape}, expected {first_shape}"
            )
        assert self.pop.shape == first_shape, (
            f"POP shape {self.pop.shape} does not match NTL shape {first_shape}"
        )

        # Load optional built-up surface
        self.built_up = None
        if built_up_path and Path(built_up_path).exists():
            with rasterio.open(built_up_path) as src:
                self.built_up = src.read(1).astype(np.float32)
            assert self.built_up.shape == first_shape, f"Built-up shape {self.built_up.shape} does not match NTL shape {first_shape}"
            print(f"[Dataset] Loaded built-up land cover from {built_up_path}")

        # Load optional built-up volume
        self.built_up_volume = None
        if built_up_volume_path and Path(built_up_volume_path).exists():
            with rasterio.open(built_up_volume_path) as src:
                self.built_up_volume = src.read(1).astype(np.float32)
            assert self.built_up_volume.shape == first_shape, f"Built-up volume shape {self.built_up_volume.shape} does not match NTL shape {first_shape}"
            print(f"[Dataset] Loaded built-up volume from {built_up_volume_path}")

        # Determine channel count
        bu_channels = 0
        if self.built_up_as_channel:
            bu_channels += (1 if self.built_up is not None else 0)
            bu_channels += (1 if self.built_up_volume is not None else 0)
        self.in_channels = 2 + bu_channels

        self.h, self.w = first_shape
        self.T = len(self.ntl_stack)

        # Precompute robust normalization percentiles once
        self.ntl_p99s = []
        for t in range(self.T):
            arr = self.ntl_stack[t]
            vals = arr[arr > 0]
            p99 = np.percentile(vals, 99.0) if vals.size > 0 else 1.0
            if not (np.isfinite(p99) and p99 > 0):
                p99 = float(arr.max()) + 1e-6
            self.ntl_p99s.append(p99)

        pop_vals = self.pop[self.pop > 0]
        self.pop_p99 = np.percentile(pop_vals, 99.0) if pop_vals.size > 0 else 1.0
        if not (np.isfinite(self.pop_p99) and self.pop_p99 > 0):
            self.pop_p99 = float(self.pop.max()) + 1e-6

        # Load border mask if provided
        self.border_mask = None
        if border_mask_path and Path(border_mask_path).exists():
            with rasterio.open(border_mask_path) as src:
                self.border_mask = src.read(1).astype(np.uint8)
            assert self.border_mask.shape == first_shape, "Border mask shape mismatch"
            print(f"[Dataset] Loaded border mask from {border_mask_path}")

        # Build valid patch indices
        self.indices = []
        for y in range(0, self.h - patch_size + 1, stride):
            for x in range(0, self.w - patch_size + 1, stride):
                # Check NTL validity leniently: average valid fraction across all months
                ntl_valid_fracs = []
                for t in range(self.T):
                    patch = self.ntl_stack[t][y:y+patch_size, x:x+patch_size]
                    ntl_valid_fracs.append(np.mean((patch != nodata) & np.isfinite(patch)))
                avg_ntl_valid = np.mean(ntl_valid_fracs)

                pop_patch = self.pop[y:y+patch_size, x:x+patch_size]
                pop_valid = np.mean((pop_patch != nodata) & np.isfinite(pop_patch))

                # Border mask check
                mask_valid = 1.0
                if self.border_mask is not None:
                    mask_patch = self.border_mask[y:y+patch_size, x:x+patch_size]
                    mask_valid = np.mean(mask_patch)

                if (avg_ntl_valid >= self.valid_threshold and
                    pop_valid >= self.valid_threshold and
                    mask_valid >= self.valid_threshold):
                    self.indices.append((y, x))

        print(f"[Dataset] Total valid patches: {len(self.indices)} | valid_threshold={self.valid_threshold}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        y, x = self.indices[idx]

        # Process NTL months
        ntl_patches = []
        for t in range(self.T):
            patch = self.ntl_stack[t][y:y+self.patch_size, x:x+self.patch_size].copy()
            # Replace nodata/inf with 0
            patch[(patch == self.nodata) | ~np.isfinite(patch)] = 0.0
            # Clip negatives and cap outliers (literature upgrade)
            patch = np.clip(patch, 0.0, self.ntl_cap)
            # Robust normalization per month using precomputed 99th percentile
            patch = patch / self.ntl_p99s[t]
            ntl_patches.append(patch)

        # Process static POP proxy channel
        pop_patch = self.pop[y:y+self.patch_size, x:x+self.patch_size].copy()
        pop_patch[(pop_patch == self.nodata) | ~np.isfinite(pop_patch)] = 0.0
        # Normalize POP proxy robustly
        pop_norm = pop_patch / self.pop_p99

        # Stack: (T, 2, H, W)
        ntl_tensor = torch.from_numpy(np.stack(ntl_patches, axis=0)).unsqueeze(1)  # (T, 1, H, W)
        pop_tensor = torch.from_numpy(pop_norm).unsqueeze(0).unsqueeze(0).repeat(self.T, 1, 1, 1)  # (T, 1, H, W)
        image = torch.cat([ntl_tensor, pop_tensor], dim=1)  # (T, 2, H, W)

        # Optional: built-up channels (surface + volume)
        if self.built_up_as_channel:
            if self.built_up is not None:
                bu_patch = self.built_up[y:y+self.patch_size, x:x+self.patch_size].copy()
                bu_patch[(bu_patch == self.nodata) | ~np.isfinite(bu_patch)] = 0.0
                bu_tensor = torch.from_numpy(bu_patch).unsqueeze(0).unsqueeze(0).repeat(self.T, 1, 1, 1)
                image = torch.cat([image, bu_tensor], dim=1)
            if self.built_up_volume is not None:
                bv_patch = self.built_up_volume[y:y+self.patch_size, x:x+self.patch_size].copy()
                bv_patch[(bv_patch == self.nodata) | ~np.isfinite(bv_patch)] = 0.0
                bv_tensor = torch.from_numpy(bv_patch).unsqueeze(0).unsqueeze(0).repeat(self.T, 1, 1, 1)
                image = torch.cat([image, bv_tensor], dim=1)

        # Target: total population in patch
        target = pop_patch.sum()
        if self.log_transform_pop:
            target = np.log1p(target)
        target = torch.tensor(target, dtype=torch.float32)

        # Optional: patch-level built-up scalar (surface only, when not used as channel)
        built_up_scalar = torch.tensor(0.0, dtype=torch.float32)
        if self.built_up is not None and not self.built_up_as_channel:
            bu_patch = self.built_up[y:y+self.patch_size, x:x+self.patch_size].copy()
            bu_patch[(bu_patch == self.nodata) | ~np.isfinite(bu_patch)] = 0.0
            built_up_scalar = torch.tensor(bu_patch.mean(), dtype=torch.float32)

        return {
            "image": image,
            "target": target,
            "coords": torch.tensor([y, x], dtype=torch.long),
            "built_up_scalar": built_up_scalar,
        }


from pathlib import Path
