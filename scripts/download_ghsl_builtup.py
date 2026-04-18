"""
Download GHSL GHS-BUILT-S R2023A (2020 epoch, 100m) and align it to the
Pakistan 500m study grid.
"""
import sys
from pathlib import Path
import zipfile
import requests
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
import numpy as np

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100/V1-0/"
    "GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0.zip"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0.zip"
REF_PATH = PROJECT_ROOT / "data" / "aligned" / "border_mask.tif"
OUT_PATH = PROJECT_ROOT / "data" / "aligned" / "built_up_2020_ghsl_100m_aligned.tif"

CHUNK_SIZE = 1024 * 1024  # 1 MiB
# ------------------------------------------------------------------

def download_with_resume(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    mode = "wb"
    if dest.exists():
        existing = dest.stat().st_size
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"[Download] Resuming from {existing:,} bytes …")
    else:
        existing = 0
        print(f"[Download] Starting fresh download ({url.split('/')[-1]}) …")

    r = requests.get(url, headers=headers, stream=True, timeout=60)
    if r.status_code in (200, 206):
        total = int(r.headers.get("Content-Length", 0)) + existing
        downloaded = existing
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total and downloaded % (10 * CHUNK_SIZE) == 0:
                        pct = downloaded / total * 100
                        print(f"  … {downloaded:,} / {total:,} bytes ({pct:.1f}%)")
        print(f"[Download] Finished – {downloaded:,} bytes saved to {dest}")
    else:
        raise RuntimeError(f"Download failed: HTTP {r.status_code}")


def extract_zip(zip_path: Path, extract_to: Path):
    print(f"[Extract] Unzipping {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=extract_to)
    tifs = list(extract_to.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError("No TIFF found in extracted archive")
    print(f"[Extract] Found {tifs[0].name}")
    return tifs[0]


def align_raster(src_path: Path, ref_path: Path, out_path: Path):
    print(f"[Align] Reprojecting {src_path.name} to match {ref_path.name} …")
    with rasterio.open(ref_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height
        dst_nodata = 0.0

    with rasterio.open(src_path) as src:
        # GHSL BUILT-S values are m² of built-up surface per 100m pixel.
        # Max for a fully built pixel = 100m × 100m = 10_000 m².
        # We normalise to a 0-1 fraction.
        SCALE = 10_000.0

        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "dtype": "float32",
            "nodata": dst_nodata,
            "count": 1,
            "compress": "lzw",
        })

        with rasterio.open(out_path, "w", **kwargs) as dst:
            data = np.empty((dst_height, dst_width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.average,
                src_nodata=src.nodata if src.nodata is not None else -200,
                dst_nodata=dst_nodata,
            )
            # clip negative values and scale to fraction
            data = np.clip(data, 0, None) / SCALE
            # ensure no-data stays 0
            data[np.isnan(data)] = 0.0
            dst.write(data, 1)

    print(f"[Align] Saved aligned raster to {out_path}")
    # quick stats
    print(f"[Stats] min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")


def main():
    if not REF_PATH.exists():
        print(f"ERROR: reference raster not found: {REF_PATH}")
        sys.exit(1)

    download_with_resume(URL, ZIP_PATH)
    tif_path = extract_zip(ZIP_PATH, RAW_DIR)
    align_raster(tif_path, REF_PATH, OUT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
