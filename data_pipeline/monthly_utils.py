"""
Utilities for discovering and sorting monthly NTL raster files.
"""
import os
import re
from pathlib import Path
from typing import List, Tuple


def discover_monthly_files(directory: str) -> List[str]:
    """
    Scans a directory for GeoTIFF files and returns them sorted by date
    based on YYYYMM or YYYY_MM patterns in the filename.
    """
    directory = Path(directory)
    files = [f for f in directory.rglob("*.tif") if f.suffix.lower() in (".tif", ".tiff")]

    dated_files = []
    for f in files:
        date = extract_date_from_filename(f.name)
        if date:
            dated_files.append((date, str(f)))

    # Sort by (year, month)
    dated_files.sort(key=lambda x: x[0])
    return [path for _, path in dated_files]


def extract_date_from_filename(filename: str) -> Tuple[int, int]:
    """
    Extracts (year, month) from filenames like:
      pak_ntl_202001.tif  -> (2020, 1)
      pak_ntl_2020_01.tif -> (2020, 1)
    Returns None if no date found.
    """
    # Try YYYYMM first
    m = re.search(r"(\d{4})(\d{2})(?=\D*\.tif)", filename)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        if 1 <= month <= 12:
            return (year, month)

    # Try YYYY_MM
    m = re.search(r"(\d{4})_(\d{2})(?=\D*\.tif)", filename)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        if 1 <= month <= 12:
            return (year, month)

    return None


def group_by_year(paths: List[str]) -> dict:
    """
    Groups monthly file paths by year.
    Returns {year_int: [list of monthly paths in order]}
    """
    grouped = {}
    for p in paths:
        date = extract_date_from_filename(Path(p).name)
        if date:
            year, _ = date
            grouped.setdefault(year, []).append(p)
    # Sort months within each year
    for year in grouped:
        grouped[year].sort(key=lambda x: extract_date_from_filename(Path(x).name))
    return grouped
