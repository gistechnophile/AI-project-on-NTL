"""
Data Quality Framework — 5 Engineering Dimensions (Session 4)
Implemented for aligned raster pairs (NTL, Population).
"""
import numpy as np
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics.pairwise import rbf_kernel


def compute_mmd(x, y, gamma=1.0, max_samples=5000):
    """
    Maximum Mean Discrepancy (MMD) using RBF kernel.
    Session 4: Distribution mismatch detection.
    Subsamples large arrays to avoid memory explosion.
    """
    rng = np.random.default_rng(42)
    if len(x) > max_samples:
        x = rng.choice(x, size=max_samples, replace=False)
    if len(y) > max_samples:
        y = rng.choice(y, size=max_samples, replace=False)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    xx = rbf_kernel(x, x, gamma)
    yy = rbf_kernel(y, y, gamma)
    xy = rbf_kernel(x, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()


def audit_raster_pair(ntl_array, pop_array, nodata=-9999.0):
    """
    Returns a quality scorecard dict for the aligned NTL + POP raster pair.
    """
    # Flatten and mask nodata
    ntl_flat = ntl_array.flatten()
    pop_flat = pop_array.flatten()
    valid_mask = (ntl_flat != nodata) & (pop_flat != nodata) & np.isfinite(ntl_flat) & np.isfinite(pop_flat)
    ntl_valid = ntl_flat[valid_mask]
    pop_valid = pop_flat[valid_mask]

    n_total = ntl_flat.size
    n_valid = ntl_valid.size

    # 1. Completeness
    completeness = n_valid / n_total
    completeness_status = "PASS" if completeness > 0.95 else "WARN" if completeness > 0.85 else "FAIL"

    # 2. Consistency (NTL saturation / zero-brightness checks)
    ntl_zeros = np.mean(ntl_valid <= 0)
    ntl_saturated = np.mean(ntl_valid >= np.percentile(ntl_valid, 99.5))
    consistency_status = "WARN" if (ntl_saturated > 0.05 or ntl_zeros > 0.5) else "PASS"

    # 3. Accuracy (label noise proxy: high-pop but zero-light pixels)
    suspicious = np.sum((pop_valid > 100) & (ntl_valid <= 0)) / max(n_valid, 1)
    accuracy_status = "WARN" if suspicious > 0.05 else "PASS"

    # 4. Timeliness (assume static for now; user updates year)
    timeliness_status = "PASS"

    # 5. Relevance: Mutual Information between NTL and log(pop+1)
    if len(ntl_valid) > 100:
        mi = mutual_info_regression(
            ntl_valid.reshape(-1, 1),
            np.log1p(pop_valid),
            discrete_features=False,
            random_state=42,
        )[0]
        relevance_status = "PASS" if mi > 0.1 else "WARN"
    else:
        mi = 0.0
        relevance_status = "FAIL"

    # Distribution: MMD between this NTL and a reference (here we use a synthetic normal as placeholder)
    # In practice, compare against training-year NTL distribution.
    mmd_val = compute_mmd(ntl_valid, np.random.normal(ntl_valid.mean(), ntl_valid.std() + 1e-6, size=len(ntl_valid)))
    distribution_status = "PASS" if mmd_val < 0.05 else "WARN"

    # Aggregate score: geometric mean of the 5 dimensions (simplified to 0/1)
    scores = {
        "completeness": 1.0 if completeness_status == "PASS" else 0.5 if completeness_status == "WARN" else 0.0,
        "consistency": 1.0 if consistency_status == "PASS" else 0.5,
        "accuracy": 1.0 if accuracy_status == "PASS" else 0.5,
        "timeliness": 1.0,
        "relevance": 1.0 if relevance_status == "PASS" else 0.5 if relevance_status == "WARN" else 0.0,
    }
    q_total = np.exp(np.mean([np.log(v + 1e-6) for v in scores.values()]))

    return {
        "completeness": (completeness, completeness_status),
        "consistency": (1.0 - ntl_saturated - ntl_zeros, consistency_status),
        "accuracy": (1.0 - suspicious, accuracy_status),
        "timeliness": (1.0, timeliness_status),
        "relevance": (mi, relevance_status),
        "distribution_mmd": (mmd_val, distribution_status),
        "q_total": q_total,
        "n_valid_pixels": n_valid,
        "n_total_pixels": n_total,
    }
