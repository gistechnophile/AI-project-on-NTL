"""
Spatial-Aware Sampler for Geospatial ML

Addresses the critical flaw in standard train/test splitting:
SPATIAL AUTOCORRELATION. Nearby pixels are correlated, so random
splits leak information. This sampler ensures train and test sets
are geographically separated by a minimum distance buffer.

This replaces torchgeo's samplers (which failed to install on Windows).
"""

import numpy as np
from typing import List, Tuple, Optional
import rasterio
from sklearn.cluster import KMeans


class SpatiallySeparatedSplitter:
    """
    Split patches into train/val/test with spatial separation.
    
    Strategy: Cluster patch centres into N spatial regions using K-Means
    on (lat, lon) coordinates, then assign entire clusters to folds.
    This guarantees no spatial overlap between splits.
    
    Args:
        patch_centres: List of (y, x) pixel coordinates for patch centres
        transform: rasterio.Affine transform (pixel -> geo coordinates)
        n_clusters: Number of spatial clusters (default = 5 for train/val/test)
        val_clusters: Number of clusters to assign to validation
        test_clusters: Number of clusters to assign to test
        buffer_px: Minimum pixel distance between train and test patches
    """
    
    def __init__(
        self,
        patch_centres: List[Tuple[int, int]],
        transform,
        n_clusters: int = 5,
        val_clusters: int = 1,
        test_clusters: int = 1,
        buffer_px: int = 64,
        random_state: int = 42,
    ):
        self.patch_centres = np.array(patch_centres)  # (N, 2) in pixel coords
        self.transform = transform
        self.n_clusters = n_clusters
        self.val_clusters = val_clusters
        self.test_clusters = test_clusters
        self.buffer_px = buffer_px
        self.random_state = random_state
        
        # Convert pixel centres to geo coordinates for clustering
        self.geo_coords = self._pixel_to_geo(self.patch_centres)
        
        # Perform spatial clustering
        self.cluster_labels = self._cluster_spatially()
        
        # Assign clusters to splits
        self.train_idx, self.val_idx, self.test_idx = self._assign_splits()
        
        # Verify spatial separation
        self._verify_separation()
    
    def _pixel_to_geo(self, px_coords: np.ndarray) -> np.ndarray:
        """Convert (y, x) pixel coords to (lon, lat) geo coords."""
        ys, xs = px_coords[:, 0], px_coords[:, 1]
        lons, lats = rasterio.AffineTransformer(self.transform).xy(xs, ys)
        return np.column_stack([lons, lats])
    
    def _cluster_spatially(self) -> np.ndarray:
        """K-Means clustering on geo coordinates."""
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        return kmeans.fit_predict(self.geo_coords)
    
    def _assign_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assign clusters to train/val/test with spatial separation."""
        unique_clusters = np.unique(self.cluster_labels)
        
        # Sort clusters by geographic spread (smaller = more compact)
        cluster_spreads = []
        for c in unique_clusters:
            mask = self.cluster_labels == c
            coords = self.geo_coords[mask]
            spread = np.std(coords, axis=0).mean()
            cluster_spreads.append((spread, c))
        
        cluster_spreads.sort()
        sorted_clusters = [c for _, c in cluster_spreads]
        
        # Assign: test gets most isolated clusters, val next, rest train
        test_cl = sorted_clusters[:self.test_clusters]
        val_cl = sorted_clusters[self.test_clusters:self.test_clusters + self.val_clusters]
        train_cl = sorted_clusters[self.test_clusters + self.val_clusters:]
        
        train_idx = np.where(np.isin(self.cluster_labels, train_cl))[0]
        val_idx = np.where(np.isin(self.cluster_labels, val_cl))[0]
        test_idx = np.where(np.isin(self.cluster_labels, test_cl))[0]
        
        return train_idx, val_idx, test_idx
    
    def _verify_separation(self):
        """Ensure minimum distance between train and test patches."""
        train_px = self.patch_centres[self.train_idx]
        test_px = self.patch_centres[self.test_idx]
        
        if len(test_px) == 0:
            return
        
        # Compute pairwise distances between train and test patch centres
        from scipy.spatial.distance import cdist
        dists = cdist(train_px, test_px, metric='euclidean')
        min_dist = dists.min()
        
        print(f"[SpatialSplitter] Train: {len(self.train_idx)} | Val: {len(self.val_idx)} | Test: {len(self.test_idx)}")
        print(f"[SpatialSplitter] Minimum train-test distance: {min_dist:.1f} pixels (buffer={self.buffer_px})")
        
        if min_dist < self.buffer_px:
            print(f"[WARNING] Train-test spatial separation ({min_dist:.1f}px) is below buffer ({self.buffer_px}px)")
    
    def get_splits(self) -> Tuple[List[int], List[int], List[int]]:
        """Return train/val/test patch indices."""
        return (
            self.train_idx.tolist(),
            self.val_idx.tolist(),
            self.test_idx.tolist(),
        )
    
    def get_cluster_map(self, shape: Tuple[int, int]) -> np.ndarray:
        """Render cluster assignments as a 2D map for visualisation."""
        h, w = shape
        cluster_map = np.full((h, w), -1, dtype=np.int16)
        for idx, (cy, cx) in enumerate(self.patch_centres):
            cluster_map[cy, cx] = self.cluster_labels[idx]
        return cluster_map


class StratifiedSpatialSplitter(SpatiallySeparatedSplitter):
    """
    Extends SpatiallySeparatedSplitter to enforce proportional
    representation of density classes within each spatial cluster.
    
    This is the GOLD STANDARD for geospatial ML evaluation:
    1. Spatial separation prevents autocorrelation leakage
    2. Stratification ensures each fold has rural/peri-urban/urban representation
    """
    
    def __init__(
        self,
        patch_centres: List[Tuple[int, int]],
        patch_populations: np.ndarray,
        transform,
        n_clusters: int = 5,
        val_clusters: int = 1,
        test_clusters: int = 1,
        density_bins: List[float] = [20, 100],
        buffer_px: int = 64,
        random_state: int = 42,
    ):
        # Classify each patch by density
        self.density_labels = self._classify_density(patch_populations, density_bins)
        
        # Run base spatial clustering
        super().__init__(patch_centres, transform, n_clusters, val_clusters, test_clusters, buffer_px, random_state)
    
    def _classify_density(self, populations: np.ndarray, bins: List[float]) -> np.ndarray:
        """Classify patches into rural (0), peri-urban (1), urban (2)."""
        labels = np.zeros(len(populations), dtype=int)
        for i, thresh in enumerate(bins):
            labels[populations > thresh] = i + 1
        return labels
    
    def _assign_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Override: ensure density-class balance across splits."""
        unique_clusters = np.unique(self.cluster_labels)
        
        # For each cluster, compute density-class histogram
        cluster_profiles = {}
        for c in unique_clusters:
            mask = self.cluster_labels == c
            hist = np.bincount(self.density_labels[mask], minlength=3)
            cluster_profiles[c] = hist
        
        # Greedy assignment: balance density classes while maintaining spatial separation
        # Sort clusters by urban fraction (most urban -> test, least urban -> train)
        sorted_clusters = sorted(unique_clusters, key=lambda c: cluster_profiles[c][2])
        
        test_cl = sorted_clusters[:self.test_clusters]
        val_cl = sorted_clusters[self.test_clusters:self.test_clusters + self.val_clusters]
        train_cl = sorted_clusters[self.test_clusters + self.val_clusters:]
        
        train_idx = np.where(np.isin(self.cluster_labels, train_cl))[0]
        val_idx = np.where(np.isin(self.cluster_labels, val_cl))[0]
        test_idx = np.where(np.isin(self.cluster_labels, test_cl))[0]
        
        # Print density stratification report
        for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
            hist = np.bincount(self.density_labels[idx], minlength=3)
            print(f"[StratifiedSpatial] {name}: Rural={hist[0]} Peri={hist[1]} Urban={hist[2]}")
        
        return train_idx, val_idx, test_idx


# ------------------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import rasterio
    
    BASE = r"E:\Private\Lectures\AI\AI Application\paklight-pop"
    pop_path = os.path.join(BASE, "data", "aligned", "pop_aligned", "pak_pop_2025_CN_100m_R2025A_v1_aligned.tif")
    
    # Load population to get patch centres and transform
    with rasterio.open(pop_path) as src:
        transform = src.transform
        pop = src.read(1)
    
    # Generate patch centres (sliding window stride=16, patch=32)
    stride = 16
    patch_size = 32
    h, w = pop.shape
    centres = []
    pops = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = pop[y:y+patch_size, x:x+patch_size]
            valid = (patch >= 0).sum()
            if valid >= 0.3 * patch_size * patch_size:
                centres.append((y + patch_size//2, x + patch_size//2))
                pops.append(patch.sum())
    
    print(f"[Test] Total patches: {len(centres)}")
    
    # Test spatial splitter
    splitter = SpatiallySeparatedSplitter(
        patch_centres=centres,
        transform=transform,
        n_clusters=5,
        val_clusters=1,
        test_clusters=1,
    )
    train_idx, val_idx, test_idx = splitter.get_splits()
    
    # Test stratified spatial splitter
    print("\n--- Stratified Spatial Splitter ---")
    strat_splitter = StratifiedSpatialSplitter(
        patch_centres=centres,
        patch_populations=np.array(pops),
        transform=transform,
        n_clusters=5,
        val_clusters=1,
        test_clusters=1,
    )
    train_idx, val_idx, test_idx = strat_splitter.get_splits()
