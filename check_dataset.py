from data_pipeline.dataset import TemporalPopulationRasterDataset

ds = TemporalPopulationRasterDataset(
    ntl_dir='data/aligned/ntl_monthly_aligned',
    pop_path='data/aligned/pop_aligned/pak_pop_2025_CN_100m_R2025A_v1_aligned.tif',
    patch_size=32, stride=16,
    border_mask_path='data/aligned/border_mask.tif',
)
print(f'Patches: {len(ds)}')
print(f'Temporal length: {ds.T}')
sample = ds[0]
print(f'Image shape: {sample["image"].shape}')
print(f'Target: {sample["target"].item():.4f}')
