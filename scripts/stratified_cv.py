"""
Stratified 3-Fold Cross-Validation by density class.
Ensures each fold has proportional Rural / Peri-urban / Urban representation.
"""
import os
import sys
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.dataset import TemporalPopulationRasterDataset
from models.population_cnn import TemporalPopulationRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Best config
CONFIG = {
    "ntl_dir": "data/aligned/ntl_monthly_aligned",
    "pop": "data/aligned/pop_aligned/pak_pop_2025_CN_100m_R2025A_v1_aligned.tif",
    "border_mask": "data/aligned/border_mask.tif",
    "patch_size": 32,
    "stride": 16,
    "valid_threshold": 0.3,
    "built_up_path": "data/aligned/built_up_2020_ghsl_100m_aligned.tif",
    "built_up_volume_path": "data/aligned/built_up_volume_2020_ghsl_100m_aligned.tif",
    "built_up_as_channel": True,
    "epochs": 10,
    "batch_size": 8,
    "lr": 1e-3,
}


def compute_density_labels(dataset):
    """Assign density class label to each patch index based on GT pop sum."""
    labels = []
    for idx in range(len(dataset)):
        y, x = dataset.indices[idx]
        pop_patch = dataset.pop[y:y + dataset.patch_size, x:x + dataset.patch_size]
        total = np.nansum(pop_patch)
        if total < 20:
            labels.append(0)  # Rural
        elif total <= 100:
            labels.append(1)  # Peri-urban
        else:
            labels.append(2)  # Urban core
    return np.array(labels)


def stratified_kfold_split(labels, n_folds=3, seed=42):
    """Manual stratified split ensuring proportional class representation."""
    rng = np.random.default_rng(seed)
    fold_indices = [[] for _ in range(n_folds)]

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        # Distribute evenly across folds
        for i, idx in enumerate(cls_idx):
            fold_indices[i % n_folds].append(idx)

    # Build train/val pairs
    splits = []
    for fold in range(n_folds):
        val_idx = np.array(fold_indices[fold])
        train_idx = np.concatenate([fold_indices[i] for i in range(n_folds) if i != fold])
        rng.shuffle(train_idx)
        splits.append((train_idx, val_idx))
    return splits


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = total_log_mse = total_count_mae = 0.0
    for batch in loader:
        x = batch["image"].to(DEVICE)
        y = batch["target"].to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss, log_mse, count_mae = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_log_mse += log_mse.item() * x.size(0)
        total_count_mae += count_mae.item() * x.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_log_mse / n, total_count_mae / n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = total_log_mse = total_count_mae = 0.0
    for batch in loader:
        x = batch["image"].to(DEVICE)
        y = batch["target"].to(DEVICE)
        pred = model(x)
        loss, log_mse, count_mae = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
        total_log_mse += log_mse.item() * x.size(0)
        total_count_mae += count_mae.item() * x.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_log_mse / n, total_count_mae / n


class HuberLossWrapper(torch.nn.Module):
    def __init__(self, beta=1.0, rel_mae_weight=0.1):
        super().__init__()
        self.huber = torch.nn.SmoothL1Loss(beta=beta)
        self.rel_mae_weight = rel_mae_weight

    def forward(self, pred_log, target_log):
        huber_loss = self.huber(pred_log, target_log)
        pred_count = torch.expm1(torch.clamp(pred_log, min=-2.0, max=16.0))
        target_count = torch.expm1(target_log)
        rel_mae = (torch.abs(pred_count - target_count) / (target_count + 1.0)).mean()
        total = huber_loss + self.rel_mae_weight * rel_mae
        return total, huber_loss, rel_mae


def run_fold(fold_idx, train_idx, val_idx, dataset):
    out_dir = PROJECT_ROOT / "outputs" / f"stratified_cv_fold_{fold_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    model = TemporalPopulationRegressor(
        pretrained=True,
        in_channels=dataset.in_channels,
        use_built_up_scalar=False,
    ).to(DEVICE)

    criterion = HuberLossWrapper(beta=1.0, rel_mae_weight=0.1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val = float("inf")
    best_epoch = 0
    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, _, _ = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, _, _ = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "args": CONFIG,
                "T": dataset.T,
            }
            torch.save(ckpt, out_dir / "best_model.pt")

    print(f"  Fold {fold_idx}: best epoch {best_epoch}, val_loss={best_val:.4f}")
    return out_dir / "best_model.pt", best_val


def inference_and_eval(ckpt_path, fold_idx, dataset, val_idx):
    """Run inference on validation patches only and compute Pearson R against GT."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = TemporalPopulationRegressor(
        pretrained=True,
        in_channels=dataset.in_channels,
        use_built_up_scalar=False,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_ds = Subset(dataset, val_idx.tolist())
    preds = []
    gts = []
    loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(DEVICE)
            y = batch["target"].to(DEVICE)
            pred = model(x)
            preds.extend(torch.expm1(pred).cpu().numpy().tolist())
            gts.extend(torch.expm1(y).cpu().numpy().tolist())

    preds = np.array(preds)
    gts = np.array(gts)
    r = np.corrcoef(preds, gts)[0, 1]
    scale = gts.sum() / preds.sum() if preds.sum() > 0 else 1.0
    mae = np.abs(preds - gts).mean()
    # per-pixel MAE: divide by avg pixels per patch (32*32=1024)
    mae_per_pixel = mae / 1024.0

    print(f"  Fold {fold_idx}: R={r:.4f} | PatchMAE={mae:.2f} | PixelMAE={mae_per_pixel:.2f} | Scale={scale:.4f}")
    return {"fold": fold_idx, "r": float(r), "mae": float(mae), "mae_per_pixel": float(mae_per_pixel), "scale": float(scale), "val_loss": float(ckpt["val_loss"])}


def main():
    print("[Stratified CV] Loading dataset...")
    dataset = TemporalPopulationRasterDataset(
        ntl_dir=str(PROJECT_ROOT / CONFIG["ntl_dir"]),
        pop_path=str(PROJECT_ROOT / CONFIG["pop"]),
        patch_size=CONFIG["patch_size"],
        stride=CONFIG["stride"],
        border_mask_path=str(PROJECT_ROOT / CONFIG["border_mask"]),
        valid_threshold=CONFIG["valid_threshold"],
        built_up_path=str(PROJECT_ROOT / CONFIG["built_up_path"]),
        built_up_volume_path=str(PROJECT_ROOT / CONFIG["built_up_volume_path"]),
        built_up_as_channel=CONFIG["built_up_as_channel"],
    )
    print(f"[Stratified CV] Total patches: {len(dataset)}")

    labels = compute_density_labels(dataset)
    print(f"[Stratified CV] Class distribution: Rural={(labels==0).sum()}, Peri-urban={(labels==1).sum()}, Urban={(labels==2).sum()}")

    splits = stratified_kfold_split(labels, n_folds=3, seed=42)

    results = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx+1}/3 | Train={len(train_idx)} | Val={len(val_idx)}")
        print(f"{'='*50}")
        ckpt_path, best_val = run_fold(fold_idx, train_idx, val_idx, dataset)
        metrics = inference_and_eval(ckpt_path, fold_idx, dataset, val_idx)
        results.append(metrics)

    # Summary
    print(f"\n{'='*50}")
    print("STRATIFIED CROSS-VALIDATION SUMMARY")
    print(f"{'='*50}")
    rs = [m["r"] for m in results]
    maes = [m["mae"] for m in results]
    maes_px = [m["mae_per_pixel"] for m in results]
    scales = [m["scale"] for m in results]
    print(f"R              : {np.mean(rs):.4f} ± {np.std(rs):.4f}")
    print(f"MAE (patch)    : {np.mean(maes):.2f} ± {np.std(maes):.2f}")
    print(f"MAE (per-pixel): {np.mean(maes_px):.2f} ± {np.std(maes_px):.2f}")
    print(f"Scale factor   : {np.mean(scales):.4f} ± {np.std(scales):.4f}")

    summary_path = PROJECT_ROOT / "outputs" / "stratified_cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "results": results,
            "summary": {
                "r_mean": float(np.mean(rs)),
                "r_std": float(np.std(rs)),
                "mae_mean": float(np.mean(maes)),
                "mae_std": float(np.std(maes)),
                "mae_per_pixel_mean": float(np.mean(maes_px)),
                "mae_per_pixel_std": float(np.std(maes_px)),
                "scale_mean": float(np.mean(scales)),
                "scale_std": float(np.std(scales)),
            }
        }, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
