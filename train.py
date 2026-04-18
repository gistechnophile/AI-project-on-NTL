"""
Training script for the Temporal Pakistan NTL -> Population model.
Session 1: Reproducibility & Engineering Integrity.
Session 3: ResNet + Temporal Conv training with MSE loss.
Session 5: FLOPs and compute cost awareness.
"""
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data_pipeline.dataset import TemporalPopulationRasterDataset
from models.population_cnn import TemporalPopulationRegressor, count_flops


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CombinedLoss(nn.Module):
    """
    Combined MSE on log1p(pop) + Relative MAE on raw population count.
    The RAE term fixes the scale bias of pure log-MSE, while clamping
    prevents explosive gradients from torch.expm1 on outlier predictions.
    """
    def __init__(self, rel_mae_weight=0.1):
        super().__init__()
        self.rel_mae_weight = rel_mae_weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred_log, target_log):
        log_mse = self.mse(pred_log, target_log).mean()

        pred_count = torch.expm1(torch.clamp(pred_log, min=-2.0, max=16.0))
        target_count = torch.expm1(target_log)

        rel_mae = torch.abs(pred_count - target_count) / (target_count + 1.0)
        rel_mae = rel_mae.mean()

        total = log_mse + self.rel_mae_weight * rel_mae
        return total, log_mse, rel_mae


class HuberLossWrapper(nn.Module):
    """
    Smooth L1 (Huber) loss on log1p(pop) with optional relative MAE logging.
    More robust to urban-core outliers than MSE.
    """
    def __init__(self, beta=1.0, rel_mae_weight=0.0):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=beta)
        self.rel_mae_weight = rel_mae_weight

    def forward(self, pred_log, target_log):
        huber_loss = self.huber(pred_log, target_log)

        pred_count = torch.expm1(torch.clamp(pred_log, min=-2.0, max=16.0))
        target_count = torch.expm1(target_log)

        rel_mae = torch.abs(pred_count - target_count) / (target_count + 1.0)
        rel_mae = rel_mae.mean()

        total = huber_loss + self.rel_mae_weight * rel_mae
        return total, huber_loss, rel_mae


def compute_pop_weights(target_log, power=0.5):
    """Compute per-sample weights based on population count^power."""
    counts = torch.expm1(target_log)
    weights = (counts + 1.0) ** power
    return weights


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_log_mse = 0.0
    total_count_mae = 0.0
    for batch in tqdm(loader, desc="Training"):
        x = batch["image"].to(device)  # (B, T, C, H, W)
        y = batch["target"].to(device)
        bu = batch.get("built_up_scalar")
        if bu is not None:
            bu = bu.to(device)
        optimizer.zero_grad()
        pred = model(x, built_up_scalar=bu)
        loss, log_mse, count_mae = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_log_mse += log_mse.item() * x.size(0)
        total_count_mae += count_mae.item() * x.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_log_mse / n, total_count_mae / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_log_mse = 0.0
    total_count_mae = 0.0
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["target"].to(device)
        bu = batch.get("built_up_scalar")
        if bu is not None:
            bu = bu.to(device)
        pred = model(x, built_up_scalar=bu)
        loss, log_mse, count_mae = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
        total_log_mse += log_mse.item() * x.size(0)
        total_count_mae += count_mae.item() * x.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_log_mse / n, total_count_mae / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntl_dir", required=True, help="Directory with aligned monthly NTL .tif files")
    parser.add_argument("--pop", required=True, help="Path to aligned WorldPOP .tif")
    parser.add_argument("--border_mask", default=None, help="Path to border mask .tif (optional)")
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrained", action="store_true",
                        help="Use ImageNet-pretrained ResNet-18 encoder")
    parser.add_argument("--loss_type", type=str, default="combined",
                        choices=["combined", "huber"],
                        help="Loss function: combined (log-MSE+relMAE) or huber (SmoothL1)")
    parser.add_argument("--huber_beta", type=float, default=1.0,
                        help="Beta parameter for Huber loss")
    parser.add_argument("--rel_mae_weight", type=float, default=0.1,
                        help="Weight for relative MAE term in combined loss")
    parser.add_argument("--valid_threshold", type=float, default=0.3,
                        help="Minimum valid pixel fraction for NTL, POP and border mask")
    parser.add_argument("--built_up_as_channel", action="store_true",
                        help="Use built-up raster as 3rd image channel (requires built_up_path)")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (0 = main process only)")
    parser.add_argument("--pop_weight_power", type=float, default=0.0,
                        help="Power for population-based sample weighting (0 = uniform)")
    parser.add_argument("--total_pop_weight", type=float, default=0.0,
                        help="Weight for batch-level total-population regularization term")
    parser.add_argument("--built_up_path", default=None,
                        help="Path to aligned built-up land cover .tif (optional 3rd channel)")
    parser.add_argument("--built_up_volume_path", default=None,
                        help="Path to aligned built-up volume .tif (optional extra channel)")
    args = parser.parse_args()

    if args.built_up_as_channel and not (args.built_up_path or args.built_up_volume_path):
        raise ValueError("--built_up_as_channel requires --built_up_path or --built_up_volume_path")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Dataset
    dataset = TemporalPopulationRasterDataset(
        ntl_dir=args.ntl_dir,
        pop_path=args.pop,
        patch_size=args.patch_size,
        stride=args.stride,
        border_mask_path=args.border_mask,
        valid_threshold=args.valid_threshold,
        built_up_path=args.built_up_path,
        built_up_volume_path=args.built_up_volume_path,
        built_up_as_channel=args.built_up_as_channel,
    )
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model: adaptive channels + optional built-up scalar
    in_channels = dataset.in_channels
    use_bu_scalar = (dataset.built_up is not None) and (not args.built_up_as_channel)
    model = TemporalPopulationRegressor(
        pretrained=args.pretrained,
        in_channels=in_channels,
        use_built_up_scalar=use_bu_scalar,
    ).to(device)
    dummy_shape = (1, dataset.T, in_channels, args.patch_size, args.patch_size)
    flops, params = count_flops(model, input_shape=dummy_shape)
    print(f"[Model] Params: {params/1e6:.2f}M | Forward FLOPs: {flops/1e9:.2f} GFLOPs | Temporal length T={dataset.T} | in_channels={in_channels} | use_bu_scalar={use_bu_scalar} | pretrained={args.pretrained}")

    if args.loss_type == "huber":
        criterion = HuberLossWrapper(beta=args.huber_beta, rel_mae_weight=args.rel_mae_weight).to(device)
    else:
        criterion = CombinedLoss(rel_mae_weight=args.rel_mae_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_log_mse, train_count_mae = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_log_mse, val_count_mae = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{args.epochs} | Train: total={train_loss:.4f} logMSE={train_log_mse:.4f} relMAE={train_count_mae:.4f}  |  Val: total={val_loss:.4f} logMSE={val_log_mse:.4f} relMAE={val_count_mae:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
                "T": dataset.T,
            }, ckpt_path)
            print(f"[Checkpoint] Saved best model to {ckpt_path}")

    # Save config for reproducibility
    with open(os.path.join(args.output_dir, "train_config.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()
