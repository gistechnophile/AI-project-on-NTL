"""
Architecture Deepening — Track 5 Training Pipeline v3.0
=======================================================

Supports three architecture families:
  1. TemporalPopulationRegressor   — ResNet-18/34/50 + 1D temporal conv
  2. TemporalAttentionRegressor    — ResNet-18/34/50 + multi-head self-attention
  3. DeepEnsemble                  — Bag N independent models (different seeds)

Usage (single model):
    python train_v3_architecture.py \
        --architecture attention \
        --backbone resnet18 \
        --ntl_dir data/aligned/ntl_monthly_aligned \
        --pop data/aligned/pop_aligned/pak_pop_2025_CN_100m_R2025A_v1_aligned.tif \
        --border_mask data/aligned/border_mask.tif \
        --built_up_path data/aligned/built_up_2020_ghsl_100m_aligned.tif \
        --built_up_volume_path data/aligned/built_up_volume_2020_ghsl_100m_aligned.tif \
        --pretrained --loss_type huber \
        --epochs 10 --batch_size 8

Usage (ensemble — train 5 models sequentially):
    python train_v3_architecture.py \
        --architecture ensemble \
        --ensemble_n 5 \
        --backbone resnet18 \
        ... (same data args)

Usage (compare all architectures on same split):
    python train_v3_architecture.py \
        --compare_all \
        ... (same data args)
"""

import os
import sys
import argparse
import json
import time
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.architectures import (
    TemporalPopulationRegressor,
    TemporalAttentionRegressor,
    DeepEnsemble,
    count_params,
)
from data_pipeline.dataset import TemporalPopulationRasterDataset
from utils.train_utils import train_one_epoch, evaluate
from mlops.mlflow_tracker import ExperimentTracker

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="Track 5: Architecture Deepening")

    # Data paths
    p.add_argument("--ntl_dir", required=True)
    p.add_argument("--pop", required=True)
    p.add_argument("--border_mask", required=True)
    p.add_argument("--built_up_path", default=None)
    p.add_argument("--built_up_volume_path", default=None)

    # Architecture selection
    p.add_argument("--architecture", type=str, default="temporal_conv",
                   choices=["temporal_conv", "attention", "ensemble"],
                   help="Temporal aggregator type")
    p.add_argument("--backbone", type=str, default="resnet18",
                   choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--n_heads", type=int, default=4,
                   help="Attention heads (attention only)")
    p.add_argument("--n_attn_layers", type=int, default=2,
                   help="Attention encoder layers (attention only)")
    p.add_argument("--ensemble_n", type=int, default=3,
                   help="Number of models in ensemble")
    p.add_argument("--compare_all", action="store_true",
                   help="Run all architecture variants and compare")

    # Training
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss_type", type=str, default="huber", choices=["mse", "huber"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=5,
                   help="Early stopping patience (epochs without improvement)")

    # MLOps
    p.add_argument("--experiment_name", type=str, default="paklight-track5")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--no_mlflow", action="store_true")
    p.add_argument("--output_dir", default="outputs/track5")

    return p.parse_args()


def build_dataset(args):
    """Load dataset with all available channels."""
    built_up_as_channel = bool(args.built_up_path or args.built_up_volume_path)
    dataset = TemporalPopulationRasterDataset(
        ntl_dir=args.ntl_dir,
        pop_path=args.pop,
        border_mask_path=args.border_mask,
        built_up_path=args.built_up_path,
        built_up_volume_path=args.built_up_volume_path,
        built_up_as_channel=built_up_as_channel,
    )
    return dataset


def get_data_loaders(dataset, args, seed=42):
    """Random 80/10/10 split (fixed seed for reproducible comparison)."""
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    return train_loader, val_loader, test_loader


def build_loss(args):
    """Loss function with relative MAE regularisation."""
    def combined_loss(pred, target):
        if args.loss_type == "huber":
            base = nn.SmoothL1Loss(beta=1.0)(pred, target)
        else:
            base = nn.MSELoss()(pred, target)

        pred_orig = torch.expm1(pred.clamp(-2, 16))
        target_orig = torch.expm1(target.clamp(-2, 16))
        rel_mae = (pred_orig - target_orig).abs() / (target_orig + 1)
        rel_mae = rel_mae[~torch.isnan(rel_mae)].mean()

        return base + 0.1 * rel_mae
    return combined_loss


def build_model(args, architecture="temporal_conv", seed=None):
    """Instantiate a single model."""
    in_channels = 2  # NTL + POP proxy
    if args.built_up_path and args.built_up_volume_path:
        in_channels = 4
    elif args.built_up_path or args.built_up_volume_path:
        in_channels = 3

    use_bu_scalar = (args.built_up_path is not None) and (args.built_up_volume_path is None)

    kwargs = dict(
        pretrained=args.pretrained,
        backbone_name=args.backbone,
        in_channels=in_channels,
        use_built_up_scalar=use_bu_scalar,
    )

    if architecture == "attention":
        model = TemporalAttentionRegressor(
            n_heads=args.n_heads,
            n_attn_layers=args.n_attn_layers,
            **kwargs,
        )
    else:
        model = TemporalPopulationRegressor(**kwargs)

    return model


def train_single_model(
    model, train_loader, val_loader, test_loader, args, device,
    tracker=None, run_suffix="", seed=42
):
    """Train one model and return test metrics + checkpoint path."""
    torch.manual_seed(seed)
    model = model.to(device)

    total_params, trainable_params = count_params(model)
    print(f"[Model{run_suffix}] Backbone: {args.backbone} | Params: {total_params/1e6:.2f}M")

    if tracker:
        tracker.log_params({
            f"params_total{run_suffix}": total_params,
            f"params_trainable{run_suffix}": trainable_params,
            f"seed{run_suffix}": seed,
        })

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    loss_fn = build_loss(args)

    best_val_r = -float("inf")
    patience_counter = 0
    ckpt_path = os.path.join(args.output_dir, f"best_model{run_suffix}.pt")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_mae, val_r = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0
        scheduler.step(val_loss)

        metrics = {
            f"train_loss{run_suffix}": train_loss,
            f"val_loss{run_suffix}": val_loss,
            f"val_mae{run_suffix}": val_mae,
            f"val_r{run_suffix}": val_r,
            f"epoch_time{run_suffix}": epoch_time,
        }

        print(f"  Epoch {epoch+1}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | val_MAE={val_mae:.2f} | val_R={val_r:.4f} | "
              f"time={epoch_time:.1f}s")

        if tracker:
            tracker.log_metrics(metrics, step=epoch)

        # Early stopping on val R
        if val_r > best_val_r:
            best_val_r = val_r
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
                "args": vars(args),
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  [Early stop] No improvement for {args.patience} epochs")
                break

    # Load best and evaluate on test
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_mae, test_r = evaluate(model, test_loader, device)
    print(f"  [Test{run_suffix}] MAE={test_mae:.2f} | R={test_r:.4f}")

    if tracker:
        tracker.log_metrics({
            f"test_loss{run_suffix}": test_loss,
            f"test_mae{run_suffix}": test_mae,
            f"test_r{run_suffix}": test_r,
            f"best_val_r{run_suffix}": best_val_r,
        })

    return {
        "test_loss": test_loss,
        "test_mae": test_mae,
        "test_r": test_r,
        "best_val_r": best_val_r,
        "params": total_params,
        "ckpt_path": ckpt_path,
    }


def train_ensemble(args, dataset, train_loader, val_loader, test_loader, device, tracker=None):
    """Train N models with different seeds and create ensemble."""
    print(f"\n[Ensemble] Training {args.ensemble_n} models with different seeds...")
    models = []
    all_results = []

    for i in range(args.ensemble_n):
        seed = args.seed + i * 100
        print(f"\n[Ensemble] Model {i+1}/{args.ensemble_n} (seed={seed})")
        model = build_model(args, architecture="temporal_conv", seed=seed)
        result = train_single_model(
            model, train_loader, val_loader, test_loader, args, device,
            tracker=tracker, run_suffix=f"_ens{i}", seed=seed
        )
        all_results.append(result)
        models.append(model)

    # Build ensemble
    ensemble = DeepEnsemble(models).to(device)
    ens_loss, ens_mae, ens_r = evaluate(ensemble, test_loader, device)
    print(f"\n[Ensemble] Test MAE={ens_mae:.2f} | Test R={ens_r:.4f}")

    # Individual vs ensemble comparison
    individual_r = [r["test_r"] for r in all_results]
    print(f"[Ensemble] Individual R: {individual_r}")
    print(f"[Ensemble] Mean individual R: {np.mean(individual_r):.4f} ± {np.std(individual_r):.4f}")

    if tracker:
        tracker.log_metrics({
            "ensemble_test_mae": ens_mae,
            "ensemble_test_r": ens_r,
            "ensemble_mean_individual_r": np.mean(individual_r),
            "ensemble_std_individual_r": np.std(individual_r),
        })

    # Save ensemble
    ens_path = os.path.join(args.output_dir, "ensemble.pt")
    torch.save({
        "models": [m.state_dict() for m in models],
        "args": vars(args),
        "results": all_results,
    }, ens_path)
    print(f"[Ensemble] Saved to {ens_path}")

    return ensemble, all_results


def compare_all_architectures(args, dataset, train_loader, val_loader, test_loader, device, tracker=None):
    """Run all architecture variants on the same data split and compare."""
    variants = [
        ("ResNet18_1DConv", "temporal_conv", "resnet18"),
        ("ResNet34_1DConv", "temporal_conv", "resnet34"),
        ("ResNet18_Attention", "attention", "resnet18"),
        ("ResNet34_Attention", "attention", "resnet34"),
    ]

    results = {}
    for name, arch, backbone in variants:
        print(f"\n{'='*60}")
        print(f"  Variant: {name}")
        print(f"{'='*60}")
        args.backbone = backbone
        model = build_model(args, architecture=arch)
        result = train_single_model(
            model, train_loader, val_loader, test_loader, args, device,
            tracker=tracker, run_suffix=f"_{name}", seed=args.seed
        )
        results[name] = result

    # Summary table
    print(f"\n{'='*60}")
    print("  ARCHITECTURE COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':<22} {'Params(M)':>10} {'Test MAE':>10} {'Test R':>8} {'Best Val R':>10}")
    print("-" * 62)
    for name, res in results.items():
        print(f"{name:<22} {res['params']/1e6:>10.2f} {res['test_mae']:>10.2f} "
              f"{res['test_r']:>8.4f} {res['best_val_r']:>10.4f}")

    # Save comparison JSON
    comp_path = os.path.join(args.output_dir, "architecture_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Compare] Results saved to {comp_path}")

    if tracker:
        tracker.log_artifact(comp_path)

    return results


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Device: {device}")

    # --- Load dataset once ---
    dataset = build_dataset(args)
    print(f"[Data] Total patches: {len(dataset)} | T={dataset.T} | Channels={dataset.in_channels}")

    # --- MLflow ---
    tracker = None
    if not args.no_mlflow:
        tracker = ExperimentTracker(
            experiment_name=args.experiment_name,
            run_name=args.run_name or f"track5_{args.architecture}_{args.backbone}_{int(time.time())}",
            tags={"track": "5", "architecture": args.architecture, "backbone": args.backbone},
        )
        tracker.__enter__()
        tracker.log_params(vars(args))

    try:
        if args.compare_all:
            train_loader, val_loader, test_loader = get_data_loaders(dataset, args, seed=args.seed)
            compare_all_architectures(args, dataset, train_loader, val_loader, test_loader, device, tracker)

        elif args.architecture == "ensemble":
            train_loader, val_loader, test_loader = get_data_loaders(dataset, args, seed=args.seed)
            train_ensemble(args, dataset, train_loader, val_loader, test_loader, device, tracker)

        else:
            train_loader, val_loader, test_loader = get_data_loaders(dataset, args, seed=args.seed)
            model = build_model(args, architecture=args.architecture)
            train_single_model(model, train_loader, val_loader, test_loader, args, device, tracker)

    except Exception as e:
        import traceback
        traceback.print_exc()
        if tracker:
            tracker.__exit__(type(e), e, None)
        raise
    finally:
        if tracker:
            tracker.__exit__(None, None, None)

    print("\n[Done] Track 5 training complete!")


if __name__ == "__main__":
    main()
