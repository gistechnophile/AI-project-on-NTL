"""
Training utilities shared across training scripts.
"""
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Single training epoch.
    Supports criterion as either a callable or an nn.Module.
    Returns: average loss (float)
    """
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        x = batch["image"].to(device)
        y = batch["target"].to(device)
        bu = batch.get("built_up_scalar")
        if bu is not None and getattr(model, "use_built_up_scalar", False):
            bu = bu.to(device)

        optimizer.zero_grad()
        pred = model(x, built_up_scalar=bu)

        if isinstance(criterion, nn.Module):
            loss = criterion(pred, y)
        else:
            loss = criterion(pred, y)

        if isinstance(loss, tuple):
            loss = loss[0]  # Some losses return (total, component1, component2)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)

    return total_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, device, return_predictions=False):
    """
    Evaluation loop. Returns loss, MAE (count), and Pearson R.
    If return_predictions=True, also returns (preds, targets) arrays.
    """
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    n_samples = 0
    criterion = nn.SmoothL1Loss()

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["target"].to(device)
        bu = batch.get("built_up_scalar")
        if bu is not None and getattr(model, "use_built_up_scalar", False):
            bu = bu.to(device)

        pred = model(x, built_up_scalar=bu)
        loss = criterion(pred, y)

        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)

        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

    all_preds = torch.tensor(all_preds)
    all_targets = torch.tensor(all_targets)

    # MAE on count scale
    pred_counts = torch.expm1(all_preds.clamp(-2, 16))
    target_counts = torch.expm1(all_targets)
    mae = (pred_counts - target_counts).abs().mean().item()

    # Pearson R on log scale
    if len(all_preds) > 1:
        r, _ = pearsonr(all_preds.numpy(), all_targets.numpy())
    else:
        r = 0.0

    avg_loss = total_loss / n_samples

    if return_predictions:
        return avg_loss, mae, float(r), all_preds.numpy(), all_targets.numpy()
    return avg_loss, mae, float(r)
