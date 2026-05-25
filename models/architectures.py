"""
Architecture Deepening — Track 5
==================================

Advanced model architectures for population estimation:
  1. TemporalAttentionRegressor  — Multi-head self-attention over time
  2. TemporalPopulationRegressor  — Original (ResNet-18 + 1D conv) with backbone selection
  3. DeepEnsemble               — Wrapper for N models with different seeds

Session 7 / 8 applications: Attention mechanisms, ensemble methods, architecture search.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Optional


# ── Backbone helpers ──────────────────────────────────────────────────────

def build_backbone(name: str, pretrained: bool = False, in_channels: int = 2):
    """
    Build a ResNet backbone without final FC layer.
    Supports resnet18, resnet34, resnet50.
    """
    name = name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = 512
    elif name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        backbone = models.resnet34(weights=weights)
        feature_dim = 512
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        feature_dim = 2048
    else:
        raise ValueError(f"Unknown backbone: {name}")

    # Adapt first conv for multi-channel input
    if in_channels != 3:
        backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    backbone.fc = nn.Identity()
    return backbone, feature_dim


# ── 1. Original with backbone selection ───────────────────────────────────

class TemporalPopulationRegressor(nn.Module):
    """
    Shared ResNet spatial encoder + 1D temporal convolution.
    Input:  (B, T, C, H, W)
    Output: (B,)
    """
    def __init__(
        self,
        pretrained=False,
        backbone_name="resnet18",
        in_channels=2,
        temporal_hidden=128,
        use_built_up_scalar=False,
    ):
        super().__init__()
        self.use_built_up_scalar = use_built_up_scalar
        self.backbone, self.feature_dim = build_backbone(
            backbone_name, pretrained, in_channels
        )

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.feature_dim, temporal_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(temporal_hidden),
            nn.Conv1d(temporal_hidden, temporal_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(temporal_hidden),
            nn.AdaptiveAvgPool1d(1),
        )

        head_in = temporal_hidden + (1 if use_built_up_scalar else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x, built_up_scalar=None):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)  # (B*T, feature_dim)
        feats = feats.view(B, T, self.feature_dim).transpose(1, 2)
        fused = self.temporal_conv(feats).squeeze(-1)  # (B, temporal_hidden)

        if self.use_built_up_scalar and built_up_scalar is not None:
            fused = torch.cat([fused, built_up_scalar.unsqueeze(-1)], dim=-1)

        out = self.head(fused).squeeze(-1).clamp(-2, 16)
        return out


# ── 2. Self-attention temporal aggregator ─────────────────────────────────

class TemporalAttentionRegressor(nn.Module):
    """
    Shared ResNet spatial encoder + multi-head self-attention over time.

    Replaces the 1D convolution with Transformer-style self-attention:
      - Each month is a "token" with feature_dim dimensions
      - Learnable positional encoding for temporal order
      - Multi-head attention captures month-to-month dependencies
      - Feed-forward + LayerNorm for each month
      - Global average pooling over time -> regression head

    Key advantage: Can learn to attend to specific months (e.g., lockdowns,
    festivals) rather than only local temporal neighbourhoods like 1D conv.
    """
    def __init__(
        self,
        pretrained=False,
        backbone_name="resnet18",
        in_channels=2,
        n_heads=4,
        n_attn_layers=2,
        temporal_hidden=128,
        dropout=0.1,
        use_built_up_scalar=False,
    ):
        super().__init__()
        self.use_built_up_scalar = use_built_up_scalar
        self.backbone, self.feature_dim = build_backbone(
            backbone_name, pretrained, in_channels
        )

        # Learnable positional encoding for each month position
        # Max 72 months (2020–2025)
        self.pos_encoding = nn.Parameter(torch.randn(1, 72, self.feature_dim) * 0.02)

        # Temporal self-attention blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=n_heads,
            dim_feedforward=temporal_hidden * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=n_attn_layers)

        # Post-attention pooling
        self.temporal_pool = nn.Sequential(
            nn.Linear(self.feature_dim, temporal_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        head_in = temporal_hidden + (1 if use_built_up_scalar else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, built_up_scalar=None):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)  # (B*T, feature_dim)
        feats = feats.view(B, T, self.feature_dim)  # (B, T, feature_dim)

        # Add positional encoding (use first T positions)
        feats = feats + self.pos_encoding[:, :T, :]

        # Self-attention over time
        # Create causal mask? For population, we can attend to all months
        # since it's not a forecasting task — we know all months at once
        attn_out = self.attention(feats)  # (B, T, feature_dim)

        # Mean pool over time
        pooled = attn_out.mean(dim=1)  # (B, feature_dim)
        fused = self.temporal_pool(pooled)  # (B, temporal_hidden)

        if self.use_built_up_scalar and built_up_scalar is not None:
            fused = torch.cat([fused, built_up_scalar.unsqueeze(-1)], dim=-1)

        out = self.head(fused).squeeze(-1).clamp(-2, 16)
        return out


# ── 3. Deep Ensemble wrapper ──────────────────────────────────────────────

class DeepEnsemble(nn.Module):
    """
    Deep ensemble: average predictions from N independently trained models.

    Reference: Lakshminarayanan et al. (2017), "Simple and Scalable Predictive
    Uncertainty Estimation using Deep Ensembles", NeurIPS.

    The ensemble prediction variance is a proxy for epistemic uncertainty
    (model uncertainty) — distinct from MC Dropout's aleatoric uncertainty.
    """
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    @property
    def n_models(self):
        return len(self.models)

    def forward(self, x, built_up_scalar=None, return_variance=False):
        """
        Forward through all models. Returns mean prediction.
        If return_variance=True, also returns epistemic variance.
        """
        preds = torch.stack([
            m(x, built_up_scalar) for m in self.models
        ], dim=0)  # (N_models, B)

        mean_pred = preds.mean(dim=0)  # (B,)

        if return_variance:
            var_pred = preds.var(dim=0, unbiased=True)  # (B,)
            return mean_pred, var_pred
        return mean_pred

    def predict_with_uncertainty(self, x, built_up_scalar=None):
        """Alias that always returns mean + variance."""
        return self.forward(x, built_up_scalar, return_variance=True)


# ── Parameter / FLOP counting ─────────────────────────────────────────────

def count_params(model):
    """Total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_summary(model_name: str, model: nn.Module, input_shape=(1, 12, 2, 32, 32)):
    """Print a compact summary."""
    total, trainable = count_params(model)
    # Rough FLOPs: forward ≈ 2 * params * T
    T = input_shape[1]
    flops = 2 * total * T
    print(f"\n{'='*50}")
    print(f"  Model: {model_name}")
    print(f"  Params: {total/1e6:.2f}M (trainable: {trainable/1e6:.2f}M)")
    print(f"  FLOPs (fwd): ~{flops/1e9:.2f} GFLOPs")
    print(f"{'='*50}")
    return total, flops


if __name__ == "__main__":
    # ── Sanity checks ─────────────────────────────────────────────────────
    x = torch.randn(4, 12, 2, 32, 32)
    bu = torch.randn(4)

    print("\n>>> TemporalPopulationRegressor (ResNet-18, 1D conv)")
    m1 = TemporalPopulationRegressor(pretrained=False, backbone_name="resnet18", in_channels=2)
    model_summary("ResNet-18 + 1D conv", m1)
    y1 = m1(x)
    print(f"  Output shape: {y1.shape} | range: [{y1.min():.2f}, {y1.max():.2f}]")

    print("\n>>> TemporalPopulationRegressor (ResNet-34, 1D conv)")
    m1b = TemporalPopulationRegressor(pretrained=False, backbone_name="resnet34", in_channels=2)
    model_summary("ResNet-34 + 1D conv", m1b)
    y1b = m1b(x)
    print(f"  Output shape: {y1b.shape}")

    print("\n>>> TemporalAttentionRegressor (ResNet-18, Self-Attention)")
    m2 = TemporalAttentionRegressor(
        pretrained=False, backbone_name="resnet18", in_channels=2,
        n_heads=4, n_attn_layers=2
    )
    model_summary("ResNet-18 + Self-Attention", m2)
    y2 = m2(x)
    print(f"  Output shape: {y2.shape} | range: [{y2.min():.2f}, {y2.max():.2f}]")

    print("\n>>> TemporalAttentionRegressor with built_up_scalar")
    m2b = TemporalAttentionRegressor(
        pretrained=False, backbone_name="resnet18", in_channels=2,
        use_built_up_scalar=True,
    )
    y2b = m2b(x, bu)
    print(f"  Output shape: {y2b.shape}")

    print("\n>>> DeepEnsemble (3 models)")
    ensemble = DeepEnsemble([
        TemporalPopulationRegressor(pretrained=False, backbone_name="resnet18", in_channels=2)
        for _ in range(3)
    ])
    model_summary("DeepEnsemble (3× ResNet-18)", ensemble)
    mean, var = ensemble.predict_with_uncertainty(x)
    print(f"  Mean shape: {mean.shape} | Var shape: {var.shape}")
    print(f"  Var range: [{var.min():.4f}, {var.max():.4f}]")

    print("\n✅ All architecture checks passed!")
