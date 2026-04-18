"""
Temporal Population Regressor using monthly NTL sequences + POP proxy.
Session 3: CNNs (ResNet-18) + Session 6: Sequence modeling (1D temporal conv).
Session 5: FLOPs estimation and compute reality.
Upgrades: 2-channel input [NTL, POP_proxy]
"""
import torch
import torch.nn as nn
import torchvision.models as models


class TemporalPopulationRegressor(nn.Module):
    """
    Shared ResNet-18 spatial encoder + 1D temporal convolution over months.
    Input:  (B, T, C, H, W)  where C=2 by default [NTL, POP_proxy]
    Output: (B,)
    Optionally accepts a patch-level built-up scalar concatenated to the fused features.
    """
    def __init__(
        self,
        pretrained=False,
        in_channels=2,
        feature_dim=512,
        temporal_hidden=128,
        use_built_up_scalar=False,
    ):
        super().__init__()
        self.use_built_up_scalar = use_built_up_scalar
        # Shared ResNet-18 spatial encoder (no final FC)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Adapt first conv for multi-channel input
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Remove original FC; we will use the 512-dim global average pool output
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feature_dim = feature_dim

        # Temporal fusion: 1D conv over the monthly feature vectors
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(feature_dim, temporal_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(temporal_hidden),
            nn.Conv1d(temporal_hidden, temporal_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(temporal_hidden),
            nn.AdaptiveAvgPool1d(1),  # aggregates over T
        )

        head_in = temporal_hidden + (1 if use_built_up_scalar else 0)
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x, built_up_scalar=None):
        """
        x: (B, T, C, H, W)
        built_up_scalar: (B,) or None
        """
        B, T, C, H, W = x.shape
        # Collapse batch and time for shared encoding
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)  # (B*T, feature_dim)
        # Reshape to (B, feature_dim, T) for 1D conv
        feats = feats.view(B, T, self.feature_dim).transpose(1, 2)
        fused = self.temporal_conv(feats).squeeze(-1)  # (B, temporal_hidden)
        if self.use_built_up_scalar and built_up_scalar is not None:
            built_up_scalar = built_up_scalar.unsqueeze(-1)  # (B, 1)
            fused = torch.cat([fused, built_up_scalar], dim=-1)  # (B, temporal_hidden+1)
        out = self.head(fused).squeeze(-1).clamp(-2, 16)  # (B,)
        return out


def count_flops(model, input_shape=(1, 12, 2, 32, 32)):
    """
    Rough FLOP estimator.
    Rule of thumb: forward FLOPs ≈ 2 * parameters * T (since backbone is shared but run T times).
    """
    total_params = sum(p.numel() for p in model.parameters())
    T = input_shape[1]
    forward_flops = 2 * total_params * T  # rough scaling
    return forward_flops, total_params


if __name__ == "__main__":
    model = TemporalPopulationRegressor(pretrained=False, in_channels=2)
    x = torch.randn(4, 12, 2, 32, 32)  # batch=4, 12 months, 2 channels
    y = model(x)
    flops, params = count_flops(model, input_shape=x.shape)
    print(f"Output shape: {y.shape}")
    print(f"Params: {params/1e6:.2f}M, Forward FLOPs: {flops/1e9:.2f}GFLOPs")
