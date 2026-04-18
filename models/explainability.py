"""
Grad-CAM explainability for the Temporal Population Regressor.
Session 3: CNN interpretability and engineering integrity.
For 2-channel inputs, we compute Grad-CAM on the temporal mean image.
"""
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
class RegressionTarget:
    """Custom target for regression tasks: backpropagate the model output itself."""
    def __init__(self, scalar_index=0):
        self.scalar_index = scalar_index

    def __call__(self, model_output):
        if model_output.ndim == 0:
            return model_output
        if model_output.ndim == 1:
            return model_output[self.scalar_index]
        return model_output[:, self.scalar_index]


class TemporalToSingleChannelWrapper(torch.nn.Module):
    """
    Wraps TemporalPopulationRegressor so GradCAM can operate on
    a multi-channel image (the mean across months).
    """
    def __init__(self, model, temporal_length=12):
        super().__init__()
        self.model = model
        self.T = temporal_length

    def forward(self, x):
        # x comes in as (B, C, H, W) from GradCAM
        # We replicate it T times to satisfy the temporal model
        x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1)  # (B, T, C, H, W)
        return self.model(x)


def get_gradcam_heatmap(model, image_tensor, target_layer_name="backbone.layer4"):
    """
    Generates a Grad-CAM heatmap.
    Args:
        model: TemporalPopulationRegressor instance
        image_tensor: torch.Tensor of shape (1, C, H, W) — typically the mean of all months
        target_layer_name: layer to hook for Grad-CAM
    Returns:
        grayscale_heatmap: np.array of shape (H, W) in [0, 1]
    """
    # Infer temporal length from model or default to 12
    T = getattr(model, "_last_T", 12)
    wrapped = TemporalToSingleChannelWrapper(model, temporal_length=T)
    wrapped.eval()
    # Resolve nested attribute path safely (e.g. "backbone.layer4")
    target_layer = model
    for attr in target_layer_name.split("."):
        target_layer = getattr(target_layer, attr)

    cam = GradCAM(model=wrapped, target_layers=[target_layer])
    targets = [RegressionTarget(scalar_index=0)]

    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    return grayscale_cam[0, :]
