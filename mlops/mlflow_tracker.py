"""
MLOps: MLflow Experiment Tracking Integration

Tracks every training run with:
- Hyperparameters (auto-logged)
- Metrics per epoch (train/val loss, R, MAE, scale factor)
- Model artifacts (checkpoint files)
- Dataset lineage (input file hashes, preprocessing params)
- System metrics (GPU memory, training time)

Usage:
    from mlflow_tracker import ExperimentTracker
    
    with ExperimentTracker(experiment_name="paklight-pop") as tracker:
        tracker.log_params({"lr": 1e-3, "batch_size": 8, "backbone": "resnet18"})
        for epoch in range(epochs):
            # ... train ...
            tracker.log_metrics({"train_loss": loss, "val_r": r}, step=epoch)
        tracker.log_model(model, "best_model")
        tracker.log_artifact("outputs/prediction_maps.png")
"""

import os
import time
import hashlib
import tempfile
from typing import Dict, Any, Optional
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn


class ExperimentTracker:
    """
    Context manager for MLflow experiment tracking.
    Handles run creation, parameter logging, metric logging, and artifact storage.
    """
    
    def __init__(
        self,
        experiment_name: str = "paklight-pop",
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        self.experiment_name = experiment_name
        # Windows paths need file:// prefix for MLflow
        if tracking_uri is None:
            p = os.path.join(os.getcwd(), "mlruns").replace("\\", "/")
            self.tracking_uri = f"file:///{p}"
        else:
            self.tracking_uri = tracking_uri
        self.run_name = run_name or f"run_{int(time.time())}"
        self.tags = tags or {}
        self.run_id = None
        self.start_time = None
    
    def __enter__(self):
        """Start MLflow run."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        self.run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
        self.run_id = self.run.info.run_id
        self.start_time = time.time()
        
        print(f"[MLflow] Started run: {self.run_name}")
        print(f"[MLflow] Tracking URI: {self.tracking_uri}")
        print(f"[MLflow] Experiment: {self.experiment_name}")
        
        # Auto-log system info
        self._log_system_info()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End MLflow run."""
        duration = time.time() - self.start_time
        mlflow.log_metric("total_duration_seconds", duration)
        
        if exc_type is not None:
            mlflow.set_tag("run_status", "FAILED")
            mlflow.set_tag("error_type", exc_type.__name__)
            mlflow.set_tag("error_message", str(exc_val))
            print(f"[MLflow] Run FAILED after {duration:.1f}s: {exc_val}")
        else:
            mlflow.set_tag("run_status", "COMPLETED")
            print(f"[MLflow] Run COMPLETED in {duration:.1f}s")
        
        mlflow.end_run()
    
    def _log_system_info(self):
        """Log system and environment information."""
        import platform
        import torch
        
        mlflow.set_tag("python_version", platform.python_version())
        mlflow.set_tag("pytorch_version", torch.__version__)
        mlflow.set_tag("cuda_available", str(torch.cuda.is_available()))
        
        if torch.cuda.is_available():
            mlflow.set_tag("gpu_name", torch.cuda.get_device_name(0))
            mlflow.set_tag("gpu_memory_gb", str(torch.cuda.get_device_properties(0).total_memory / 1e9))
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters. Flattens nested dicts."""
        flat = self._flatten_dict(params)
        for k, v in flat.items():
            mlflow.log_param(k, v)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics at a given step (epoch)."""
        for k, v in metrics.items():
            if v is not None and not (isinstance(v, float) and (v != v)):  # skip NaN
                mlflow.log_metric(k, v, step=step)
    
    def log_model(self, model: nn.Module, artifact_path: str = "model"):
        """Log PyTorch model as MLflow artifact."""
        mlflow.pytorch.log_model(model, artifact_path)
        print(f"[MLflow] Model logged to {artifact_path}")
    
    def log_checkpoint(self, model: nn.Module, filepath: str, epoch: int, metrics: Dict[str, float]):
        """Save and log a model checkpoint."""
        # Save locally first
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, filepath)
        
        # Log to MLflow
        mlflow.log_artifact(filepath, artifact_path="checkpoints")
        print(f"[MLflow] Checkpoint logged: {filepath}")
    
    def log_artifact(self, filepath: str, artifact_path: Optional[str] = None):
        """Log any file (plots, configs, etc.) as artifact."""
        if os.path.exists(filepath):
            mlflow.log_artifact(filepath, artifact_path=artifact_path)
            print(f"[MLflow] Artifact logged: {filepath}")
    
    def log_dataset_lineage(self, file_paths: Dict[str, str], preprocessing: Optional[Dict] = None):
        """
        Log dataset provenance with file hashes.
        Enables reproducibility: you can verify exactly which data was used.
        """
        for name, path in file_paths.items():
            if os.path.exists(path):
                file_hash = self._compute_hash(path)
                mlflow.log_param(f"data_{name}_path", path)
                mlflow.log_param(f"data_{name}_hash", file_hash)
                mlflow.log_param(f"data_{name}_size_mb", os.path.getsize(path) / (1024**2))
        
        if preprocessing:
            self.log_params({f"preprocess_{k}": v for k, v in preprocessing.items()})
    
    def log_figure(self, fig, artifact_path: str = "figure.png"):
        """Log a matplotlib figure directly."""
        import matplotlib.pyplot as plt
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(tmp.name, artifact_path="figures")
            os.unlink(tmp.name)
    
    @staticmethod
    def _flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flatten nested dictionaries for MLflow params."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ExperimentTracker._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    @staticmethod
    def _compute_hash(filepath: str, algorithm: str = "md5") -> str:
        """Compute file hash for data lineage tracking."""
        h = hashlib.new(algorithm)
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


# ------------------------------------------------------------------------------
# Compare runs utility
# ------------------------------------------------------------------------------
def list_experiments(tracking_uri: Optional[str] = None):
    """List all experiments and their best runs."""
    if tracking_uri is None:
        p = os.path.join(os.getcwd(), "mlruns").replace("\\", "/")
        tracking_uri = f"file:///{p}"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    
    for exp in client.search_experiments():
        print(f"\nExperiment: {exp.name} (ID: {exp.experiment_id})")
        runs = client.search_runs(exp.experiment_id, order_by=["metrics.val_r DESC"], max_results=5)
        for run in runs:
            val_r = run.data.metrics.get("val_r", "N/A")
            print(f"  Run: {run.info.run_name} | val_r={val_r} | Status={run.info.status}")


# ------------------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== MLflow Tracker Self-Test ===")
    
    with ExperimentTracker(
        experiment_name="test-paklight",
        run_name="test_run",
        tags={"version": "v2.0", "author": "AI-agent"},
    ) as tracker:
        # Log params
        tracker.log_params({
            "model": "resnet18",
            "lr": 1e-3,
            "batch_size": 8,
            "epochs": 10,
            "optimizer": {"name": "AdamW", "betas": [0.9, 0.999]},
        })
        
        # Simulate training
        for epoch in range(3):
            time.sleep(0.1)
            tracker.log_metrics({
                "train_loss": 0.5 - epoch * 0.1,
                "val_loss": 0.6 - epoch * 0.08,
                "val_r": 0.6 + epoch * 0.1,
                "val_mae": 5.0 - epoch * 0.5,
            }, step=epoch)
        
        # Log a dummy model
        dummy_model = torch.nn.Linear(10, 1)
        tracker.log_model(dummy_model, "model")
    
    print("\n=== Listing Experiments ===")
    list_experiments()
