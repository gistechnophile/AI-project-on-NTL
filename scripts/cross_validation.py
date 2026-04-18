"""
3-Fold Cross-Validation for the best model config.
Trains on 3 different random splits and reports mean ± std metrics.
"""
import os
import sys
import subprocess
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path(sys.executable)
TRAIN_PY = PROJECT_ROOT / "train.py"
INFER_PY = PROJECT_ROOT / "inference.py"
EVAL_PY = PROJECT_ROOT / "scripts" / "eval_by_density.py"

# Best model config
COMMON_ARGS = [
    "--ntl_dir", "data/aligned/ntl_monthly_aligned",
    "--pop", "data/aligned/pop_aligned/pak_pop_2025_CN_100m_R2025A_v1_aligned.tif",
    "--border_mask", "data/aligned/border_mask.tif",
    "--epochs", "10",
    "--batch_size", "8",
    "--lr", "0.001",
    "--rel_mae_weight", "0.1",
    "--valid_threshold", "0.3",
    "--patch_size", "32",
    "--stride", "16",
    "--num_workers", "0",
    "--pretrained",
    "--loss_type", "huber",
    "--huber_beta", "1.0",
    "--built_up_path", "data/aligned/built_up_2020_ghsl_100m_aligned.tif",
    "--built_up_volume_path", "data/aligned/built_up_volume_2020_ghsl_100m_aligned.tif",
    "--built_up_as_channel",
]

FOLDS = [42, 123, 999]  # 3 different seeds

results = []

for fold_idx, seed in enumerate(FOLDS):
    out_dir = PROJECT_ROOT / "outputs" / f"cv_fold_{fold_idx}_seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx+1}/3 | Seed {seed}")
    print(f"{'='*60}")
    
    # Train
    train_cmd = [str(PYTHON), str(TRAIN_PY)] + COMMON_ARGS + [
        "--seed", str(seed),
        "--output_dir", str(out_dir),
    ]
    print("Training...")
    result = subprocess.run(train_cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[ERROR] Fold {fold_idx+1} training failed")
        continue
    
    # Inference
    ckpt = out_dir / "best_model.pt"
    infer_cmd = [
        str(PYTHON), str(INFER_PY),
        "--checkpoint", str(ckpt),
        "--output_dir", str(out_dir),
        "--evaluate",
        "--scale_to_gt",
        "--built_up_path", "data/aligned/built_up_2020_ghsl_100m_aligned.tif",
        "--built_up_volume_path", "data/aligned/built_up_volume_2020_ghsl_100m_aligned.tif",
        "--built_up_as_channel",
        "--pretrained",
    ]
    print("Running inference...")
    infer_log = out_dir / "infer_log.txt"
    with open(infer_log, "w", encoding="utf-8") as f:
        subprocess.run(infer_cmd, cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT)
    
    # Parse metrics from log
    metrics = {"fold": fold_idx, "seed": seed, "r": None, "scaled_r": None, "scale": None}
    if infer_log.exists():
        text = infer_log.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if "Pearson R=" in line and "Scaled Eval" not in line:
                parts = line.split("|")
                for p in parts:
                    if "Pearson R=" in p:
                        metrics["r"] = float(p.split("=")[1].strip())
            if "[Scale] Applied post-hoc scale factor:" in line:
                metrics["scale"] = float(line.split(":")[-1].strip())
            if "[Scaled Eval]" in line:
                parts = line.split("|")
                for p in parts:
                    if "Pearson R=" in p:
                        metrics["scaled_r"] = float(p.split("=")[1].strip())
    
    results.append(metrics)
    print(f"Fold {fold_idx+1} results: R={metrics['r']} | ScaledR={metrics['scaled_r']} | Scale={metrics['scale']}")

# Summary
print(f"\n{'='*60}")
print("CROSS-VALIDATION SUMMARY")
print(f"{'='*60}")
import numpy as np
rs = [m["r"] for m in results if m["r"] is not None]
scaled_rs = [m["scaled_r"] for m in results if m["scaled_r"] is not None]
scales = [m["scale"] for m in results if m["scale"] is not None]

if rs:
    print(f"R           : {np.mean(rs):.4f} ± {np.std(rs):.4f}")
if scaled_rs:
    print(f"Scaled R    : {np.mean(scaled_rs):.4f} ± {np.std(scaled_rs):.4f}")
if scales:
    print(f"Scale factor: {np.mean(scales):.4f} ± {np.std(scales):.4f}")

summary_path = PROJECT_ROOT / "outputs" / "cv_summary.json"
with open(summary_path, "w") as f:
    json.dump({"results": results, "summary": {
        "r_mean": float(np.mean(rs)) if rs else None,
        "r_std": float(np.std(rs)) if rs else None,
        "scaled_r_mean": float(np.mean(scaled_rs)) if scaled_rs else None,
        "scaled_r_std": float(np.std(scaled_rs)) if scaled_rs else None,
    }}, f, indent=2)
print(f"\nSaved summary to {summary_path}")
