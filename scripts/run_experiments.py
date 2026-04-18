"""
Experiment runner for GHSL 100m built-up ablation study.
Runs training + inference for a fixed matrix of configs and writes a summary CSV.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path(sys.executable)
TRAIN_PY = PROJECT_ROOT / "train.py"
INFER_PY = PROJECT_ROOT / "inference.py"
EVAL_PY = PROJECT_ROOT / "scripts" / "eval_by_density.py"

# Common training arguments
COMMON_TRAIN_ARGS = [
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
    "--seed", "42",
    "--num_workers", "0",
]

EXPERIMENTS = [
    {
        "name": "baseline",
        "train_extra": [],
        "infer_extra": [],
        "depends_on_ghsl": False,
    },
    {
        "name": "ghsl_scalar",
        "train_extra": [
            "--built_up_path", "data/aligned/built_up_2020_ghsl_100m_aligned.tif",
        ],
        "infer_extra": [
            "--built_up_path", "data/aligned/built_up_2020_ghsl_100m_aligned.tif",
        ],
        "depends_on_ghsl": True,
    },
    {
        "name": "ghsl_channel",
        "train_extra": [
            "--built_up_path", "data/aligned/built_up_2020_ghsl_100m_aligned.tif",
            "--built_up_as_channel",
        ],
        "infer_extra": [
            "--built_up_path", "data/aligned/built_up_2020_ghsl_100m_aligned.tif",
            "--built_up_as_channel",
        ],
        "depends_on_ghsl": True,
    },
    {
        "name": "pretrained_huber",
        "train_extra": [
            "--pretrained",
            "--loss_type", "huber",
            "--huber_beta", "1.0",
        ],
        "infer_extra": [
            "--pretrained",
        ],
        "depends_on_ghsl": False,
    },
    {
        "name": "pretrained_ghsl_channel_huber",
        "train_extra": [
            "--pretrained",
            "--loss_type", "huber",
            "--huber_beta", "1.0",
            "--built_up_path", "data/aligned/built_up_2020_ghsl_100m_aligned.tif",
            "--built_up_as_channel",
        ],
        "infer_extra": [
            "--pretrained",
            "--built_up_path", "data/aligned/built_up_2020_ghsl_100m_aligned.tif",
            "--built_up_as_channel",
        ],
        "depends_on_ghsl": True,
    },
    {
        "name": "pretrained_ghsl_vol_channel_huber",
        "train_extra": [
            "--pretrained",
            "--loss_type", "huber",
            "--huber_beta", "1.0",
            "--built_up_volume_path", "data/aligned/built_up_volume_2020_ghsl_100m_aligned.tif",
            "--built_up_as_channel",
        ],
        "infer_extra": [
            "--pretrained",
            "--built_up_volume_path", "data/aligned/built_up_volume_2020_ghsl_100m_aligned.tif",
            "--built_up_as_channel",
        ],
        "depends_on_ghsl": True,
    },
    {
        "name": "pretrained_ghsl_surface_vol_channel_huber",
        "train_extra": [
            "--pretrained",
            "--loss_type", "huber",
            "--huber_beta", "1.0",
            "--built_up_path", "data/aligned/built_up_2020_ghsl_100m_aligned.tif",
            "--built_up_volume_path", "data/aligned/built_up_volume_2020_ghsl_100m_aligned.tif",
            "--built_up_as_channel",
        ],
        "infer_extra": [
            "--pretrained",
            "--built_up_path", "data/aligned/built_up_2020_ghsl_100m_aligned.tif",
            "--built_up_volume_path", "data/aligned/built_up_volume_2020_ghsl_100m_aligned.tif",
            "--built_up_as_channel",
        ],
        "depends_on_ghsl": True,
    },
]


def run_cmd(cmd_list, desc):
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print(f"Command: {' '.join(str(c) for c in cmd_list)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd_list, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[ERROR] {desc} failed with exit code {result.returncode}")
        return False
    return True


def parse_infer_log(log_path: Path):
    """Extract MAE, RMSE, Pearson R from inference stdout log."""
    metrics = {
        "mae": None,
        "rmse": None,
        "r": None,
        "scaled_mae": None,
        "scaled_rmse": None,
        "scaled_r": None,
        "scale_factor": None,
    }
    if not log_path.exists():
        return metrics
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if "Pearson R=" in line and "Scaled Eval" not in line:
            parts = line.split("|")
            for p in parts:
                if "MAE=" in p:
                    metrics["mae"] = float(p.split("=")[1].strip())
                if "RMSE=" in p:
                    metrics["rmse"] = float(p.split("=")[1].strip())
                if "Pearson R=" in p:
                    metrics["r"] = float(p.split("=")[1].strip())
        if "[Scale] Applied post-hoc scale factor:" in line:
            metrics["scale_factor"] = float(line.split(":")[-1].strip())
        if "[Scaled Eval]" in line:
            parts = line.split("|")
            for p in parts:
                if "MAE=" in p:
                    metrics["scaled_mae"] = float(p.split("=")[1].strip())
                if "RMSE=" in p:
                    metrics["scaled_rmse"] = float(p.split("=")[1].strip())
                if "Pearson R=" in p:
                    metrics["scaled_r"] = float(p.split("=")[1].strip())
    return metrics


def main():
    ghsl_path = PROJECT_ROOT / "data" / "aligned" / "built_up_2020_ghsl_100m_aligned.tif"
    ghsl_vol_path = PROJECT_ROOT / "data" / "aligned" / "built_up_volume_2020_ghsl_100m_aligned.tif"
    ghsl_ready = ghsl_path.exists()
    ghsl_vol_ready = ghsl_vol_path.exists()

    results = []
    for exp in EXPERIMENTS:
        if exp["depends_on_ghsl"] and not ghsl_ready:
            print(f"[Skip] {exp['name']} — GHSL surface aligned raster not ready yet ({ghsl_path})")
            continue
        if "volume" in exp["name"] and not ghsl_vol_ready:
            print(f"[Skip] {exp['name']} — GHSL volume aligned raster not ready yet ({ghsl_vol_path})")
            continue

        out_dir = PROJECT_ROOT / "outputs" / f"exp_{exp['name']}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Training
        train_cmd = [str(PYTHON), str(TRAIN_PY)] + COMMON_TRAIN_ARGS + exp["train_extra"] + ["--output_dir", str(out_dir)]
        ok = run_cmd(train_cmd, f"Training {exp['name']}")
        if not ok:
            continue

        # Inference
        ckpt = out_dir / "best_model.pt"
        infer_log = out_dir / "infer_log.txt"
        infer_cmd = [
            str(PYTHON), str(INFER_PY),
            "--checkpoint", str(ckpt),
            "--output_dir", str(out_dir),
            "--evaluate",
            "--scale_to_gt",
        ] + exp["infer_extra"]

        # Run inference and capture stdout
        print(f"\n{'='*60}")
        print(f"Running inference for {exp['name']}")
        print(f"{'='*60}\n")
        with open(infer_log, "w", encoding="utf-8") as log_f:
            result = subprocess.run(infer_cmd, cwd=PROJECT_ROOT, stdout=log_f, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(f"[ERROR] Inference for {exp['name']} failed with exit code {result.returncode}")
            continue

        # Density-stratified evaluation on scaled prediction
        scaled_tif = out_dir / "pred_population_scaled.tif"
        if scaled_tif.exists():
            eval_cmd = [
                str(PYTHON), str(EVAL_PY),
                "--pred", str(scaled_tif),
                "--gt", "data/aligned/pop_aligned/pak_pop_2025_CN_100m_R2025A_v1_aligned.tif",
            ]
            print(f"\n{'='*60}")
            print(f"Running density-stratified eval for {exp['name']}")
            print(f"{'='*60}\n")
            subprocess.run(eval_cmd, cwd=PROJECT_ROOT)

        metrics = parse_infer_log(infer_log)
        metrics["experiment"] = exp["name"]
        results.append(metrics)

        # Print live summary
        print(f"[Result] {exp['name']} | R={metrics['r']} | Scaled R={metrics['scaled_r']} | Scale={metrics['scale_factor']}")

    # Write summary CSV
    summary_path = PROJECT_ROOT / "outputs" / "experiment_summary.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        header = "experiment,mae,rmse,r,scaled_mae,scaled_rmse,scaled_r,scale_factor\n"
        f.write(header)
        for r in results:
            line = (
                f"{r['experiment']},"
                f"{r['mae'] if r['mae'] is not None else ''},"
                f"{r['rmse'] if r['rmse'] is not None else ''},"
                f"{r['r'] if r['r'] is not None else ''},"
                f"{r['scaled_mae'] if r['scaled_mae'] is not None else ''},"
                f"{r['scaled_rmse'] if r['scaled_rmse'] is not None else ''},"
                f"{r['scaled_r'] if r['scaled_r'] is not None else ''},"
                f"{r['scale_factor'] if r['scale_factor'] is not None else ''}\n"
            )
            f.write(line)

    print(f"\n[Done] Summary written to {summary_path}")
    for r in results:
        print(f"  {r['experiment']:40s}  R={r['r']}  ScaledR={r['scaled_r']}  scale={r['scale_factor']}")


if __name__ == "__main__":
    main()
