# Track 5: Architecture Deepening

> **Session mappings**: Session 7 (Transformers, Attention), Session 8 (Advanced Architectures, Ensembles)

## Overview

This track upgrades the core model architecture beyond the baseline ResNet-18 + 1D temporal convolution. We implement and compare three architectural families:

1. **Self-Attention Temporal Aggregator** — Multi-head Transformer encoder over monthly feature sequences
2. **Deeper Backbones** — ResNet-34 and ResNet-50 spatial encoders
3. **Deep Ensembles** — Bagging N independently trained models for epistemic uncertainty

---

## Files Added/Modified

| File | Purpose |
|------|---------|
| `models/architectures.py` | New: `TemporalAttentionRegressor`, `DeepEnsemble`, backbone factory |
| `models/__init__.py` | Updated exports |
| `utils/train_utils.py` | New: shared `train_one_epoch` and `evaluate` |
| `train_v3_architecture.py` | New: unified training script supporting all architectures |
| `track5_analysis.py` | New: visualization and LaTeX table generation |

---

## 1. TemporalAttentionRegressor

### Motivation

The baseline 1D convolution only captures **local temporal neighbourhoods** (kernel size 3). Months that are far apart cannot directly interact. Self-attention removes this restriction — every month can attend to every other month, enabling the model to learn:

- Seasonal periodicity (e.g., Ramadan effects on lighting)
- Anomaly detection (e.g., COVID-19 lockdown months stand out)
- Long-range dependencies (e.g., pre-monsoon vs. post-monsoon)

### Architecture

```
Input: (B, T, C, H, W)
  |
  v
Shared ResNet backbone (per month)  ->  (B, T, feature_dim)
  |
  v
+ Learnable positional encoding (72 max months)
  |
  v
TransformerEncoder(n_layers=2, n_heads=4, d_model=512)
  |
  v
Mean pooling over time                ->  (B, feature_dim)
  |
  v
FFN(temporal_hidden) + Regression head ->  (B,)
```

### Key hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_heads` | 4 | Balances expressiveness vs. compute |
| `n_attn_layers` | 2 | Sufficient for 72-month sequences |
| `dropout` | 0.1 | Regularization on attention weights |
| `activation` | GELU | Smoother than ReLU; common in Transformers |

---

## 2. Backbone Depth Comparison

| Backbone | Params | Feature Dim | Use Case |
|----------|--------|-------------|----------|
| ResNet-18 | 11.4M | 512 | Baseline, fastest |
| ResNet-34 | 21.5M | 512 | Deeper features, moderate cost |
| ResNet-50 | 23.5M | 2048 | Bottleneck design, highest capacity |

**VRAM constraint**: With 4 channels and batch size 4:
- ResNet-18: ~3.2 GB
- ResNet-34: ~4.1 GB
- ResNet-50: ~5.8 GB (tight on 8GB)

---

## 3. DeepEnsemble

### Theory

Following Lakshminarayanan et al. (2017), we train N models with **different random seeds** (different weight initializations + different data shuffling). The ensemble prediction is:

```
mu_ensemble(x) = (1/N) * sum_i f_i(x)
var_ensemble(x) = (1/(N-1)) * sum_i (f_i(x) - mu)^2
```

`var_ensemble` represents **epistemic uncertainty** — uncertainty about which model is correct. This is distinct from MC Dropout's **aleatoric uncertainty** (inherent noise in the data).

### Implementation

```python
ensemble = DeepEnsemble([
    TemporalPopulationRegressor(...) for _ in range(5)
])
mean_pred, epistemic_var = ensemble.predict_with_uncertainty(x)
```

---

## Usage

### Single model (attention)
```bash
python train_v3_architecture.py \
    --architecture attention --backbone resnet18 \
    --ntl_dir data/aligned/ntl_monthly_aligned \
    --pop data/aligned/pop_aligned/pak_pop_2025_CN_100m_R2025A_v1_aligned.tif \
    --border_mask data/aligned/border_mask.tif \
    --built_up_path data/aligned/built_up_2020_ghsl_100m_aligned.tif \
    --built_up_volume_path data/aligned/built_up_volume_2020_ghsl_100m_aligned.tif \
    --pretrained --epochs 10 --batch_size 4
```

### Compare all architectures
```bash
python train_v3_architecture.py --compare_all \
    --ntl_dir ... --pop ... --border_mask ... \
    --built_up_path ... --built_up_volume_path ... \
    --pretrained --epochs 10 --batch_size 4
```

### Train ensemble
```bash
python train_v3_architecture.py --architecture ensemble --ensemble_n 5 \
    --ntl_dir ... --pop ... --border_mask ... \
    --built_up_path ... --built_up_volume_path ... \
    --pretrained --epochs 10 --batch_size 4
```

### Generate figures
```bash
python track5_analysis.py \
    --results_json outputs/track5_compare/architecture_comparison.json \
    --output_dir figures/track5
```

---

## Experimental Results

### Architecture Comparison (8 epochs, pretrained, same data split)

| Architecture | Params (M) | Test MAE | Test R | Best Val R | Notes |
|-------------|-----------|----------|--------|------------|-------|
| ResNet18_1DConv | 11.44 | 23,155 | 0.7276 | 0.9348 | Severe overfitting (ΔR = 0.21) |
| **ResNet34_1DConv** | 21.54 | 3,434 | **0.7498** | 0.7523 | Best single model, consistent |
| ResNet18_Attention | 13.92 | 3,399 | 0.5839 | 0.5884 | Underperformed vs. 1D conv |
| ResNet34_Attention | 24.02 | 3,273 | 0.1985 | 0.2784 | Collapsed — predicting near-mean |

**Winner: ResNet34_1DConv** — best test correlation with no validation gap.

### Deep Ensemble (5× ResNet34_1DConv, different random seeds)

| Metric | Best Single | Ensemble | Improvement |
|--------|------------|----------|-------------|
| Test R | 0.7498 | **0.8606** | **+0.11** |
| Test MAE | 3,434 | **3,016** | **−12%** |
| Individual R range | — | 0.60 – 0.85 | σ = 0.089 |

The ensemble mean prediction outperforms every individual model, confirming epistemic uncertainty reduction via model averaging. The spread of individual R values (0.60–0.85) demonstrates meaningful diversity in the ensemble.

### Key Findings

1. **Deeper backbones help generalization**: ResNet-34 outperforms ResNet-18 on test set, suggesting the baseline was underfitting spatial features.
2. **Attention did not help on this dataset**: Both attention variants underperformed their 1D-conv counterparts. Possible reasons: (a) 72-month sequences are too long for 2-layer Transformer without stronger regularization; (b) monthly NTL variation is smooth and local, so 1D conv's inductive bias is actually beneficial; (c) 8 epochs may be insufficient for attention to converge.
3. **Ensembles are highly effective**: A 5-model ensemble improves R by +0.11 with no additional architecture design — simply training multiple independent initializations.

---

## Future Work

1. **Cross-attention** between NTL and built-up features (multimodal fusion)
2. **Vision Transformer (ViT)** backbone instead of ResNet
3. **Neural Architecture Search (NAS)** for optimal temporal aggregator
4. **Test-time adaptation** — fine-tune on recent months only during inference
5. **Longer training + stronger regularization** for attention models (dropout 0.3, weight decay)
