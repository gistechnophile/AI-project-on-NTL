# PakLight-Pop

**AI-Native Population Estimation for Pakistan via Temporal Nighttime Lights + 3D Built-Up Data**

> *Course Application Project for "From First Principles to Superintelligence"*  
> *Concept alignment: Data Quality (S4), CNN+ResNet+GradCAM (S3), Sequence Modeling (S6), Compute Reality (S5), RAG (S8), Agents (S9), Agentic Coding (S11)*

---

## 1. Engineering Problem

Traditional census data is expensive and updated infrequently. Nighttime light (NTL) satellite imagery correlates with human settlement density, but raw radiance alone produces high error in rural/dark regions and **oversaturated urban cores** — a dense Karachi slum and a dense high-rise district can look equally bright to the satellite.

**PakLight-Pop** solves this by learning a **multimodal temporal mapping** (72 months of NTL + static population proxy + GHSL built-up surface + GHSL built-up volume) from aligned WorldPop 2025 ground truth. The volume channel is the key innovation: it provides 3D building information that breaks the NTL saturation ceiling.

---

## 2. System Architecture

```
72 monthly NTL GeoTIFFs (Pakistan, 2020–2025)
      + WorldPop 2025 500m
      + GHSL Built-Up Surface 100m (2020)
      + GHSL Built-Up Volume 100m (2020)
              │
              ▼
┌─────────────────────────────┐
│  Data Alignment & QA        │  <-- rasterio, common 500m grid
│  Border mask + valid-patch  │  <-- quality_audit.json
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│ TemporalPopulationRegressor │  <-- Shared ResNet-18 encoder
│   4-channel input           │      + 1D temporal conv
│   [NTL, POP_proxy,          │  <-- Pretrained ImageNet weights
│    BU_surface, BU_volume]   │  <-- Huber loss (urban-core robust)
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Inference + Post-hoc Scale  │  <-- Optional --scale_to_gt
│ Density-stratified eval     │  <-- Rural / Peri-urban / Urban core
└─────────────────────────────┘
```

---

## 3. Best Model (v2.0)

**Checkpoint:** `outputs/best_model.pt`  
**Experiment:** `pretrained_ghsl_surface_vol_channel_huber`

| Metric | Value | Notes |
|--------|-------|-------|
| **Overall Pearson R** | **0.8811** | Highest spatial correlation achieved |
| **Overall MAE** | **2.24** | Per 500m pixel (people count) |
| **Scale factor** | **1.0116** | Essentially exact national total (7.45M) |
| **Rural R** | **0.9583** | Nearly perfect |
| **Peri-urban R** | **0.8886** | Excellent |
| **Urban core R** | **0.3393** | Still the hardest class, but massively improved |
| **Urban core bias** | **−28.2** | Down from −56.8 (surface only) and −175.9 (baseline) |

### Ablation Study Summary

| Experiment | R | Scale | Urban Core Bias | Verdict |
|------------|---|-------|-----------------|---------|
| **pretrained_ghsl_surface_vol_channel_huber** | **0.8811** | **1.01** | **−28.2** | 🏆 **Best** |
| pretrained_ghsl_channel_huber (3-ch surface) | 0.8746 | 1.21 | −56.8 | Good |
| pretrained_huber (no built-up) | 0.7668 | 1.20 | — | Solid baseline |
| ghsl_scalar | 0.6551 | 0.23 | — | Marginal |
| baseline | 0.6420 | 0.24 | — | Weak |
| ghsl_channel (no pretrain) | 0.5890 | 0.62 | — | Worse |
| pretrained_ghsl_vol_channel_huber (vol only) | 0.6118 | 1.23 | −113.6 | ❌ Volume alone hurts |

**Key insight:** GHSL Building **Surface** helps (+0.11 R with pretraining). Building **Volume** alone confuses the model (−0.26 R). But **Surface + Volume together** creates synergy — volume provides the 3D context needed to break NTL saturation, while surface anchors the spatial footprint.

---

## 4. Project Structure

```
paklight-pop/
├── app/
│   └── streamlit_app.py          # Interactive web application
├── data_pipeline/
│   ├── align_rasters.py          # Reproject all rasters to common grid
│   ├── quality_audit.py          # Quality framework
│   ├── monthly_utils.py          # Discover monthly NTL files
│   └── dataset.py                # PyTorch Dataset (4-channel support)
├── models/
│   ├── population_cnn.py         # TemporalPopulationRegressor
│   └── explainability.py         # Grad-CAM wrapper
├── scripts/
│   ├── download_ghsl_builtup.py  # Download & align GHSL Surface
│   ├── download_ghsl_volume.py   # Download & align GHSL Volume
│   ├── run_experiments.py        # Automated ablation study runner
│   ├── compare_builtup.py        # Visualize old vs new built-up
│   └── eval_by_density.py        # Stratified density evaluation
├── train.py                      # Training script (reproducible)
├── inference.py                  # Inference + evaluation script
├── prepare_data.py               # One-command alignment + audit
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 5. Quick Start

### Step 0: Environment Setup (Windows)

Because geospatial packages (`rasterio`, `torch`) have heavy binary dependencies on Windows, **Conda is strongly recommended**:

```bash
conda env create -f environment.yml
conda activate paklight-pop
```

Or, if you prefer `venv` and have GDAL pre-installed:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 1: Prepare Your Data

Place your files in a `data/` folder:

```
data/
├── ntl_monthly/                  # 72 monthly VIIRS NTL .tif files
├── pop/
│   └── pak_pop_2025.tif          # WorldPop 2025 population grid
└── border_mask.tif               # Optional: country border mask
```

Run alignment and quality audit:

```bash
python prepare_data.py \
  --ntl_dir data/ntl_monthly \
  --pop data/pop/pak_pop_2025.tif \
  --output data/aligned
```

### Step 2: Download GHSL Built-Up Data (Optional but Recommended)

```bash
# Built-up surface (100m)
python scripts/download_ghsl_builtup.py

# Built-up volume (100m)
python scripts/download_ghsl_volume.py
```

These scripts download the global GHSL R2023A datasets, extract the Pakistan region, and align them to the 500m study grid.

### Step 3: Train the Best Model

**Best configuration** (4 channels, pretrained, Huber loss):

```bash
python train.py \
  --ntl_dir data/aligned/ntl_monthly_aligned \
  --pop data/aligned/pop_aligned/pak_pop_2025.tif \
  --border_mask data/aligned/border_mask.tif \
  --built_up_path data/aligned/built_up_2020_ghsl_100m_aligned.tif \
  --built_up_volume_path data/aligned/built_up_volume_2020_ghsl_100m_aligned.tif \
  --built_up_as_channel \
  --pretrained \
  --loss_type huber \
  --huber_beta 1.0 \
  --epochs 10 \
  --batch_size 8 \
  --lr 0.001 \
  --output_dir checkpoints
```

**Simpler baseline** (2 channels, no built-up):

```bash
python train.py \
  --ntl_dir data/aligned/ntl_monthly_aligned \
  --pop data/aligned/pop_aligned/pak_pop_2025.tif \
  --border_mask data/aligned/border_mask.tif \
  --epochs 10 \
  --batch_size 8 \
  --output_dir checkpoints
```

Checkpoint saved to: `checkpoints/best_model.pt`

### Step 4: Run Inference

```bash
python inference.py \
  --ntl_dir data/aligned/ntl_monthly_aligned \
  --pop data/aligned/pop_aligned/pak_pop_2025.tif \
  --border_mask data/aligned/border_mask.tif \
  --checkpoint checkpoints/best_model.pt \
  --built_up_path data/aligned/built_up_2020_ghsl_100m_aligned.tif \
  --built_up_volume_path data/aligned/built_up_volume_2020_ghsl_100m_aligned.tif \
  --built_up_as_channel \
  --pretrained \
  --evaluate \
  --scale_to_gt \
  --output_dir outputs
```

### Step 5: Density-Stratified Evaluation

```bash
python scripts/eval_by_density.py \
  --pred outputs/pred_population_scaled.tif \
  --gt data/aligned/pop_aligned/pak_pop_2025.tif
```

### Step 6: Launch the Interactive App

```bash
# Using the project virtual environment (Windows)
C:\pakvenv\Scripts\python.exe -m streamlit run app/streamlit_app.py

# Or if activated:
streamlit run app/streamlit_app.py
```

Open your browser at `http://localhost:8501`. The app auto-detects the checkpoint configuration (channels, pretrained, GHSL paths) and loads the 4-channel best model automatically.

---

## 6. Course Concepts Applied

| Feature | Course Session | Evidence in Code |
|---|---|---|
| **ResNet + CNN** | S3: Classic Foundations | `models/population_cnn.py` — ResNet-18 backbone |
| **Sequence Modeling** | S6: Sequence Modeling | `models/population_cnn.py` — 1D temporal conv over 72 months |
| **Grad-CAM Explainability** | S3: CNN Interpretability | `models/explainability.py` — heatmaps for engineering integrity |
| **5-Dimension Data Quality** | S4: Data Thinking | `data_pipeline/quality_audit.py` — completeness, consistency, accuracy, relevance (MI), timeliness |
| **Distribution Shift (MMD)** | S4: Distribution Mismatch | `data_pipeline/quality_audit.py` — `compute_mmd()` |
| **FLOP Estimation** | S5: Compute Reality | `models/population_cnn.py` — `count_flops()` |
| **RAG Literature Retrieval** | S8: Retrieval-Augmented Generation | `report_engine/rag_engine.py` — ChromaDB + embeddings |
| **ReAct Agentic Report** | S9: Agents and Tool Use | `report_engine/agent_reporter.py` — perceive → reason → act (RAG) → observe |
| **Agentic Coding Workflow** | S11: Claude Code | Entire scaffold generated via agentic loop (read → write → verify) |
| **Reproducibility** | S1: The Rules | `requirements.txt`, deterministic seeds in `train.py`, checkpoint metadata |
| **Multimodal Fusion** | S10: Multimodal AI | 4-channel input: NTL + POP proxy + GHSL Surface + GHSL Volume |
| **Loss Engineering** | S7: Optimization | Huber loss for urban-core outlier robustness |

---

## 7. Extending the Project

### For Other Years
If you have NTL for 2012, 2015, 2020, etc., simply align each year to the same WorldPop grid and run inference through the trained model to generate **temporal population estimates**.

### Switching to Black Marble NTL
The current model uses NOAA VIIRS monthly composites. For even cleaner signals, swap to **NASA Black Marble** (VNP46A3) — better atmospheric correction and reduced stray light. The pipeline supports any GeoTIFF stack.

### Adding OSM Road Density
OpenStreetMap road intersection density is a free, strong correlate of urban population. Extract OSM roads, rasterize to the study grid, and add as a 5th channel. **Note:** The Overpass API can block requests when VPN is in Auto/TUN mode; use Global VPN mode or download Geofabrik OSM extracts as an alternative.

### Fine-Grained Output
Replace the scalar regression head with a **U-Net decoder** to produce pixel-wise population density maps instead of patch totals.

---

## 8. Troubleshooting

**`ImportError: DLL load failed while importing _C` (torch)**
- Install the [Microsoft C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).
- Or use the Conda environment (`environment.yml`) which handles Windows binaries better.

**`No module named rasterio`**
- `rasterio` requires GDAL. Use Conda: `conda install -c conda-forge rasterio`.

**GHSL download fails with ConnectionResetError**
- The JRC FTP server can be sensitive to proxy routing. If using a VPN TUN proxy, switch to **Global mode** (not Auto) for large file transfers.

**Streamlit map not showing**
- Ensure you have an active internet connection for the Leaflet tiles, or run offline with a local tile server.

---

## 9. License & Attribution

This is a student course project. All scientific findings are grounded in retrievable literature via the RAG engine. Model weights and training configs are versioned for reproducibility.

**Data sources:**
- WorldPop 2025: [worldpop.org](https://www.worldpop.org)
- GHSL R2023A (Built-Up Surface, Volume): [JRC GHSL](https://human-settlement.emergency.copernicus.eu)
- VIIRS Nighttime Lights: [NOAA Earth Observation Group](https://eogdata.mines.edu)
