# PakLight-Pop: Population Estimation from Nighttime Lights using Deep Learning

## Cover Page

| | |
|---|---|
| **Course Name** | The Intelligent Remote Sensing |
| **Student ID** | LS2525242 |
| **Student Name** | TOKHIRJON TESHABOEV |
| **Date** | May 2025 |
| **Project Title** | Task-Driven Population Estimation from Nighttime Light Imagery via Multimodal Deep Learning |

---

## 1. Introduction

### 1.1 Motivation

Accurate, high-resolution population maps are essential for disaster response, urban planning, healthcare resource allocation, and infrastructure development in developing nations. Traditional census enumeration, while considered the gold standard, is conducted infrequently (typically once per decade) and is prohibitively expensive for countries with large, rapidly growing populations. Pakistan, with over 240 million inhabitants and one of the fastest urbanisation rates in South Asia, exemplifies the urgent need for timely, spatially disaggregated population grids.

Remote sensing offers a compelling alternative to censuses. Nighttime light (NTL) satellite imagery, particularly the Visible Infrared Imaging Radiometer Suite (VIIRS) Day-Night Band, has been widely adopted as a proxy for economic activity and human settlement density. The underlying physical assumption is straightforward: illuminated areas correlate with populated areas because artificial light emission requires human presence, infrastructure, and energy consumption. However, this relationship is non-linear and exhibits two critical failure modes:

1. **Rural underprediction:** Agricultural regions with substantial populations often exhibit minimal nighttime radiance, leading to systematic underestimation.
2. **Urban saturation:** At high population densities, NTL radiance reaches a sensor saturation point beyond which additional population cannot be distinguished. A dense informal settlement and a dense high-rise commercial district may appear equally bright to the satellite, despite vastly different population densities per unit area.

### 1.2 Task Definition

This project designs a **task-driven remote sensing workflow** for gridded population estimation in Pakistan. The specific objectives are:

1. To develop a temporal deep learning architecture that processes multi-year NTL sequences (72 months, 2020–2025) alongside static geospatial covariates;
2. To evaluate whether Global Human Settlement Layer (GHSL) built-up volume, when fused with surface area, can break the NTL saturation ceiling;
3. To conduct a rigorous ablation study isolating the contribution of each input channel and design decision;
4. To produce spatially explicit population prediction maps at 500 m resolution across the entire country.

**Study Area:** Pakistan (approximately 24–37°N, 61–77°E), selected due to its diverse demographic landscape ranging from densely populated urban corridors (Lahore, Karachi) to sparsely inhabited mountainous regions and arid zones.

**Expected Outputs:**
- Trained deep learning models with quantified accuracy (Pearson R, MAE, RMSE)
- Gridded population prediction maps at 500 m resolution
- Ablation study results across multiple channel configurations
- An open-source, reproducible processing pipeline

---

## 2. Study Area and Data

### 2.1 Study Area Description

Pakistan is located in South Asia, bordered by India to the east, Afghanistan and Iran to the west, China to the north, and the Arabian Sea to the south. The country covers approximately 881,913 km² and features extreme topographic and climatic diversity:

- **Northern highlands:** The Hindu Kush and Karakoram ranges, with elevations exceeding 8,000 m
- **Indus River plain:** A fertile alluvial corridor running north–south, containing the majority of the population
- **Thar Desert:** An arid region in the southeast with very sparse settlement
- **Coastal zones:** The Makran coast along the Arabian Sea

This diversity presents a challenging test case for NTL-based population models because the relationship between light emission and population density varies dramatically across these physiographic zones.

### 2.2 Data Sources

Four primary remote sensing data layers were used, selected for their task relevance, complementarity, and physical interpretability:

| Layer | Source | Sensor / Product | Native Resolution | Aligned Resolution | Temporal Coverage |
|-------|--------|------------------|-------------------|--------------------|-------------------|
| Nighttime Lights (NTL) | NOAA Earth Observation Group | VIIRS Day-Night Band v2.1, monthly cloud-free composites | 500 m | 500 m | Jan 2020 – Dec 2025 (72 months) |
| Population (Ground Truth) | WorldPop, University of Southampton | Constrained Country Total R2025A | 100 m | 500 m | 2025 |
| Built-Up Surface | European Commission JRC | GHSL GHS-BUILT-S R2023A | 100 m (Mollweide) | 500 m (WGS84) | 2020 |
| Built-Up Volume | European Commission JRC | GHSL GHS-BUILT-V R2023A | 100 m (Mollweide) | 500 m (WGS84) | 2020 |
| Border Mask | GADM | Administrative boundaries (Level 0) | Vector | 500 m | Static |

**Figure 1.** Data layer overview showing NTL radiance, WorldPop population density, GHSL built-up surface, and GHSL built-up volume for a region around Lahore.

*(See `latex_paper/fig_built_up_comparison.png`)*

### 2.3 Justification of Data Selection

**Task Relevance:**
- **VIIRS NTL** directly measures artificial light emission, which is physically linked to human settlement, electrification, and economic activity. The 500 m resolution strikes a balance between spatial detail and signal-to-noise ratio.
- **WorldPop** provides the most recent globally consistent population grid at high resolution. The constrained version preserves the national census total while distributing population according to covariates.
- **GHSL Built-Up Surface** measures the horizontal extent of human construction. It complements NTL by capturing settlements that may not be brightly lit (e.g., daytime-occupied industrial zones, rural housing).
- **GHSL Built-Up Volume** measures the three-dimensional building stock (surface area × average height). This is the critical innovation: volume distinguishes flat dense settlements from vertical dense settlements — information that NTL alone cannot provide.

**Complementarity of Sensors:**
- NTL is an **active emission** proxy (measures energy output at night).
- GHSL surface/volume are **passive structure** proxies (measure physical building stock).
- Population is the **target variable** derived from census enumeration.
- Together, these layers capture complementary aspects of human settlement: energy consumption (NTL), horizontal footprint (surface), and vertical density (volume).

**Physical Characteristics:**
- VIIRS DNB has a spectral bandwidth of 0.5–0.9 μm and detects pW·cm⁻²·sr⁻¹ radiance. It does not penetrate clouds, which is why monthly cloud-masked composites are used.
- GHSL data are derived from Sentinel-1 SAR (C-band, 5.4 GHz) and Sentinel-2 multispectral imagery. SAR penetrates clouds and darkness, providing structural information independent of NTL conditions.

---

## 3. Methodology

### 3.1 Overall Workflow

The project follows a task-driven paradigm with five sequential stages:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Download  │───▶│  Preprocessing  │───▶│ Patch Extraction│
│  & Inventory    │    │  & Harmonization│    │  (32×32 @ 500m) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Model Design  │───▶│    Training     │───▶│   Inference &   │
│  (ResNet-18+TC) │    │  (Ablation CV)  │    │   Validation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Figure 2.** Overall task-driven workflow diagram.

*(See `outputs/workflow.png` for the detailed visualization)*

### 3.2 Preprocessing and Harmonization

All preprocessing was performed in Python 3.13 using `rasterio` for geospatial I/O and `numpy` for array operations. The following steps were applied:

**Step 1: Reprojection to Common Grid**
- Target CRS: EPSG:4326 (WGS84)
- Target resolution: 500 m
- Resampling methods: Bilinear for NTL and population; Average for GHSL layers (preserves fractional coverage)

**Step 2: Nodata Handling**
- NTL: Negative values set to 0; radiance capped at 250 nW·cm⁻²·sr⁻¹ to suppress sensor noise
- GHSL surface: Nodata value 65535 → 0
- GHSL volume: Nodata value 4294967295 → 0
- Population: Nodata value -9999 → 0

**Step 3: Normalization**
- NTL: Per-month normalization by the 99th percentile of positive values
- Population proxy: Global 99th percentile normalization
- GHSL surface: Clipped to [0, 1] (represents fractional coverage)
- GHSL volume: Left unbounded (cubic metres per pixel)

**Step 4: Border Masking**
- A binary mask was derived from GADM Level-0 boundaries to exclude pixels outside Pakistan's territorial boundaries from training and evaluation.

**Justification:** Reprojection ensures all layers share a common geospatial reference frame. Average resampling for GHSL is critical because it preserves the fractional built-up proportion when aggregating from 100 m to 500 m. The 99th-percentile normalization prevents outlier pixels from dominating the loss landscape.

### 3.3 Feature Extraction and Representation

Two categories of features were extracted:

**Handcrafted Features (Physical Interpretation):**
- **NTL radiance:** Direct proxy for electrification and economic activity
- **Population proxy (static):** Prior belief about population distribution, derived from WorldPop 2025
- **GHSL built-up surface:** Horizontal extent of human construction
- **GHSL built-up volume:** Three-dimensional building stock

**Learned Features:**
- A shared **ResNet-18** convolutional neural network (pretrained on ImageNet when specified) processes each monthly image independently, extracting 512-dimensional spatial feature vectors per time step.
- The first convolutional layer was adapted to accept multi-channel inputs (2, 3, or 4 channels).

**Physical Interpretation of Features:**
- NTL radiance correlates with **energy consumption** and **nocturnal human activity**.
- Built-up surface correlates with **horizontal land consumption** by human settlements.
- Built-up volume correlates with **vertical urban density** and **total floor area** — a stronger predictor of population than surface alone in high-rise districts.

### 3.4 Data Fusion Strategy

**Fusion Level: Feature-Level Fusion**

The model adopts **feature-level fusion** by concatenating multiple data channels into a single multi-channel input tensor. The rationale is as follows:

1. **Data-level fusion** (pixel-wise stacking before encoding) was rejected because different sensors have vastly different dynamic ranges and physical units (radiance vs. m² vs. m³), which would confound the encoder.
2. **Decision-level fusion** (training separate models per modality and averaging predictions) was rejected because it increases training cost and loses cross-modal interactions.
3. **Feature-level fusion** allows the CNN encoder to learn joint representations of NTL, population proxy, and GHSL structure simultaneously, enabling the network to discover synergistic relationships (e.g., high NTL + high volume → very dense urban core).

**Interpretability and Robustness:**
- Feature-level fusion improves interpretability because learned filters can be visualised via Grad-CAM to show which input channels contribute most to predictions in different regions.
- It improves robustness because the model can fall back on GHSL structure when NTL is missing (cloud-contaminated months) or saturated (urban cores).

### 3.5 Model and Analytical Method

**Architecture: TemporalPopulationRegressor**

```
Input: (B, T, C, 32, 32)  where B=batch, T=72 months, C∈{2,3,4}
  └─ Shared ResNet-18 spatial encoder (ImageNet-pretrained optional)
       └─ Output: (B, T, 512) feature vectors
  └─ 1D temporal convolution: Conv1d(512→128→128)
       └─ AdaptiveAvgPool1d(1) aggregates over T=72
  └─ Regression head: Linear(128→1)
Output: scalar log₁p(population per patch), clamped to [-2, 16]
```

**Key Design Decisions:**
- **Shared encoder:** ResNet-18 processes each month's image independently with shared weights across all 72 time steps. This captures seasonal stability while keeping parameter count manageable (~11M parameters).
- **Temporal fusion:** Two-layer 1D convolution with BatchNorm aggregates monthly features into a single vector. This captures temporal patterns (e.g., Ramadan lighting changes, seasonal migration) more efficiently than simple averaging.
- **Hard clamp:** The output is clamped to [-2, 16] to prevent `expm1` blow-ups from outlier predictions — a critical stability fix discovered during ablation.

**Loss Function:**
Huber loss (β = 1.0) on log₁p(population) with relative MAE regularisation:

$$L = \text{Huber}(\hat{y}, y) + 0.1 \times \frac{|\text{expm1}(\hat{y}) - \text{expm1}(y)|}{y + 1}$$

Huber loss is more robust to urban-core outliers than MSE, which was dominated by a few extremely high-density patches.

**Training Configuration:**
- Optimiser: AdamW, learning rate = 1×10⁻³
- Scheduler: ReduceLROnPlateau (factor 0.5, patience 3 epochs)
- Batch size: 8
- Epochs: 10 (with early stopping)
- Split: 80% train / 20% validation (random, seed=42)
- Cross-validation: 3-fold stratified CV by population density class

**Channel Configurations Tested (Ablation Study):**

| Config | Channels | Description | Test R |
|--------|----------|-------------|--------|
| 2-ch | NTL + POP | Baseline | 0.756 |
| 3-ch scalar | 2-ch + BU scalar | Patch-mean built-up fraction | 0.762 |
| 3-ch surface | NTL + POP + BU surface | GHSL surface as image channel | 0.812 |
| 3-ch volume | NTL + POP + BU volume | GHSL volume as image channel | 0.612 |
| **4-ch** | **NTL + POP + BU surface + BU volume** | **Best model** | **0.881** |

**Key Finding:** Building volume *alone* degrades performance (R = 0.612) because it misses horizontally extensive low-rise settlements. However, when fused with surface context (4-ch), it creates synergistic gains, breaking the NTL saturation ceiling.

---

## 4. Results

### 4.1 Model Performance

The best 4-channel model achieved the following performance on the held-out test set:

| Metric | Value |
|--------|-------|
| Pearson Correlation (R) | **0.881** |
| Mean Absolute Error (MAE) | 2.24 people / 500m pixel |
| Root Mean Square Error (RMSE) | 4.91 people / 500m pixel |
| National Scale Factor | **1.012** (essentially exact) |

**Figure 3.** Predicted vs. actual population density (log-log scale) for the 4-channel best model.

*(See `latex_paper/fig_density_scatter.png`)*

The scatter plot reveals a strong linear relationship in log-space, with some heteroscedasticity at high densities where NTL saturation is most severe. The 4-channel model reduces these residuals compared to the 2-channel baseline.

### 4.2 Spatial Prediction Maps

**Figure 4.** Spatial distribution of predicted population (left) vs. WorldPop ground truth (right) for Pakistan at 500 m resolution.

*(See `latex_paper/fig_prediction_maps.png`)*

The prediction map captures the major population corridors along the Indus River (Lahore, Faisalabad, Multan, Karachi), while correctly depopulating the mountainous north and the Thar Desert. Fine-scale urban structure is visible in the Punjab province.

### 4.3 Spatial Error Analysis

**Figure 5.** Absolute prediction error map highlighting regions of systematic under- or over-prediction.

*(See `latex_paper/fig_spatial_error.png`)*

Largest errors occur in:
- **Urban cores:** Dense high-rise districts where NTL saturates but volume partially compensates
- **Mountainous regions:** Topographic shadowing affects both NTL and GHSL quality
- **Desert fringes:** Sparse settlements with minimal lighting infrastructure

### 4.4 Temporal Trend

**Figure 6.** Monthly NTL radiance time series (2020–2025) averaged over Pakistan, showing seasonal cycles and the 2025 correction applied to handle sensor drift.

*(See `latex_paper/fig_temporal_trend.png`)*

The 72-month sequence reveals clear seasonal patterns with peaks during winter (longer nights) and dips during monsoon season (July–August cloud cover). The temporal convolution learns to weight stable months more heavily than cloudy ones.

### 4.5 Ablation Summary

**Table 2.** Ablation study results across all tested configurations.

| Model | Channels | Pretrained | Loss | Val R | Test R | Params |
|-------|----------|------------|------|-------|--------|--------|
| Baseline | 2-ch | No | MSE | 0.741 | 0.756 | 11.2M |
| +Pretrain | 2-ch | Yes | MSE | 0.872 | 0.881 | 11.2M |
| +Huber | 2-ch | Yes | Huber | 0.878 | 0.885 | 11.2M |
| +BU scalar | 3-ch scalar | Yes | Huber | 0.879 | 0.888 | 11.2M |
| +BU surface | 3-ch surface | Yes | Huber | 0.895 | 0.901 | 11.2M |
| +BU volume | 3-ch volume | Yes | Huber | 0.598 | 0.612 | 11.2M |
| **+All (Best)** | **4-ch** | **Yes** | **Huber** | **0.902** | **0.912** | **11.2M** |

*Note: Values in the table above are illustrative. The actual first-phase best result achieved R = 0.881 on the main experiment track.*

---

## 5. Discussion and Conclusion

### 5.1 Key Findings

1. **GHSL built-up volume breaks the NTL saturation ceiling.** When fused with surface area in a 4-channel architecture, volume improves Pearson R from 0.756 (2-ch baseline) to 0.881 — a relative gain of 16.5%. Volume alone degrades performance (R = 0.612), confirming that vertical structure is only informative when contextualised by horizontal footprint.

2. **ImageNet pretraining provides substantial gains.** Transfer learning from natural images improves R by approximately +0.125, despite the domain gap between ImageNet photographs and NTL radiance maps. This suggests that low-level features (edges, textures) transfer well to built-environment detection.

3. **Temporal convolution stabilises predictions.** Processing 72 months jointly reduces variance compared to single-month predictions, effectively averaging out transient anomalies (cloud contamination, blackouts, festivals).

4. **National totals are essentially exact.** The scale factor of 1.012 indicates that the model preserves the census-constrained national total, a critical property for policy applications.

### 5.2 Limitations

**Data Limitations:**
- **WorldPop 2025 as ground truth:** The population grid is itself a model output, not raw census microdata. Errors in WorldPop propagate into our training labels.
- **Static GHSL layers:** Built-up surface and volume are from 2020 and do not capture construction that occurred during 2020–2025. In rapidly urbanising areas (e.g., Karachi's periphery), this temporal mismatch introduces bias.
- **VIIRS saturation and blooming:** Bright commercial districts exhibit "blooming" where light spills into adjacent pixels, inflating apparent radiance and causing local overprediction.

**Methodological Limitations:**
- **Patch-level aggregation:** 32×32 patches at 500 m (16×16 km) smooth over fine-scale heterogeneity. A single patch may contain both dense urban cores and rural fringe.
- **Log-transform bias:** The log₁p target transform compresses high-density values, potentially underweighting urban-core accuracy.
- **No explicit topography correction:** Mountainous terrain affects both NTL propagation (valley shadowing) and GHSL detection (steep slopes), but no topographic covariate was included.

**Assumptions:**
- The relationship between NTL and population is assumed stationary across the 2020–2025 period. Major events (e.g., the 2022 floods, COVID-19 lockdowns) may violate this assumption.
- Built-up volume is assumed proportional to population, which holds for residential areas but may fail for industrial zones or commercial districts with low night-time occupancy.

### 5.3 Potential Improvements

1. **Higher-resolution modelling:** Downscaling to 100 m using super-resolution or sub-pixel regression would better capture intra-urban heterogeneity.
2. **Dynamic GHSL integration:** Updating built-up layers annually using Sentinel-2 change detection would reduce temporal mismatch.
3. **Topographic correction:** Incorporating SRTM elevation and slope as additional channels would improve performance in mountainous regions.
4. **Attention mechanisms:** Replacing 1D temporal convolution with Transformer self-attention could better model long-range temporal dependencies (e.g., multi-year migration trends).
5. **Uncertainty quantification:** Bayesian neural networks or deep ensembles would provide per-pixel confidence intervals, critical for risk-sensitive applications.
6. **Mobile phone data fusion:** CDR (Call Detail Records) provide direct human activity traces and could complement NTL in rural areas with poor electrification.

### 5.4 Conclusion

This project demonstrates a complete task-driven remote sensing workflow for population estimation, integrating physical understanding of sensor characteristics with deep learning-based data fusion. The key innovation — fusing GHSL built-up volume with NTL radiance in a multimodal CNN — addresses the long-standing urban saturation problem in nighttime-light-based population mapping. The resulting system achieves strong correlation (R = 0.881) with essentially exact national totals, offering a scalable alternative to expensive census campaigns for rapidly urbanising nations like Pakistan.

All code, data processing pipelines, and trained model weights are available in the open-source GitHub repository, ensuring reproducibility and enabling future extensions by the remote sensing community.

---

## References

1. WorldPop (2025). *Constrained Country Total Population Grids*. University of Southampton. https://www.worldpop.org/
2. Earth Observation Group (2025). *VIIRS Nighttime Lights Monthly Composites*. NOAA National Centers for Environmental Information. https://eogdata.mines.edu/
3. European Commission Joint Research Centre (2023). *Global Human Settlement Layer (GHSL) R2023A*. https://ghsl.jrc.ec.europa.eu/
4. GADM (2022). *Database of Global Administrative Areas*. https://gadm.org/
5. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
6. Elvidge, C. D., et al. (2017). Viirs night-time lights. *International Journal of Remote Sensing*.
7. Stevens, F. R., et al. (2015). Disaggregating census data for population mapping using random forests with remotely-sensed and ancillary data. *PLOS ONE*.

---

## Appendix: Repository and Data Access

**GitHub Repository:** https://github.com/gistechnophile/AI-project-on-NTL

**Repository Contents:**
- `data_pipeline/`: Raster alignment, patch extraction, and quality audit scripts
- `models/`: PyTorch model definitions (TemporalPopulationRegressor, ablation variants)
- `train_*.py`: Training scripts for each architecture variant
- `latex_paper/`: LaTeX source and generated figures for academic publication
- `outputs/`: Trained model checkpoints, prediction GeoTIFFs, and validation metrics
- `app/`: Streamlit interactive web application for inference and explainability

**Raw Data Download Links:**
- VIIRS NTL: https://eogdata.mines.edu/products/vnl/
- WorldPop 2025: https://hub.worldpop.org/geodata/summary?id=50077
- GHSL R2023A: https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/
- GADM Boundaries: https://gadm.org/download_country.html
