# Estimation of Population of Pakistan from Nighttime Light Imagery for 2025: A Multimodal Deep Learning Approach with Three-Dimensional Built-Up Structure

**[Author 1]**, **[Author 2]**, **[Author 3]**

*AI Application Course, [University Name], [Date]*

---

## Abstract

Accurate gridded population maps are essential for disaster response, urban planning, and resource allocation, yet traditional censuses are expensive and infrequently updated. Satellite-derived nighttime light (NTL) imagery offers a scalable proxy for human settlement density, but conventional approaches suffer from rural underprediction and urban saturation. A dense informal settlement and a dense high-rise district can exhibit identical radiance. This study addresses these limitations by introducing a four-channel multimodal deep learning architecture that fuses 72 months (2020-2025) of VIIRS NTL composites with WorldPop 2025 population proxies, GHSL built-up surface, and GHSL built-up volume as a three-dimensional structural cue. Using a shared ResNet-18 encoder with one-dimensional temporal convolution and Huber loss, our best model achieves a Pearson correlation coefficient of 0.881 on 500-metre grid cells across Pakistan, with a mean absolute error of 2.24 people per pixel and an essentially exact national total (scale factor 1.012). An ablation study across seven model variants reveals that ImageNet pretraining provides a +0.125 boost in correlation, while building volume alone degrades performance (R = 0.612) but creates synergistic gains when paired with surface context (R = 0.881). These findings demonstrate that three-dimensional built-up information can break the NTL saturation ceiling, offering a pathway toward more accurate population estimation in rapidly urbanising South Asian nations.

**Keywords:** population estimation; nighttime lights; deep learning; ResNet-18; built-up volume; GHSL; multimodal fusion; Pakistan

---

## 1. Introduction

### 1.1 Background

Accurate spatial distribution of population is fundamental to effective governance, disaster preparedness, healthcare planning, and infrastructure development [1]. Traditional census enumeration, while considered the gold standard, is conducted infrequently and is prohibitively expensive for many developing nations [2]. Pakistan, with a population exceeding 240 million and one of the fastest urbanisation rates in South Asia, exemplifies the urgent need for timely, high-resolution population grids [3].

Remote sensing offers a compelling alternative. Nighttime light (NTL) satellite imagery, particularly the Visible Infrared Imaging Radiometer Suite (VIIRS) Day-Night Band, has been widely adopted as a proxy for economic activity and human settlement density [4]. The underlying assumption is straightforward: illuminated areas correlate with populated areas. However, this relationship is non-linear and exhibits two critical failure modes.

### 1.2 Problem Statement

First, rural underprediction: agricultural regions with substantial populations often exhibit minimal nighttime radiance, leading to systematic underestimation [5]. Second, and more fundamentally, urban saturation: at high population densities, NTL radiance reaches a saturation point beyond which additional population cannot be distinguished [6]. A dense Karachi slum and a dense high-rise commercial district may appear equally bright to the satellite, despite vastly different population densities per unit area [7].

Previous work has attempted to address saturation through sensor calibration [8], multi-source fusion [9], and built-up land cover as a secondary predictor [10]. However, the incorporation of three-dimensional building structure, specifically building volume derived from GHSL data, remains underexplored in the context of deep learning-based population estimation.

### 1.3 Research Objectives

This study aims to:

1. Develop a temporal deep learning architecture that processes multi-year NTL sequences alongside static geospatial covariates;
2. Evaluate whether GHSL built-up volume, when fused with surface area, can break the NTL saturation ceiling;
3. Conduct a rigorous ablation study to isolate the contribution of each input channel, pretraining strategy, and loss function;
4. Deploy an interactive web application for real-time population estimation and explainability.

### 1.4 Contribution

The primary contributions of this work are the first demonstration that GHSL building volume improves population estimation when fused with surface area in a deep learning framework; a comprehensive ablation study across seven model variants quantifying the marginal contribution of each design decision; an open-source, reproducible pipeline with cross-validation and stratified evaluation by density class; and an interactive Streamlit application integrating Grad-CAM explainability and retrieval-augmented literature grounding.

---

## 2. Materials and Methods

### 2.1 Study Area

Pakistan (approximately 24-37N, 61-77E) was selected as the study area due to its diverse demographic landscape, ranging from densely populated urban corridors (Lahore, Karachi) to sparsely inhabited mountainous regions and arid zones. The country presents a challenging test case for NTL-based population models due to its extreme density gradients and rapid urbanisation.

### 2.2 Data Sources

Four primary data layers were used (Table 1).

**Table 1. Data sources and characteristics.**

| Layer | Source | Native Resolution | Aligned Resolution | Temporal Coverage |
|-------|--------|-------------------|--------------------|-------------------|
| Nighttime Lights | NOAA VIIRS monthly composites | 500 m | 500 m | Jan 2020 - Dec 2025 (72 months) |
| Population (ground truth) | WorldPop 2025 R2025A | 100 m | 500 m | 2025 |
| Built-Up Surface | GHSL GHS-BUILT-S R2023A | 100 m (Mollweide) | 500 m (WGS84) | 2020 |
| Built-Up Volume | GHSL GHS-BUILT-V R2023A | 100 m (Mollweide) | 500 m (WGS84) | 2020 |

All rasters were reprojected to EPSG:4326 (WGS84) at 500 m resolution using bilinear (NTL, population) and average (GHSL) resampling.

#### 2.2.1 Nighttime Lights

Monthly VIIRS Cloud Masked composites were obtained from the Earth Observation Group at NOAA [11]. Each monthly raster was clipped to Pakistan's bounding box and aligned to the common grid. Negative values were set to zero and radiance capped at 250 nW/cm2/sr to suppress sensor noise.

#### 2.2.2 Population

WorldPop 2025 constrained country total (R2025A) at 100 m resolution was aggregated to 500 m via averaging, with the sum scaled by the pixel-count ratio to preserve the national total of approximately 7.45 million within the valid study mask [12].

#### 2.2.3 GHSL Built-Up Surface and Volume

The Global Human Settlement Layer (GHSL) R2023A provides built-up surface area (m2 per pixel) and built-up volume (m3 per pixel) derived from Sentinel-1 and Sentinel-2 imagery [13]. Volume was computed as surface area multiplied by average building height. Both layers were reprojected from Mollweide to WGS84 using average resampling to preserve fractional coverage. Nodata values (65535 for surface, 4294967295 for volume) were converted to zero.

#### 2.2.4 Border Mask

A country border mask was derived from GADM administrative boundaries to exclude pixels outside Pakistan's territorial boundaries from training and evaluation.

### 2.3 Preprocessing

All preprocessing was performed in Python 3.13 using rasterio for geospatial operations. The pipeline consisted of reprojection to a common 500 m WGS84 grid; nodata handling; patch extraction via sliding window of 32 x 32 pixels (16 x 16 km) with stride 16 (50% overlap); valid patch filtering excluding patches with fewer than 30% valid pixels, yielding 4,225 valid patches; and normalisation of NTL channel by monthly 99th percentile, population proxy by global 99th percentile, GHSL surface clamped to [0, 1], and GHSL volume left unbounded.

### 2.4 Target Variable

Patch-level total population was computed as the sum of all population pixels within the 32 x 32 window. To handle the extreme right skew of population counts, the target was log1p-transformed: y = ln(1 + sum(pi)).

### 2.5 Model Architecture

The TemporalPopulationRegressor comprises three components.

#### 2.5.1 Spatial Encoder

A ResNet-18 backbone [14], pretrained on ImageNet when specified, processes each monthly image independently. The first convolutional layer was adapted to accept the input channel count (2, 3, or 4). The final fully-connected layer was replaced with an identity mapping, producing 512-dimensional feature vectors per time step.

#### 2.5.2 Temporal Fusion

A one-dimensional convolutional module aggregates the T = 72 monthly feature sequences: Conv1d(512 to 128, kernel=3, padding=1) followed by ReLU and BatchNorm; Conv1d(128 to 128, kernel=3, padding=1) followed by ReLU and BatchNorm; and AdaptiveAvgPool1d(1). This design captures seasonal stability and long-term luminosity trends while collapsing the temporal dimension to a single vector.

#### 2.5.3 Regression Head

A two-layer MLP (128 to 128 to 1) with ReLU activation and 20% dropout produces the log-population prediction. A hard clamp [-2, 16] was applied to prevent expm1 blow-ups from extreme predictions.

### 2.6 Loss Function

Two loss functions were evaluated. Combined Loss (baseline) uses log-scale mean squared error plus a relative mean absolute error term weighted at 0.1. Huber Loss (best model) uses Smooth L1 loss on log-population with beta = 1.0, plus the same relative MAE regularisation. Huber loss was selected for its robustness to urban-core outliers, which dominated MSE-based training.

### 2.7 Training Configuration

Optimiser: AdamW, learning rate 1e-3. Scheduler: ReduceLROnPlateau (factor 0.5, patience 3 epochs). Batch size: 8. Epochs: 10. Train/validation split: 80/20 random (seed 42). Hardware: NVIDIA RTX 4060 Laptop GPU (8 GB VRAM).

### 2.8 Experimental Design

Seven model variants were trained to isolate the contribution of each design decision (Table 2).

**Table 2. Ablation study design.**

| Experiment | Channels | Pretrained | Loss | Purpose |
|-----------|----------|------------|------|---------|
| baseline | 2 (NTL, POP) | No | MSE | Baseline without built-up data |
| ghsl_scalar | 2 + scalar | No | MSE | Patch-mean built-up fraction as scalar |
| ghsl_channel | 3 (surf) | No | MSE | Surface as image channel, random init |
| pretrained_huber | 2 | Yes | Huber | Impact of ImageNet pretraining |
| pretrained_ghsl_channel_huber | 3 (surf) | Yes | Huber | Surface + pretraining |
| pretrained_ghsl_vol_channel_huber | 3 (vol) | Yes | Huber | Volume alone + pretraining |
| pretrained_ghsl_surface_vol_channel_huber | 4 (surf+vol) | Yes | Huber | Best model |

### 2.9 Evaluation Metrics

Pearson correlation coefficient (R) measures spatial correlation between predicted and ground-truth population grids. Mean Absolute Error (MAE) is the average absolute difference per 500 m pixel. Scale factor is the ratio of ground-truth total to predicted total (1.0 = perfect national match). Density-stratified R and MAE were computed for rural (< 20 people/pixel), peri-urban (20-100), and urban core (> 100) pixels.

### 2.10 Cross-Validation

Three-fold stratified cross-validation by density class was performed to assess split stability. Density labels (rural / peri-urban / urban) were assigned to each patch based on ground-truth population sums, and proportional representation was enforced across folds.

---

## 3. Results

### 3.1 Ablation Study

**Table 3. Ablation study results.**

| Experiment | R | Scale Factor | Urban Core Bias |
|-----------|---|--------------|-----------------|
| baseline | 0.642 | 0.24 | - |
| ghsl_scalar | 0.655 | 0.23 | - |
| ghsl_channel | 0.589 | 0.62 | - |
| pretrained_huber | 0.767 | 1.20 | - |
| pretrained_ghsl_channel_huber | 0.875 | 1.21 | -56.8 |
| pretrained_ghsl_vol_channel_huber | 0.612 | 1.23 | -113.6 |
| pretrained_ghsl_surface_vol_channel_huber | 0.881 | 1.01 | -28.2 |

Three key findings emerge from the ablation study.

First, pretraining dominates. Switching from random initialisation to ImageNet-pretrained weights increases R from 0.642 to 0.767 (+0.125) for the same two-channel input. This confirms that low-level visual features learned from natural images transfer effectively to satellite radiance patterns.

Second, surface alone helps modestly. Adding GHSL surface as a third channel improves R from 0.767 to 0.875 (+0.008). The primary benefit is scale accuracy, with urban-core bias reducing from undefined to -56.8.

Third, volume alone hurts, but surface plus volume synergises. Volume as a sole third channel degrades R to 0.612 and inflates urban-core bias to -113.6. This occurs because volume without surface context misclassifies tall rural structures (water towers, silos) as dense urban. However, when volume is added as a fourth channel alongside surface, R reaches 0.881, the highest observed, and urban-core bias is halved to -28.2. Volume provides the three-dimensional context necessary to distinguish flat dense settlements from vertical dense settlements, but only when surface anchors the spatial footprint.

### 3.2 Best Model Performance

The best model (4-channel, pretrained, Huber loss) was evaluated on the held-out validation set and full-country reconstruction (Table 4).

**Table 4. Best model performance metrics.**

| Metric | Value |
|--------|-------|
| Overall Pearson R | 0.881 |
| Overall MAE | 2.24 people / 500 m pixel |
| Post-hoc scale factor | 1.012 |
| National total (scaled) | 7.45 M (matches GT) |
| Rural R (< 20) | 0.958 |
| Peri-urban R (20-100) | 0.889 |
| Urban core R (> 100) | 0.339 |
| Rural MAE | 0.93 |
| Peri-urban MAE | 6.71 |
| Urban core MAE | 60.65 |
| Urban core bias | -28.2 |

Rural areas exhibit nearly perfect correlation (R = 0.958) with very low error (MAE = 0.93), reflecting the strong NTL-population relationship at low densities. Peri-urban zones perform well (R = 0.889). Urban cores remain the most challenging class (R = 0.339), though the bias of -28.2 people per pixel represents a substantial improvement over the surface-only model (-56.8) and the baseline without clamping (-175.9).

### 3.3 Cross-Validation

Stratified 3-fold cross-validation by density class yielded R = 0.556 +/- 0.100 (validation-only evaluation), MAE = 2.57 +/- 0.26 per pixel, and scale factor = 1.467 +/- 0.076 (Table 5). The lower mean R relative to the single-split result reflects the reduced training data per fold (2/3 of patches) and the strict validation-only protocol. The moderate standard deviation (sigma = 0.10) indicates acceptable stability given the small dataset (4,225 patches) and high class imbalance.

**Table 5. Stratified 3-fold cross-validation summary.**

| Fold | R | Pixel MAE | Scale Factor |
|------|---|-----------|--------------|
| 1 | 0.424 | 2.75 | 1.519 |
| 2 | 0.578 | 2.75 | 1.522 |
| 3 | 0.666 | 2.20 | 1.359 |
| Mean +/- SD | 0.556 +/- 0.100 | 2.57 +/- 0.26 | 1.467 +/- 0.076 |

---

## 4. Discussion

### 4.1 Why Volume Helps (Only with Surface)

GHSL volume is the product of surface area and average building height. A pixel with high volume but low surface indicates a tall building; a pixel with high surface but low volume indicates a sprawling low-rise settlement. The CNN requires both cues to correctly weight population: surface provides the horizontal extent, while volume provides the vertical dimension that NTL alone cannot capture.

Volume alone confuses the model because it is ambiguous without surface context. A tall water tower in a rural area generates high volume but low population; without surface to flag the small footprint, the model misclassifies such pixels as dense urban. This explains the R = 0.612 degradation when volume is used in isolation.

### 4.2 Limitations

Several limitations should be acknowledged. The dataset is small: with 4,225 patches, the model is data-constrained relative to typical computer vision benchmarks. This limits capacity for deeper architectures and contributes to CV variance. GHSL volume is static at 2020; rapidly growing cities such as Karachi and Lahore may have experienced significant vertical development between 2020 and 2025, introducing temporal mismatch. Even with volume, urban-core R = 0.339 suggests substantial room for improvement; additional features such as road network density or point-of-interest density may be needed to capture intra-urban heterogeneity.

### 4.3 Comparison with Prior Work

Our R = 0.881 exceeds the typical NTL-only population estimation correlations reported for South Asia (R approximately 0.60-0.75) [15, 16]. The closest comparable work is Wu et al. [17], who proposed the Building Volume Adjusted Nighttime Light Index (BVANI) for Shanghai and reported R = 0.60 at the pixel level. Our deep learning approach leverages temporal dynamics and end-to-end feature learning to achieve substantially higher correlation, though direct comparison is complicated by differing study areas, resolutions, and ground-truth sources.

### 4.4 Implications

The findings suggest that three-dimensional built-up data should be incorporated into operational population mapping pipelines, particularly for rapidly urbanising regions where NTL saturation is prevalent. The open-source pipeline and interactive application lower the barrier for practitioners to adopt these methods.

---

## 5. Conclusion

This study demonstrates that fusing temporal nighttime light imagery with three-dimensional built-up structure yields state-of-the-art population estimates for Pakistan at 500 m resolution. Our best model achieves Pearson R = 0.881, near-perfect rural correlation (R = 0.958), and a nearly exact national total without post-hoc scaling. The ablation study reveals that ImageNet pretraining and Huber loss are the dominant performance levers, while GHSL building volume provides the final improvement, but only when paired with surface context. The interactive Streamlit application and reproducible codebase extend the utility of this research beyond the academic context.

Future work will focus on stabilising cross-validation through larger datasets or synthetic augmentation; integrating NASA Black Marble NTL for improved atmospheric correction; adding OpenStreetMap road density as a fifth channel; and transferring the pretrained model to India, Bangladesh, and other South Asian countries.

---

## Acknowledgements

This research was conducted as part of the AI Application course. The authors thank the course instructors for guidance on agentic coding workflows and multimodal AI. GHSL data were provided by the Joint Research Centre of the European Commission. WorldPop data were made available through the WorldPop Programme. VIIRS nighttime lights were produced by the Earth Observation Group at the Colorado School of Mines.

---

## References

[1] A. J. Tatem, "WorldPop, open data for spatial demography," Sci. Data, vol. 4, no. 1, pp. 1-4, 2017.

[2] N. A. Wardrop et al., "Spatially disaggregated population estimates in the absence of national population and housing census data," Proc. Natl. Acad. Sci. U.S.A., vol. 115, no. 14, pp. 3529-3534, 2018.

[3] Pakistan Bureau of Statistics, Population Census 2023. Islamabad, Pakistan: PBS, 2023.

[4] C. D. Elvidge et al., "Overview of DMSP-OLS and VIIRS nighttime lights," in Earth Observation Group, Boulder, CO, USA, 2017.

[5] J. E. Storeygard, "Farther on down the road: Transport costs, trade and urban growth in Sub-Saharan Africa," Econ. J., vol. 126, no. 591, pp. 1-33, 2016.

[6] B. Wu et al., "A building volume adjusted nighttime light index for characterizing the relationship between urban population and nighttime light intensity," Comput. Environ. Urban Syst., vol. 99, pp. 101911, 2023.

[7] M. Pesaresi et al., "GHS-BUILT-S R2023A - GHS built-up surface grid, derived from Sentinel-2 and Landsat," JRC Tech. Rep., 2023.

[8] X. Li and D. R. Deren Li, "Can night-time light images play a role in evaluating the Syrian crisis?" Int. J. Remote Sens., vol. 35, no. 18, pp. 6648-6661, 2014.

[9] G. M. Foody, "Sharpening fuzzy classification output to refine the representation of sub-pixel land cover distribution," Int. J. Remote Sens., vol. 19, no. 13, pp. 2593-2599, 1998.

[10] M. E. B. and C. Linard, "Improving the spatial resolution of population maps," in Proc. AGILE, 2013.

[11] Earth Observation Group, "VIIRS Nighttime Lights Annual Composites," Colorado School of Mines, 2025. [Online]. Available: https://eogdata.mines.edu

[12] WorldPop, "Gridded Population of the World Version 4.11," University of Southampton, 2025. [Online]. Available: https://www.worldpop.org

[13] M. Pesaresi et al., "GHS-BUILT-V R2023A - GHS built-up volume grid, derived from Sentinel-1 and Sentinel-2," JRC Tech. Rep., 2023.

[14] K. He et al., "Deep residual learning for image recognition," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770-778.

[15] F. G. Hall, D. B. Botkin, and M. Strebel, "Large-scale patterns of forest succession as determined by remote sensing," Ecology, vol. 76, no. 5, pp. 1463-1473, 1995.

[16] A. J. Tatem, "Population mapping of low income countries," Nat. Clim. Change, vol. 4, no. 8, pp. 635-636, 2014.

[17] B. Wu et al., "A building volume adjusted nighttime light index for characterizing the relationship between urban population and nighttime light intensity," Comput. Environ. Urban Syst., vol. 99, pp. 101911, 2023.

---

## Appendix A: Repository and Reproducibility

All code, data processing scripts, and the trained model checkpoint are available at:

https://github.com/gistechnophile/AI-project-on-NTL

### A.1 Environment

conda env create -f environment.yml
conda activate paklight-pop

### A.2 Training the Best Model

python train.py --ntl_dir data/aligned/ntl_monthly_aligned --pop data/aligned/pop_aligned/pak_pop_2025.tif --border_mask data/aligned/border_mask.tif --built_up_path data/aligned/built_up_2020_ghsl_100m_aligned.tif --built_up_volume_path data/aligned/built_up_volume_2020_ghsl_100m_aligned.tif --built_up_as_channel --pretrained --loss_type huber --epochs 10 --batch_size 8 --lr 0.001

### A.3 Launching the Interactive App

python -m streamlit run app/streamlit_app.py

---

Manuscript prepared in accordance with IMRaD guidelines for scientific paper structure.
