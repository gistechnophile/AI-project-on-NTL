"""
Interactive Streamlit Application for Pakistan Population Estimation from Monthly NTL.
Upgrades implemented:
  - 4-channel input [NTL, POP_proxy, BU_surface, BU_volume]
  - NTL clipping (negatives -> 0, cap at 250)
  - Border mask support
  - Auto-detects checkpoint config (channels, pretrained, etc.)
Integrates:
  - Data Quality Audit (Session 4)
  - Temporal ResNet CNN Prediction (Sessions 3, 6)
  - Grad-CAM Explainability (Session 3)
  - RAG Literature Retrieval (Session 8)
  - Agentic Report Generation (Session 9)
"""
import os
import sys
import glob
import numpy as np
import rasterio
import torch
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_pipeline.align_rasters import extract_patch
from data_pipeline.quality_audit import audit_raster_pair
from data_pipeline.monthly_utils import group_by_year, extract_date_from_filename
from models.population_cnn import TemporalPopulationRegressor
from models.explainability import get_gradcam_heatmap
from report_engine.agent_reporter import PopulationReportAgent

st.set_page_config(page_title="NTL in Population | Pakistan Population Estimation", layout="wide")

# ------------------------------------------------------------------
# Custom CSS for modern styling
# ------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Gradient header banner */
    .header-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .header-subtitle {
        font-size: 1.15rem;
        font-weight: 300;
        color: #94a3b8;
        margin-bottom: 0.75rem;
        line-height: 1.5;
    }
    
    .header-badge {
        display: inline-block;
        background: rgba(96, 165, 250, 0.15);
        color: #60a5fa;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid rgba(96, 165, 250, 0.25);
        margin-top: 0.5rem;
    }
    
    /* Section cards */
    .section-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.08);
        margin-bottom: 1.5rem;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Streamlit dark theme enhancement */
    .stApp {
        background: linear-gradient(180deg, #0a0e1a 0%, #0f172a 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #1e293b);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4) !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #cbd5e1;
        background: rgba(30, 41, 59, 0.6);
        border-radius: 8px;
    }
    
    /* Caption text */
    .stCaption {
        color: #64748b !important;
        font-size: 0.85rem;
    }
    
    /* Select box */
    .stSelectbox label {
        color: #94a3b8 !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Sidebar: Configuration
# ------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

ntl_dir = st.sidebar.text_input(
    "Aligned Monthly NTL directory",
    value="data/aligned/ntl_monthly_aligned",
)
pop_path = st.sidebar.text_input(
    "Aligned WorldPOP .tif path",
    value="data/aligned/pop_aligned/pak_pop_2025_CN_100m_R2025A_v1_aligned.tif",
)
border_mask_path = st.sidebar.text_input(
    "Border mask .tif path (optional)",
    value="data/aligned/border_mask.tif",
)
checkpoint_path = st.sidebar.text_input(
    "Model checkpoint",
    value="outputs/best_model.pt",
)
# Auto-detected from checkpoint
bu_surface_path = None
bu_volume_path = None
use_pretrained = False
in_channels = 2
bu_as_channel = False

st.sidebar.markdown("---")
st.sidebar.info(
    "This app applies the AI Flywheel from the course: \n"
    "1. Query a region & year \n"
    "2. AI decomposes (quality audit) \n"
    "3. AI predicts (Temporal CNN, 4-ch input) \n"
    "4. You verify (Grad-CAM + RAG report)\n\n"
    "Note: The model predicts 2025 population using whichever year's NTL you select. "
    "Population numbers will be similar across years — see 'Understanding Your Results' for details."
)

# ------------------------------------------------------------------
# Main Header
# ------------------------------------------------------------------
st.markdown("""
<div class="header-banner">
    <div class="header-title">🛰️ NTL in Population</div>
    <div class="header-subtitle">
        Estimation of Population of Pakistan from Nighttime Light Imagery for 2025
    </div>
    <div class="header-badge">AI Application Course Project &bull; ResNet-18 + GHSL Built-Up Volume &bull; R = 0.881</div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Year & Map Selection
# ------------------------------------------------------------------
available_years = []
if os.path.isdir(ntl_dir):
    all_files = glob.glob(os.path.join(ntl_dir, "**/*.tif"), recursive=True)
    grouped = group_by_year(all_files)
    available_years = sorted(grouped.keys())

selected_year = st.selectbox(
    "Select year for prediction",
    options=available_years if available_years else list(range(2020, 2026)),
    index=0,
)

st.markdown("""
<div style="margin-top: 1rem; margin-bottom: 0.5rem;">
    <span style="font-size: 1.3rem; font-weight: 600; color: #e2e8f0;">📍 1. Select Region of Interest</span>
    <div style="height: 2px; width: 60px; background: linear-gradient(90deg, #60a5fa, #a78bfa); margin-top: 0.4rem; border-radius: 2px;"></div>
</div>
""", unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])

with col1:
    m = folium.Map(location=[30.3753, 69.3451], zoom_start=6)  # Center of Pakistan
    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "rectangle": True,
        },
    ).add_to(m)
    map_data = st_folium(m, width=700, height=500)

bounds = None
if map_data and map_data.get("all_drawings"):
    try:
        geo = map_data["all_drawings"][0]["geometry"]["coordinates"][0]
        lons = [p[0] for p in geo]
        lats = [p[1] for p in geo]
        bounds = (min(lons), min(lats), max(lons), max(lats))
        st.success(f"Selected bounds: {bounds}")
    except Exception:
        st.info("Draw a rectangle on the map to define the analysis area.")

# ------------------------------------------------------------------
# Prediction & Audit Panel
# ------------------------------------------------------------------
st.markdown("""
<div style="margin-top: 2rem; margin-bottom: 0.5rem;">
    <span style="font-size: 1.3rem; font-weight: 600; color: #e2e8f0;">🤖 2. Run AI Analysis</span>
    <div style="height: 2px; width: 60px; background: linear-gradient(90deg, #60a5fa, #a78bfa); margin-top: 0.4rem; border-radius: 2px;"></div>
</div>
""", unsafe_allow_html=True)

if st.button("🔍 Predict Population", disabled=(bounds is None)):
    if not os.path.exists(pop_path):
        st.error("WorldPOP raster not found. Please check the path in the sidebar.")
    else:
        with st.spinner("Loading monthly data, auditing, and predicting..."):
            # Gather monthly files for selected year
            all_files = glob.glob(os.path.join(ntl_dir, "**/*.tif"), recursive=True)
            grouped = group_by_year(all_files)
            year_files = grouped.get(selected_year, [])
            if len(year_files) == 0:
                st.error(f"No monthly NTL files found for year {selected_year} in {ntl_dir}")
                st.stop()

            # Extract patches for each month
            monthly_patches = []
            for fpath in year_files:
                patch, _ = extract_patch(fpath, bounds)
                monthly_patches.append(patch)

            # Use first month's shape as reference; resize others if needed
            ref_shape = monthly_patches[0].shape
            from skimage.transform import resize
            for i in range(len(monthly_patches)):
                if monthly_patches[i].shape != ref_shape:
                    monthly_patches[i] = resize(
                        monthly_patches[i], ref_shape, order=1, preserve_range=True
                    )

            # Load POP patch for the same bounds
            pop_patch, _ = extract_patch(pop_path, bounds)
            if pop_patch.shape != ref_shape:
                pop_patch = resize(pop_patch, ref_shape, order=1, preserve_range=True)

            # Load border mask patch if available
            border_mask_patch = None
            if border_mask_path and os.path.exists(border_mask_path):
                bmp, _ = extract_patch(border_mask_path, bounds)
                if bmp.shape != ref_shape:
                    bmp = resize(bmp, ref_shape, order=0, preserve_range=True)
                border_mask_patch = bmp

            # Quality audit on the mean of months vs pop
            ntl_stack = np.stack(monthly_patches, axis=0)
            ntl_stack = np.where((ntl_stack == -9999) | ~np.isfinite(ntl_stack), np.nan, ntl_stack)
            mean_ntl = np.nanmean(ntl_stack, axis=0)
            
            # Also mask POP nodata for audit
            pop_for_audit = np.where((pop_patch == -9999) | ~np.isfinite(pop_patch), np.nan, pop_patch)
            scorecard = audit_raster_pair(mean_ntl, pop_for_audit)

            # Load model — auto-detect config from checkpoint
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            T_actual = len(year_files)

            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location=device)
                T_ckpt = ckpt.get("T", T_actual)
                args = ckpt.get("args", {})
                
                use_pretrained = args.get("pretrained", False)
                bu_as_channel = args.get("built_up_as_channel", False)
                bu_surface_path = args.get("built_up_path")
                bu_volume_path = args.get("built_up_volume_path")
                
                # Count channels
                in_channels = 2
                if bu_as_channel:
                    if bu_surface_path and os.path.exists(bu_surface_path):
                        in_channels += 1
                    if bu_volume_path and os.path.exists(bu_volume_path):
                        in_channels += 1
                
                if T_ckpt != T_actual:
                    st.warning(
                        f"Checkpoint was trained with T={T_ckpt} months, but selected year has {T_actual} months. "
                        f"Using {T_actual} months anyway."
                    )
                
                st.info(f"Loaded checkpoint: {in_channels} channels | pretrained={use_pretrained} | epoch={ckpt.get('epoch', '?')} | val_loss={ckpt.get('val_loss', '?'):.4f}")
            else:
                st.warning("No checkpoint found. Using random weights for demo.")

            model = TemporalPopulationRegressor(
                pretrained=use_pretrained,
                in_channels=in_channels,
                use_built_up_scalar=False,
            ).to(device)
            
            if os.path.exists(checkpoint_path):
                model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            model._last_T = T_actual  # for Grad-CAM wrapper

            # Preprocess temporal stack
            proc_ntl = []
            for t in range(T_actual):
                patch = monthly_patches[t].astype(np.float32)
                patch[(patch == -9999) | ~np.isfinite(patch)] = 0.0
                patch = np.clip(patch, 0.0, 250.0)
                p99 = np.percentile(patch[patch > 0], 99.0)
                if np.isfinite(p99) and p99 > 0:
                    patch = patch / p99
                else:
                    patch = patch / (patch.max() + 1e-6)
                proc_ntl.append(patch)
            proc_ntl = np.stack(proc_ntl, axis=0)  # (T, H, W)

            # Preprocess POP proxy channel
            pop_proc = pop_patch.astype(np.float32)
            pop_proc[(pop_proc == -9999) | ~np.isfinite(pop_proc)] = 0.0
            pop_p99 = np.percentile(pop_proc[pop_proc > 0], 99.0)
            if np.isfinite(pop_p99) and pop_p99 > 0:
                pop_proc = pop_proc / pop_p99
            else:
                pop_proc = pop_proc / (pop_proc.max() + 1e-6)

            # Preprocess optional built-up channels
            bu_surface_proc = None
            bu_volume_proc = None
            if bu_as_channel:
                if bu_surface_path and os.path.exists(bu_surface_path):
                    bu_patch, _ = extract_patch(bu_surface_path, bounds)
                    if bu_patch.shape != ref_shape:
                        bu_patch = resize(bu_patch, ref_shape, order=1, preserve_range=True)
                    bu_surface_proc = bu_patch.astype(np.float32)
                    bu_surface_proc[(bu_surface_proc == -9999) | ~np.isfinite(bu_surface_proc)] = 0.0
                if bu_volume_path and os.path.exists(bu_volume_path):
                    bv_patch, _ = extract_patch(bu_volume_path, bounds)
                    if bv_patch.shape != ref_shape:
                        bv_patch = resize(bv_patch, ref_shape, order=1, preserve_range=True)
                    bu_volume_proc = bv_patch.astype(np.float32)
                    bu_volume_proc[(bu_volume_proc == -9999) | ~np.isfinite(bu_volume_proc)] = 0.0

            h, w = proc_ntl.shape[1], proc_ntl.shape[2]
            patch_size = 32

            # Sliding window inference over the selected region
            predictions = []
            heatmaps = []
            for y in range(0, h - patch_size + 1, patch_size // 2):
                for x in range(0, w - patch_size + 1, patch_size // 2):
                    # Border mask check: skip mostly-masked windows
                    if border_mask_patch is not None:
                        sub_mask = border_mask_patch[y:y+patch_size, x:x+patch_size]
                        if np.mean(sub_mask) < 0.5:
                            continue

                    sub_ntl = proc_ntl[:, y:y+patch_size, x:x+patch_size]  # (T, H, W)
                    sub_pop = pop_proc[y:y+patch_size, x:x+patch_size]     # (H, W)

                    # Build multi-channel tensor: (1, T, C, H, W)
                    ntl_tensor = torch.from_numpy(sub_ntl).unsqueeze(1).to(device)  # (T, 1, H, W)
                    pop_tensor = torch.from_numpy(sub_pop).unsqueeze(0).unsqueeze(0).repeat(T_actual, 1, 1, 1).to(device)
                    channels = [ntl_tensor, pop_tensor]
                    
                    if bu_surface_proc is not None:
                        sub_bu_s = bu_surface_proc[y:y+patch_size, x:x+patch_size]
                        bu_s_tensor = torch.from_numpy(sub_bu_s).unsqueeze(0).unsqueeze(0).repeat(T_actual, 1, 1, 1).to(device)
                        channels.append(bu_s_tensor)
                    
                    if bu_volume_proc is not None:
                        sub_bu_v = bu_volume_proc[y:y+patch_size, x:x+patch_size]
                        bu_v_tensor = torch.from_numpy(sub_bu_v).unsqueeze(0).unsqueeze(0).repeat(T_actual, 1, 1, 1).to(device)
                        channels.append(bu_v_tensor)
                    
                    tensor = torch.cat(channels, dim=1).unsqueeze(0)  # (1, T, C, H, W)

                    with torch.no_grad():
                        pred = model(tensor).item()
                    predictions.append(np.expm1(pred))  # reverse log1p

                    # Grad-CAM on first valid window (for speed)
                    if len(heatmaps) == 0:
                        try:
                            mean_ch = [sub_ntl.mean(axis=0)[None, :, :], sub_pop[None, :, :]]
                            if bu_surface_proc is not None:
                                mean_ch.append(bu_surface_proc[y:y+patch_size, x:x+patch_size][None, :, :])
                            if bu_volume_proc is not None:
                                mean_ch.append(bu_volume_proc[y:y+patch_size, x:x+patch_size][None, :, :])
                            mean_img = np.concatenate(mean_ch, axis=0)  # (C, H, W)
                            mean_tensor = torch.from_numpy(mean_img).unsqueeze(0).to(device)
                            hm = get_gradcam_heatmap(model, mean_tensor)
                            heatmaps.append(hm)
                        except Exception as e:
                            st.warning(f"Grad-CAM skipped: {e}")
                            heatmaps.append(np.zeros((patch_size, patch_size)))

            if len(predictions) == 0:
                st.error("No valid patches found in the selected region (possibly masked by border filter).")
                st.stop()

            total_pop = sum(predictions)
            confidence = "High" if scorecard["q_total"] > 0.85 else "Medium" if scorecard["q_total"] > 0.6 else "Low"

        # Display results
        st.markdown("""
<div style="margin-top: 2rem; margin-bottom: 0.5rem;">
    <span style="font-size: 1.3rem; font-weight: 600; color: #e2e8f0;">📊 3. Results</span>
    <div style="height: 2px; width: 60px; background: linear-gradient(90deg, #60a5fa, #a78bfa); margin-top: 0.4rem; border-radius: 2px;"></div>
</div>
""", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Year", selected_year)
        c2.metric("Estimated Population", f"{total_pop:,.0f}")
        c3.metric("Data Quality (Q_total)", f"{scorecard['q_total']:.2f}")
        c4.metric("Confidence", confidence)

        with st.expander("📊 Data Quality Scorecard"):
            st.write(f"**Completeness:** {scorecard['completeness'][1]} ({scorecard['completeness'][0]:.1%})")
            st.caption("Fraction of pixels in your selected region that contain valid satellite data. Large rectangles often include water, mountains, or international borders with no data. Completeness < 85% triggers Low confidence.")
            
            st.write(f"**Consistency:** {scorecard['consistency'][1]}")
            st.caption("Checks for abnormal NTL saturation (>5% clipped pixels) or excessive darkness (>50% zero-brightness). PASS means the lights behave normally.")
            
            st.write(f"**Accuracy:** {scorecard['accuracy'][1]}")
            st.caption("Measures how many 'high-population but zero-light' pixels exist. High values indicate the model may underestimate population in unlit settlements.")
            
            st.write(f"**Relevance (MI):** {scorecard['relevance'][1]} (MI = {scorecard['relevance'][0]:.3f})")
            st.caption("Mutual Information between NTL and population. Higher MI = stronger NTL-population relationship in this region. > 0.1 is considered usable.")
            
            st.write(f"**Distribution Shift (MMD):** {scorecard['distribution_mmd'][1]} (MMD = {scorecard['distribution_mmd'][0]:.4f})")
            st.caption("Maximum Mean Discrepancy — how different the NTL distribution is from a reference 'normal' distribution. Large mixed urban-rural regions naturally score WARN due to bimodal distributions.")
            
            st.write(f"**Overall Quality Score (Q_total):** {scorecard['q_total']:.3f}")
            st.caption("Geometric mean of all five dimensions. A single FAIL dimension (especially Completeness) can collapse the entire score to near-zero.")

        # Grad-CAM visualization
        if heatmaps:
            st.markdown("""
<div style="margin-top: 2rem; margin-bottom: 0.5rem;">
    <span style="font-size: 1.3rem; font-weight: 600; color: #e2e8f0;">🔥 4. Explainability (Grad-CAM)</span>
    <div style="height: 2px; width: 60px; background: linear-gradient(90deg, #60a5fa, #a78bfa); margin-top: 0.4rem; border-radius: 2px;"></div>
</div>
""", unsafe_allow_html=True)
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].imshow(mean_ntl, cmap="magma", vmin=0)
            ax[0].set_title(f"Mean Nighttime Lights ({selected_year})")
            ax[0].axis("off")
            ax[1].imshow(mean_ntl, cmap="magma")
            ax[1].imshow(heatmaps[0], cmap="jet", alpha=0.5)
            ax[1].set_title("Attention Heatmap")
            ax[1].axis("off")
            st.pyplot(fig)

        # Understanding Results section
        st.markdown("""
<div style="margin-top: 2rem; margin-bottom: 0.5rem;">
    <span style="font-size: 1.3rem; font-weight: 600; color: #e2e8f0;">💡 5. Understanding Your Results</span>
    <div style="height: 2px; width: 60px; background: linear-gradient(90deg, #60a5fa, #a78bfa); margin-top: 0.4rem; border-radius: 2px;"></div>
</div>
""", unsafe_allow_html=True)
        with st.expander("📖 Click to learn what these numbers mean"):
            st.markdown("""
            **Why is my predicted population similar for 2020 and 2025?**
            
            This is expected behavior. The model was trained to predict **2025 population** from NTL patterns.
            When you select 2020 vs 2025, only the NTL input changes — the population target remains 2025.
            Think of it as: "Given the 2020 light pattern, what would the 2025 population be?"
            True temporal backcasting would require training data from multiple years of population censuses.
            
            ---
            
            **Why is my Confidence 'Low'?**
            
            Confidence is primarily driven by **Completeness** — the fraction of your selected rectangle that contains valid satellite data.
            - Large rectangles (>100×100 km) often span water bodies, mountains, or border gaps → Completeness drops → Confidence = Low
            - **Fix:** Zoom in and select a smaller region (~20–50 km) centered on a city like Lahore or Faisalabad
            
            ---
            
            **What is a 'good' Q_total score?**
            
            | Q_total | Confidence | Interpretation |
            |---------|-----------|----------------|
            | > 0.85 | High | Small, data-rich region — reliable estimate |
            | 0.6–0.85 | Medium | Moderate gaps — estimate is usable with caution |
            | < 0.6 | Low | Large data gaps — treat estimate as rough guidance |
            
            ---
            
            **Tips for best results:**
            1. Select a **tight rectangle** around a city center (e.g., 74.2–74.5 E, 31.4–31.7 N for Lahore)
            2. Avoid selecting areas that include the ocean, high mountains, or international borders
            3. The Grad-CAM heatmap shows which luminous clusters most influenced the prediction
            """)

        # Agentic Report
        st.markdown("""
<div style="margin-top: 2rem; margin-bottom: 0.5rem;">
    <span style="font-size: 1.3rem; font-weight: 600; color: #e2e8f0;">📄 6. Agentic Engineering Report</span>
    <div style="height: 2px; width: 60px; background: linear-gradient(90deg, #60a5fa, #a78bfa); margin-top: 0.4rem; border-radius: 2px;"></div>
</div>
""", unsafe_allow_html=True)
        with st.spinner("Retrieving literature and generating report..."):
            agent = PopulationReportAgent()
            report_md = agent.generate_report(
                region_name=f"Custom Bounds {bounds}",
                predicted_pop=total_pop,
                confidence=confidence,
                quality_scorecard=scorecard,
                year=selected_year,
            )
        st.markdown(report_md)

        st.download_button(
            label="📥 Download Report (Markdown)",
            data=report_md,
            file_name=f"population_estimate_report_{selected_year}.md",
            mime="text/markdown",
        )

st.markdown("---")
st.caption("Built with agentic coding principles from Session 11 | Reproducibility enforced via checkpoints & requirements.txt")
