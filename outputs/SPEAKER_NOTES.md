# Speaker Notes: NTL in Population
**Total Time: 4 minutes** (2.5 min slides + 1.5 min live demo)

---

## Slide 1: Title (15 sec) — 0:00 to 0:15
**What to say:**
"Good morning everyone. I'm [Your Name], and today I'll present 'NTL in Population' — our project on estimating Pakistan's population from nighttime light imagery. This is a team effort with [Author 2] on model engineering and [Author 3] on evaluation."

**Key point:** Set the stage. Mention it's a 3-person team.

---

## Slide 2: The Problem (20 sec) — 0:15 to 0:35
**What to say:**
"Traditional censuses are expensive and happen once a decade. Satellite nighttime lights offer a scalable alternative — but they fail in two ways. First, rural underprediction: dark areas don't mean zero people. Second, urban saturation: a dense slum and a skyscraper district can look equally bright. Our innovation is adding 3D building volume to break this saturation ceiling."

**Key point:** Urban saturation is the core problem. Volume is the solution.

---

## Slide 3: Data Pipeline (15 sec) — 0:35 to 0:50
**What to say:**
"We used four data sources. Seventy-two months of VIIRS nighttime lights from 2020 to 2025. WorldPop 2025 as our ground truth. And critically, GHSL built-up surface and building volume from 2020. Everything was aligned to a common 500-meter grid, giving us 4,225 valid training patches."

**Key point:** Mention the 72 months and the 4-channel input.

---

## Slide 4: Model Architecture (25 sec) — 0:50 to 1:15
**What to say:**
"Our model, TemporalPopulationRegressor, takes a 4-channel spatiotemporal input. Each month contributes NTL, population proxy, built-up surface, and building volume. A shared ResNet-18 encoder — pretrained on ImageNet — extracts spatial features from each month independently. A 1D temporal convolution aggregates 72 months into a single vector. And a regression head outputs log population per patch, with a hard clamp to prevent blow-ups."

**Key point:** Emphasize pretrained ResNet-18 and the 4-channel input.

---

## Slide 5: Ablation Study (20 sec) — 1:15 to 1:35
**What to say:**
"We ran five ablation experiments. Baseline with random weights gives R of 0.64. Adding ImageNet pretraining jumps us to 0.77 — that's a massive gain. Adding GHSL surface gets us to 0.88. But here's the surprising part: volume ALONE drops R to 0.61. Volume only works when paired with surface context. Together, they achieve our best R of 0.881 and reduce urban core bias from minus 57 to minus 28."

**Key point:** Volume alone hurts. Surface + volume is the winning combination.

---

## Slide 6: Best Model Results (20 sec) — 1:35 to 1:55
**What to say:**
"Our best model achieves Pearson R of 0.881 — that's state-of-the-art correlation for Pakistan at 500-meter resolution. MAE is 2.24 people per pixel. The scale factor is 1.012, meaning our predicted national total is essentially exact at 7.45 million. Rural correlation is nearly perfect at 0.958. Urban cores remain hardest at R 0.34, but that's a huge improvement from earlier baselines."

**Key point:** R = 0.881 and scale factor 1.01 are the headline numbers.

---

## Slide 7: Live Demo (5 sec transition) — 1:55 to 2:00
**What to say:**
"Let me show you this in action with our Streamlit app."

**Action:** Switch to browser, launch Streamlit.

---

## LIVE DEMO (90 sec) — 2:00 to 3:30
**What to do:**
1. **Map selection** (20 sec): Draw a rectangle around Lahore or Faisalabad.
2. **Click Predict** (10 sec): Let it load.
3. **Show results** (30 sec): Point out the estimated population, data quality scorecard, and confidence level.
4. **Grad-CAM** (20 sec): "The heatmap shows exactly which luminous clusters drove the prediction."
5. **Literature** (10 sec): "The report pulls real academic papers for grounding."

**Tip:** Have the app already running in a second browser tab to save time.

---

## Slide 8: Conclusion (15 sec) — 3:30 to 3:45
**What to say:**
"To conclude: fusing temporal NTL with 3D built-up structure achieves R 0.88. Pretraining and Huber loss are the biggest levers. Volume only works with surface context. And our interactive app makes this research accessible to end users. Future work includes stratified cross-validation stability, NASA Black Marble NTL, and transfer learning to neighboring countries."

**Key point:** Quick recap of the three main findings.

---

## Slide 9: Thank You (15 sec) — 3:45 to 4:00
**What to say:**
"Thank you. Questions?"

**Action:** Pause confidently. Be ready for: "Why does volume alone hurt?" or "What about other countries?"

---

## Anticipated Q&A

**Q: Why does volume alone hurt?**
A: "Volume is surface times average height. Without surface context, tall rural structures like water towers get misclassified as dense urban. Surface anchors the spatial footprint; volume adds the 3D nuance."

**Q: Can this work for India or Bangladesh?**
A: "Yes — the architecture is geography-agnostic. We'd need to realign the NTL and WorldPop rasters, but the pretrained model should transfer well."

**Q: Why is urban core R only 0.34?**
A: "Urban cores are inherently heterogeneous — informal settlements, commercial zones, and high-rises all mix together. We'd need additional features like road density or POI data to push this further."

**Q: Why are 2020 and 2025 predictions so similar?**
A: "The model predicts 2025 population regardless of year selected — the year only changes the NTL input. True temporal forecasting would require multi-year population ground truth."

---

## Pre-Presentation Checklist

- [ ] Fill in `[Your Name]`, `[Author 2]`, `[Author 3]` on Slide 1
- [ ] Fill in `[Author 1]`, `[Author 2]`, `[Author 3]` on Slide 9
- [ ] Start Streamlit app in background before presenting
- [ ] Open app at `http://localhost:8501` in second browser tab
- [ ] Test rectangle draw + predict on Lahore region
- [ ] Have PROJECT_REPORT.md open in case professor asks for details
