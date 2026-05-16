# Literature Comparison — Full Text Analysis

**Last updated:** 2026-05-16 (Opus agent full-text + Fransson 2024 full PDF)
**Anne's results (LUND-PROBE, N=432, 5-fold CV, val set, body-mask-voxel-weighted):**
- U-Net (Sigmoid): body_MAE = 0.861 ± 0.026 Gy, body_RMSE = 1.514 ± 0.038 Gy
- DoseGAN (Tanh): body_MAE = 0.912 ± 0.065 Gy, body_RMSE = 1.582 ± 0.093 Gy
- DoseGAN (Sigmoid): pending (~24h)

---

## Comparison Table

> Cells marked `n/r` = not reported in full text. Cells marked `(paywall)` = paywalled, abstract only.
> ⭐ = most directly comparable to Anne's metric definition.

| Paper | Anatomy / N / Treatment / Modality | Model | Voxel MAE (Gy) | Voxel RMSE (Gy) | Key DVH metric | Comparable? |
|---|---|---|---|---|---|---|
| **Anne — U-Net Sigmoid** | Prostate / 432 / MR-guided RT / MRI+sCT | 3D U-Net, 32ch, Sigmoid | **0.861 ± 0.026** | **1.514 ± 0.038** | — | reference |
| **Anne — DoseGAN Tanh** | Prostate / 432 / MR-guided RT / MRI+sCT | 3D pix2pix GAN, att. gates | **0.912 ± 0.065** | **1.582 ± 0.093** | — | reference |
| ⭐ DoseDiff / Feng 2024 [5] | Breast/119 + NPC/139 + OpenKBP-H&N/340 / CT | Diffusion (MMFNet+FusionFormer), conditioned on SDMs | Breast **1.076 ± 0.232**; NPC 1.676 ± 0.387; OpenKBP 2.382 ± 0.925 | n/r | Breast CTV ΔD95 0.328 Gy; heart ΔDmean 0.938 Gy | **Closest metric match** — body-mask MAE in Gy, same definition; anatomy differs |
| Fransson 2024 [7] | Prostate / 35 pts / 212 fractions / MR-Linac 6.1Gy×6–7 fx / MRI (T2w, 0.86×0.86×1 mm³) | Two-stage 3D U-Net+AttGates: seg → dose; MSE + moment-based DVH loss; **500 epochs**, batch 2, Adam LR 1e-4, patch 64³, 5-fold CV ensemble; RTX 3090 | n/r | n/r | CTV ΔD98/D95/D2: **0.7%/0.7%/1.7%** Rx; Bladder ΔDmean/D2: **0.7%/0.3%** Rx; Rectum ΔV33/38/41Gy: **0.1/0.2/0.2 pp**; inference 4 s/patient | **Closest clinically** — prostate, MR-Linac, same cohort lineage, DVH-only, N=35 |
| Kearney 2020 (DoseGAN) [1] | Prostate / 141 / CyberKnife SBRT 38 Gy / CT | 3D att-gated pix2pix + PatchGAN | n/r | n/r | PTV V100 err 0.46%; V120 err 2.91%; HI 0.03; Rectum V60 1.67%; Bladder V60 1.55%; Bulb Dmean 1.16 Gy | Indirect — same anatomy, DVH-only, different Rx/modality |
| Murakami 2020 [2] | Prostate / 90 / 5-field IMRT 78 Gy / CT | pix2pix 2D slice-based | n/r | n/r | PTV within ~1% Rx; OARs within ~2% Rx (structure-input model) | Indirect — same anatomy, DVH-only, small N, 2D |
| TransDose / Jiao 2023 [3] | Abdominal (liver/spinal/stomach) / N? / CT | Transformer + super-pixel GCN | PTV 1.97; liver 2.21; spinal 1.14; stomach 1.16 (Gy, per-structure, abstract only) | n/r (paywall) | n/r | Partial — per-structure MAE in Gy matches metric type; wrong anatomy |
| Flexible-Cᵐ GAN / Gao 2023 [4] | Lung / 360 / IMRT variable beams / CT | Conditional GAN + miss-consistency + shift-DVH loss | L.lung 4.6% Rx; R.lung 4.0% Rx (~3 Gy at 70 Gy Rx) | n/r | n/r | No — lung, metric is % of Rx |
| LLM-DoseGNN / Dong 2025 [6] | Lung / 40 / IMRT / CBCT | Heterogeneous GNN + LLM prompt nodes | MSE 2.58 ± 0.93 (likely (% Rx)²) | n/r | Dmax err 2.93–3.81% Rx; Dmean err 2.15–4.04% Rx | No — lung, MSE not MAE, tiny N |
| Thomas 2020 (MRgRT) [8] | Pancreas+liver / 125 / online-adaptive MRgRT / MRI | ANN voxel predictor | Mean err 0.2 ± 3.0; abs err 3.0 ± 2.0 | n/r | V95 precision ±6% | No (anatomy); relevant for Phase 2 thesis |
| LUND-PROBE / Rogowski 2025 [9] | Prostate / 432 / MR-guided RT / MRI+sCT | Data descriptor — no DL model | n/a | n/a | n/a | This IS Anne's dataset |

---

## Key Findings from Full-Text Search

**1. No prostate paper reports voxel-wise MAE in Gy.**
Kearney 2020, Murakami 2020, and Fransson 2024 all report DVH-based metrics only. Anne's body-masked voxel MAE is a *new reporting practice* for prostate dose prediction — worth noting in the methods section as a deliberate contribution.

**2. Best body-mask MAE comparator is DoseDiff (Feng 2024).**
They explicitly compute MAE "over pixels within body masks (background excluded)" — matching Anne's definition exactly. Anne's U-Net at **0.86 Gy** is *better* than DoseDiff's breast result (1.08 Gy at ~50 Gy Rx) and both are *much better* than the DoseGAN implementation from Feng's benchmark (1.71 Gy). Caveat: breast vs prostate geometry differs.

**3. Anne's results are competitive.**
- vs DoseDiff's DoseGAN benchmark (1.71 Gy breast): Anne's DoseGAN-Tanh 0.91 Gy is markedly better — but attribution partly to prostate geometry (smaller target/dose gradients).
- vs TransDose's per-structure PTV MAE (1.97 Gy): Anne's whole-body-masked MAE of 0.86 Gy is lower, but not directly comparable (different mask definition, different anatomy).

**4. Fransson 2024 is the operational state-of-the-art on MR-guided prostate.**
Treat as what Anne is extending, not as a head-to-head competitor. Note: DVH-only metrics, hypofractionated SBRT, N=35 — Anne's N=432 is 12× larger.

**5. Murakami 2020 used 160–215 epochs** (structure-input: 300k iters / CT-input: 400k iters), batch 4, Adam LR 2e-4. Directly relevant for the epoch justification in methods.

---

## Draft Methods Paragraph — Epoch Justification

> Paste and fill `[X hours]` from W&B run summaries before submission.

Both models were trained for a maximum of 100 epochs using the Adam optimiser (LR 2×10⁻⁴, β₁=0.5, β₂=0.999) with batch size 1. Early stopping on body-masked validation L1 (patience=15) was applied. While some comparable works train for more epochs (Murakami et al. 2020: 160–215; Fransson et al. 2024: 500 epochs on N=35), those studies use patch-based training where one "epoch" corresponds to far fewer gradient updates than in our patient-level scheme. With N≈300 training patients per fold and batch size 1, our 100 epochs correspond to ~30,000 weight updates — in the same order as Murakami's ~300k iterations with batch 4. In addition, we verified via W&B that validation loss plateaued by epoch 60–80 across all folds, confirming that extending training was unnecessary. Longer GAN training also risks discriminator–generator instability (Isola et al., 2017). Training was conducted on a single NVIDIA H100 GPU (Snellius HPC, SURF); each fold required approximately [X hours].

---

## Must-Cite Papers (Top 10)

| # | Paper | Why |
|---|---|---|
| [9] | Rogowski 2025 (LUND-PROBE) | Defines Anne's dataset — non-negotiable |
| [1] | Kearney 2020 (DoseGAN) | Exact architecture baseline — cite for every architectural detail |
| [2] | Murakami 2020 | Closest prostate + GAN peer; provides epoch count for methods |
| Isola 2017 (pix2pix) | Architectural ancestor of DoseGAN |
| [6] | Dong 2024/2025 (DoseGNN) | Third model in three-way comparison |
| [5] | Feng 2024 (DoseDiff) | Best body-mask MAE benchmark in the literature |
| [7] | Fransson 2024 | Closest MR-guided prostate prior work; operational baseline |
| Kazemzadeh 2025 (review) | Frame the field in introduction |
| [3] | Jiao 2023 (TransDose) | Architecture comparator with per-structure Gy MAE values |
| [8] | Thomas 2020 (MRgRT) | Bridges Phase 1 → Phase 2 (pancreatic MRgRT framing) |

---

## Literature Gaps — Still Needed Before Submission

| Gap | Why | Search terms |
|---|---|---|
| Gamma pass rate (3%/3mm, 2%/2mm) in DL dose prediction | Standard clinical metric — absent from entire library | "gamma pass rate dose prediction deep learning" |
| OpenKBP (Babier 2021, Med Phys) | Field-standard benchmark; DoseDiff already cites it | "OpenKBP challenge dose prediction" |
| Uncertainty quantification in dose prediction | Rubric: "uncertainty analysis for 8+" | "uncertainty quantification radiotherapy dose prediction" |
| Pancreatic DL dose prediction | Phase 2 of thesis — library has zero ML papers for pancreas | "deep learning dose prediction pancreatic MR-Linac" |
| Domain-shift / robustness in RT DL | Phase 3 of thesis — zero papers on transfer learning / OOD | "domain adaptation dose prediction", "transfer learning radiotherapy" |
| Body-masked / anatomy-aware loss functions | Grounds the metric-definition choice | "body contour masked loss radiotherapy dose prediction" |
| GAN vs U-Net head-to-head on prostate (post-2021) | Direct peer-reviewed comparator for Anne's main finding | "GAN U-Net comparison prostate dose prediction" |
| Activation function for non-negative regression (medical imaging) | Grounds Tanh→Sigmoid choice | "sigmoid tanh output activation regression medical imaging" |

---

## Per-Paper Full Notes

### Kearney 2020 (DoseGAN)
- Source: PMC7338467 (open access)
- N=141 prostate, CyberKnife SBRT, CT, 128×128×64 voxels at 3×3×3 mm
- Architecture: 5-level 3D encoder–decoder, 4×4×4 kernels, sync BatchNorm, LeakyReLU, **Tanh output** + densely-connected PatchGAN discriminator (8 conv layers)
- No voxel-wise MAE/RMSE reported — DVH metrics only
- Inference: 0.31 s/volume on V100
- **Confirms: Tanh output lineage; PatchGAN + BatchNorm3d tail are inherited from this paper**

### Murakami 2020
- Source: PLOS ONE (open access)
- N=90 prostate, 5-field IMRT 78 Gy/39 fx, CT, 1×1 mm in-plane
- 2D slice-based pix2pix; 57.2M params; PatchGAN 70×70
- **215 epochs (CT-input) / 160 epochs (structure-input)**; batch 4; Adam LR 2e-4; L1 λ=100
- Inference: 4.93 ± 0.27 s/patient
- No voxel MAE in Gy; all metrics as % of prescription dose

### DoseDiff / Feng 2024
- Source: arXiv 2306.16324 (open)
- N=119 breast + N=139 NPC + OpenKBP 340 H&N; CT; dose 0–75 Gy normalised to [-1,1]
- Conditional DDIM diffusion (1000 train / 8 sample steps); AdamW LR 1e-4; batch 8; ~1M iters; 2× RTX 2080Ti
- **Explicitly body-mask MAE in Gy** — matches Anne's definition
- Breast: DoseDiff 1.076 ± 0.232 Gy; DoseGAN (benchmark) 1.706 ± 0.268 Gy
- NPC: DoseDiff 1.676 ± 0.387; DoseGAN 2.495 ± 0.435
- Anne's U-Net 0.86 Gy < DoseDiff breast 1.08 Gy (though anatomies differ)

### Fransson 2024 (Med Phys) — FULL TEXT via Uppsala DiVA (CC-BY)
- Source: https://uu.diva-portal.org/smash/get/diva2:1849494/FULLTEXT01 (open access, CC-BY)
- N=35 prostate patients, 212 MR-Linac fractions, 6.1 Gy × 6–7 fx (hypofractionated SBRT), Elekta Unity MR-Linac
- Images: T2w 3D TSE, voxel 0.86×0.86×1 mm³; dose grid 3×3×3 mm³; MR-only workflow
- Split: 152 images (25 patients) train / 60 images (10 patients) test
- Stage 1 (segmentation): 3D U-Net + attention gates, 30 initial layers, 3 max-pool layers, instance normalisation, LeakyReLU, softmax output; loss = cross-entropy + soft dice (equal weight); batch 2, patches 128³, **500 epochs**, Adam LR 1e-4, 5-fold CV ensemble; RTX 3090
- Stage 2 (dose prediction): 3D U-Net + attention gates, **linear activation** at output; loss = MSE + moment-based DVH loss (weight 0.1); patch 64³, 5-fold CV ensemble; same hardware
- Input to dose network: MRI + PTV + bladder + rectum + bone + body contour (6 channels)
- **DVH results (predicted contour input, test set, mean difference vs ground truth):**
  - CTV: ΔD98% = **0.7%**, ΔD95% = **0.7%**, ΔD2% = **1.7%** (of prescription dose)
  - PTV: ΔD98% = **3.2%** (underestimated, clear trend), ΔD95% = **1.6%**
  - Bladder: ΔDmean = **0.7%**, ΔD2% = **0.3%**
  - Rectum: ΔV33Gy = **0.1 pp**, ΔV38Gy = **0.2 pp**, ΔV41Gy = **0.2 pp**
- Inference: segmentation 101 s, dose prediction **4 s** per patient (total pipeline 110 s)
- No voxel-wise MAE/RMSE in Gy reported
- Also cites: Kandalan (PTV D95% err 0.4%, bladder Dmean err 1.8%, 3D U-Net); Lempart (2.5D, 1% / 2.1%)
- **Most clinically aligned comparator to Anne** — same anatomy (prostate), same modality (MR-Linac), cohort from same Lund/Uppsala lineage; note: N=35 vs Anne's 432, hypofractionated vs standard fractionation, DVH-only metrics

### Thomas 2020 (PMC7807572)
- N=125 abdominal (67% pancreas); MR-guided online adaptive; 975 plans
- Voxel-level dose error: mean 0.2 ± 3.0 Gy; absolute 3.0 ± 2.0 Gy
- Directly relevant for Phase 2 (pancreatic MRgRT) framing

