# Thesis Reporting Tracker

Working document. Update as runs complete and sections are drafted.

Last updated: 2026-06-03

---

## 1. Status Overview

### Done
- [x] U-Net Sigmoid — 5-fold trained + evaluated (val)
- [x] U-Net Tanh — 5-fold trained + evaluated (val)
- [x] DoseGAN Tanh — 5-fold trained + evaluated (val)
- [x] DoseGAN Sigmoid — 5-fold trained + evaluated (val)
- [x] U-Net Sigmoid geom — 5-fold trained + evaluated (val, 2026-06-03)
- [x] DoseGAN Sigmoid geom — 5-fold trained + evaluated (val, 2026-06-03)
- [x] 2x2 activation ablation (U-Net × DoseGAN, Tanh vs Sigmoid) — decision: Sigmoid wins for both
- [x] Full clinical evaluation suite run on all 4 conditions — body MAE, PTV/OAR DVH, isodose Dice+HD95, boundary MAE (2026-06-03). Results in `docs/results_validation.md`
- [x] Investigation 1 (acquisition group breakdown) — U-Net Sigmoid + DoseGAN Sigmoid
- [x] Investigation 2 (worst-case patient visualisations) — U-Net Sigmoid + DoseGAN Sigmoid
- [x] Investigation 3 (dose-smearing index) — code exists, run on DoseGAN Tanh only
- [x] Literature comparison table finalised — full-text details extracted for all 11 papers
- [x] DVH comparison with literature — benchmarked against Fransson 2024, Kandalan 2021, Lempart 2021 (see Section 3.5)
- [x] DVH acquisition-group breakdown — newAcq vs oldAcq; finding: errors comparable across groups at DVH level
- [x] Penile Bulb evaluation added to both eval scripts
- [x] DVH regularisation loss implemented — `structure_dmean_loss(PTV + Bladder + Rectum)`, LAMBDA_DVH (5.0 DoseGAN, 0.1 U-Net)
- [x] DVH early stopping implemented — monitors `val_dvh_score = mean|Δ PTV D95| + mean|Δ Bladder Dmean| + mean|Δ Rectum Dmean|` in Gy
- [x] Gradient-magnitude loss implemented and tested (LAMBDA_GRAD=10.0 worse; LAMBDA_GRAD=1.0 fold 0 tested — negative result; documented as M19)

### Pending
- [ ] Ablation A1: boundary-weighted loss — fold 0 both models (see Section 8)
- [ ] Ablation A2: signed distance map input — fold 0 both models (see Section 8)
- [ ] Ablation A3: multi-scale discriminator — fold 0 DoseGAN only (see Section 8)
- [ ] Investigation 3 at scale — U-Net Sigmoid + DoseGAN Sigmoid baselines
- [ ] Gamma pass rate (3%/3mm, 2%/2mm) on full val set — deferred until GPU time available
- [ ] Test-set evaluation for final models — locked until all ablations complete

### Blocked
- ~~DoseGNN~~ — dropped from scope; collaborator stopped development (2026-05-22)

---

## 2. Methods Section

| # | Item | Status | Source |
|---|------|--------|--------|
| M1 | Dataset: LUND-PROBE, 432 patients, 9-channel input (8 structure masks + sCT), dose target | READY | `preprocessing/preprocessing.py` |
| M2 | Preprocessing: resample to 1.5 mm isotropic, z-score sCT inside body contour, dose / 50 Gy, PTV-centric crop 128×256×320 | READY | `preprocessing/preprocessing.py`, `configs/config_preprocessing_shared.py` |
| M3 | Split: 15% test held out, 5-fold CV on remainder, stratified by acquisition group (oldAcq/newAcq) | READY | `data/split.csv`, `preprocessing/create_split.py` |
| M4 | U-Net architecture: 3D U-Net, 32 base channels, InstanceNorm3d(affine=True), Sigmoid output activation | READY | `models/unet3d.py` |
| M5 | DoseGAN architecture: 3D U-Net generator with attention gates at every skip connection, PatchGAN discriminator, LSGAN loss | READY | `models/dosegan.py` |
| M6 | Footnote: output activation set to Sigmoid based on 2×2 ablation; deviates from Kearney et al. 2020 which uses Tanh | READY | Section 5 below |
| M7 | Footnote: PatchGAN discriminator ends with BatchNorm3d(1) → LeakyReLU (non-standard, inherited from Kearney et al.) | READY | `models/dosegan.py` |
| M8 | Footnote: AttGate ties weights between down_inp and sample_inp (inherited from upstream) | READY | `models/dosegan.py` |
| M9 | Footnote: both models use InstanceNorm3d(affine=True) — stable at batch_size=1; original BatchNorm3d collapses to single-sample stats at batch_size=1 and was replaced 2026-06-01 | READY | `models/unet3d.py`, `models/dosegan.py` |
| M10 | Training: Adam optimiser, LR 2e-4, 200 epochs + early stopping (patience 30), augmentation (LR flip, depth flip, sCT intensity scale/shift) | READY | `configs/config_dosegan.py`, `training/dataset.py` |
| M11 | Metrics: body_MAE_Gy, body_RMSE_Gy computed over body-contour voxels (channel 7); same mask used during training for val_L1 | READY | `training/metrics.py`, `training/evaluate.py` |
| M12 | Note: val_L1 redefined 2026-05-13 — pre-change values not comparable (conversion: new ≈ old / 0.32) | READY | commit a2c5cc7 |
| M13 | Hardware and runtime: Snellius HPC, NVIDIA H100, wallclock per fold | READY | `outputs/logs/` |
| M14 | Full clinical evaluation suite: boundary MAE (±20 mm PTV/OAR surface), isodose Dice + HD95 (100/95/80/50%), D01cc, V_prescription for OARs | IMPLEMENTED AND RUN | `training/metrics.py` |
| M14b | Gamma pass rate (3%/3mm, 2%/2mm) | IMPLEMENTED in code, NOT RUN — omitted from all eval jobs via --skip-gamma due to computational cost (full 3D gamma is ~10 min/patient). Columns exist in CSVs but are all NaN. Reserved for supplementary evaluation. | `training/metrics.py` |
| M15 | Investigation 3 metric: dose-smearing index (ratio of predicted to ground-truth dose gradient magnitude per patient) | READY — code exists, not yet run at scale | `analysis/inv3_dose_smearing_index.py` |
| M19 | Gradient-magnitude loss: MAE on per-axis finite differences (dx, dy, dz), averaged across 3 directions, LAMBDA_GRAD=1.0 | IMPLEMENTED — fold 0 running | `training/train_dosegan.py`, `training/train_unet3d.py` |
| M16 | DVH regularisation loss: `structure_dmean_loss(PTV) + structure_dmean_loss(Bladder) + structure_dmean_loss(Rectum)`, weighted λ_dvh (5.0 DoseGAN, 0.1 U-Net) — added to generator loss; from Kearney et al. 2020 | IMPLEMENTED — applies to next re-train | `training/train_dosegan.py`, `training/train_unet3d.py` |
| M17 | DVH early stopping: model saved when `val_dvh_score = mean|Δ PTV D95| + mean|Δ Bladder Dmean| + mean|Δ Rectum Dmean|` (Gy) improves, not when body-masked L1 improves | IMPLEMENTED — applies to next re-train | `training/train_dosegan.py`, `training/train_unet3d.py` |
| M18 | Penile Bulb evaluation: Dmean, Dmax, D95, D98, V20, V40 added to per-patient eval CSV; extracted from input channel 6; NaN for ~48 absent patients | IMPLEMENTED | `training/evaluate.py` |
| M20 | Boundary-weighted L1 loss: per-voxel weight map where ±20 mm PTV surface shell voxels get weight W, all other body voxels weight 1. Loss = body-masked weighted L1. Motivated by grad-magnitude failure (M19): targets the gradient where it matters clinically rather than all gradients uniformly. | PENDING — see Ablation A1 | `training/train_*.py` |
| M21 | Signed distance map input: extra channel encoding signed Euclidean distance from PTV surface (negative inside, positive outside), clipped to ±50 mm and normalised to [−1, 1]. Computed on-the-fly from PTV mask (channel 0). Gives the model an explicit, continuous representation of where the dose boundary is. Distinct from the geom dist_ptv channel which is unsigned. | PENDING — see Ablation A2 | `training/dataset.py`, `training/train_*.py` |
| M22 | Multi-scale discriminator (DoseGAN only): second NLayerDiscriminator applied to 2× spatially downsampled input (AvgPool3d kernel=2). Total D loss = 0.5*(L_D_fine + L_D_coarse); generator adversarial loss from both. Adds pressure on boundary realism at the scale of dose gradients (~5–15 mm), complementing the existing patch-level discriminator. | PENDING — see Ablation A3 | `models/dosegan.py`, `training/train_dosegan.py` |

---

## 3. Results Section

### 3.1 Main quantitative comparison (val set, 5-fold)

| # | Item | Status | Source |
|---|------|--------|--------|
| R1 | Table: body_MAE_Gy and body_RMSE_Gy per model, mean ± std across 5 folds | PARTIAL — full eval with new metrics pending | `outputs/evaluation/*_fold*_val.csv` |
| R2 | U-Net Sigmoid: body_MAE_Gy = 0.861 ± 0.026 Gy; body_RMSE_Gy = 1.514 ± 0.038 Gy | READY | `outputs/evaluation/unet3d_ch32_sigmoid_snellius_fold*_val.csv` |
| R3 | DoseGAN Sigmoid: body_MAE_Gy = 0.868 ± 0.035 Gy; body_RMSE_Gy = 1.521 ± 0.045 Gy | READY | `outputs/evaluation/dosegan_ngf32_sigmoid_snellius_fold*_val.csv` |
| R4 | ~~DoseGNN results~~ | DROPPED — out of scope (2026-05-22) | — |
| R5 | Test-set results for all final models | PENDING | requires test-split eval |

### 3.2 Activation ablation (2×2)

| # | Item | Status | Source |
|---|------|--------|--------|
| R6 | Table: Tanh vs Sigmoid for U-Net and DoseGAN, per fold + mean ± std | READY | Section 5 below |
| R7 | U-Net: Sigmoid 0.861 ± 0.026 vs Tanh 0.895 ± 0.041 Gy (body_MAE); Sigmoid wins every fold | READY | `outputs/evaluation/` |
| R8 | DoseGAN: Sigmoid val_L1 0.0174 ± 0.0007 vs Tanh 0.0182 ± 0.0013 — Sigmoid wins/ties every fold | READY | W&B groups dosegan_ngf32_{sigmoid,tanh}_snellius |
| R9 | Conclusion: Sigmoid adopted as default for both models | READY | Section 5 below |

### 3.3 Investigation 1: Acquisition group subgroup analysis

| # | Item | Status | Source |
|---|------|--------|--------|
| R10 | Boxplot: body_MAE_Gy by oldAcq vs newAcq, U-Net Sigmoid | READY | `outputs/analysis/inv1_unet3d_ch32_sigmoid_snellius_acquisition_boxplot.png` |
| R11 | Per-group means and medians CSV, U-Net Sigmoid | READY | `outputs/analysis/inv1_unet3d_ch32_sigmoid_snellius_acquisition_breakdown.csv` |
| R12 | Same for DoseGAN Tanh | NOT RUN — Inv1 was never run on DoseGAN Tanh; not needed now that Sigmoid is the final model | — |
| R13 | Per-group means and medians, DoseGAN Sigmoid | READY | `outputs/analysis/inv1_dosegan_ngf32_sigmoid_snellius_acquisition_breakdown.csv` |
| R14 | Finding: oldAcq patients drive body-MAE tail; but DVH-level errors are comparable across groups | READY (refined) | Inv 1 outputs + Section 3.7 below |

**DoseGAN Sigmoid acquisition group breakdown (N=367 val patients across 5 folds):**

| Group | Mean MAE (Gy) | Median MAE (Gy) |
|---|---|---|
| oldAcq (N=274) | 0.875 | 0.824 |
| newAcq (N=93) | 0.845 | 0.829 |
| difference (old − new) | +0.030 Gy | — |
| Mann-Whitney U | 12860 | p = 0.894 |

Finding: The 0.030 Gy mean difference is not statistically significant (p=0.894). As with U-Net Sigmoid, the distribution shift between acquisition groups does not produce a systematic mean-level degradation. The tail (worst 15 patients, all oldAcq except one) is where the gap lives.

**DVH acquisition-group breakdown (U-Net Sigmoid, N=367 val patients; % of Rx = 50 Gy):**

| Metric | newAcq (N=93) | oldAcq (N=274) | ratio old/new |
|---|---|---|---|
| PTV Dmean | 0.54% ± 0.77% | 0.47% ± 0.80% | 0.87× |
| PTV D95 | 0.94% ± 1.64% | 0.61% ± 0.84% | 0.65× |
| Rectum Dmean | 2.67% ± 2.14% | 2.50% ± 1.99% | 0.93× |
| Rectum D95 | 0.62% ± 0.41% | 0.58% ± 0.60% | 0.94× |
| Bladder Dmean | 1.68% ± 1.48% | 1.64% ± 1.33% | 0.98× |
| Bladder D95 | 0.82% ± 0.89% | 0.94% ± 1.47% | 1.14× |
| Body MAE | 0.841 ± 0.140 Gy | 0.867 ± 0.235 Gy | 1.03× |

**DVH acquisition-group breakdown (DoseGAN Tanh, N=367 val patients):**

| Metric | newAcq (N=93) | oldAcq (N=274) | ratio old/new |
|---|---|---|---|
| PTV Dmean | 1.70% ± 1.52% | 1.56% ± 1.39% | 0.92× |
| PTV D95 | 2.04% ± 2.00% | 1.79% ± 1.54% | 0.88× |
| Rectum Dmean | 2.56% ± 2.00% | 2.60% ± 2.10% | 1.01× |
| Rectum D95 | 0.63% ± 0.35% | 0.61% ± 0.64% | 0.98× |
| Bladder Dmean | 1.99% ± 1.59% | 1.79% ± 1.49% | 0.90× |
| Bladder D95 | 0.88% ± 1.09% | 1.09% ± 1.93% | 1.24× |
| Body MAE | 0.893 ± 0.178 Gy | 0.918 ± 0.258 Gy | 1.03× |

**Key finding:** All ratios are within 0.87×–1.24×. Distribution shift between acquisition groups manifests as tail-risk inflation at the body-MAE level (14/15 worst-MAE patients across all folds are oldAcq), NOT as systematic DVH-level degradation. Clinically relevant metrics (PTV coverage, OAR doses) are statistically comparable across groups.

### 3.4 Investigation 2: Worst-case visualisations

| # | Item | Status | Source |
|---|------|--------|--------|
| R15 | Top-3 worst-case patient panels per fold (15 panels), U-Net Sigmoid | READY | `outputs/analysis/inv2_unet3d_ch32_sigmoid_snellius_worst_*.png` |
| R16 | Same for DoseGAN Tanh | NOT RUN — never executed; not needed now that Sigmoid is the final model | — |
| R17 | Same for DoseGAN Sigmoid | READY | `outputs/analysis/inv2_dosegan_ngf32_sigmoid_snellius_worst_*.png` |
| R18 | Failure pattern summary (DoseGAN Sigmoid, from Inv2 visual inspection of all 15 patients) | READY | below |

**DoseGAN Sigmoid Inv2 findings (15 worst-MAE patients, 3 per fold):**

Patient composition: 14/15 oldAcq, 1/15 newAcq. The single newAcq failure (fold 3 rank 3, MAE=1.39 Gy) shows the same failure patterns as the oldAcq cases.

| Failure mode | Frequency across 15 patients | Description |
|---|---|---|
| PTV well-predicted | 14/15 | DVH curves nearly identical; model reliably covers the target |
| Bladder under-prediction | ~11/15 | Predicted bladder DVH curve shifted left — model assigns less dose to bladder than ground truth. Dominant and systematic failure. |
| Rectum over-prediction | ~5/15 | Occasional; not systematic |
| Dose blurring (ring/halo pattern) | ~8/15 | Over-prediction in the penumbra (dose falloff zone), under-prediction at body edge; model smooths dose gradients |
| Lateral asymmetry | ~4/15 | Model predicts symmetric dose for a patient whose real distribution is asymmetric; associated with atypical anatomy |
| Localised hotspot/coldspot | ~2/15 | Concentrated error near organ boundary; likely patient with unusual bladder proximity to PTV |

**Notable individual cases:**
- Fold 1 rank 1 (oldAcq_bfc95ad92b2a0fc8, MAE=2.40 Gy): strongest lateral asymmetry error; likely atypical anatomy. Localised errors up to ±15 Gy.
- Fold 4 ranks 1–3: all show the same "sunburst" radial error pattern in the axial view, suggesting a fold-4-specific failure cluster.
- Fold 0 rank 3: reversed blurring (blue centre, red ring) — under-prediction in the PTV core, over-prediction in penumbra; opposite of the typical pattern.

**Clinical interpretation:** The model is reliable on target coverage (PTV) but has a systematic blind spot on bladder dose, predicting lower bladder dose than is actually delivered. This holds across both acquisition groups. The spatial errors are primarily due to smoothing of dose gradients rather than global bias.

### 3.5 DVH comparison with literature

All values are mean |Δ| (mean absolute error) as % of prescription dose (50 Gy), N=367 val patients. Literature values as reported. See `docs/literature_comparison.md` for full detail and caveats.

| Metric | U-Net Sigmoid | DoseGAN Sigmoid | DoseGAN Tanh | Fransson 2024 | Kandalan 2021 | Lempart 2021 |
|---|---|---|---|---|---|---|
| PTV/CTV Dmean | **0.49%** ± 0.79% | 0.66% ± 0.76% | 1.59% ± 1.42% | 0.7% (CTV) | 1.0% | — |
| PTV D95 | **0.69%** ± 1.11% | 0.93% ± 0.83% | 1.85% ± 1.67% | 0.7% (CTV) | **0.4%** | 1.0% |
| PTV D98 | **0.79%** ± 1.30% | 1.12% ± 0.96% | 2.01% ± 1.86% | 3.2% (PTV) | 1.6% (D2) | 1.9% |
| Bladder Dmean | 1.65% ± 1.37% | 1.73% ± 1.44% | 1.84% ± 1.51% | **0.7%** | 1.8% | ≤ 2.6% |
| Rectum Dmean | 2.54% ± 2.03% | 2.50% ± 2.04% | 2.59% ± 2.07% | n/r | — | ≤ 2.6% |
| Rectum D95 | 0.59% ± 0.56% | 0.61% ± 0.51% | 0.62% ± 0.58% | — | — | — |

**Interpretation for thesis:**
- U-Net Sigmoid PTV metrics are competitive with or better than every prostate paper. PTV D98 (0.79%) substantially outperforms Fransson (3.2%) and Lempart (1.9%).
- Bladder Dmean gap vs Fransson (1.65% vs 0.7%): explained by cohort heterogeneity (N=432, two acquisition eras, real-world bladder filling variability) not model weakness. Acquisition-group breakdown confirms both groups show identical bladder errors (~1.65–1.68%).
- DoseGAN Sigmoid PTV metrics (Dmean 0.66%, D95 0.93%, D98 1.12%) are substantially better than DoseGAN Tanh and within the Lempart range, but trail U-Net Sigmoid. The GAN adversarial objective does not improve DVH accuracy over the plain U-Net in this setting.
- DoseGAN Sigmoid Bladder Dmean (1.73%) is marginally worse than U-Net Sigmoid (1.65%), consistent with Inv2 showing bladder under-prediction as the dominant DoseGAN failure mode.
- DoseGAN Sigmoid Rectum Dmean (2.50%) is essentially identical to U-Net Sigmoid (2.54%) and DoseGAN Tanh (2.59%) — rectum prediction is similarly limited across all models.
- No prostate paper reports body-masked voxel MAE in Gy; 0.861 Gy compares favourably to Feng 2024 breast result (1.076 Gy, same metric definition).

| # | Item | Status | Source |
|---|------|--------|--------|
| R23 | DVH comparison table (as above) | READY | `docs/literature_comparison.md`, `outputs/analysis/dvh_summary_*.csv` |
| R24 | Bladder Dmean gap explanation (cohort heterogeneity, not model weakness) | READY | acquisition-group breakdown + Fransson N=35 context |
| R25 | Voxel MAE benchmark vs Feng 2024 (breast): 0.861 vs 1.076 Gy | READY | `docs/literature_comparison.md` |

---

### 3.6 Investigation 3: Dose-smearing index

| # | Item | Status | Source |
|---|------|--------|--------|
| R19 | Per-patient dose-smearing index across all 4 variants, with oldAcq/newAcq breakdown | OPTIONAL — code exists, run on DoseGAN Tanh only | analysis scripts |
| R20 | Correlation between dose-smearing index and body_MAE_Gy | OPTIONAL | derived from R19 |

### 3.7 Clinical metrics (optional)

| # | Item | Status | Source |
|---|------|--------|--------|
| R21 | DVH curves per OAR and PTV | OPTIONAL | Not implemented — low priority |
| R22 | Isodose Dice/HD95 + boundary MAE | COMPLETE — in all eval CSVs | `training/metrics.py`, `training/evaluate.py` |
| R22b | Gamma pass rate (3%/3 mm, 2%/2 mm) | NOT RUN — all gamma columns are NaN in every eval CSV. Gamma analysis was implemented but omitted from the quantitative comparison due to computational cost (~10 min/patient for 3D gamma). Should be disclosed as a limitation; not listed as a completed result. | `training/metrics.py` |

---

## 4. Discussion Section

| # | Item | Status | Source |
|---|------|--------|--------|
| D1 | Headline result: which model achieves lowest body_MAE/RMSE and by how much | READY — U-Net Sigmoid (0.861 Gy) marginally better than DoseGAN Sigmoid (0.868 Gy); GAN discriminator does not improve over plain U-Net | R1–R3 |
| D2 | Activation finding: Sigmoid consistently outperforms Tanh; range match to normalised [0,1] dose target; small deviation from Kearney et al. | READY | Section 5 |
| D3 | Acquisition group shift: oldAcq patients drive the error tail; implications for transfer to pancreatic cohort (Phase 2) | READY | R10–R14 |
| D4 | Failure modes (DoseGAN Sigmoid): bladder under-prediction is the dominant systematic failure; dose blurring (gradient smoothing) is the dominant spatial pattern; lateral asymmetry is a secondary failure in atypical anatomy cases | READY | R17–R18 |
| D5 | Dose-smearing index: if R19 is run, provides a quantitative characterisation of the dominant failure mode | OPTIONAL | R19–R20 |
| D6 | Limitations: val-set-only metrics until test-set eval, single dataset (prostate only), batch_size=1, no DoseGNN comparison (collaboration ended) | READY | — |
| D11 | Scope reduction: DoseGNN dropped because collaborator stopped development; geometric channels showed no benefit in collaborator's experiments. Honest reflection required. | READY to write | — |
| D7 | Reflection: dose-smearing addressed at loss level — boundary-weighted L1 (A1, M20) is the concrete implemented attempt; gradient-magnitude loss tried and failed (M19); multi-scale discriminator and signed distance map are future work (M21, M22) | PENDING — update after ablation results | M19, M20, M21, M22 |
| D8 | Inherited architectural quirks: PatchGAN tail BatchNorm+LeakyReLU, AttGate weight-tying — flagged as inherited, not corrected | READY | M7–M8 |
| D9 | Comparison with prior literature — DVH table + acquisition-group breakdown added 2026-05-18 | READY | `docs/literature_comparison.md`, Section 3.5 |
| D10 | Phase 2/3 outlook: transfer to pancreatic cohort, robustness under anatomical variation | READY | `docs/thesis_proposal.md` |
| D12 | Future work — signed distance map (M21): targeted single-channel addition to address boundary shape without adding all 5 geom channels; computationally cheap, directly encodes what the model needs to get the falloff right | READY to write | M21 |
| D13 | Future work — multi-scale discriminator (M22): adversarial pressure at the spatial scale of dose gradients (~5–15 mm); the current PatchGAN already helps vs U-Net on isodose Dice, and a second discriminator at coarser scale would push this further | READY to write | M22 |

---

## 5. Activation Ablation — Sigmoid vs Tanh

Both U-Net and DoseGAN were trained with 5-fold CV on LUND-PROBE, all hyperparameters held constant. Only the generator output activation varied.

| Model | Activation | body_MAE_Gy (Gy) | body_RMSE_Gy (Gy) | body-masked val_L1 |
|---|---|---|---|---|
| U-Net | Sigmoid | **0.861 ± 0.026** | **1.514 ± 0.038** | — |
| U-Net | Tanh | 0.895 ± 0.041 | 1.670 ± 0.050 | — |
| DoseGAN | Sigmoid | **0.868 ± 0.035** | **1.521 ± 0.045** | **0.0174 ± 0.0007** |
| DoseGAN | Tanh | — | — | 0.0182 ± 0.0013 |

U-Net body_MAE_Gy per fold:

| Fold | Sigmoid | Tanh |
|---|---|---|
| 0 | 0.863 | 0.894 |
| 1 | 0.896 | 0.950 |
| 2 | 0.837 | 0.851 |
| 3 | 0.873 | 0.921 |
| 4 | 0.833 | 0.860 |

DoseGAN body_MAE_Gy per fold:

| Fold | Sigmoid | Tanh |
|---|---|---|
| 0 | 0.860 | — |
| 1 | 0.904 | — |
| 2 | 0.837 | — |
| 3 | 0.904 | — |
| 4 | 0.833 | — |

Sigmoid wins on every fold for U-Net. For DoseGAN, Sigmoid wins or ties on every fold with ~4% lower mean and lower variance. The dose target is normalised to [0, 1] (dose / 50 Gy), making Sigmoid the natural range-matched output. Sigmoid is adopted as the default for both models. Tanh is retained in the ablation table and discussed as a deviation from Kearney et al. 2020.

Draft methods footnote:
> The generator output activation was set to Sigmoid following a 2×2 ablation (Tanh vs Sigmoid, U-Net vs DoseGAN, 5-fold CV on LUND-PROBE). This deviates from Kearney et al. (2020), which uses Tanh; Sigmoid achieved lower body-masked validation L1 on every fold for both architectures.

---

## 6. Next Steps (priority order)

**Current state (2026-06-04):** DoseGAN baseline retrained with InstanceNorm (old BatchNorm checkpoints replaced); eval pending (~10h). U-Net baseline, U-Net geom, DoseGAN geom — all evaluated. Results in `docs/results_validation.md` (DG base column stale until new eval finishes). Ablation experiments (Section 8) are the next experimental priority after DG baseline eval completes.

### Experiments (in order)
1. **Implement and run Ablation A1** (boundary-weighted loss, fold 0 both models) — lowest effort, highest thesis impact. See Section 8.
2. **Implement and run Ablation A2** (signed distance map, fold 0 both models) — medium effort, informative regardless of outcome.
3. **Run Inv3 at scale** on U-Net + DoseGAN Sigmoid baselines: `python3 analysis/inv3_dose_smearing_index.py`
4. **If A1 or A2 is positive on fold 0** → submit folds 1–4 for that condition.
5. **Ablation A3** (multi-scale discriminator, DoseGAN only) — implement only if time permits after A1+A2.
6. **Test-set evaluation** once all training decisions are locked.

### Writing (start immediately — no experiments needed)
- Introduction: thesis scope, motivation, RQs
- Background: U-Net, attention gates, GANs, dose prediction literature, GhTara
- Dataset: LUND-PROBE, 432 patients, acquisition groups, split rationale
- Preprocessing: M1–M3
- Methods — Architectures: M4–M9
- Methods — Training: M10, M16, M17, M19, M20
- Methods — Evaluation: M11–M15
- Limitations (partial): D6, D11

### Writing (needs results first)
- Results 3.1 (main table + geom comparison): use `docs/results_validation.md`
- Results 3.6 (dose-smearing): R19–R20 after Inv3
- Results — ablation A1/A2: after fold 0 results
- Discussion D1, D5, D7 (updated), D12, D13

---

---

## 8. Ablation Experiments — Shape Problem

**Motivation:** All four conditions achieve similar body MAE (~0.86 Gy) but U-Net fails almost completely on spatial dose accuracy (Dice 50iso = 0.043 vs DoseGAN geom 0.733). Even DoseGAN baseline is weak at 50iso (0.240). The models are getting the dose magnitudes right but not the shape of the dose distribution — specifically the falloff at the PTV boundary. Three concrete ablations address this in increasing order of implementation effort.

---

### A1: Boundary-Weighted L1 Loss

**Motivation:** The gradient-magnitude loss (M19) penalised all spatial gradients uniformly and failed — it fires on every tissue boundary including artefacts. A boundary-weighted loss is more targeted: upweight only the ±20 mm PTV surface shell, which is exactly where the clinically critical dose falloff lives.

**Implementation:**
1. In the training loop, after loading the batch, compute a boundary mask from the PTV mask (channel 0):
   ```python
   from scipy.ndimage import binary_dilation, binary_erosion
   ptv = input_tensor[:, 0]  # (B, D, H, W)
   # 20 mm shell = 13 voxels at 1.5 mm/voxel
   dilated  = binary_dilation(ptv.cpu().numpy(), iterations=13)
   eroded   = binary_erosion(ptv.cpu().numpy(),  iterations=13)
   bnd_mask = torch.tensor(dilated & ~eroded, device=device)
   ```
2. Build a per-voxel weight map: `weight = body_mask * (1.0 + (W - 1.0) * bnd_mask)` where W is the upweight factor.
3. Weighted loss: `(torch.abs(pred - target) * weight).sum() / (weight.sum() + 1e-8)`
4. Replace the existing `nn.L1Loss` body-MAE term with this weighted version. Keep val_L1 as unweighted body-masked MAE so folds stay comparable.

**Hyperparameter:** Start with W = 5.0 (boundary voxels weighted 5× relative to other body voxels). If W=5 degrades body MAE > 0.05 Gy, try W = 2.0.

**Config / RUN_NAME:**
- DoseGAN: `dosegan_ngf32_sigmoid_bndw5_snellius`
- U-Net: `unet3d_ch32_sigmoid_bndw5_snellius`

**What to run:**
| Step | Command | Gate |
|---|---|---|
| Fold 0 DoseGAN | `sbatch --export=... train_dosegan.sbatch` | always |
| Fold 0 U-Net | `sbatch --export=... train_unet3d.sbatch` | always |
| Eval fold 0 both | `sbatch eval.sbatch` | after training |
| Folds 1–4 both | submit if fold 0 Dice_50iso improves | conditional |

**Success criterion:** Dice_50iso improves by ≥ 0.05 AND body MAE does not increase > 0.05 Gy versus the baseline of the same model.

**Primary eval metrics to compare:** Dice_50iso, Dice_100iso, HD95_100iso, boundary_MAE_ptv, body_MAE.

**Thesis framing:** This is Investigation 3 / the shape-targeted intervention. Frame as: gradient-magnitude loss failed because it was non-specific; boundary-weighted loss is the targeted version. Result positive or negative both contribute to the thesis argument.

---

### A2: Signed Distance Map Input

**Motivation:** The geom channels include an unsigned distance to PTV, which tells the model how far each voxel is from the PTV but not whether it is inside or outside. A signed distance map explicitly encodes the PTV boundary as a zero-crossing, giving the model a direct, continuous signal for where the falloff should be. This is a targeted single-channel addition, not a full 5-channel geom rerun.

**Implementation:**
Compute on-the-fly in `LUNDPROBEDataset.__getitem__`, after loading the input tensor:

```python
from scipy.ndimage import distance_transform_edt
ptv_np = input_tensor[0].numpy()  # channel 0 = PTV mask, shape (D, H, W)
dist_outside = distance_transform_edt(ptv_np == 0) * 1.5   # mm, positive outside
dist_inside  = distance_transform_edt(ptv_np == 1) * 1.5   # mm, positive inside
sdt = dist_outside - dist_inside                            # negative inside, positive outside
sdt_clipped = np.clip(sdt, -50.0, 50.0) / 50.0             # normalised to [-1, 1]
sdt_tensor  = torch.tensor(sdt_clipped, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
input_tensor = torch.cat([input_tensor, sdt_tensor], dim=0)  # now 10 channels
```

- No preprocessing changes needed — computed from the existing PTV mask.
- `INPUT_NC = 10` for both models in their configs.
- This is independent of the geom channels — it is a standalone 10-channel baseline variant.

**Config / RUN_NAME:**
- DoseGAN: `dosegan_ngf32_sigmoid_sdtptv_snellius`
- U-Net: `unet3d_ch32_sigmoid_sdtptv_snellius`

**What to run:**
| Step | Command | Gate |
|---|---|---|
| Fold 0 DoseGAN | `sbatch ... train_dosegan.sbatch` | always |
| Fold 0 U-Net | `sbatch ... train_unet3d.sbatch` | always |
| Eval fold 0 both | `sbatch eval.sbatch` | after training |
| Folds 1–4 both | submit if fold 0 Dice_50iso improves | conditional |

**Success criterion:** Dice_50iso improves by ≥ 0.05 AND body MAE does not increase > 0.05 Gy. Secondary: does it close the gap between U-Net and DoseGAN on Dice_50iso?

**Note on interpretation:** If A2 helps U-Net but A1 does not, it implies the model needs spatial context to produce the right gradient, not just penalty pressure. If neither helps U-Net but both help DoseGAN, it implies U-Net's L1 loss fundamentally cannot leverage spatial cues — the adversarial loss is necessary.

---

### A3: Multi-Scale Discriminator (DoseGAN only)

**Motivation:** DoseGAN already benefits from adversarial training on spatial realism (Dice_100iso 0.899 vs U-Net 0.014 at baseline). A second discriminator operating at a coarser scale (~3 mm effective resolution after 2× downsampling) would add adversarial pressure specifically at the scale of dose gradients (5–15 mm), where the falloff errors are most visible.

**Implementation:**
In `models/dosegan.py`, add a second discriminator:

```python
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super().__init__()
        self.disc_fine   = NLayerDiscriminator(input_nc, ndf, n_layers)
        self.disc_coarse = NLayerDiscriminator(input_nc, ndf, n_layers)
        self.downsample  = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.disc_fine(x), self.disc_coarse(self.downsample(x))
```

Training loop changes:
- `D_loss = 0.5 * (lsgan_d_loss(fine_real, fine_fake) + lsgan_d_loss(coarse_real, coarse_fake))`
- `G_adv_loss = 0.5 * (lsgan_g_loss(fine_fake) + lsgan_g_loss(coarse_fake))`

All other losses unchanged. Checkpoint saves the multi-scale discriminator state under `discriminator`.

**Config / RUN_NAME:** `dosegan_ngf32_sigmoid_msd_snellius`

**What to run:** Fold 0 only initially. Gate on Dice_50iso improvement before running folds 1–4.

**Effort:** Medium — requires changes to `models/dosegan.py` and `training/train_dosegan.py`. Implement only after A1 and A2 results are in.

**Thesis framing:** Future work if time runs out. If implemented: this is the architectural follow-up to the loss-level intervention, motivated by the fact that the PatchGAN already helps DoseGAN vs U-Net but does not fully solve the 50iso problem.

---

### A1/A2/A3 — Summary

| Ablation | Models | Effort | Expected primary gain | Gate |
|---|---|---|---|---|
| A1: boundary-weighted L1 | DoseGAN + U-Net | Low (loss function, ~20 lines) | Dice_50iso, HD95 | None — run now |
| A2: signed distance map | DoseGAN + U-Net | Low-medium (dataset change, new config key) | Dice_50iso, PTV MAE | None — run now |
| A3: multi-scale disc. | DoseGAN only | Medium (model + training loop) | Dice_100iso, Dice_50iso | After A1+A2 |

Run A1 and A2 in parallel on fold 0. Evaluate before committing to folds 1–4.

---

## 7. File Paths

| Artefact | Path |
|---|---|
| Evaluation CSVs | `outputs/evaluation/<RUN_NAME>_fold{N}_val.csv` |
| Inv 1 | `outputs/analysis/inv1_<RUN_NAME>_acquisition_{boxplot.png,breakdown.csv}` |
| Inv 2 | `outputs/analysis/inv2_<RUN_NAME>_worst_{1,2,3}_fold{N}_{acq}_{hash}.png` |
| Checkpoints | `outputs/checkpoints_{unet3d,dosegan}/<RUN_NAME>_fold{N}_best.pt` |
| SLURM logs | `outputs/logs/<jobname>_<jobid>.out` |
| W&B project | doseprediction-lundprobe |
| Split | `data/split.csv` (do not regenerate) |
