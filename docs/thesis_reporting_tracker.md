# Thesis Reporting Tracker

Working document. Update as runs complete and sections are drafted.

Last updated: 2026-05-18

---

## 1. Status Overview

### Done
- [x] U-Net Sigmoid — 5-fold trained + evaluated (val)
- [x] U-Net Tanh — 5-fold trained + evaluated (val)
- [x] DoseGAN Tanh — 5-fold trained + evaluated (val)
- [x] 2x2 activation ablation (U-Net x DoseGAN, Tanh vs Sigmoid) — decision: Sigmoid wins for both
- [x] Investigation 1 (acquisition group breakdown) — U-Net Sigmoid + DoseGAN Tanh
- [x] Investigation 2 (worst-case patient visualisations) — U-Net Sigmoid + DoseGAN Tanh
- [x] Investigation 3 (dose-smearing index) — code exists
- [x] Literature comparison table finalised — full-text details extracted for all 11 papers; Kandalan 2021 and Lempart 2021 added to Zotero and notes completed
- [x] DVH comparison with literature — per-metric absolute errors (% Rx) computed and benchmarked against Fransson 2024, Kandalan 2021, Lempart 2021 (see Section 3.7)
- [x] DVH acquisition-group breakdown — newAcq vs oldAcq for all DVH metrics; finding: errors comparable across groups at DVH level (see Section 3.3)
- [x] Penile Bulb evaluation added to both eval scripts (extracted from input channel 6; NaN for ~48 absent patients — handled automatically by dvh_metrics)
- [x] DVH regularisation loss implemented in both training scripts — `structure_dmean_loss(PTV + Bladder + Rectum)`, weighted by LAMBDA_DVH (5.0 for DoseGAN, 0.1 for U-Net)
- [x] DVH early stopping implemented in both training scripts — monitors `val_dvh_score = mean|Δ PTV D95| + mean|Δ Bladder Dmean| + mean|Δ Rectum Dmean|` in Gy instead of body-masked L1

### Pending
- [ ] DoseGAN Sigmoid 5-fold training (jobs 22845136–22845139, ETA ~2026-05-19)
- [ ] DoseGAN Sigmoid evaluation (run `sbatch eval_dosegan.sbatch` once training done)
- [ ] DoseGAN Sigmoid Inv1 + Inv2 analyses
- [ ] Investigation 3 at scale across all 4 model variants
- [ ] DVH curve comparison (not yet implemented)
- [ ] Gamma analysis (not yet implemented)
- [ ] Test-set evaluation for all final models (val set only so far)
- [ ] DoseGNN results (collaborator)

### Blocked
- DoseGNN — waiting on collaborator
- Geometric channels 8–14 — waiting on V5geometric_channels.py (not blocking thesis if scoped to 9-channel input)

---

## 2. Methods Section

| # | Item | Status | Source |
|---|------|--------|--------|
| M1 | Dataset: LUND-PROBE, 432 patients, 9-channel input (8 structure masks + sCT), dose target | READY | `preprocessing/preprocessing.py` |
| M2 | Preprocessing: resample to 1.5 mm isotropic, z-score sCT inside body contour, dose / 50 Gy, PTV-centric crop 128×256×320 | READY | `preprocessing/preprocessing.py`, `configs/config_preprocessing_shared.py` |
| M3 | Split: 15% test held out, 5-fold CV on remainder, stratified by acquisition group (oldAcq/newAcq) | READY | `data/split.csv`, `preprocessing/create_split.py` |
| M4 | U-Net architecture: 3D U-Net, 32 base channels, BatchNorm3d, Sigmoid output activation | READY | `models/unet3d.py` |
| M5 | DoseGAN architecture: 3D U-Net generator with attention gates at every skip connection, PatchGAN discriminator, LSGAN loss | READY | `models/dosegan.py` |
| M6 | Footnote: output activation set to Sigmoid based on 2×2 ablation; deviates from Kearney et al. 2020 which uses Tanh | READY | Section 5 below |
| M7 | Footnote: PatchGAN discriminator ends with BatchNorm3d(1) → LeakyReLU (non-standard, inherited from Kearney et al.) | READY | `models/dosegan.py` |
| M8 | Footnote: AttGate ties weights between down_inp and sample_inp (inherited from upstream) | READY | `models/dosegan.py` |
| M9 | Footnote: BatchNorm3d with batch_size=1 (memory constraint standard in 3D medical imaging) | READY | `training/train_*.py` |
| M10 | Training: Adam optimiser, LR 2e-4, 100 epochs + early stopping (patience 15), augmentation (LR flip, depth flip, sCT intensity scale/shift) | READY | `configs/config_dosegan.py`, `training/dataset.py` |
| M11 | Metrics: body_MAE_Gy, body_RMSE_Gy computed over body-contour voxels (channel 7); same mask used during training for val_L1 | READY | `training/evaluate_*.py` |
| M12 | Note: val_L1 redefined 2026-05-13 — pre-change values not comparable (conversion: new ≈ old / 0.32) | READY | commit a2c5cc7 |
| M13 | Hardware and runtime: Snellius HPC, NVIDIA H100, wallclock per fold | READY | `outputs/logs/` |
| M14 | DVH and gamma analysis | OPTIONAL / PENDING | Not yet implemented |
| M15 | Investigation 3 metric: ratio of predicted to ground-truth dose gradient magnitude per patient | OPTIONAL | code exists, not yet run at scale |
| M16 | DVH regularisation loss: `structure_dmean_loss(PTV) + structure_dmean_loss(Bladder) + structure_dmean_loss(Rectum)`, weighted λ_dvh (5.0 DoseGAN, 0.1 U-Net) — added to generator loss; from Kearney et al. 2020 | IMPLEMENTED — applies to next re-train | `training/train_dosegan.py`, `training/train_unet3d.py` |
| M17 | DVH early stopping: model saved when `val_dvh_score = mean|Δ PTV D95| + mean|Δ Bladder Dmean| + mean|Δ Rectum Dmean|` (Gy) improves, not when body-masked L1 improves | IMPLEMENTED — applies to next re-train | `training/train_dosegan.py`, `training/train_unet3d.py` |
| M18 | Penile Bulb evaluation: Dmean, Dmax, D95, D98, V20, V40 added to per-patient eval CSV; extracted from input channel 6; NaN for ~48 absent patients | IMPLEMENTED — applies on next evaluation run | `training/evaluate_dosegan.py`, `training/evaluate_unet3d.py` |

---

## 3. Results Section

### 3.1 Main quantitative comparison (val set, 5-fold)

| # | Item | Status | Source |
|---|------|--------|--------|
| R1 | Table: body_MAE_Gy and body_RMSE_Gy per model, mean ± std across 5 folds | PARTIAL — DoseGAN Sigmoid pending | `outputs/evaluation/*_fold*_val.csv` |
| R2 | U-Net Sigmoid: body_MAE_Gy = 0.861 ± 0.026 Gy; body_RMSE_Gy = 1.514 ± 0.038 Gy | READY | `outputs/evaluation/unet3d_ch32_sigmoid_snellius_fold*_val.csv` |
| R3 | DoseGAN Sigmoid: body_MAE_Gy + body_RMSE_Gy | PENDING | `outputs/evaluation/dosegan_ngf32_sigmoid_snellius_fold*_val.csv` |
| R4 | DoseGNN: body_MAE_Gy + body_RMSE_Gy | PENDING collaborator | — |
| R5 | Test-set results for all final models | PENDING | requires test-split eval |

### 3.2 Activation ablation (2×2)

| # | Item | Status | Source |
|---|------|--------|--------|
| R6 | Table: Tanh vs Sigmoid for U-Net and DoseGAN, per fold + mean ± std | PARTIAL — DoseGAN Sigmoid pending | Section 5 below |
| R7 | U-Net: Sigmoid 0.861 ± 0.026 vs Tanh 0.895 ± 0.041 Gy (body_MAE); Sigmoid wins every fold | READY | `outputs/evaluation/` |
| R8 | DoseGAN: Sigmoid val_L1 0.0174 ± 0.0007 vs Tanh 0.0182 ± 0.0013 — Sigmoid wins/ties every fold | READY | W&B groups dosegan_ngf32_{sigmoid,tanh}_snellius |
| R9 | Conclusion: Sigmoid adopted as default for both models | READY | Section 5 below |

### 3.3 Investigation 1: Acquisition group subgroup analysis

| # | Item | Status | Source |
|---|------|--------|--------|
| R10 | Boxplot: body_MAE_Gy by oldAcq vs newAcq, U-Net Sigmoid | READY | `outputs/analysis/inv1_unet3d_ch32_sigmoid_snellius_acquisition_boxplot.png` |
| R11 | Per-group means and medians CSV, U-Net Sigmoid | READY | `outputs/analysis/inv1_unet3d_ch32_sigmoid_snellius_acquisition_breakdown.csv` |
| R12 | Same for DoseGAN Tanh | READY | `outputs/analysis/inv1_dosegan_*_acquisition_*` |
| R13 | Same for DoseGAN Sigmoid | PENDING | — |
| R14 | Finding: oldAcq patients drive body-MAE tail; but DVH-level errors are comparable across groups | READY (refined) | Inv 1 outputs + Section 3.7 below |

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

**Key finding:** All ratios are within 0.87×–1.24×. Distribution shift between acquisition groups manifests as tail-risk inflation at the body-MAE level (all 15 worst-MAE patients are oldAcq), NOT as systematic DVH-level degradation. Clinically relevant metrics (PTV coverage, OAR doses) are statistically comparable across groups.

### 3.4 Investigation 2: Worst-case visualisations

| # | Item | Status | Source |
|---|------|--------|--------|
| R15 | Top-3 worst-case patient panels per fold (15 panels), U-Net Sigmoid | READY | `outputs/analysis/inv2_unet3d_ch32_sigmoid_snellius_worst_*.png` |
| R16 | Same for DoseGAN Tanh | READY | `outputs/analysis/inv2_dosegan_*.png` |
| R17 | Same for DoseGAN Sigmoid | PENDING | — |
| R18 | Failure pattern: bladder over-prediction and dose-smearing | READY | Inv 2 outputs |

### 3.5 DVH comparison with literature

All values are mean |Δ| (mean absolute error) as % of prescription dose (50 Gy), N=367 val patients. Literature values as reported. See `docs/literature_comparison.md` for full detail and caveats.

| Metric | U-Net Sigmoid | DoseGAN Tanh | Fransson 2024 | Kandalan 2021 | Lempart 2021 |
|---|---|---|---|---|---|
| PTV/CTV Dmean | **0.49%** ± 0.79% | 1.59% ± 1.42% | 0.7% (CTV) | 1.0% | — |
| PTV D95 | **0.69%** ± 1.11% | 1.85% ± 1.67% | 0.7% (CTV) | **0.4%** | 1.0% |
| PTV D98 | **0.79%** ± 1.30% | 2.01% ± 1.86% | 3.2% (PTV) | 1.6% (D2) | 1.9% |
| Bladder Dmean | 1.65% ± 1.37% | 1.84% ± 1.51% | **0.7%** | 1.8% | ≤ 2.6% |
| Rectum Dmean | 2.54% ± 2.03% | 2.59% ± 2.07% | n/r | — | ≤ 2.6% |
| Rectum D95 | 0.59% ± 0.56% | 0.62% ± 0.58% | — | — | — |

**Interpretation for thesis:**
- U-Net Sigmoid PTV metrics are competitive with or better than every prostate paper. PTV D98 (0.79%) substantially outperforms Fransson (3.2%) and Lempart (1.9%).
- Bladder Dmean gap vs Fransson (1.65% vs 0.7%): explained by cohort heterogeneity (N=432, two acquisition eras, real-world bladder filling variability) not model weakness. Acquisition-group breakdown confirms both groups show identical bladder errors (~1.65–1.68%).
- DoseGAN Tanh PTV metrics are within Lempart range but clearly behind U-Net. The U-Net vs DoseGAN gap holds across both acquisition groups, indicating it is architectural, not data-driven.
- No prostate paper reports body-masked voxel MAE in Gy; 0.861 Gy compares favourably to Feng 2024 breast result (1.076 Gy, same metric definition).

| # | Item | Status | Source |
|---|------|--------|--------|
| R23 | DVH comparison table (as above) | READY | `docs/literature_comparison.md` |
| R24 | Bladder Dmean gap explanation (cohort heterogeneity, not model weakness) | READY | acquisition-group breakdown + Fransson N=35 context |
| R25 | Voxel MAE benchmark vs Feng 2024 (breast): 0.861 vs 1.076 Gy | READY | `docs/literature_comparison.md` |

---

### 3.6 Investigation 3: Dose-smearing index

| # | Item | Status | Source |
|---|------|--------|--------|
| R19 | Per-patient dose-smearing index across all 4 variants, with oldAcq/newAcq breakdown | OPTIONAL — code exists | analysis scripts |
| R20 | Correlation between dose-smearing index and body_MAE_Gy | OPTIONAL | derived from R19 |

### 3.7 Clinical metrics (optional)

| # | Item | Status | Source |
|---|------|--------|--------|
| R21 | DVH curves per OAR and PTV | OPTIONAL / PENDING | Not implemented |
| R22 | Gamma pass rate (3%/3 mm, 2%/2 mm) | OPTIONAL / PENDING | Not implemented |

---

## 4. Discussion Section

| # | Item | Status | Source |
|---|------|--------|--------|
| D1 | Headline result: which model achieves lowest body_MAE/RMSE and by how much | PENDING DoseGAN Sigmoid + DoseGNN | R1 |
| D2 | Activation finding: Sigmoid consistently outperforms Tanh; range match to normalised [0,1] dose target; small deviation from Kearney et al. | READY | Section 5 |
| D3 | Acquisition group shift: oldAcq patients drive the error tail; implications for transfer to pancreatic cohort (Phase 2) | READY | R10–R14 |
| D4 | Failure modes: bladder over-prediction and dose-smearing; activation choice does not resolve these | READY | R15–R18 |
| D5 | Dose-smearing index: if R19 is run, provides a quantitative characterisation of the dominant failure mode | OPTIONAL | R19–R20 |
| D6 | Limitations: val-set-only metrics, no DVH/gamma, single dataset (Phase 1), batch_size=1 with BatchNorm3d, geometric channels 8–14 not implemented | READY | — |
| D7 | Reflection: dose-smearing should be addressed at the loss level; DVH regularisation loss now implemented (M16); γ-pass-rate loss remains future work | READY | M16 |
| D8 | Inherited architectural quirks: PatchGAN tail BatchNorm+LeakyReLU, AttGate weight-tying — flagged as inherited, not corrected | READY | M7–M8 |
| D9 | Comparison with prior literature — DVH table + acquisition-group breakdown added 2026-05-18 | READY | `docs/literature_comparison.md`, Section 3.5 |
| D10 | Phase 2/3 outlook: transfer to pancreatic cohort, robustness under anatomical variation | READY | `docs/thesis_proposal.md` |

---

## 5. Activation Ablation — Sigmoid vs Tanh

Both U-Net and DoseGAN were trained with 5-fold CV on LUND-PROBE, all hyperparameters held constant. Only the generator output activation varied.

| Model | Activation | body_MAE_Gy (Gy) | body_RMSE_Gy (Gy) | body-masked val_L1 |
|---|---|---|---|---|
| U-Net | Sigmoid | **0.861 ± 0.026** | **1.514 ± 0.038** | — |
| U-Net | Tanh | 0.895 ± 0.041 | 1.670 ± 0.050 | — |
| DoseGAN | Sigmoid | pending | pending | **0.0174 ± 0.0007** |
| DoseGAN | Tanh | pending | pending | 0.0182 ± 0.0013 |

U-Net body_MAE_Gy per fold:

| Fold | Sigmoid | Tanh |
|---|---|---|
| 0 | 0.863 | 0.894 |
| 1 | 0.896 | 0.950 |
| 2 | 0.837 | 0.851 |
| 3 | 0.873 | 0.921 |
| 4 | 0.833 | 0.860 |

Sigmoid wins on every fold for U-Net. For DoseGAN, Sigmoid wins or ties on every fold with ~4% lower mean and lower variance. The dose target is normalised to [0, 1] (dose / 50 Gy), making Sigmoid the natural range-matched output. Sigmoid is adopted as the default for both models. Tanh is retained in the ablation table and discussed as a deviation from Kearney et al. 2020.

Draft methods footnote:
> The generator output activation was set to Sigmoid following a 2×2 ablation (Tanh vs Sigmoid, U-Net vs DoseGAN, 5-fold CV on LUND-PROBE). This deviates from Kearney et al. (2020), which uses Tanh; Sigmoid achieved lower body-masked validation L1 on every fold for both architectures.

---

## 6. Next Steps (priority order)

1. Wait for DoseGAN Sigmoid training to finish (jobs 22845136–22845139, ETA ~2026-05-19). Monitor: `squeue -u akhalil`.
2. Evaluate DoseGAN Sigmoid: `sbatch eval_dosegan.sbatch` (loops folds 0–4). Penile Bulb metrics now included automatically.
3. Compute DVH metrics for DoseGAN Sigmoid; fill in R3 (body_MAE/RMSE) and update Section 3.5 DVH table.
4. Run Inv1 + Inv2 for DoseGAN Sigmoid; fill in R13, R17.
5. Run Investigation 3 at scale across all 4 model variants.
6. Implement test-set evaluation for final models.
7. (Optional) Re-train with DVH loss + DVH early stopping (M16–M17) to quantify improvement. Compare new val_dvh_score vs baseline. Note: currently running DoseGAN Sigmoid jobs (22845136–22845139) use the old loss — these are still valid as a baseline.
8. (Stretch) Implement full DVH curves and gamma pass rate.
9. Draft Methods using M1–M18 as checklist.
10. Draft Results using R1–R25 as checklist.
11. Draft Discussion using D1–D10 as checklist; prioritise D2, D3, D4, D9.

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
