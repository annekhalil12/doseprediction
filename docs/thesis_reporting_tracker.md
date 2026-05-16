# Thesis Reporting Tracker

Working document. Update as runs complete and sections are drafted.

Last updated: 2026-05-16

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

### Pending
- [ ] DoseGAN Sigmoid 5-fold training (running, ETA ~24h)
- [ ] DoseGAN Sigmoid evaluation
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
| R14 | Finding: oldAcq patients show higher error (distribution shift) | READY | Inv 1 outputs |

### 3.4 Investigation 2: Worst-case visualisations

| # | Item | Status | Source |
|---|------|--------|--------|
| R15 | Top-3 worst-case patient panels per fold (15 panels), U-Net Sigmoid | READY | `outputs/analysis/inv2_unet3d_ch32_sigmoid_snellius_worst_*.png` |
| R16 | Same for DoseGAN Tanh | READY | `outputs/analysis/inv2_dosegan_*.png` |
| R17 | Same for DoseGAN Sigmoid | PENDING | — |
| R18 | Failure pattern: bladder over-prediction and dose-smearing | READY | Inv 2 outputs |

### 3.5 Investigation 3: Dose-smearing index

| # | Item | Status | Source |
|---|------|--------|--------|
| R19 | Per-patient dose-smearing index across all 4 variants, with oldAcq/newAcq breakdown | OPTIONAL — code exists | analysis scripts |
| R20 | Correlation between dose-smearing index and body_MAE_Gy | OPTIONAL | derived from R19 |

### 3.6 Clinical metrics (optional)

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
| D7 | Reflection: dose-smearing should be addressed at the loss level (gradient-matching or perceptual loss); DVH/gamma should be implemented earlier | READY | — |
| D8 | Inherited architectural quirks: PatchGAN tail BatchNorm+LeakyReLU, AttGate weight-tying — flagged as inherited, not corrected | READY | M7–M8 |
| D9 | Comparison with prior literature | READY | `docs/literature_comparison.md` |
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

1. Wait for DoseGAN Sigmoid training to finish. Monitor with `squeue -u akhalil` and `tail -f outputs/logs/<jobid>.out`.
2. Evaluate DoseGAN Sigmoid: `for F in 0 1 2 3 4; do python3 -m training.evaluate_dosegan --fold $F; done`
3. Fill in R3, R6 (DoseGAN row) with body_MAE_Gy and body_RMSE_Gy.
4. Run Inv1 + Inv2 for DoseGAN Sigmoid.
5. Run Investigation 3 at scale across all 4 model variants.
6. Implement test-set evaluation for final models.
7. (Stretch) Implement DVH + gamma analysis.
8. Draft Methods using M1–M13 as checklist.
9. Draft Results using R1–R18 as checklist.
10. Draft Discussion using D1–D10 as checklist; prioritise D2, D4, D5.

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
