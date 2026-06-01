# Experimental Results — Baseline (validation set, 5-fold CV)

Results on the LUND-PROBE validation set (367 patients across 5 folds). All metrics are fold mean ± fold std. Dose values in Gy; dose normalised by 50 Gy prescription.

---

## SRQ 1 — Model comparison (9-channel baseline, without geometric channels)

### Voxel-level accuracy

| Metric | U-Net Sigmoid | DoseGAN Sigmoid |
|---|---|---|
| body_MAE (Gy) | **0.861 ± 0.023** | 0.868 ± 0.031 |
| body_RMSE (Gy) | **1.514 ± 0.034** | 1.522 ± 0.040 |
| ptv_MAE (Gy) | 0.515 ± 0.110 | **0.474 ± 0.108** |

### Boundary MAE — ±20 mm band around structure surface

| Structure | U-Net Sigmoid | DoseGAN Sigmoid |
|---|---|---|
| PTV boundary (Gy) | 1.577 ± 0.071 | **1.573 ± 0.065** |
| Rectum boundary (Gy) | **1.381 ± 0.054** | 1.398 ± 0.071 |
| Bladder boundary (Gy) | **1.107 ± 0.035** | 1.127 ± 0.046 |

### DVH accuracy

| Metric | U-Net Sigmoid | DoseGAN Sigmoid |
|---|---|---|
| PTV D95 error (Gy) | **−0.164 ± 0.260** | 0.183 ± 0.398 |
| PTV Dmean error (Gy) | **−0.020 ± 0.190** | 0.179 ± 0.307 |
| PTV D0.1cc error (Gy) | 0.852 ± 0.548 | **−0.032 ± 0.312** |
| Rectum D0.1cc error (Gy) | 0.217 ± 0.265 | **0.093 ± 0.349** |
| Bladder D0.1cc error (Gy) | 0.349 ± 0.485 | **0.050 ± 0.318** |

![Body error comparison](figures/fig_body_error_comparison.png)
![Per-fold MAE](figures/fig_per_fold_mae.png)
![DVH errors](figures/fig_dvh_errors.png)
![Structure MAE](figures/fig_structure_mae.png)
![New metrics](figures/fig_new_metrics.png)

---

## Ablations (fold 0 only unless stated)

| Ablation | U-Net val_L1 | DoseGAN val_L1 | Decision |
|---|---|---|---|
| Sigmoid vs Tanh (5-fold) | 0.0172 ± 0.0005 vs 0.0179 ± 0.0008 | 0.0174 ± 0.0007 vs 0.0182 ± 0.0013 | Sigmoid adopted for both |
| Gradient-magnitude loss λ=1.0 | 0.0174 (+1.2%) | 0.0183 (+4.6%) | Negative — not used |
| BCE vs LSGAN (DoseGAN only) | — | 0.0195 vs 0.0174 | LSGAN retained |

---

## Investigation 1 — Acquisition group robustness (oldAcq vs newAcq)

Mann-Whitney U on body_MAE: **p = 0.923** — no significant mean shift between acquisition groups. However, all 15 worst-MAE patients (top-3 per fold) are from oldAcq (probability < 1.2% under the null), indicating tail-risk concentration rather than a mean effect.

![Acquisition breakdown](figures/fig_acquisition_breakdown.png)
![DoseGAN acquisition boxplot](figures/inv1_dosegan_acquisition_boxplot.png)
![U-Net acquisition boxplot](figures/inv1_unet3d_acquisition_boxplot.png)

---

## Investigation 2 — Failure mode analysis (worst-case patients)

Dominant failure mode: **bladder over-prediction** (+0.40 Gy mean, 65% of patients over-predict) combined with **dose-smearing** at structure boundaries. Worst cases are concentrated in oldAcq patients.

### Dose maps — worst-case patient (fold 0, oldAcq)

| | DoseGAN | U-Net |
|---|---|---|
| **Worst case** | ![](figures/dose_dosegan_worst_case.png) | ![](figures/dose_unet3d_worst_case.png) |
| **Median case** | ![](figures/dose_dosegan_median_case.png) | ![](figures/dose_unet3d_median_case.png) |

---

## SRQ 2 — Geometric channels

*Training in progress.* 5-fold results for U-Net+geom and DoseGAN+geom will be added here once the 14-channel training runs complete.

## SRQ 3 — Clinical relevance

*Pending full evaluation run.* DVH metrics (D95, Dmean, D0.1cc) for both conditions will be added here.
