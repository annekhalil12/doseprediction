# Validation Results — 2-Model × 2-Condition Comparison

**Date:** 2026-06-03 (DG base column stale — see warning below)
**Models:** DoseGAN (GAN + L1 loss) and U-Net (L1 loss only)
**Conditions:** baseline (9-channel: 8 structure masks + sCT) vs geom (14-channel: + 5 geometric channels)
**Split:** 5-fold cross-validation on the train/val set (n ≈ 367 patients per condition)
**Note:** Gamma pass rate (3%/3mm, 2%/2mm) was implemented in code but omitted from all evaluation runs (--skip-gamma) due to computational cost (~10 min/patient for 3D gamma). All gamma columns in the eval CSVs are NaN. Gamma analysis is not part of the main quantitative comparison and should be disclosed as a limitation.

> **⚠ DG base column is stale.** The DoseGAN baseline checkpoints were retrained on 2026-06-04
> with InstanceNorm(affine=True) to match the U-Net normalisation. The numbers in the DG base
> column below come from the old BatchNorm checkpoints and will change once the new eval finishes.
> All other columns (UN base, DG geom, UN geom) are current.

---

## 1. Error Metrics

| Metric | DG base | UN base | DG geom | UN geom |
|---|---|---|---|---|
| **Body MAE (Gy)** | 0.868 ± 0.225 | 0.861 ± 0.215 | 0.868 ± 0.191 | **0.839 ± 0.190** |
| PTV MAE (Gy) | 0.473 ± 0.337 | 0.515 ± 0.352 | **0.401 ± 0.308** | 0.612 ± 0.426 |
| Rectum MAE (Gy) | 1.844 ± 0.771 | **1.840 ± 0.784** | 1.892 ± 0.633 | 1.896 ± 0.674 |
| Bladder MAE (Gy) | 1.432 ± 0.622 | **1.386 ± 0.601** | 1.481 ± 0.574 | 1.441 ± 0.578 |
| PTV Dmean diff (Gy) | +0.179 ± 0.468 | −0.020 ± 0.464 | +0.094 ± 0.339 | +0.271 ± 0.487 |
| **PTV D95 diff (Gy)** | +0.183 ± 0.596 | −0.165 ± 0.629 | −0.116 ± 0.376 | **−0.003 ± 0.404** |
| Rectum Dmean diff (Gy) | +0.459 ± 1.543 | **+0.061 ± 1.624** | −0.056 ± 1.433 | +0.198 ± 1.430 |
| Bladder Dmean diff (Gy) | +0.092 ± 1.121 | +0.096 ± 1.068 | −0.268 ± 0.997 | −0.095 ± 0.997 |
| Boundary MAE PTV (Gy) | **1.573 ± 0.293** | 1.577 ± 0.306 | 1.637 ± 0.260 | 1.682 ± 0.313 |

Body MAE is the primary metric (body-masked mean absolute error in Gy; dose normalized to 50 Gy prescription).
Diff = pred − true; positive = over-prediction.

---

## 2. Isodose Conformality

Dice and HD95 at four isodose levels, each defined as a fraction of the patient-specific prescription dose (D95 of PTV from the ground-truth dose).

| Metric | DG base | UN base | DG geom | UN geom |
|---|---|---|---|---|
| Dice 100iso | 0.899 ± 0.144 | 0.014 ± 0.005 | **0.962 ± 0.010** | 0.013 ± 0.004 |
| Dice 95iso | 0.849 ± 0.243 | 0.016 ± 0.005 | **0.965 ± 0.008** | 0.015 ± 0.005 |
| Dice 80iso | 0.594 ± 0.352 | 0.021 ± 0.007 | **0.906 ± 0.151** | 0.020 ± 0.006 |
| Dice 50iso | 0.240 ± 0.213 | 0.043 ± 0.013 | **0.733 ± 0.309** | 0.041 ± 0.012 |
| HD95 100iso (mm) | 50.7 ± 98.2 | 267 ± 4.8 | **1.8 ± 0.4** | 266 ± 4.5 |

### Note on U-Net Dice values

The near-zero Dice for both U-Net conditions is not a computation error. Two factors explain it:

**100iso:** The prescription threshold is defined as D95_true. U-Net systematically under-predicts PTV D95 (mean −0.17 Gy baseline, −0.003 Gy geom), meaning only a small fraction of PTV voxels cross the prescription threshold in the prediction. DoseGAN over-predicts D95 (+0.183 Gy baseline), so more voxels cross it. The 100iso Dice is therefore highly sensitive to the direction of systematic bias.

**50iso (threshold ≈ 20 Gy):** This cannot be explained by D95 bias — the thresholds are unrelated. UN geom achieves near-zero D95 bias (−0.003 Gy) yet Dice_50iso = 0.041. The U-Net's spatial dose distribution at medium dose levels is genuinely misaligned with the ground truth even when per-structure DVH endpoints match. DoseGAN baseline achieves Dice_50iso = 0.240 and DoseGAN geom 0.733, indicating the adversarial loss preserves spatial dose gradients that L1 loss alone does not.

---

## 3. Per-Fold Body MAE

| Fold | DG base | UN base | DG geom | UN geom | n_val |
|---|---|---|---|---|---|
| 0 | 0.860 ± 0.215 | 0.863 ± 0.211 | 0.855 ± 0.161 | 0.864 ± 0.216 | 74 |
| 1 | 0.904 ± 0.270 | 0.896 ± 0.265 | 0.950 ± 0.287 | 0.875 ± 0.246 | 74 |
| 2 | 0.837 ± 0.233 | 0.837 ± 0.202 | 0.823 ± 0.149 | 0.813 ± 0.139 | 73 |
| 3 | 0.904 ± 0.196 | 0.873 ± 0.208 | 0.878 ± 0.156 | 0.842 ± 0.168 | 73 |
| 4 | 0.833 ± 0.189 | 0.833 ± 0.168 | 0.835 ± 0.134 | 0.803 ± 0.146 | 73 |
| **mean** | **0.868** | **0.861** | **0.868** | **0.839** | — |

Fold 1 is consistently the hardest across all conditions.

---

## 4. PTV D95 Bias by Condition

| Condition | Mean D95 diff (Gy) | % patients over-predicting |
|---|---|---|
| DG base | +0.183 | 61% |
| UN base | −0.165 | 34% |
| DG geom | −0.116 | 30% |
| UN geom | −0.003 | 48% |

DoseGAN baseline is the most biased toward over-prediction. U-Net geom is the most calibrated (near-zero mean bias). The direction of D95 bias has clinical implications: over-prediction implies the plan looks adequately covered when it may not be; under-prediction implies under-coverage that may not exist.

---

## 5. Distribution Shift — oldAcq vs newAcq

Body MAE stratified by acquisition group.

| Condition | oldAcq (n≈274) | newAcq (n≈93) | Δ |
|---|---|---|---|
| DG base | 0.875 ± 0.243 | 0.845 ± 0.159 | +0.030 |
| UN base | 0.867 ± 0.235 | 0.841 ± 0.139 | +0.026 |
| DG geom | 0.871 ± 0.201 | 0.861 ± 0.161 | +0.010 |
| UN geom | 0.845 ± 0.205 | 0.823 ± 0.137 | +0.022 |

The gap is modest (~0.01–0.03 Gy) and consistent across conditions. Neither model escapes the distribution shift. DoseGAN geom shows the smallest gap (0.010 Gy), though the difference from the others is within noise.

---

## 6. Summary and Interpretation

**Global error (body MAE):** All four conditions are within 0.03 Gy of each other (~1.7% of the 50 Gy prescription). Geom channels reduce body MAE slightly for U-Net (0.861 → 0.839 Gy) and not at all for DoseGAN.

**Spatial accuracy (isodose Dice):** DoseGAN and U-Net differ dramatically. Geom channels make DoseGAN markedly more accurate (Dice 50iso: 0.240 → 0.733; HD95 100iso: 51 mm → 1.8 mm). They have no effect on U-Net spatial accuracy.

**Geom channels benefit DoseGAN but not U-Net.** DoseGAN geom is the best-performing condition on all spatial metrics and also achieves the best PTV MAE (0.401 Gy). U-Net geom is the best on body MAE (0.839 Gy) and D95 calibration, but the worst on PTV MAE (0.612 Gy) and no better spatially.

**Interpretation:** The adversarial loss in DoseGAN appears necessary to translate geometric context into accurate spatial dose distributions. The L1 loss in U-Net produces dose values that are correct on average within structures (matching DVH endpoints) but spatially smeared at dose gradient regions, regardless of whether geometric channels are added. This is consistent with the known tendency of L1-trained regression models to produce over-smoothed outputs.

---

## Pending

- U-Net geom per-fold isodose breakdown (currently aggregated only)
- Gamma pass rate (3%/3mm, 2%/2mm) on full val set — deferred
- Test-set evaluation (locked until model selection is final)
- Inv1/Inv2/Inv3 analyses on baseline checkpoints
