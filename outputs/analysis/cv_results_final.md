# Cross-validation results — final (2026-06-20)

Generated from: `clinical_report.csv`, `paired_comparison.csv`,
`interaction_test.csv`, `inv1_val_acquisition_breakdown.csv`.

All values: fold mean ± fold std (95% CI half-width). Five-fold CV, n=367
validation patients total. GPU: NVIDIA H100. Statistical tests: paired
t-test + Wilcoxon signed-rank, BH-FDR correction per comparison across all
metrics. CIs: t-distribution, df=4. Boundary metrics and interaction test
now included in the default run.

---

## 1. Global dose accuracy

| Metric | U-Net base | U-Net geom | DoseGAN base | DoseGAN geom |
|---|---|---|---|---|
| Body MAE (Gy) | 0.827 ± 0.026 (±0.032) | 0.831 ± 0.028 (±0.035) | 0.902 ± 0.056 (±0.069) | 0.824 ± 0.031 (±0.039) |
| Body RMSE (Gy) | 1.544 ± 0.033 (±0.041) | 1.528 ± 0.043 (±0.053) | 1.650 ± 0.089 (±0.110) | 1.498 ± 0.075 (±0.093) |
| PTV MAE (Gy) | 0.485 ± 0.093 (±0.115) | 0.457 ± 0.109 (±0.136) | 0.383 ± 0.038 (±0.047) | 0.429 ± 0.104 (±0.130) |
| Rectum MAE (Gy) | 1.749 ± 0.061 (±0.076) | 1.850 ± 0.124 (±0.154) | 1.949 ± 0.109 (±0.136) | 1.963 ± 0.064 (±0.079) |
| Bladder MAE (Gy) | 1.354 ± 0.050 (±0.062) | 1.407 ± 0.060 (±0.075) | 1.517 ± 0.091 (±0.113) | 1.463 ± 0.083 (±0.103) |

---

## 2a. DVH accuracy (mean |error|)

| Metric | U-Net base | U-Net geom | DoseGAN base | DoseGAN geom |
|---|---|---|---|---|
| PTV D95 (Gy) | 0.195 ± 0.051 (±0.064) | 0.211 ± 0.051 (±0.063) | 0.230 ± 0.059 (±0.073) | 0.214 ± 0.064 (±0.079) |
| PTV D98 (Gy) | 0.253 ± 0.056 (±0.069) | 0.271 ± 0.051 (±0.063) | 0.322 ± 0.041 (±0.051) | 0.293 ± 0.057 (±0.071) |
| PTV Dmean (Gy) | 0.145 ± 0.101 (±0.125) | 0.165 ± 0.156 (±0.193) | 0.172 ± 0.074 (±0.092) | 0.245 ± 0.179 (±0.222) |
| PTV D0.1cc (Gy) | 1.752 ± 1.052 (±1.306) | 0.574 ± 0.358 (±0.444) | 0.812 ± 0.287 (±0.356) | 1.054 ± 0.166 (±0.206) |
| Rectum Dmean (Gy) | 1.091 ± 0.113 (±0.140) | 1.109 ± 0.105 (±0.130) | 1.154 ± 0.099 (±0.123) | 1.171 ± 0.065 (±0.081) |
| Rectum D0.1cc (Gy) | 1.288 ± 0.963 (±1.195) | 0.477 ± 0.328 (±0.408) | 0.350 ± 0.102 (±0.126) | 0.479 ± 0.210 (±0.261) |
| Rectum V_Rx (%) | 0.847 ± 0.185 (±0.230) | 1.110 ± 0.293 (±0.364) | 0.709 ± 0.147 (±0.182) | 0.790 ± 0.061 (±0.076) |
| Bladder Dmean (Gy) | 0.763 ± 0.049 (±0.061) | 0.762 ± 0.041 (±0.050) | 0.794 ± 0.078 (±0.096) | 0.784 ± 0.092 (±0.114) |
| Bladder D0.1cc (Gy) | 1.588 ± 0.962 (±1.194) | 0.562 ± 0.207 (±0.257) | 0.662 ± 0.203 (±0.252) | 0.790 ± 0.183 (±0.227) |
| Bladder V_Rx (%) | 0.660 ± 0.143 (±0.178) | 0.707 ± 0.075 (±0.093) | 0.678 ± 0.034 (±0.042) | 0.807 ± 0.263 (±0.326) |

---

## 2b. DVH bias (mean signed error, pred − true)

| Metric | U-Net base | U-Net geom | DoseGAN base | DoseGAN geom |
|---|---|---|---|---|
| PTV D95 (Gy) | −0.041 ± 0.095 | −0.008 ± 0.079 | +0.152 ± 0.082 | +0.047 ± 0.190 |
| PTV D98 (Gy) | +0.007 ± 0.090 | +0.018 ± 0.033 | +0.239 ± 0.065 | +0.124 ± 0.204 |
| PTV Dmean (Gy) | +0.120 ± 0.121 | +0.107 ± 0.188 | −0.041 ± 0.164 | −0.200 ± 0.229 |
| PTV D0.1cc (Gy) | +1.699 ± 1.148 | +0.424 ± 0.470 | −0.763 ± 0.315 | −1.025 ± 0.193 |
| Rectum Dmean (Gy) | +0.012 ± 0.072 | +0.094 ± 0.120 | −0.054 ± 0.250 | −0.057 ± 0.360 |
| Rectum D0.1cc (Gy) | +1.247 ± 1.021 | +0.328 ± 0.425 | −0.242 ± 0.139 | −0.421 ± 0.272 |
| Rectum V_Rx (%) | −0.443 ± 0.336 | −0.755 ± 0.377 | −0.068 ± 0.286 | −0.172 ± 0.552 |
| Bladder Dmean (Gy) | +0.009 ± 0.113 | −0.071 ± 0.126 | −0.126 ± 0.242 | +0.062 ± 0.248 |
| Bladder D0.1cc (Gy) | +1.532 ± 1.060 | +0.352 ± 0.363 | −0.616 ± 0.206 | −0.757 ± 0.209 |
| Bladder V_Rx (%) | −0.180 ± 0.245 | −0.018 ± 0.428 | −0.056 ± 0.178 | −0.554 ± 0.448 |

U-Net over-predicts D0.1cc and D0.1cc for rectum/bladder (positive bias).
DoseGAN under-predicts D0.1cc (negative bias). These are opposite in sign.

---

## 3. Spatial boundary quality

| Metric | U-Net base | U-Net geom | DoseGAN base | DoseGAN geom |
|---|---|---|---|---|
| PTV boundary MAE (Gy) | 1.607 ± 0.037 (±0.045) | 1.619 ± 0.077 (±0.096) | 1.699 ± 0.026 (±0.032) | 1.646 ± 0.062 (±0.077) |
| Rectum boundary MAE (Gy) | 1.358 ± 0.032 (±0.040) | 1.406 ± 0.085 (±0.106) | 1.483 ± 0.065 (±0.081) | 1.453 ± 0.042 (±0.052) |
| Bladder boundary MAE (Gy) | 1.101 ± 0.033 (±0.041) | 1.121 ± 0.048 (±0.059) | 1.221 ± 0.078 (±0.097) | 1.160 ± 0.054 (±0.066) |
| Dice 100% iso | 0.960 ± 0.003 (±0.004) | 0.958 ± 0.004 (±0.005) | 0.965 ± 0.001 (±0.002) | 0.964 ± 0.002 (±0.003) |
| Dice 95% iso | 0.966 ± 0.001 (±0.002) | 0.964 ± 0.003 (±0.004) | 0.968 ± 0.001 (±0.001) | 0.967 ± 0.001 (±0.001) |
| Dice 80% iso | 0.954 ± 0.001 (±0.001) | 0.954 ± 0.002 (±0.002) | 0.951 ± 0.001 (±0.001) | 0.951 ± 0.002 (±0.002) |
| Dice 50% iso | 0.885 ± 0.003 (±0.003) | 0.888 ± 0.003 (±0.004) | 0.879 ± 0.008 (±0.010) | 0.886 ± 0.006 (±0.007) |
| HD95 100% iso (mm) | 2.526 ± 1.245 (±1.546) | 1.926 ± 0.168 (±0.208) | 1.679 ± 0.056 (±0.069) | 1.710 ± 0.089 (±0.111) |
| Outside-body leakage (Gy) | 40.167 ± 0.445 (±0.553) | 41.007 ± 1.506 (±1.870) | 3.603 ± 7.954 (±9.876) | 5.584 ± 5.817 (±7.223) |
| Leakage vol frac | 0.927 ± 0.009 | 0.950 ± 0.018 | 0.200 ± 0.435 | 0.593 ± 0.537 |

U-Net leakage is systematic (all 5 folds ~40 Gy). DoseGAN leakage is
bimodal: most folds near zero, one or two folds fail with high leakage;
the large fold std reflects this instability rather than a consistent value.

---

## 4. Paired statistical comparisons (n=367, BH-FDR corrected)

Format: mean difference (A − B) ± SD, 95% bootstrap CI, Cohen's dz, t-test p.
Negative = A is better for MAE/RMSE/boundary/HD95; positive = A is better for Dice.

### 4a. Geometric channel effect — U-Net (geom − base)

| Metric | Δ mean | 95% CI | dz | t p (FDR) |
|---|---|---|---|---|
| Body MAE (Gy) | +0.005 | [−0.002, +0.012] | +0.07 | 0.198 |
| Body RMSE (Gy) | −0.015 | [−0.025, −0.006] | −0.16 | 0.002 |
| PTV MAE (Gy) | −0.029 | [−0.045, −0.011] | −0.17 | 0.001 |
| Rectum MAE (Gy) | +0.101 | [+0.071, +0.131] | +0.34 | <0.001 |
| Bladder MAE (Gy) | +0.053 | [+0.029, +0.077] | +0.22 | <0.001 |
| PTV boundary MAE (Gy) | +0.011 | [−0.004, +0.027] | +0.08 | 0.146 |
| Rectum boundary MAE (Gy) | +0.047 | [+0.029, +0.065] | +0.27 | <0.001 |
| Bladder boundary MAE (Gy) | +0.020 | [+0.005, +0.035] | +0.13 | 0.013 |
| Dice 100% iso | −0.002 | [−0.002, −0.001] | −0.24 | <0.001 |
| Dice 50% iso | +0.003 | [+0.002, +0.004] | +0.25 | <0.001 |

Geometric channels do not improve U-Net body MAE. Rectum and bladder MAE
worsen significantly. Boundary MAE worsens at rectum and bladder.

### 4b. Geometric channel effect — DoseGAN (geom − base)

| Metric | Δ mean | 95% CI | dz | t p (FDR) |
|---|---|---|---|---|
| Body MAE (Gy) | −0.078 | [−0.090, −0.067] | −0.70 | <0.001 |
| Body RMSE (Gy) | −0.152 | [−0.170, −0.136] | −0.91 | <0.001 |
| PTV MAE (Gy) | +0.046 | [+0.033, +0.060] | +0.34 | <0.001 |
| Rectum MAE (Gy) | +0.014 | [−0.021, +0.049] | +0.04 | 0.453 |
| Bladder MAE (Gy) | −0.053 | [−0.087, −0.019] | −0.16 | 0.002 |
| PTV boundary MAE (Gy) | −0.054 | [−0.070, −0.037] | −0.34 | <0.001 |
| Rectum boundary MAE (Gy) | −0.030 | [−0.054, −0.006] | −0.13 | 0.016 |
| Bladder boundary MAE (Gy) | −0.061 | [−0.081, −0.040] | −0.30 | <0.001 |
| Dice 100% iso | −0.001 | [−0.002, −0.000] | −0.16 | 0.002 |
| Dice 50% iso | +0.007 | [+0.006, +0.009] | +0.49 | <0.001 |

Geometric channels reduce DoseGAN body MAE by 0.078 Gy (dz=−0.70) and
improve all three boundary MAEs. The improvement is consistent in direction
across all five folds.

### 4c. Model effect at baseline (DoseGAN − U-Net)

| Metric | Δ mean | 95% CI | dz | t p (FDR) |
|---|---|---|---|---|
| Body MAE (Gy) | +0.076 | [+0.066, +0.086] | +0.76 | <0.001 |
| Body RMSE (Gy) | +0.106 | [+0.089, +0.123] | +0.64 | <0.001 |
| PTV MAE (Gy) | −0.102 | [−0.111, −0.094] | −1.20 | <0.001 |
| Rectum MAE (Gy) | +0.200 | [+0.158, +0.245] | +0.48 | <0.001 |
| Bladder MAE (Gy) | +0.162 | [+0.126, +0.199] | +0.45 | <0.001 |
| PTV boundary MAE (Gy) | +0.092 | [+0.073, +0.111] | +0.49 | <0.001 |
| Rectum boundary MAE (Gy) | +0.125 | [+0.101, +0.149] | +0.53 | <0.001 |
| Bladder boundary MAE (Gy) | +0.120 | [+0.099, +0.140] | +0.59 | <0.001 |
| Dice 100% iso | +0.006 | [+0.005, +0.006] | +0.85 | <0.001 |
| Dice 50% iso | −0.007 | [−0.008, −0.005] | −0.41 | <0.001 |

U-Net is better than DoseGAN baseline on body MAE, all OAR MAEs, and all
boundary MAEs. DoseGAN baseline is better on PTV MAE (dz=−1.20, the
largest effect size across all comparisons).

### 4d. Model effect at geom (DoseGAN geom − U-Net geom)

| Metric | Δ mean | 95% CI | dz | t p (FDR) |
|---|---|---|---|---|
| Body MAE (Gy) | −0.007 | [−0.017, +0.002] | −0.07 | 0.154 |
| Body RMSE (Gy) | −0.031 | [−0.046, −0.015] | −0.20 | <0.001 |
| PTV MAE (Gy) | −0.027 | [−0.041, −0.013] | −0.20 | <0.001 |
| Rectum MAE (Gy) | +0.113 | [+0.068, +0.161] | +0.25 | <0.001 |
| Bladder MAE (Gy) | +0.056 | [+0.019, +0.093] | +0.16 | 0.003 |
| PTV boundary MAE (Gy) | +0.027 | [+0.008, +0.045] | +0.15 | 0.005 |
| Rectum boundary MAE (Gy) | +0.047 | [+0.024, +0.072] | +0.20 | <0.001 |
| Bladder boundary MAE (Gy) | +0.039 | [+0.017, +0.062] | +0.18 | 0.001 |
| Dice 100% iso | +0.006 | [+0.006, +0.007] | +1.05 | <0.001 |
| HD95 100% iso (mm) | −0.216 | [−0.252, −0.181] | −0.62 | <0.001 |

At the geom condition, body MAE is no longer significantly different between
models. DoseGAN geom is better on body RMSE, PTV MAE, and HD95; U-Net geom
is better on rectum/bladder MAE and all boundary MAEs.

---

## 5. Interaction test: model × geometry (n=367, BH-FDR corrected)

interaction = (DoseGAN geom − DoseGAN base) − (U-Net geom − U-Net base)
Negative = geom benefit larger for DoseGAN (DoseGAN gains more, or loses less).

| Metric | Interaction | 95% CI | dz | t p (FDR) |
|---|---|---|---|---|
| Body MAE (Gy) | −0.083 | [−0.097, −0.069] | −0.59 | <0.001 |
| Body RMSE (Gy) | −0.137 | [−0.158, −0.117] | −0.67 | <0.001 |
| PTV MAE (Gy) | +0.075 | [+0.055, +0.095] | +0.38 | <0.001 |
| Rectum MAE (Gy) | −0.087 | [−0.134, −0.040] | −0.19 | <0.001 |
| Bladder MAE (Gy) | −0.106 | [−0.150, −0.062] | −0.25 | <0.001 |
| PTV boundary MAE (Gy) | −0.065 | [−0.087, −0.043] | −0.30 | <0.001 |
| Rectum boundary MAE (Gy) | −0.078 | [−0.107, −0.047] | −0.26 | <0.001 |
| Bladder boundary MAE (Gy) | −0.080 | [−0.108, −0.053] | −0.29 | <0.001 |
| Dice 100% iso | +0.001 | [−0.000, +0.002] | +0.08 | 0.113 |
| Dice 50% iso | +0.004 | [+0.003, +0.006] | +0.25 | <0.001 |
| HD95 100% iso (mm) | +0.629 | [−0.026, +1.880] | +0.06 | 0.291 |

The interaction is significant for body MAE, body RMSE, all three boundary
MAEs, and bladder/rectum MAE (all p<0.001 after FDR). The geom effect on
body MAE is 0.083 Gy larger for DoseGAN than for U-Net. The interaction on
boundary MAE ranges from −0.065 to −0.080 Gy, consistently favouring
DoseGAN. Dice 100% iso and HD95 show no significant interaction.

---

## 6. Acquisition subgroup — val split, body MAE (exploratory)

Mann-Whitney U test (two-sided, pooled across folds), with BH-FDR correction
across the four model conditions. Analysis is exploratory: body MAE is the
only metric tested, and the val split is pooled across folds without
accounting for fold as a clustering variable.

Positive Δ = oldAcq higher (worse) MAE than newAcq.

| Condition | n oldAcq | n newAcq | oldAcq mean (Gy) | newAcq mean (Gy) | Δ mean (Gy) | MWU p (raw) | MWU p (FDR) |
|---|---|---|---|---|---|---|---|
| U-Net base | 274 | 93 | 0.819 | 0.849 | −0.030 | 0.036 | 0.144 |
| U-Net geom | 274 | 93 | 0.831 | 0.832 | −0.001 | 0.489 | 0.489 |
| DoseGAN base | 274 | 93 | 0.903 | 0.902 | +0.000 | 0.431 | 0.489 |
| DoseGAN geom | 274 | 93 | 0.819 | 0.841 | −0.022 | 0.121 | 0.242 |

After FDR correction no condition shows a significant acquisition effect.
The nominally significant U-Net base result (raw p=0.036) does not survive
correction and should not be presented as evidence of an acquisition effect.
The expected oldAcq disadvantage is not confirmed on the validation split.
The direction in U-Net base is reversed (newAcq higher MAE than oldAcq).

---

---

## 7. Activation ablation (sigmoid vs tanh) — confirmed from ablation_sigmoid_vs_tanh.csv

Five-fold CV on baseline checkpoints (no geometric channels). Metric: body MAE (Gy).
Source: `outputs/evaluation/ablation_sigmoid_vs_tanh.csv` (20 rows, 4 conditions × 5 folds).

| Model | Sigmoid mean ± std (Gy) | Tanh mean ± std (Gy) | Δ (sig − tanh) | paired t | p |
|---|---|---|---|---|---|
| DoseGAN | 0.868 ± 0.035 | 0.912 ± 0.065 | −0.044 ± 0.047 | t=−2.11 | 0.102 |
| U-Net | 0.861 ± 0.026 | 0.895 ± 0.041 | −0.034 ± 0.016 | t=−4.74 | 0.009 |

Sigmoid is numerically better in both models. The U-Net advantage is
statistically significant (p=0.009); the DoseGAN advantage does not reach
significance at five folds (p=0.102), though the direction is consistent
across all five folds (fold-level delta ranges from −0.014 to −0.113 Gy).
The decision to use Sigmoid is empirically supported and consistent with
the CLAUDE.md note that Sigmoid wins on all 10 fold-model combinations.

Note: these are the pre-retrain checkpoints used for the 2×2 ablation
(2026-05-18). The main CV tables above use the retrained checkpoints
(2026-06-18). The activation choice is locked; the ablation is a
retrospective confirmation, not a repeated experiment.

---

## 8. Leakage — inference policy (PENDING DECISION)

U-Net produces raw predictions with ~40 Gy outside the body mask
(systematic, all 5 folds). DoseGAN leakage is variable (fold std 5–8 Gy).
All body, structure, and boundary metrics already use body-masked
evaluation, so reported accuracy numbers are unaffected by the raw leakage.

Two policy options before the test set runs:

Option A — Retain raw outputs, report leakage as a finding.
  The leakage table stands as a real behavioural difference between models.
  The test set would also be evaluated on raw outputs.

Option B — Apply body mask to all predictions at inference, report this
  as a post-processing step applied equally to both models.
  Existing body/structure/boundary metrics are unchanged.
  The leakage rows in the table would become identically zero and can be dropped.

This decision must be locked before running evaluate.py on the test set.

---

## What remains pending

- Leakage/inference policy decision (see section 8)
- Test-set evaluation (all four conditions, `evaluate.py --split test`)
- Test-set acquisition subgroup (`inv1_acquisition_breakdown.py --split test`)
- Abstract and conclusion claims dependent on test-set numbers
