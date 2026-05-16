# Thesis Reporting Tracker

Working document. Update as runs finish and sections get drafted.

Last updated: 2026-05-16

---

## 1. Status Overview

### Done
- [x] U-Net Sigmoid — 5-fold trained + evaluated (val)
- [x] U-Net Tanh — 5-fold trained + evaluated (val)
- [x] DoseGAN Tanh — 5-fold trained + evaluated (val)
- [x] 2x2 activation ablation (U-Net x DoseGAN, Tanh vs Sigmoid) — decision: **Sigmoid wins for both**
- [x] Investigation 1 (acquisition group breakdown) — U-Net Sigmoid + DoseGAN Tanh
- [x] Investigation 2 (worst-case patient visualisations) — U-Net Sigmoid + DoseGAN Tanh
- [x] Investigation 3 (dose-smearing index) — code exists

### Pending
- [ ] DoseGAN Sigmoid 5-fold training (running, ETA ~24h)
- [ ] DoseGAN Sigmoid evaluation (depends on training)
- [ ] DoseGAN Sigmoid Inv1 + Inv2 analyses (depends on eval)
- [ ] Investigation 3 at scale across all 4 model variants
- [ ] DVH curve comparison (not implemented)
- [ ] Gamma analysis (not implemented)
- [ ] Test-set evaluation for all final models (val-set only so far)
- [ ] DoseGNN results (collaborator)

### Blocked
- DoseGNN results — waiting on collaborator
- Geometric channels 8-14 — waiting on `V5geometric_channels.py` (not blocking thesis if scoped to 9-channel input)

---

## 2. Methods Section

| # | Item | Status | Source |
|---|------|--------|--------|
| M1 | Dataset description: LUND-PROBE 432 patients, NIfTI, 9-channel input (8 masks + sCT), dose target | READY | `preprocessing/preprocessing.py`, `docs/thesis_proposal.md` |
| M2 | Preprocessing pipeline: resample to 1.5 mm isotropic, z-score sCT inside body, dose / 50 Gy, asymmetric PTV-centric crop 128x256x320 (40 below, 88 above) | READY | `preprocessing/preprocessing.py`, `configs/config_preprocessing_shared.py` |
| M3 | Patient-level split: 15% test held out, 5-fold CV on remainder, stratified by acquisition group (oldAcq/newAcq) | READY | `data/split.csv`, `preprocessing/create_split.py` |
| M4 | U-Net architecture: 3D U-Net, 32 base channels, BatchNorm3d, output activation **Sigmoid** | READY | `models/` (U-Net file) |
| M5 | DoseGAN architecture: 3D U-Net generator with AttGate at every skip connection, PatchGAN discriminator (NLayerDiscriminator + BlockDiscriminator), LSGAN loss | READY | `models/dosegan.py` |
| M6 | **Footnote: activation choice.** Generator output activation set to Sigmoid based on 2x2 ablation (see Results); deviates from GhTara baseline which uses Tanh | READY | This tracker, section 5 |
| M7 | **Footnote: PatchGAN discriminator ends with `BatchNorm3d(1) -> LeakyReLU`** (non-standard, inherited from GhTara upstream) | READY | `models/dosegan.py` |
| M8 | **Footnote: AttGate ties weights between `down_inp` and `sample_inp`** (inherited from upstream) | READY | `models/dosegan.py` |
| M9 | **Footnote: BatchNorm3d with batch_size=1** in both models (common in 3D medical imaging due to memory constraints) | READY | `training/train_*.py` |
| M10 | Training setup: optimizer, LR, epochs, augmentation (random LR flip, depth flip, sCT intensity scale/shift; masks not augmented), batch_size=1 | READY | `configs/config_dosegan.py`, `training/dataset.py` |
| M11 | Metric definitions: body_MAE_Gy, body_RMSE_Gy (over body-contour voxels = channel 7); val_L1 body-masked throughout training | READY | `training/evaluate_*.py` |
| M12 | **Caveat: val_L1 was redefined on 2026-05-13** — pre/post values not directly comparable (conversion: new ≈ old / 0.32) | READY | `CLAUDE.md` Gotchas, commit `a2c5cc7` |
| M13 | Hardware / runtime (Snellius HPC, GPU, training wallclock per fold) | READY | SLURM logs in `outputs/logs/` |
| M14 | DVH and gamma analysis methods | OPTIONAL / PENDING | Not implemented yet |
| M15 | Investigation 3 (dose-smearing index) — metric definition: ratio of predicted to ground-truth dose gradient magnitude | OPTIONAL | code exists, not run at scale |

---

## 3. Results Section

### 3.1 Main quantitative comparison (per-model, val set, 5-fold)

| # | Item | Status | Source |
|---|------|--------|--------|
| R1 | Table: per-model val body_MAE_Gy and body_RMSE_Gy, mean ± std across 5 folds | PARTIAL — DoseGAN Sigmoid pending (~24h) | `outputs/evaluation/*_fold*_val.csv` |
| R2 | U-Net Sigmoid (main): body_MAE_Gy = **0.861 ± 0.026** Gy; body_RMSE_Gy = **1.514 ± 0.038** Gy | READY | `outputs/evaluation/unet3d_ch32_sigmoid_snellius_fold*_val.csv` |
| R3 | DoseGAN Sigmoid (main): body_MAE_Gy + body_RMSE_Gy | PENDING DoseGAN Sigmoid training | will be `outputs/evaluation/dosegan_ngf32_sigmoid_snellius_fold*_val.csv` |
| R4 | DoseGNN (main): body_MAE_Gy + body_RMSE_Gy | PENDING collaborator | — |
| R5 | Test-set numbers for all three main models | PENDING | needs eval script on test split |

### 3.2 Activation ablation (2x2)

| # | Item | Status | Source |
|---|------|--------|--------|
| R6 | Table: Tanh vs Sigmoid for U-Net and DoseGAN, val body_MAE_Gy / body_RMSE_Gy per fold + mean ± std | PARTIAL — DoseGAN Sigmoid pending | see section 5 below |
| R7 | U-Net Sigmoid vs Tanh: 0.861 ± 0.026 vs 0.895 ± 0.041 Gy (body_MAE); 1.514 vs 1.670 (body_RMSE) — Sigmoid wins | READY | this tracker |
| R8 | DoseGAN Sigmoid vs Tanh (body-masked val_L1, normalised, W&B): 0.0174 ± 0.0007 vs 0.0182 ± 0.0013 — Sigmoid wins/ties every fold | READY | W&B group `dosegan_ngf32_{sigmoid,tanh}_snellius` |
| R9 | Conclusion sentence: Sigmoid chosen as default for both models | READY | this tracker, section 5 |

### 3.3 Investigation 1: Acquisition group subgroup analysis

| # | Item | Status | Source |
|---|------|--------|--------|
| R10 | Boxplot: body_MAE_Gy split by oldAcq vs newAcq, U-Net Sigmoid | READY | `outputs/analysis/inv1_unet3d_ch32_sigmoid_snellius_acquisition_boxplot.png` |
| R11 | CSV with per-group means/medians, U-Net Sigmoid | READY | `outputs/analysis/inv1_unet3d_ch32_sigmoid_snellius_acquisition_breakdown.csv` |
| R12 | Same boxplot + CSV for DoseGAN Tanh | READY | `outputs/analysis/inv1_dosegan_*_acquisition_*` |
| R13 | Same boxplot + CSV for DoseGAN Sigmoid | PENDING DoseGAN Sigmoid eval | — |
| R14 | Finding to report: oldAcq patients have higher error (distribution shift effect) | READY | Inv 1 outputs |

### 3.4 Investigation 2: Worst-case failure mode visualisations

| # | Item | Status | Source |
|---|------|--------|--------|
| R15 | Top-3 worst-case patient axial/sagittal/coronal panels per fold (15 patients), U-Net Sigmoid | READY | `outputs/analysis/inv2_unet3d_ch32_sigmoid_snellius_worst_*.png` |
| R16 | Same for DoseGAN Tanh | READY | `outputs/analysis/inv2_dosegan_*.png` |
| R17 | Same for DoseGAN Sigmoid | PENDING DoseGAN Sigmoid eval | — |
| R18 | Failure pattern narrative: bladder over-prediction + dose-smearing (predicts average rather than sharp gradients) | READY | Inv 2 outputs + `findings_dosegan_failure_modes` memory |

### 3.5 Investigation 3: Dose-smearing index (originality lever)

| # | Item | Status | Source |
|---|------|--------|--------|
| R19 | Per-patient dose-smearing index across all 4 model variants, with subgroup breakdown (oldAcq/newAcq) | OPTIONAL — code exists, run pending | `preprocessing/`/`analysis/` (Inv 3 script) |
| R20 | Correlation between dose-smearing index and body_MAE_Gy | OPTIONAL | derived from R19 |

### 3.6 Clinical metrics (optional but recommended)

| # | Item | Status | Source |
|---|------|--------|--------|
| R21 | DVH curve comparison per OAR + PTV | OPTIONAL / PENDING | Not implemented |
| R22 | Gamma pass rate (3%/3 mm, 2%/2 mm) | OPTIONAL / PENDING | Not implemented |

---

## 4. Discussion Section

| # | Item | Status | Source |
|---|------|--------|--------|
| D1 | Headline result: which model wins on body_MAE_Gy / body_RMSE_Gy and by how much (U-Net vs DoseGAN vs DoseGNN) | PENDING DoseGAN Sigmoid + DoseGNN | R1 |
| D2 | **Activation finding:** Sigmoid mildly but consistently outperforms Tanh; argued via `Sigmoid(0) ≈ 0.5` being neutral and Tanh saturation at the [0,1] range edge — frame as a small but defensible deviation from GhTara | READY | section 5 |
| D3 | **Acquisition-group distribution shift:** oldAcq patients drive most of the error tail; implications for transfer to pancreatic cohort in Phase 2 | READY | R10-R14 |
| D4 | **Failure-mode discussion:** bladder over-prediction + dose-smearing; argue this is the dominant failure mode for both U-Net and DoseGAN — activation choice does not fix it | READY | R15-R18, `findings_dosegan_failure_modes` memory |
| D5 | **Originality lever (dose-smearing index):** if R19 is run, this is the strongest novelty claim — quantifies a failure mode that body_MAE_Gy averages over | OPTIONAL | R19-R20 |
| D6 | **Limitations:** val-only metrics (no test-set yet), no DVH/gamma, single dataset for Phase 1, batch_size=1 → BatchNorm3d acts as InstanceNorm, geometric channels 8-14 not implemented | READY | this tracker + CLAUDE.md |
| D7 | **What I would do differently:** address dose-smearing at the loss level (gradient-matching loss, perceptual loss); evaluate with DVH/gamma earlier; preload pickles to reduce I/O variance | READY | reflection |
| D8 | Inherited GhTara quirks (PatchGAN tail BN+LeakyReLU, AttGate weight-tying) — flag as inherited, not bugs; note that fixing them is future work | READY | M7-M8 |
| D9 | Comparison with prior dose-prediction literature (GhTara, others) | READY | `docs/thesis_proposal.md`, `reference_dosegan_ghtara` memory |
| D10 | Phase 2 / Phase 3 outlook: transfer to pancreatic cohort, robustness under anatomical variation | READY | `docs/thesis_proposal.md` |

---

## 5. Activation Ablation — Why Sigmoid Is the Main Model

**Setup.** Both U-Net and DoseGAN trained 5-fold CV on LUND-PROBE val split, all other hyperparameters held constant. Only the generator output activation differs (Tanh vs Sigmoid).

**Results.**

| Model | Activation | body_MAE_Gy (Gy) | body_RMSE_Gy (Gy) | body-masked val_L1 (norm) |
|-------|------------|------------------|--------------------|---------------------------|
| U-Net | Sigmoid    | **0.861 ± 0.026** | **1.514 ± 0.038** | — |
| U-Net | Tanh       | 0.895 ± 0.041     | 1.670 ± 0.050     | — |
| DoseGAN | Sigmoid  | pending           | pending            | **0.0174 ± 0.0007** |
| DoseGAN | Tanh     | pending           | pending            | 0.0182 ± 0.0013 |

U-Net per-fold body_MAE_Gy:

| Fold | Sigmoid | Tanh |
|------|---------|------|
| 0 | 0.863 | 0.894 |
| 1 | 0.896 | 0.950 |
| 2 | 0.837 | 0.851 |
| 3 | 0.873 | 0.921 |
| 4 | 0.833 | 0.860 |

**Decision rationale.**
- Sigmoid wins on every fold for U-Net (lower body_MAE_Gy, lower body_RMSE_Gy, lower variance).
- Sigmoid wins or ties on every fold for DoseGAN body-masked val_L1, ~4% lower mean, lower variance.
- Tanh's only theoretical advantage (`Tanh(0) = 0` is a free correct prediction at init for the dominant zero-target voxels) does not outweigh the empirical improvement.
- Dose target is normalised to [0, 1] (dose / 50 Gy), so Sigmoid is the natural range match.

**Reporting choice.** Sigmoid is the **main reported model** for both U-Net and DoseGAN. Tanh appears in the ablation table (R6) and is discussed in D2 as a deviation from the GhTara baseline.

**Methods footnote text (draft).**
> Generator output activation was set to Sigmoid following a 2x2 activation ablation (Tanh vs Sigmoid, U-Net vs DoseGAN, 5-fold CV). This deviates from the GhTara baseline which uses Tanh; on body-masked val L1 Sigmoid won or tied on every fold for both architectures.

---

## 6. Do Next (priority order)

1. **Wait for DoseGAN Sigmoid training** (~24h). Monitor with `squeue -u akhalil` and `tail -f outputs/logs/<jobid>.out`.
2. **Evaluate DoseGAN Sigmoid** on val split: `for F in 0 1 2 3 4; do python3 -m training.evaluate_dosegan --fold $F; done` → 5 CSVs in `outputs/evaluation/`.
3. **Fill in R3, R6 (DoseGAN row), R8** in the table above with body_MAE_Gy + body_RMSE_Gy. Update this tracker.
4. **Run Inv 1 + Inv 2 for DoseGAN Sigmoid** → fills R13, R17.
5. **Run Investigation 3 at scale** across all 4 model variants (U-Net Sigmoid/Tanh, DoseGAN Sigmoid/Tanh). This is the originality lever — prioritise if time allows.
6. **Implement + run test-set evaluation** for the final main models (U-Net Sigmoid, DoseGAN Sigmoid, DoseGNN once received).
7. **(Stretch) Implement DVH + gamma** — strongest clinical-credibility addition; only attempt if Investigation 3 is done.
8. **Draft Methods section** using rows M1-M13 as a checklist; add M14-M15 only if implemented.
9. **Draft Results section** using R1-R18 as a checklist; add R19-R22 only if implemented.
10. **Draft Discussion** using D1-D10 as a checklist; emphasise D2, D4, D5 (originality levers).

---

## 7. File Path Reference

| Artefact | Path |
|---|---|
| Evaluation CSVs | `outputs/evaluation/<RUN_NAME>_fold{N}_val.csv` |
| Inv 1 (acquisition) | `outputs/analysis/inv1_<RUN_NAME>_acquisition_{boxplot.png,breakdown.csv}` |
| Inv 2 (worst-case) | `outputs/analysis/inv2_<RUN_NAME>_worst_{1,2,3}_fold{N}_{oldAcq,newAcq}_<hash>.png` |
| Checkpoints | `outputs/checkpoints_{unet3d,dosegan}/<RUN_NAME>_fold{N}_best.pt` |
| SLURM logs | `outputs/logs/<jobname>_<jobid>.out` |
| W&B project | `doseprediction-lundprobe`, groups `{unet3d_ch32,dosegan_ngf32}_{sigmoid,tanh}_snellius` |
| Split | `data/split.csv` (do not regenerate) |
