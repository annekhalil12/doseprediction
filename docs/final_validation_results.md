# Final Validation Results — 2-Model × 2-Condition Comparison

**Status:** PENDING — DoseGAN baseline evaluation jobs must complete before tables can be filled.
See Blocker 4 notes in project tracking.

**Supersedes:** `docs/archive/results_validation_stale_2026-06-03.md` (stale: DoseGAN baseline column
was from old BatchNorm checkpoints; patient-level std was used without being labelled as such).

---

## Canonical result pipeline

```
outputs/evaluation/<run_name>_fold{0..4}_val.csv   (one file per model × condition × fold)
    ↓
analysis/clinical_report.py
    ↓
outputs/analysis/clinical_report.csv   (machine-readable, all conditions × metrics)
outputs/analysis/clinical_report.png   (three-panel summary figure)
    ↓
tables in this document
```

To regenerate after any eval update:
```bash
export PYTHONPATH=/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction
python3 -m analysis.clinical_report
```

---

## Standard deviation convention

All tables in this document report **fold-level standard deviation**, not patient-level.

Specifically: for each metric, the per-patient values within each fold are averaged to give
one fold mean. The reported mean ± std is then the mean and standard deviation across those
5 fold means (ddof=1). This follows the standard approach for k-fold cross-validation reporting
and gives the variability of the model's performance estimate, not the variability of patient outcomes.

Patient-level std within a single condition would typically be ~3–5× larger. The two numbers
are not interchangeable and must not be compared across sources that use different conventions.

---

## Required evaluation files before tables can be filled

| File | Status |
|---|---|
| `unet3d_ch32_sigmoid_snellius_fold{0..4}_val.csv` | complete |
| `unet3d_ch32_sigmoid_geom_snellius_fold{0..4}_val.csv` | complete |
| `dosegan_ngf32_sigmoid_snellius_fold{0..4}_val.csv` | **PENDING** (Blocker 4) |
| `dosegan_ngf32_sigmoid_geom_snellius_fold{0..4}_val.csv` | complete |

All complete CSVs also need `--force` re-evaluation because the isodose metrics were fixed
(Blocker 3: body-masking before thresholding; leakage columns added). Re-run commands:

```bash
# DoseGAN baseline — missing, run fresh
for FOLD in 0 1 2 3 4; do
  sbatch --export=ALL,MODEL=dosegan,FOLD=$FOLD,GEOM=0,RUN_NAME=dosegan_ngf32_sigmoid_snellius eval.sbatch
done

# All other conditions — overwrite stale isodose columns
for FOLD in 0 1 2 3 4; do
  sbatch --export=ALL,MODEL=unet3d,FOLD=$FOLD,GEOM=0,FORCE=1 eval.sbatch
  sbatch --export=ALL,MODEL=unet3d,FOLD=$FOLD,GEOM=1,FORCE=1 eval.sbatch
  sbatch --export=ALL,MODEL=dosegan,FOLD=$FOLD,GEOM=1,FORCE=1 eval.sbatch
done
```

---

## 1. Global dose accuracy

*Fold mean ± fold std across 5 folds. Body MAE is the primary metric.*

| Metric | Unit | DoseGAN base | DoseGAN geom | U-Net base | U-Net geom |
|---|---|---|---|---|---|
| Body MAE | Gy | PENDING | | | |
| Body RMSE | Gy | PENDING | | | |
| PTV MAE | Gy | PENDING | | | |
| Rectum MAE | Gy | PENDING | | | |
| Bladder MAE | Gy | PENDING | | | |

*Fill from `outputs/analysis/clinical_report.csv` section "1. Global dose accuracy" after re-running `clinical_report.py`.*

---

## 2. DVH clinical endpoints

*Mean absolute error of DVH endpoints. Fold mean ± fold std.*

| Metric | Unit | DoseGAN base | DoseGAN geom | U-Net base | U-Net geom |
|---|---|---|---|---|---|
| PTV D95 \|err\| | Gy | PENDING | | | |
| PTV D98 \|err\| | Gy | PENDING | | | |
| PTV Dmean \|err\| | Gy | PENDING | | | |
| Rectum Dmean \|err\| | Gy | PENDING | | | |
| Bladder Dmean \|err\| | Gy | PENDING | | | |
| Rectum V_Rx \|err\| | % | PENDING | | | |
| Bladder V_Rx \|err\| | % | PENDING | | | |

*Fill from `outputs/analysis/clinical_report.csv` section "2. DVH clinical endpoints".*

---

## 3. Spatial boundary quality

*Body-masked isodose Dice and HD95 (body mask applied before thresholding — see Blocker 3 fix).*

| Metric | Unit | DoseGAN base | DoseGAN geom | U-Net base | U-Net geom |
|---|---|---|---|---|---|
| PTV boundary MAE | Gy | PENDING | | | |
| Dice 100% iso | — | PENDING | | | |
| Dice 95% iso | — | PENDING | | | |
| Dice 80% iso | — | PENDING | | | |
| Dice 50% iso | — | PENDING | | | |
| HD95 100% iso | mm | PENDING | | | |
| Leakage mean pred | Gy | PENDING | | | |
| Leakage vol frac | — | PENDING | | | |

*Fill from `outputs/analysis/clinical_report.csv` section "3. Spatial boundary quality".*
*Leakage columns are new — add them to the SPATIAL_BOUNDARY list in `clinical_report.py` if not already present.*

---

## 4. Gamma pass rate

Gamma pass rate (3%/3mm, 2%/2mm) was omitted from all evaluation runs (`--skip-gamma`) due to
computational cost (~10 min/patient for 3D gamma). All gamma columns in the eval CSVs are NaN.
This is disclosed as a limitation in the thesis methods section.

---

## Notes on interpretation

- The Dice and HD95 numbers reported here use **body-masked thresholding** (both pred and true zeroed
  outside the body before computing isodose regions). Earlier numbers in the archived file did not
  apply this mask; the U-Net near-zero Dice values there may have been partly an outside-body
  leakage artifact rather than true spatial failure. The leakage columns will quantify this.
- The std convention change (patient-level → fold-level) will reduce reported ± values substantially.
  This is not an improvement in model performance; it is a correction in reporting precision.
