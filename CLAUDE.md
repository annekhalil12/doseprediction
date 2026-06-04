# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D dose prediction for prostate radiotherapy using a GAN (DoseGAN). Input: 9-channel tensor (8 structure masks + sCT). Output: predicted dose volume. Dataset: LUND-PROBE cohort (432 patients, NIfTI format).

## Environment

This project runs on Snellius HPC (SURF). Packages are installed to `~/.local/lib/python3.9/` (user install — the `venv/` in the repo is a Windows artefact and does not activate on Linux). Python version is 3.9.

```bash
# Install dependencies
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121  # for torch
pip install -r requirements.txt  # for everything else
```

Always run scripts from the **project root** with `PYTHONPATH` set:
```bash
export PYTHONPATH=/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction
python3 <script>
```

## Common Commands

```bash
# Sanity-check preprocessing on one patient
python3 preprocessing/test_single_patient.py

# Run full preprocessing (16 workers, ~15-20 min)
sbatch preprocess.sbatch

# Regenerate train/val/test split after preprocessing
python3 preprocessing/create_split.py

# Verify all pickles have correct shape before training
python3 -m preprocessing.check_pickle_shapes

# Smoke test — 2 batches through the full GAN pipeline
python3 training/smoke_test_dosegan.py

# Submit training job
sbatch train_dosegan.sbatch

# Monitor jobs
squeue -u akhalil
tail -f outputs/logs/<jobname>_<jobid>.out
```

## Architecture

### Data flow
```
raw_data/lund-probe/lund-probe/basePart/<patient_id>/
    sCT/image_reg2MRI.nii.gz
    MR_StorT2/dose_interpolated.nii.gz + mask_*.nii.gz
        ↓  preprocessing/preprocessing.py
data/pickles/<patient_id>.pkl             # (9,128,256,320) input + (128,256,320) dose
        ↓  training/dataset.py
LUNDPROBEDataset → DataLoader
        ↓  training/train_dosegan.py
outputs/checkpoints_dosegan/dosegan_fold0_best.pt
```

`data/` holds preprocessed inputs (pickles, `split.csv`, `preprocessing_summary.csv`).
`outputs/` holds per-run artefacts (checkpoints, SLURM logs, visualisations).

### Key config files
- `configs/config_preprocessing_shared.py` — `DATA_ROOT`, `OUTPUT_DIR` (pickles), `SUMMARY_CSV`, `SPLIT_CSV`, crop sizes, channel map
- `configs/config_dosegan.py` — all training hyperparameters, `SPLIT_CSV`, `PICKLE_DIR`, `RUN_NAME`

**Change `RUN_NAME` in `config_dosegan.py` for every meaningful experiment** — it appears in W&B.

### Model (`models/dosegan.py`)
3D U-Net generator with attention gates at every skip connection (`UnetSkipConnectionBlock3d` + `AttGate`). PatchGAN discriminator (`NLayerDiscriminator`) with `BlockDiscriminator` blocks. LSGAN loss by default (`USE_LSGAN=True`).

### Dataset (`training/dataset.py`)
Loads pickles on-demand (full cohort ~130 GB RAM if preloaded). Training augmentation: random LR flip, random depth flip, random sCT intensity scale/shift (masks are never augmented). The `channels` parameter lets different models use subsets of the 9-channel cache.

### Preprocessing (`preprocessing/preprocessing.py`)
Per patient: resample to 1.5 mm isotropic → z-score sCT inside body contour → normalise dose by 50 Gy → asymmetric PTV-centric crop (128×256×320: 40 slices below PTV centroid, 88 above). Optional structures (Genitalia, PenileBulb) are zero-filled if missing.

### Split
`data/split.csv` is committed and must not be regenerated casually — it is the permanent record of train/val/test assignments. 15% held-out test set + 5-fold CV on the remainder, stratified by acquisition group (oldAcq/newAcq).

## Paths Reference

| What | Path |
|---|---|
| Raw NIfTI data | `raw_data/lund-probe/lund-probe/basePart/` |
| Preprocessed pickles | `data/pickles/` |
| Train/val/test split | `data/split.csv` |
| Preprocessing summary | `data/preprocessing_summary.csv` |
| Checkpoints | `outputs/checkpoints_dosegan/` |
| SLURM logs | `outputs/logs/` |
| W&B runs | tracked under project `doseprediction-lundprobe` |

## About the Student

Anne Khalil — MSc Information Studies (Data Science), UvA. Background in Psychology. Has autism and ADHD. Needs clear structure, explicit next steps, and one concept at a time. Never assume prerequisite knowledge (e.g. explain what isotropic spacing means before using the term). Patient-level splits only — never fraction-level.

Full working instructions in `docs/project_instructions.md`. Full project context in `docs/thesis_proposal.md`, `docs/project_scope_omar.md`, `docs/thesis_rubric.md`, `docs/thesis_guidelines.md`.

## Project Context

See `docs/thesis_proposal.md` for the full thesis design, `docs/project_scope_omar.md` for the supervisor-defined scope, `docs/thesis_rubric.md` for the grading rubric, and `docs/thesis_guidelines.md` for section-by-section writing requirements.

**Thesis goal (revised 2026-05-22):** Compare 3D U-Net and DoseGAN for prostate dose prediction on LUND-PROBE, with depth in clinical evaluation and failure-mode analysis. DoseGNN and geometric channels are no longer in scope — Nika stopped development because geometric channels did not improve outcomes in her own experiments. The thesis is a 2-model benchmark with originality in evaluation (Inv1/2/3, full clinical metric suite, ablations).

Key implications for the work (from rubric):
- **Originality (60% weight)**: applying an existing GAN architecture to 3D dose prediction scores a 6-7; to push higher the experiments, analyses, and insights need to be novel and well-argued
- **Experimental evaluation**: quantitative error/uncertainty analysis and well-designed ablations are needed for an 8+
- **Reflection**: honest discussion of limitations and what would be done differently matters for the grade

## Geometric channels

Channels 8–14 (distance maps, angles, radiological depth) will **not** be implemented. Nika stopped DoseGNN development (2026-05-22) because geometric channels did not improve outcomes in her experiments. Current input tensor uses channels 0–7 (structure masks) + channel 8 (sCT), giving 9 channels total. The `geometric_channels_pending=True` flag in pickles can be ignored.

## Next steps

**Thesis is now a 2-model comparison (U-Net vs DoseGAN). DoseGNN is out of scope.**

1. **Check grad1.0 fold 0 results** — DoseGAN job 23059382, U-Net job 23059383, submitted 2026-05-22 with `LAMBDA_GRAD=1.0`. RUN_NAMEs: `dosegan_ngf32_sigmoid_grad1.0_snellius`, `unet3d_ch32_sigmoid_grad1.0_snellius`. If val_L1 ≤ baseline, submit folds 1–4; if not, treat as negative result and move on.
2. **Run full evaluation on baseline 5-fold checkpoints** (U-Net Sigmoid + DoseGAN Sigmoid) using the unified eval script (`training/evaluate.py`). Computes the full clinical metric suite including gamma, boundary MAE, isodose Dice/HD95.
3. **Run Inv1 + Inv2 + Inv3** on baseline checkpoints to get the investigation results.
4. **Start writing** — Introduction, Background, Dataset, Preprocessing, Methods are all writable now (no experiments needed). ~60–70% of the thesis can be written before any pending job finishes.
5. **Test-set evaluation** for final models once all training decisions are locked.

## Gotchas

- **`val_L1` was redefined on 2026-05-13 (commit `a2c5cc7`).** Before: `nn.L1Loss()(pred, target)` over all ~10.5M voxels of the (1, 128, 256, 320) volume including air. After: `(|pred - target| * body_mask).sum() / body_mask.sum()` over body voxels only (~32% of the cropped volume). For a model that predicts ≈0 outside the body, **`new_val_L1 ≈ old_val_L1 / 0.32`**. Do **not** compare `val_L1` numbers naïvely across this commit. Example: the VUMC `dosegan_fold0_baseline` run (W&B id `olnki6mt`, 2026-05-05) reported best `val_L1 = 0.0069` on the old metric, which equals roughly `0.022` in current units — comparable to the post-refactor Snellius runs, not 3× better. Convert before comparing.

- **Generator output activation is `Sigmoid` (switched in commit `a3c8ffb`, 2026-05-16).** The 2×2 ablation (Sigmoid vs Tanh × U-Net vs DoseGAN, confirmed 2026-05-18) showed Sigmoid wins on all 10 fold-model combinations. val_L1 (body-masked): DoseGAN Sigmoid 0.0174 ± 0.0007 vs Tanh 0.0182 ± 0.0013; U-Net Sigmoid 0.0172 ± 0.0005 vs Tanh 0.0179 ± 0.0008. In the thesis methods, note this as an empirically motivated deviation from GhTara (which uses Tanh). The `Tanh(0)=0` free-init argument for Tanh was outweighed by the consistent empirical advantage of Sigmoid.

- **Thesis methods footnotes (no code change needed):** DoseGAN's PatchGAN discriminator ends with `BatchNorm3d(1) → LeakyReLU` (non-standard); `AttGate` ties weights between `down_inp` and `sample_inp`; both models use `BatchNorm3d` with batch_size=1. Inherited from upstream GhTara — flag in the methods section, not bugs.
