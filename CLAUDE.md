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
outputs/pickles/<patient_id>.pkl          # (9,128,256,320) input + (128,256,320) dose
        ↓  training/dataset.py
LUNDPROBEDataset → DataLoader
        ↓  training/train_dosegan.py
outputs/checkpoints_dosegan/dosegan_fold0_best.pt
```

### Key config files
- `configs/config_preprocessing_shared.py` — `DATA_ROOT`, `OUTPUT_DIR` (pickles), crop sizes, channel map
- `configs/config_dosegan.py` — all training hyperparameters, `SPLIT_CSV`, `PICKLE_DIR`, `RUN_NAME`

**Change `RUN_NAME` in `config_dosegan.py` for every meaningful experiment** — it appears in W&B.

### Model (`models/dosegan.py`)
3D U-Net generator with attention gates at every skip connection (`UnetSkipConnectionBlock3d` + `AttGate`). PatchGAN discriminator (`NLayerDiscriminator`) with `BlockDiscriminator` blocks. LSGAN loss by default (`USE_LSGAN=True`).

### Dataset (`training/dataset.py`)
Loads pickles on-demand (full cohort ~130 GB RAM if preloaded). Training augmentation: random LR flip, random depth flip, random sCT intensity scale/shift (masks are never augmented). The `channels` parameter lets different models use subsets of the 9-channel cache.

### Preprocessing (`preprocessing/preprocessing.py`)
Per patient: resample to 1.5 mm isotropic → z-score sCT inside body contour → normalise dose by 50 Gy → asymmetric PTV-centric crop (128×256×320: 40 slices below PTV centroid, 88 above). Optional structures (Genitalia, PenileBulb) are zero-filled if missing.

### Split
`outputs/split.csv` is committed and must not be regenerated casually — it is the permanent record of train/val/test assignments. 15% held-out test set + 5-fold CV on the remainder, stratified by acquisition group (oldAcq/newAcq).

## Paths Reference

| What | Path |
|---|---|
| Raw NIfTI data | `raw_data/lund-probe/lund-probe/basePart/` |
| Preprocessed pickles | `outputs/pickles/` |
| Train/val/test split | `outputs/split.csv` |
| Checkpoints | `outputs/checkpoints_dosegan/` |
| SLURM logs | `outputs/logs/` |
| W&B runs | tracked under project `doseprediction-lundprobe` |

## Geometric channels

Channels 8–14 (distance maps, angles, radiological depth) are not yet implemented — waiting on `V5geometric_channels.py` from collaborator. All pickles have `geometric_channels_pending=True`. Current input tensor uses channels 0–7 (structure masks) + channel 15 (sCT), giving 9 channels total.
