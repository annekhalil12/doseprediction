# 3D Dose Prediction for Prostate Radiotherapy (LUND-PROBE)

3D dose prediction using a GAN (DoseGAN) and a U-Net baseline, trained on the LUND-PROBE prostate cohort (432 patients, NIfTI format). Phase 1 of an MSc thesis benchmarking deep dose-prediction architectures and analysing their behaviour under distribution shift.

## Input / output

Each patient is preprocessed to a 9-channel `(9, 128, 256, 320)` float32 input tensor (8 binary structure masks + sCT intensity) and a `(1, 128, 256, 320)` target dose volume normalised by 50 Gy.

| Channel | Content |
|---|---|
| 0 | PTV (PTVT_427) |
| 1 | Rectum |
| 2 | Bladder |
| 3 | Femoral Head L |
| 4 | Femoral Head R |
| 5 | Genitalia |
| 6 | Penile Bulb |
| 7 | BODY contour |
| 8 | sCT intensity (z-scored inside body) |

## Environment

Tested on Snellius HPC (SURF), Python 3.9, CUDA 12.1, single H100. Packages installed to `~/.local/lib/python3.9/` (the `venv/` in the repo is a Windows artefact).

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121  # for torch
pip install -r requirements.txt  # for everything else
```

Scripts run from the project root with `PYTHONPATH` set:

```bash
export PYTHONPATH=/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction
python3 <script>
```

## Common commands

```bash
# Sanity-check preprocessing on one patient
python3 preprocessing/test_single_patient.py

# Run full preprocessing (~15-20 min)
sbatch preprocess.sbatch

# Regenerate the train/val/test split (only if patient set changes)
python3 preprocessing/create_split.py

# Verify all pickles have correct shape before training
python3 -m preprocessing.check_pickle_shapes

# Smoke test — 2 batches through the full GAN pipeline
python3 training/smoke_test_dosegan.py

# Train DoseGAN (one fold)
sbatch --export=ALL,FOLD=0 train_dosegan.sbatch

# Train U-Net (one fold; --activation flag selects sigmoid|tanh)
sbatch --export=ALL,FOLD=0 train_unet3d.sbatch
sbatch --export=ALL,FOLD=0,ACT=tanh train_unet3d.sbatch

# Evaluate (writes per-patient CSV + W&B eval run)
python3 -m training.evaluate_dosegan --fold 0
python3 -m training.evaluate_unet3d  --fold 0

# Job monitoring
squeue -u $USER
tail -f outputs/logs/<jobname>_<jobid>.out
```

## Data flow

```
raw_data/lund-probe/lund-probe/basePart/<patient_id>/
    sCT/image_reg2MRI.nii.gz
    MR_StorT2/dose_interpolated.nii.gz + mask_*.nii.gz
        │
        │  preprocessing/preprocessing.py
        ▼
data/pickles/<patient_id>.pkl       # (9,128,256,320) input + (128,256,320) dose
        │
        │  training/dataset.py
        ▼
LUNDPROBEDataset → DataLoader
        │
        │  training/train_dosegan.py | train_unet3d.py
        ▼
outputs/checkpoints_{dosegan,unet3d}/<run_name>_fold<F>_best.pt
        │
        │  training/evaluate_{dosegan,unet3d}.py
        ▼
outputs/evaluation/<run_name>_fold<F>_val.csv
        │
        │  analysis/inv1_acquisition_breakdown.py
        │  analysis/inv2_worst_patient_dvh_maps.py
        ▼
outputs/analysis/*.png + *.csv
```

## Repository layout

```
configs/                hyperparameters (one config file per pipeline component)
models/                 DoseGAN (3D U-Net generator + PatchGAN discriminator) and 3D U-Net
preprocessing/          NIfTI → pickle pipeline
training/               train / smoke-test / evaluate / visualise scripts + LUNDPROBEDataset
analysis/               post-eval investigations (per-patient + per-acquisition-group)
data/                   split.csv + preprocessing_summary.csv; pickles are gitignored
outputs/                checkpoints, logs, eval CSVs, analysis figures (mostly gitignored)
docs/                   thesis docs (gitignored — personal)
*.sbatch                SLURM submission scripts
```

## Models

**DoseGAN** (`models/dosegan.py`)
3D U-Net generator with attention gates at every skip connection (`UnetSkipConnectionBlock3d` + `AttGate`). PatchGAN discriminator (`NLayerDiscriminator`). LSGAN loss by default. ~29.6M parameters. Adapted from the GhTara/Dose_Prediction repository.

**U-Net 3D baseline** (`models/unet3d.py`)
MONAI 5-level residual U-Net, 6-level `CHANNELS=(32,64,128,256,256,256)` matching DoseGAN generator's receptive field. ~21.8M parameters. Configurable output activation (`sigmoid` | `tanh`).

Both models accept the same `(9, 128, 256, 320)` input and predict a normalised dose volume `(1, 128, 256, 320)` in `[0, 1]` (multiply by 50 to get Gy).

## Split

`data/split.csv` is committed and must not be regenerated casually — it is the permanent record of train/val/test assignments. 15% held-out test set + 5-fold CV on the remainder, stratified by acquisition group (`oldAcq` / `newAcq`).

## Preprocessing

Per patient: resample to 1.5 mm isotropic → z-score sCT inside body contour → normalise dose by 50 Gy → asymmetric PTV-centric crop (128×256×320: 40 slices below PTV centroid, 88 above). Optional structures (Genitalia, PenileBulb) are zero-filled if missing.

## Experiment tracking

W&B project: `doseprediction-lundprobe`. All runs use `group=cfg.RUN_NAME` so the 5 folds of one experiment collapse into a single row. `job_type` distinguishes `train` from `eval`. Evaluation results (per-patient body MAE/RMSE + DVH metrics per structure) are logged both to per-fold CSVs (`outputs/evaluation/<run_name>_fold<F>_val.csv`) and as W&B Tables.

## Geometric channels

Channels 8–14 in `configs/config_preprocessing_shared.py` (distance maps, angles, radiological depth) are not yet implemented — waiting on `V5geometric_channels.py` from a collaborator. All pickles currently set `geometric_channels_pending=True`; models use channels 0–7 (structure masks) + channel 15 (sCT) = 9 channels total.
