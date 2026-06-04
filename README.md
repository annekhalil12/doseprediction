# 3D Dose Prediction for Prostate Radiotherapy (LUND-PROBE)

MSc thesis benchmark comparing 3D U-Net and DoseGAN for MR-guided prostate dose prediction, with and without explicit geometric input encoding. Dataset: LUND-PROBE cohort (432 patients, NIfTI format), Snellius HPC (SURF).

**Main research question:** To what extent does explicit geometric input encoding improve 3D dose prediction accuracy in prostate MR-guided radiotherapy, and does this effect generalise across a supervised regression model (U-Net) and an adversarial model (DoseGAN)?

## Sub-questions and metrics

| SRQ | Question | Metrics |
|---|---|---|
| SRQ 1 | How do U-Net and DoseGAN compare under a shared evaluation framework? | MAE_body, RMSE_body, boundary_MAE (PTV / Rectum / Bladder) — 5-fold mean ± std |
| SRQ 2 | Does adding geometric input channels improve dose prediction accuracy? | Same metrics, with vs without geom channels, per model; boundary vs global improvement ratio |
| SRQ 3 | Do predicted doses satisfy clinically relevant DVH criteria? | D95_err, Dmean_err, D0.1cc_err for PTV; D0.1cc_err for Rectum and Bladder |

## Input conditions

### Baseline — 9 channels `(9, 128, 256, 320)`

| Ch | Content |
|---|---|
| 0 | PTV (PTVT_427) binary mask |
| 1 | Rectum binary mask |
| 2 | Bladder binary mask |
| 3 | Femoral Head L binary mask |
| 4 | Femoral Head R binary mask |
| 5 | Genitalia binary mask (zeros if absent) |
| 6 | Penile Bulb binary mask (zeros if absent) |
| 7 | BODY contour binary mask |
| 8 | sCT intensity (z-scored inside body, HU window −990–2000) |

### With geometric channels — 14 channels `(14, 128, 256, 320)`

Channels 0–8 as above, plus:

| Ch | Content |
|---|---|
| 9  | dist_to_ptv_surface — Euclidean distance to nearest PTV boundary voxel, normalised [0,1] |
| 10 | dist_to_body_surface — Euclidean distance to nearest skin surface, normalised [0,1] |
| 11 | dir_z_shifted — sup/inf unit direction from PTV centroid, shifted to [0,1] |
| 12 | dir_y_shifted — ant/post unit direction from PTV centroid, shifted to [0,1] |
| 13 | dir_x_shifted — left/right unit direction from PTV centroid, shifted to [0,1] |

Geometric channels are computed once from the existing pickles by `preprocessing/add_geom_channels.py` and stored in each pickle under `geom_channels`. The dataset's `use_geom_channels=True` flag appends them at load time.

## Experiment status

All 4 conditions use InstanceNorm(affine=True), Sigmoid output, LSGAN, 200 epochs max, patience 30, 5-fold CV.

### Baseline (9-channel, no geom)
| Model | Status | W&B group |
|---|---|---|
| U-Net Sigmoid | Complete — folds 0–4 | `unet3d_ch32_sigmoid_snellius` |
| DoseGAN Sigmoid | Training complete — **evaluation CSVs pending re-run** (see Blocker 4 in `docs/final_validation_results.md`) | `dosegan_ngf32_sigmoid_snellius` |

### With-geom (14-channel) — complete
| Model | Status | W&B group |
|---|---|---|
| U-Net geom | Complete — folds 0–4 | `unet3d_ch32_sigmoid_geom_snellius` |
| DoseGAN geom | Complete — folds 0–4 | `dosegan_ngf32_sigmoid_geom_snellius` |

### Ablations — complete (all negative results)
| Ablation | Finding |
|---|---|
| Sigmoid vs Tanh (2×2, fold 0–4) | Sigmoid wins every fold for both models (~4% lower MAE) |
| Gradient-magnitude loss λ=1.0 (fold 0) | No improvement for U-Net; +4.6% worse for DoseGAN |
| BCE vs LSGAN (DoseGAN fold 0) | BCE val_L1=0.0195 vs LSGAN 0.0174 — worse |

## Environment

Snellius HPC (SURF), Python 3.9, CUDA 12.1, H100 GPU. Packages installed to `~/.local/lib/python3.9/` (the `venv/` in the repo is a Windows artefact).

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121  # torch
pip install -r requirements.txt  # everything else
```

Always run from the project root with `PYTHONPATH` set:

```bash
export PYTHONPATH=/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction
python3 <script>
```

## Common commands

```bash
# ── Preprocessing ──────────────────────────────────────────────────────────
sbatch preprocess.sbatch                        # full cohort (~15-20 min, 16 workers)
python3 preprocessing/test_single_patient.py   # sanity-check one patient
python3 preprocessing/create_split.py          # regenerate split (only if cohort changes)
python3 -m preprocessing.check_pickle_shapes   # verify all pickles before training

# Add geometric channels to existing pickles (run once)
sbatch preprocessing/add_geom_channels.sbatch

# ── Training ───────────────────────────────────────────────────────────────
sbatch --export=ALL,FOLD=0 train_dosegan.sbatch          # baseline DoseGAN fold 0
sbatch --export=ALL,FOLD=0 train_unet3d.sbatch           # baseline U-Net fold 0

# Geom-channel variants (--geom flag sets USE_GEOM_CHANNELS and INPUT_NC automatically):
for fold in 0 1 2 3 4; do sbatch --export=ALL,FOLD=$fold,GEOM=1 train_dosegan.sbatch; done
for fold in 0 1 2 3 4; do sbatch --export=ALL,FOLD=$fold,GEOM=1 train_unet3d.sbatch; done

# ── Evaluation ─────────────────────────────────────────────────────────────
# Via SLURM (recommended):
sbatch --export=ALL,MODEL=dosegan,FOLD=0,RUN_NAME=dosegan_ngf32_sigmoid_snellius,GEOM=0 eval.sbatch
sbatch --export=ALL,MODEL=unet3d,FOLD=0,RUN_NAME=unet3d_ch32_sigmoid_snellius,GEOM=0 eval.sbatch
# Geom variants (GEOM=1 is the default):
sbatch --export=ALL,MODEL=dosegan,FOLD=0,RUN_NAME=dosegan_ngf32_sigmoid_geom_snellius eval.sbatch

# Directly (no GPU job required for CPU-only metrics):
python3 -m training.evaluate --model dosegan --fold 0 --run-name dosegan_ngf32_sigmoid_snellius --no-geom

# ── Monitoring ─────────────────────────────────────────────────────────────
squeue -u $USER
tail -f outputs/logs/<jobname>_<jobid>.out
```

## Data flow

```
raw_data/lund-probe/.../basePart/<patient_id>/
    sCT/image_reg2MRI.nii.gz
    MR_StorT2/dose_interpolated.nii.gz + mask_*.nii.gz
        │
        │  preprocessing/preprocessing.py  (sbatch preprocess.sbatch)
        ▼
data/pickles/<patient_id>.pkl      # 'input' (9,128,256,320) + 'dose' (128,256,320)
        │
        │  preprocessing/add_geom_channels.py  (sbatch add_geom_channels.sbatch)
        ▼
data/pickles/<patient_id>.pkl      # + 'geom_channels' (5,128,256,320)
        │
        │  training/dataset.py  (use_geom_channels=False → 9 ch | True → 14 ch)
        ▼
LUNDPROBEDataset → DataLoader
        │
        │  training/train_dosegan.py | train_unet3d.py
        ▼
outputs/checkpoints_{dosegan,unet3d}/<run_name>_fold<F>_best.pt
        │
        │  training/evaluate.py
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
configs/            hyperparameters (one config file per component)
models/             DoseGAN (3D U-Net generator + PatchGAN discriminator) and 3D U-Net
preprocessing/      NIfTI → pickle pipeline + geometric channel augmentation
training/           train / evaluate / dataset scripts + metrics
analysis/           post-eval investigations (per-patient, per-acquisition-group)
data/               split.csv committed; pickles and outputs gitignored
*.sbatch            SLURM submission scripts
```

## Models

**DoseGAN** (`models/dosegan.py`)
3D U-Net generator with attention gates at every skip connection (`UnetSkipConnectionBlock3d` + `AttGate`). PatchGAN discriminator (`NLayerDiscriminator`). LSGAN loss (`USE_LSGAN=True`). Sigmoid output activation (empirically selected over Tanh via 5-fold ablation). ~29.6M parameters. Adapted from GhTara/Dose_Prediction.

**3D U-Net** (`models/unet3d.py`)
MONAI 6-level residual U-Net, `CHANNELS=(32,64,128,256,256,256)`, 5 downsampling strides. Sigmoid output activation. ~21.8M parameters.

Both models accept `(9, 128, 256, 320)` input for the baseline condition and `(14, 128, 256, 320)` for the with-geom condition. Output: normalised dose `(1, 128, 256, 320)` in [0,1] — multiply by 50 to recover Gy.

## Split

`data/split.csv` is committed and must not be regenerated — it is the permanent record of train/val/test assignments. 15% held-out test set + 5-fold CV on the remainder, stratified by acquisition group (`oldAcq` / `newAcq`).


## Preprocessing

Per patient: resample to 1.5 mm isotropic → z-score sCT inside body contour (HU window −990–2000) → normalise dose by 50 Gy → asymmetric PTV-centric crop (128×256×320: 40 slices below PTV centroid, 88 above). Optional structures (Genitalia, PenileBulb) zero-filled if absent.

Geometric channels are computed after cropping, in the final output coordinate space, by `preprocessing/geometric_channels.py`.

## Evaluation metrics

`training/metrics.py` provides the full clinical evaluation suite:

| Category | Metrics |
|---|---|
| Voxel-level | MAE and RMSE over body, PTV, Rectum, Bladder |
| Boundary MAE | MAE in ±20 mm band around PTV / Rectum / Bladder surface |
| DVH endpoints | Dmean, Dmax, D95, D98, D0.1cc, V20, V40 per structure (pred vs reference, diff) |
| Gamma pass rate | 3%/3 mm and 2%/2 mm (3D, body-masked, 10% low-dose cut-off) |
| Isodose conformality | Dice + HD95 at 100%, 95%, 80%, 50% isodose levels |

## Experiment tracking

W&B project: `doseprediction-lundprobe`. `group=cfg.RUN_NAME` collapses all 5 folds of one experiment into a single dashboard row. Evaluation results are logged as per-fold CSVs (`outputs/evaluation/<run_name>_fold<F>_val.csv`) and W&B Tables.
