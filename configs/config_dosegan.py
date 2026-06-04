# configs/config_dosegan.py
# All hyperparameters specific to DoseGAN training.
# Shared preprocessing settings live in configs/config_preprocessing_shared.py.
# To run a new experiment, change values here — never edit train_dosegan.py directly.

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
# data/ holds preprocessed inputs (pickles, split, summary).
# outputs/ holds per-run artefacts (checkpoints, logs, visualisations).
SPLIT_CSV  = Path("data/split.csv")
PICKLE_DIR = Path("data/pickles")
CKPT_DIR   = Path("outputs/checkpoints_dosegan")

# ── Experiment identity ────────────────────────────────────────────────────
# Change RUN_NAME when you change anything meaningful — this is what shows
# up in the W&B dashboard so you can tell runs apart at a glance.
PROJECT_NAME = "doseprediction-lundprobe"
RUN_NAME     = "dosegan_ngf32_sigmoid_snellius"   # fold number appended at runtime

# ── Cross-validation ───────────────────────────────────────────────────────
FOLD = 0  # which fold is held out as validation this run (0–4)

# ── Training ───────────────────────────────────────────────────────────────
# 200 epochs gives the model more time to converge on the full 5-fold dataset;
# early runs plateaued before 100 epochs but later folds may need the headroom.
EPOCHS      = 200
BATCH_SIZE  = 1    # 3D volumes — batch size 1 is standard
NUM_WORKERS = 4    # reduce to 0 if DataLoader throws errors on Windows

# ── Model architecture ─────────────────────────────────────────────────────
INPUT_NC          = 9         # 9 without geom (8 masks + sCT); 14 with geom (+ 5 geom channels)
USE_GEOM_CHANNELS = False
OUTPUT_NC         = 1         # predicted dose volume
OUTPUT_ACTIVATION = "sigmoid" # "sigmoid" (empirically selected) or "tanh"
NGF               = 32        # base number of generator filters — doubles at each U-Net level
NDF               = 32        # base number of discriminator filters
N_LAYERS          = 3         # number of discriminator layers

# ── Optimizers ─────────────────────────────────────────────────────────────
LR_G   = 2e-4         # generator learning rate
# Halving LR_D slows the discriminator relative to the generator, reducing
# mode collapse risk and giving the generator more useful gradient signal.
LR_D   = 1e-4         # discriminator learning rate
BETA1  = 0.5          # Adam beta1 — 0.5 is standard for GAN training
BETA2  = 0.999        # Adam beta2

# ── Loss weights ───────────────────────────────────────────────────────────
LAMBDA_VOXEL        = 100  # weight of L1 voxel loss relative to adversarial loss
LAMBDA_DVH          = 5.0  # weight for structure-Dmean DVH regularisation loss
LAMBDA_GRAD         = 0.0  # gradient-magnitude loss — confirmed no benefit (fold-0 negative result)
# Doubled to match the longer training budget; 15 was too aggressive at 200 epochs
# and would cut runs that are still slowly improving in the middle third.
EARLY_STOP_PATIENCE = 30   # stop if val_dvh_score does not improve for this many epochs
USE_LSGAN    = True   # True = LSGAN (MSE), False = vanilla GAN (BCE)