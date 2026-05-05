# configs/config_dosegan.py
# All hyperparameters specific to DoseGAN training.
# Shared preprocessing settings live in configs/config_preprocessing_shared.py.
# To run a new experiment, change values here — never edit train_dosegan.py directly.

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
SPLIT_CSV  = Path("outputs/split.csv")
PICKLE_DIR = Path("outputs/pickles")
CKPT_DIR   = Path("outputs/checkpoints_dosegan")

# ── Experiment identity ────────────────────────────────────────────────────
# Change RUN_NAME when you change anything meaningful — this is what shows
# up in the W&B dashboard so you can tell runs apart at a glance.
PROJECT_NAME = "doseprediction-lundprobe"
RUN_NAME     = "dosegan_fold0_baseline"

# ── Cross-validation ───────────────────────────────────────────────────────
FOLD = 0  # which fold is held out as validation this run (0–4)

# ── Training ───────────────────────────────────────────────────────────────
EPOCHS      = 100
BATCH_SIZE  = 1    # 3D volumes — batch size 1 is standard
NUM_WORKERS = 4    # reduce to 0 if DataLoader throws errors on Windows

# ── Model architecture ─────────────────────────────────────────────────────
INPUT_NC  = 9   # 8 structure masks + sCT intensity channel
OUTPUT_NC = 1   # predicted dose volume
NGF       = 16  # base number of generator filters — doubles at each U-Net level
NDF       = 16  # base number of discriminator filters
N_LAYERS  = 3   # number of discriminator layers

# ── Optimizers ─────────────────────────────────────────────────────────────
LR_G   = 2e-4         # generator learning rate
LR_D   = 2e-4         # discriminator learning rate
BETA1  = 0.5          # Adam beta1 — 0.5 is standard for GAN training
BETA2  = 0.999        # Adam beta2

# ── Loss weights ───────────────────────────────────────────────────────────
LAMBDA_VOXEL = 100    # weight of L1 voxel loss relative to adversarial loss
USE_LSGAN    = True   # True = LSGAN (MSE), False = vanilla GAN (BCE)