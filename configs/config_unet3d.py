# configs/config_unet3d.py
# Hyperparameters for the 3D U-Net baseline.
# Shared preprocessing settings live in configs/config_preprocessing_shared.py.
# To run a new experiment, change values here — never edit train_unet3d.py directly.

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
SPLIT_CSV  = Path("data/split.csv")
PICKLE_DIR = Path("data/pickles")
CKPT_DIR   = Path("outputs/checkpoints_unet3d")

# ── Experiment identity ────────────────────────────────────────────────────
PROJECT_NAME = "doseprediction-lundprobe"
RUN_NAME     = "unet3d_ch32_sigmoid_snellius"   # fold number appended at runtime

# ── Cross-validation ───────────────────────────────────────────────────────
FOLD = 0

# ── Training ───────────────────────────────────────────────────────────────
EPOCHS      = 200
BATCH_SIZE  = 1
NUM_WORKERS = 4

# ── Model architecture ─────────────────────────────────────────────────────
INPUT_NC      = 9    # 9 without geom (8 masks + sCT); 14 with geom (+ 5 geom channels)
USE_GEOM_CHANNELS = False
USE_FLIPS         = False  # disabled for all conditions to keep augmentation policy identical
OUTPUT_NC     = 1    # predicted dose
CHANNELS      = (32, 64, 128, 256, 256, 256)  # feature maps at each U-Net level
STRIDES       = (2, 2, 2, 2, 2)              # downsampling factor per transition
NUM_RES_UNITS = 2                         # residual units per level (0 = plain conv)
OUTPUT_ACTIVATION = "sigmoid"             # "sigmoid" | "tanh"; overridable via --activation

# ── Optimizer ──────────────────────────────────────────────────────────────
LR    = 1e-4          # slightly lower than DoseGAN — no adversarial instability to compensate for
BETA1 = 0.9           # standard Adam beta1 for supervised training
BETA2 = 0.999

# ── Loss ───────────────────────────────────────────────────────────────────
LAMBDA_DVH          = 0.1  # weight for structure-Dmean DVH regularisation loss
LAMBDA_GRAD         = 0.0  # gradient-magnitude loss — confirmed no benefit (fold-0 negative result)
EARLY_STOP_PATIENCE = 30   # stop if val_dvh_score does not improve for this many epochs
