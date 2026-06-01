"""
config_preprocessing_shared.py
================================
Central configuration for the LUND-PROBE dose prediction preprocessing pipeline.

Serves the shared preprocessing pipeline used by both models (3D U-Net and DoseGAN)
under two input conditions: 9-channel baseline (8 masks + sCT) and 14-channel
with-geometric-channels (+ 5 spatial encoding channels appended at indices 9–13).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Literal


# ---------------------------------------------------------------------------
# Paths — update DATA_ROOT to point at your local copy of LUND-PROBE
# ---------------------------------------------------------------------------

DATA_ROOT  = Path("/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction/raw_data/lund-probe/lund-probe/basePart")

# Preprocessed data products live under data/, separate from per-run outputs/.
#   data/pickles/                 — one .pkl per patient
#   data/split.csv                — committed train/val/test assignment
#   data/preprocessing_summary.csv — committed preprocessing status per patient
DATA_DIR    = Path("/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction/data")
OUTPUT_DIR  = DATA_DIR / "pickles"                    # backwards-compatible name
SUMMARY_CSV = DATA_DIR / "preprocessing_summary.csv"
SPLIT_CSV   = DATA_DIR / "split.csv"

# ---------------------------------------------------------------------------
# Preprocessing parameters
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingConfig:
    # Resample all volumes to this isotropic voxel spacing (mm).
    # Approved by Omar and Miguel. Reduces memory while preserving clinical detail.
    target_spacing: float = 1.5

    # Fixed output crop size — every patient outputs exactly this shape.
    # SI (depth) is asymmetric around the PTV centroid; see offsets below.
    crop_size_D: int = 128   # superior–inferior slices
    crop_size_H: int = 256   # anterior–posterior voxels
    crop_size_W: int = 320   # left–right voxels

    # Asymmetric SI offsets from PTV centroid (supervisor approved).
    # More slices above than below: bladder extends superior to prostate.
    si_inferior_slices: int = 40   # slices below PTV centroid
    si_superior_slices: int = 88   # slices above PTV centroid

    # sCT HU window before z-score normalisation (Omar and Miguel approved).
    sct_hu_window: Tuple[float, float] = (-990, 2000)

    # Dose normalisation: divide by this fixed scale → output in [0, 1].
    dose_max_gy: float = 50.0

    # These structures may be absent for some patients → replaced with zeros.
    optional_structures: List[str] = field(
        default_factory=lambda: ["Genitalia", "PenileBulb"]
    )


# ---------------------------------------------------------------------------
# Channel map — ORDER MATTERS for tensor assembly
# ---------------------------------------------------------------------------
# Baseline condition (9 channels, stored in every pickle's 'input' key):
#   0–7:  binary structure masks
#   8:    normalised sCT intensity
#
# With-geom condition (14 channels, channels 9–13 stored in pickle's 'geom_channels' key
# and appended by LUNDPROBEDataset when use_geom_channels=True):
#   9–13: 5 spatial encoding channels (see preprocessing/geometric_channels.py)

CHANNEL_MAP = {
    0:  "PTVT_427",
    1:  "Rectum",
    2:  "Bladder",
    3:  "FemoralHead_L",
    4:  "FemoralHead_R",
    5:  "Genitalia",
    6:  "PenileBulb",
    7:  "BODY",
    8:  "sct_intensity",
    # Geometric channels — appended at indices 9–13 (preprocessing/add_geom_channels.py)
    9:  "dist_to_ptv_surface",
    10: "dist_to_body_surface",
    11: "dir_z_shifted",
    12: "dir_y_shifted",
    13: "dir_x_shifted",
}