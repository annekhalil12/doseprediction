"""
config.py
=========
Central configuration for the LUND-PROBE dose prediction preprocessing pipeline.

Adapted from collaborator's configV5.py. DoseGNN-specific sections (encoder,
graph, GNN) are omitted here — this config serves the shared preprocessing
pipeline used by all three models (3D U-Net, DoseGAN, DoseGNN).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Literal


# ---------------------------------------------------------------------------
# Paths — update DATA_ROOT to point at your local copy of LUND-PROBE
# ---------------------------------------------------------------------------

# Example (Windows, adjust to your actual path):
#   DATA_ROOT = Path(r"\\vumc.nl\...\lund-probe\basePart")
# Example (laptop local copy):
#   DATA_ROOT = Path(r"C:\Users\anne\data\lund-probe\basePart")

DATA_ROOT  = Path(r"\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\Research_Kidney_AI_OB_ABR_MP_NGA_Joris\NIKA\LUnd_dataset\lund-probe\lund-probe\basePart")
OUTPUT_DIR = Path(r"C:\Users\P102831\thesis-doseprediction") / "outputs"

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
# Channels 0–7:  binary structure masks (available now)
# Channels 8–14: geometric features    (pending V5geometric_channels.py)
# Channel 15:    normalised sCT        (available now)

CHANNEL_MAP = {
    0:  "PTVT_427",
    1:  "Rectum",
    2:  "Bladder",
    3:  "FemoralHead_L",
    4:  "FemoralHead_R",
    5:  "Genitalia",
    6:  "PenileBulb",
    7:  "BODY",
    8:  "dist_to_ptv_surface",
    9:  "dist_to_body_surface",
    10: "azimuthal_angle",
    11: "polar_angle",
    12: "rad_depth_mean",
    13: "rad_depth_max",
    14: "rad_depth_min",
    15: "sct_intensity",
}

# Channels we can compute right now (no geometric file needed)
AVAILABLE_CHANNELS = {k: v for k, v in CHANNEL_MAP.items() if k <= 7 or k == 15}