"""
visualise_patient.py
====================
Load a preprocessed patient pickle and save a diagnostic figure to disk.

Run from the preprocessing/ directory:
    python visualise_patient.py

Output
------
A single PNG file saved to outputs/visualisations/<patient_id>.png
Open it in VS Code by clicking on it in the Explorer panel — VS Code
renders PNG files inline.

What the figure shows
---------------------
Row 1 — Axial (top-down) slice through the PTV centroid:
    sCT intensity | ground-truth dose | PTV + key OAR masks overlaid on sCT

Row 2 — Coronal (front-facing) slice through the PTV centroid:
    sCT intensity | ground-truth dose | masks overlaid on sCT

Seeing both views together lets you confirm:
  - The PTV is centred in the crop (expected from the crop anchor logic)
  - The bladder dome is captured superiorly (goal of the asymmetric SI crop)
  - The dose distribution wraps around the PTV with dose falloff into the OARs
  - Mask boundaries are sharp (confirming nearest-neighbour resampling worked)
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Imports ───────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR

# ── Load the pickle ───────────────────────────────────────────────────────────
# Find the first .pkl file in the outputs directory
pkl_files = list(OUTPUT_DIR.glob("*.pkl"))
if not pkl_files:
    raise FileNotFoundError(
        f"No .pkl files found in {OUTPUT_DIR}. Run test_single_patient.py first."
    )

pkl_path = sorted(pkl_files)[0]
print(f"Loading: {pkl_path.name}")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

patient_id   = data["patient_id"]
input_tensor = data["input"]    # (9, D, H, W)
dose         = data["dose"]     # (D, H, W)

# Channel layout in the 9-channel tensor:
#   0: PTVT_427   1: Rectum     2: Bladder
#   3: FemoralHead_L  4: FemoralHead_R  5: Genitalia
#   6: PenileBulb     7: BODY           8: sCT intensity
ptv      = input_tensor[0]   # PTV mask
rectum   = input_tensor[1]   # Rectum mask
bladder  = input_tensor[2]   # Bladder mask
body     = input_tensor[7]   # Body contour mask
sct      = input_tensor[8]   # z-scored sCT intensity

# ── Find the slice through the PTV centroid ───────────────────────────────────
# The crop was anchored on the PTV centroid, so this is roughly the centre
# of the D dimension — but we compute it explicitly to be precise.
ptv_coords  = np.argwhere(ptv > 0.5)
ptv_z_mid   = int(ptv_coords[:, 0].mean())   # axial slice index
ptv_x_mid   = int(ptv_coords[:, 2].mean())   # coronal slice index (LR axis)

print(f"PTV centroid: z={ptv_z_mid}, x={ptv_x_mid}")
print(f"Tensor shape: input={input_tensor.shape}, dose={dose.shape}")


# ── Helper: build a colour overlay of masks on a greyscale background ─────────
def overlay_masks_on_sct(sct_slice, masks_colours):
    """
    Blend binary masks as translucent colour overlays on top of a greyscale sCT slice.

    Parameters
    ----------
    sct_slice      : (H, W) float array — z-scored sCT, any range
    masks_colours  : list of (mask_2d, colour_tuple) pairs

    Returns
    -------
    (H, W, 3) RGB array suitable for imshow()
    """
    # Normalise sCT to [0, 1] for display purposes only
    lo, hi = np.percentile(sct_slice, [1, 99])
    sct_display = np.clip((sct_slice - lo) / (hi - lo + 1e-8), 0, 1)

    # Convert greyscale to RGB
    rgb = np.stack([sct_display] * 3, axis=-1)

    # Blend each mask as a translucent colour layer
    for mask, colour in masks_colours:
        alpha = 0.35
        for c_idx, c_val in enumerate(colour):
            rgb[:, :, c_idx] = np.where(
                mask > 0.5,
                (1 - alpha) * rgb[:, :, c_idx] + alpha * c_val,
                rgb[:, :, c_idx],
            )
    return np.clip(rgb, 0, 1)


# Colour scheme for masks (RGB tuples, range [0, 1])
MASK_COLOURS = {
    "PTV":     (1.0, 0.0, 0.0),   # red
    "Rectum":  (0.0, 0.8, 0.0),   # green
    "Bladder": (0.0, 0.5, 1.0),   # blue
}


# ── Build the figure ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    f"Preprocessed patient: {patient_id}\n"
    f"Shape: {input_tensor.shape}  |  Dose max: {dose.max():.3f} (×50 Gy = {dose.max()*50:.1f} Gy)",
    fontsize=12,
)

# ---------- Row 0: Axial view (looking down from above — the z slice) ---------
ax = axes[0]

# sCT
ax[0].imshow(sct[ptv_z_mid], cmap="gray", interpolation="nearest")
ax[0].set_title(f"sCT — axial z={ptv_z_mid}")
ax[0].axis("off")

# Dose
im_dose = ax[1].imshow(dose[ptv_z_mid], cmap="hot", interpolation="nearest", vmin=0, vmax=1)
ax[1].set_title("Dose (normalised) — axial")
ax[1].axis("off")
plt.colorbar(im_dose, ax=ax[1], fraction=0.046, pad=0.04, label="Dose [0-1]")

# Masks overlaid on sCT
overlay_axial = overlay_masks_on_sct(
    sct[ptv_z_mid],
    [
        (ptv[ptv_z_mid],     MASK_COLOURS["PTV"]),
        (rectum[ptv_z_mid],  MASK_COLOURS["Rectum"]),
        (bladder[ptv_z_mid], MASK_COLOURS["Bladder"]),
    ]
)
ax[2].imshow(overlay_axial, interpolation="nearest")
ax[2].set_title("Masks on sCT — axial")
ax[2].axis("off")

# ---------- Row 1: Coronal view (front-facing — the x slice) ------------------
ax = axes[1]

# sCT
ax[0].imshow(sct[:, :, ptv_x_mid], cmap="gray", interpolation="nearest", origin="lower")
ax[0].set_title(f"sCT — coronal x={ptv_x_mid}")
ax[0].axis("off")

# Dose
im_dose2 = ax[1].imshow(
    dose[:, :, ptv_x_mid], cmap="hot", interpolation="nearest",
    origin="lower", vmin=0, vmax=1
)
ax[1].set_title("Dose (normalised) — coronal")
ax[1].axis("off")
plt.colorbar(im_dose2, ax=ax[1], fraction=0.046, pad=0.04, label="Dose [0-1]")

# Masks overlaid on sCT
overlay_coronal = overlay_masks_on_sct(
    sct[:, :, ptv_x_mid],
    [
        (ptv[:, :, ptv_x_mid],     MASK_COLOURS["PTV"]),
        (rectum[:, :, ptv_x_mid],  MASK_COLOURS["Rectum"]),
        (bladder[:, :, ptv_x_mid], MASK_COLOURS["Bladder"]),
    ]
)
ax[2].imshow(overlay_coronal, interpolation="nearest", origin="lower")
ax[2].set_title("Masks on sCT — coronal")
ax[2].axis("off")

# ---------- Shared legend -------------------------------------------------------
legend_patches = [
    mpatches.Patch(color=MASK_COLOURS["PTV"],     label="PTV (target)"),
    mpatches.Patch(color=MASK_COLOURS["Rectum"],  label="Rectum (OAR)"),
    mpatches.Patch(color=MASK_COLOURS["Bladder"], label="Bladder (OAR)"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=11, frameon=True)

plt.tight_layout(rect=[0, 0.04, 1, 1])

# ── Save to disk ───────────────────────────────────────────────────────────────
vis_dir = OUTPUT_DIR / "visualisations"
vis_dir.mkdir(parents=True, exist_ok=True)
save_path = vis_dir / f"{patient_id}.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\nFigure saved to: {save_path}")
print("Open it in VS Code by clicking on it in the Explorer panel.")