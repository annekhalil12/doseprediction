"""
EDA Step 2: Visualize a single patient's sCT, dose, and structures.
Purpose: See the spatial relationship between anatomy, target, OARs, and dose.
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
BASE_PATH = Path(r"\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\Research_Kidney_AI_OB_ABR_MP_NGA_Joris\NIKA\LUnd_dataset\lund-probe\lund-probe\basePart")

patient_dirs = sorted([d for d in BASE_PATH.iterdir() if d.is_dir()])
patient = patient_dirs[0]
sct_folder = patient / "sCT"

# --- Load data ---
sct_data = nib.load(str(sct_folder / "image.nii.gz")).get_fdata()
dose_data = nib.load(str(sct_folder / "dose_interpolated.nii.gz")).get_fdata()
ctv_mask = nib.load(str(sct_folder / "mask_CTVT_427.nii.gz")).get_fdata()
ptv_mask = nib.load(str(sct_folder / "mask_PTVT_427.nii.gz")).get_fdata()
bladder_mask = nib.load(str(sct_folder / "mask_Bladder.nii.gz")).get_fdata()
rectum_mask = nib.load(str(sct_folder / "mask_Rectum.nii.gz")).get_fdata()

# --- Find the slice with the most target coverage ---
# This gives us the most informative axial slice to look at
ctv_per_slice = ctv_mask.sum(axis=(0, 1))  # sum over X,Y for each Z slice
best_slice = int(np.argmax(ctv_per_slice))
print(f"Showing axial slice {best_slice} (highest CTV coverage)")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: sCT anatomy
axes[0].imshow(sct_data[:, :, best_slice].T, cmap="gray", origin="lower")
axes[0].set_title("sCT Anatomy")
axes[0].axis("off")

# Panel 2: sCT with structure contours overlaid
axes[1].imshow(sct_data[:, :, best_slice].T, cmap="gray", origin="lower")
axes[1].contour(ctv_mask[:, :, best_slice].T, levels=[0.5], colors="red", linewidths=1.5)
axes[1].contour(ptv_mask[:, :, best_slice].T, levels=[0.5], colors="orange", linewidths=1.5)
axes[1].contour(bladder_mask[:, :, best_slice].T, levels=[0.5], colors="blue", linewidths=1.5)
axes[1].contour(rectum_mask[:, :, best_slice].T, levels=[0.5], colors="green", linewidths=1.5)
axes[1].set_title("Structures: CTV(red) PTV(orange) Bladder(blue) Rectum(green)")
axes[1].axis("off")

# Panel 3: Dose distribution overlaid on sCT
axes[2].imshow(sct_data[:, :, best_slice].T, cmap="gray", origin="lower")
dose_slice = dose_data[:, :, best_slice].T
dose_masked = np.ma.masked_where(dose_slice < 0.5, dose_slice)  # hide near-zero dose
axes[2].imshow(dose_masked, cmap="jet", alpha=0.5, origin="lower")
axes[2].contour(ctv_mask[:, :, best_slice].T, levels=[0.5], colors="red", linewidths=1)
axes[2].set_title("Dose overlay with CTV contour")
axes[2].axis("off")

plt.suptitle(f"Patient: {patient.name} — Axial slice {best_slice}", fontsize=14)
plt.tight_layout()
plt.savefig("eda_single_patient.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to eda_single_patient.png")