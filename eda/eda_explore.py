"""
EDA Step 1: Load and inspect a single patient from LUND-PROBE.
Purpose: Understand data shapes, spacing, intensity ranges, and structure masks
before writing any preprocessing or training code.
"""

import nibabel as nib
import numpy as np
from pathlib import Path

# --- Configuration ---
BASE_PATH = Path(r"\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\Research_Kidney_AI_OB_ABR_MP_NGA_Joris\NIKA\LUnd_dataset\lund-probe\lund-probe\basePart")

# Pick the first patient
patient_dirs = sorted([d for d in BASE_PATH.iterdir() if d.is_dir()])
patient = patient_dirs[0]
print(f"Patient: {patient.name}")
print(f"Total patients found: {len(patient_dirs)}")

# --- Load MR image ---
mr_path = patient / "MR_StorT2" / "image.nii.gz"
mr_img = nib.load(str(mr_path))

print(f"\n--- MR Image ---")
print(f"Shape: {mr_img.shape}")           # (X, Y, Z) voxel dimensions
print(f"Voxel spacing: {mr_img.header.get_zooms()}")  # mm per voxel
print(f"Data type: {mr_img.get_data_dtype()}")

# Load the actual voxel values into a numpy array
mr_data = mr_img.get_fdata()
print(f"Intensity range: [{mr_data.min():.1f}, {mr_data.max():.1f}]")
print(f"Mean intensity: {mr_data.mean():.1f}")

# --- Load dose ---
dose_path = patient / "MR_StorT2" / "dose_interpolated.nii.gz"
dose_img = nib.load(str(dose_path))
dose_data = dose_img.get_fdata()

print(f"\n--- Dose (interpolated) ---")
print(f"Shape: {dose_img.shape}")
print(f"Voxel spacing: {dose_img.header.get_zooms()}")
print(f"Dose range: [{dose_data.min():.2f}, {dose_data.max():.2f}] Gy")
print(f"Mean dose (nonzero): {dose_data[dose_data > 0].mean():.2f} Gy")

# --- Load structure masks ---
print(f"\n--- Structure Masks ---")
mr_folder = patient / "MR_StorT2"
mask_files = sorted([f for f in mr_folder.iterdir() if f.name.startswith("mask_")])
for mask_file in mask_files:
    mask = nib.load(str(mask_file)).get_fdata()
    volume_cc = mask.sum() * np.prod(mr_img.header.get_zooms()) / 1000  # convert mm³ to cc
    print(f"  {mask_file.name:<35s} shape={mask.shape}  volume={volume_cc:.1f} cc")