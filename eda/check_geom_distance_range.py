import pickle
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt

PICKLE_DIR = Path("data/pickles")

ptv_maxes = []
body_maxes = []
ptv_clip_fracs = []
body_clip_fracs = []

pkl_paths = sorted(PICKLE_DIR.glob("*.pkl"))
print(f"Checking {len(pkl_paths)} patients...")

for i, pkl_path in enumerate(pkl_paths):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    ptv_mask = data["ptv_mask"] > 0.5
    body_mask = data["input"][7] > 0.5  # BODY channel, confirmed earlier

    # Recompute RAW (un-normalised) distance transforms, exactly as
    # geometric_channels.py does before the /200 division.
    dist_ptv_raw = distance_transform_edt(~ptv_mask)
    dist_ptv_raw = np.where(body_mask, dist_ptv_raw, 0.0)

    dist_body_raw = distance_transform_edt(body_mask)
    dist_body_raw = np.where(body_mask, dist_body_raw, 0.0)

    ptv_max = dist_ptv_raw.max()
    body_max = dist_body_raw.max()
    ptv_maxes.append(ptv_max)
    body_maxes.append(body_max)

    # Fraction of body voxels that would be clipped (raw distance > 200)
    ptv_clip_fracs.append((dist_ptv_raw[body_mask] > 200).mean())
    body_clip_fracs.append((dist_body_raw[body_mask] > 200).mean())

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(pkl_paths)} done")

ptv_maxes = np.array(ptv_maxes)
body_maxes = np.array(body_maxes)

print("\n--- dist_to_ptv_surface (raw, unnormalised, voxels) ---")
print(f"Mean of per-patient max: {ptv_maxes.mean():.1f}")
print(f"Overall max across all patients: {ptv_maxes.max():.1f}")
print(f"Overall min across all patients: {ptv_maxes.min():.1f}")
print(f"Patients with max > 200 (would clip): {(ptv_maxes > 200).sum()} / {len(ptv_maxes)}")
print(f"Mean fraction of body voxels clipped per patient: {np.mean(ptv_clip_fracs)*100:.3f}%")

print("\n--- dist_to_body_surface (raw, unnormalised, voxels) ---")
print(f"Mean of per-patient max: {body_maxes.mean():.1f}")
print(f"Overall max across all patients: {body_maxes.max():.1f}")
print(f"Overall min across all patients: {body_maxes.min():.1f}")
print(f"Patients with max > 200 (would clip): {(body_maxes > 200).sum()} / {len(body_maxes)}")
print(f"Mean fraction of body voxels clipped per patient: {np.mean(body_clip_fracs)*100:.3f}%")