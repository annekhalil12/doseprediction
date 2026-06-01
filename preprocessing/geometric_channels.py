"""
geometric_channels.py
=====================
Compute 5 geometric input channels for LUND-PROBE dose prediction.
All channels are computed in the cropped coordinate space (128×256×320).

Channel order (matches append order in dataset.py):
  0  dist_to_ptv_surface  — Euclidean dist to nearest PTV boundary voxel [0,1]
  1  dist_to_body_surface — Euclidean dist to nearest BODY boundary voxel [0,1]
  2  dir_z_shifted        — sup/inf unit direction from PTV centroid [0,1]
  3  dir_y_shifted        — ant/post unit direction from PTV centroid [0,1]
  4  dir_x_shifted        — left/right unit direction from PTV centroid [0,1]

Directional channels are shifted from [-1,1] to [0,1] via (d+1)/2 to prevent
ReLU from zeroing negative directions. Voxels outside BODY are set to 0.

Adapted from Nika Kovacic's V6geometric_channels.py (2025). The normalization
constants (200 voxels) and directional encoding follow her implementation.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, center_of_mass

MAX_DIST_TO_PTV_SURFACE  = 200.0   # voxels — normalisation divisor
MAX_DIST_TO_BODY_SURFACE = 200.0   # voxels — normalisation divisor


def compute_geom_channels(
    ptv_mask: np.ndarray,
    body_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute all 5 geometric channels in one call.

    Parameters
    ----------
    ptv_mask  : (D, H, W) binary PTV mask (float32 or bool)
    body_mask : (D, H, W) binary BODY mask (float32 or bool)

    Returns
    -------
    (5, D, H, W) float32 array in the channel order above.

    Raises
    ------
    ValueError if the PTV mask is empty (centroid cannot be computed).
    """
    ptv_bin  = ptv_mask  > 0.5
    body_bin = body_mask > 0.5

    # ── dist_to_ptv_surface ──────────────────────────────────────────────
    # EDT on ~ptv: gives distance to nearest PTV boundary for voxels outside
    # PTV; voxels inside PTV get 0 (they are at the region of interest).
    dist_ptv = distance_transform_edt(~ptv_bin).astype(np.float32)
    dist_ptv = np.where(body_bin, dist_ptv, 0.0)
    dist_ptv = (dist_ptv / MAX_DIST_TO_PTV_SURFACE).clip(0.0, 1.0)

    # ── dist_to_body_surface ─────────────────────────────────────────────
    # EDT on body_bin: gives distance to nearest background (skin surface).
    dist_body = distance_transform_edt(body_bin).astype(np.float32)
    dist_body = np.where(body_bin, dist_body, 0.0)
    dist_body = (dist_body / MAX_DIST_TO_BODY_SURFACE).clip(0.0, 1.0)

    # ── directional unit vectors from PTV centroid ───────────────────────
    if not np.any(ptv_bin):
        raise ValueError("PTV mask is empty — cannot compute PTV centroid.")

    centroid = np.array(center_of_mass(ptv_bin), dtype=np.float64)  # [z, y, x]
    cz, cy, cx = centroid

    D, H, W = ptv_mask.shape
    zz, yy, xx = np.meshgrid(
        np.arange(D, dtype=np.float64),
        np.arange(H, dtype=np.float64),
        np.arange(W, dtype=np.float64),
        indexing="ij",
    )

    Vz = zz - cz
    Vy = yy - cy
    Vx = xx - cx
    v  = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    v_safe = np.maximum(v, 1e-6)

    # Shift (-1,1) → (0,1) so ReLU cannot zero-out any direction.
    dir_z = np.where(body_bin, (Vz / v_safe + 1.0) / 2.0, 0.0).astype(np.float32)
    dir_y = np.where(body_bin, (Vy / v_safe + 1.0) / 2.0, 0.0).astype(np.float32)
    dir_x = np.where(body_bin, (Vx / v_safe + 1.0) / 2.0, 0.0).astype(np.float32)

    return np.stack(
        [dist_ptv, dist_body, dir_z, dir_y, dir_x], axis=0
    ).astype(np.float32)   # (5, D, H, W)
