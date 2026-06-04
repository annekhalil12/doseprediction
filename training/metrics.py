# training/metrics.py
# Shared evaluation metric functions for DoseGAN and U-Net.
# All dose inputs are expected in Gy (not normalised units).
#
# Functions
# ---------
#   compute_mae / compute_rmse       : voxel-level, optionally masked
#   dvh_endpoints                    : full DVH dict per structure
#   compute_boundary_mae             : MAE in 20 mm band around PTV/OAR surface (H2)
#   compute_gamma_passrate           : 3D gamma, 3%/3mm and 2%/2mm
#   compute_isodose_metrics          : Dice + HD95 at 100/95/80/50% isodose levels (body-masked when body_mask given)
#   compute_outside_body_leakage     : mean predicted dose and leakage volume fraction outside body

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
import pymedphys

log = logging.getLogger(__name__)

VOXEL_SPACING_MM = 1.5
BOUNDARY_BAND_MM = 20.0    # each side of structure surface


# ---------------------------------------------------------------------------
# Voxel-level
# ---------------------------------------------------------------------------

def compute_mae(pred: np.ndarray, target: np.ndarray,
                mask: Optional[np.ndarray] = None) -> float:
    if mask is not None:
        m = mask > 0.5
        pred, target = pred[m], target[m]
    return float(np.mean(np.abs(pred - target)))


def compute_rmse(pred: np.ndarray, target: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> float:
    if mask is not None:
        m = mask > 0.5
        pred, target = pred[m], target[m]
    return float(np.sqrt(np.mean((pred - target) ** 2)))


# ---------------------------------------------------------------------------
# DVH endpoint helpers
# ---------------------------------------------------------------------------

def _vox(dose: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return dose[mask > 0.5]


def compute_Dmean(dose: np.ndarray, mask: np.ndarray) -> float:
    v = _vox(dose, mask)
    return float(v.mean()) if len(v) > 0 else 0.0


def compute_Dmax(dose: np.ndarray, mask: np.ndarray) -> float:
    v = _vox(dose, mask)
    return float(v.max()) if len(v) > 0 else 0.0


def compute_D95(dose: np.ndarray, mask: np.ndarray) -> float:
    """Dose received by ≥95% of the structure (5th percentile of voxel doses)."""
    v = _vox(dose, mask)
    return float(np.percentile(v, 5)) if len(v) > 0 else 0.0


def compute_D98(dose: np.ndarray, mask: np.ndarray) -> float:
    v = _vox(dose, mask)
    return float(np.percentile(v, 2)) if len(v) > 0 else 0.0


def compute_D01cc(dose: np.ndarray, mask: np.ndarray,
                  voxel_spacing_mm: float = VOXEL_SPACING_MM) -> float:
    """Mean dose of the hottest 0.1 cc — robust near-maximum metric (replaces Dmax)."""
    v = _vox(dose, mask)
    if len(v) == 0:
        return 0.0
    n = max(1, min(round(100.0 / voxel_spacing_mm ** 3), len(v)))  # 0.1 cc = 100 mm³
    return float(np.sort(v)[::-1][:n].mean())


def compute_Vx(dose: np.ndarray, mask: np.ndarray,
               dose_threshold_gy: float) -> float:
    """Percentage of structure volume receiving ≥ dose_threshold_gy."""
    v = _vox(dose, mask)
    return float(np.mean(v >= dose_threshold_gy) * 100) if len(v) > 0 else 0.0


def dvh_endpoints(pred_gy: np.ndarray, true_gy: np.ndarray,
                  mask: np.ndarray) -> Dict[str, float]:
    """
    Full DVH endpoint dict for one structure.

    Returns {metric}_{pred|true|diff} keys for:
      Dmean, Dmax, D95, D98, D01cc, V20, V40
    All values in Gy except V20/V40 which are percentages.
    """
    nan = float("nan")
    if (mask > 0.5).sum() == 0:
        result = {}
        for suffix in ("_pred", "_true", "_diff"):
            for key in ("Dmean", "Dmax", "D95", "D98", "D01cc", "V20", "V40"):
                result[key + suffix] = nan
        return result

    result = {}
    for tag, dose in (("pred", pred_gy), ("true", true_gy)):
        result[f"Dmean_{tag}"] = compute_Dmean(dose, mask)
        result[f"Dmax_{tag}"]  = compute_Dmax(dose, mask)
        result[f"D95_{tag}"]   = compute_D95(dose, mask)
        result[f"D98_{tag}"]   = compute_D98(dose, mask)
        result[f"D01cc_{tag}"] = compute_D01cc(dose, mask)
        result[f"V20_{tag}"]   = compute_Vx(dose, mask, 20.0)
        result[f"V40_{tag}"]   = compute_Vx(dose, mask, 40.0)

    for k in ("Dmean", "Dmax", "D95", "D98", "D01cc", "V20", "V40"):
        result[f"{k}_diff"] = result[f"{k}_pred"] - result[f"{k}_true"]
    return result


# ---------------------------------------------------------------------------
# Boundary MAE — tests H2 (steep dose falloff at PTV/OAR surface)
# ---------------------------------------------------------------------------

def _boundary_mask(struct_mask: np.ndarray, band_voxels: int) -> np.ndarray:
    m      = struct_mask > 0.5
    struct = generate_binary_structure(3, 1)
    outer  = binary_dilation(m, structure=struct, iterations=band_voxels)
    inner  = ~binary_dilation(~m, structure=struct, iterations=band_voxels)
    return (outer & ~inner).astype(bool)


def compute_boundary_mae(pred_gy: np.ndarray, true_gy: np.ndarray,
                         struct_mask: np.ndarray,
                         band_mm: float = BOUNDARY_BAND_MM,
                         voxel_spacing_mm: float = VOXEL_SPACING_MM) -> float:
    """MAE restricted to the ±band_mm shell around the structure surface."""
    band_vox = max(1, round(band_mm / voxel_spacing_mm))
    boundary = _boundary_mask(struct_mask, band_vox)
    if boundary.sum() == 0:
        log.warning("Empty boundary mask — returning global MAE.")
        return compute_mae(pred_gy, true_gy)
    return compute_mae(pred_gy, true_gy, mask=boundary.astype(float))


# ---------------------------------------------------------------------------
# Gamma pass rate
# ---------------------------------------------------------------------------

def compute_gamma_passrate(
    pred_gy: np.ndarray,
    true_gy: np.ndarray,
    body_mask: np.ndarray,
    voxel_spacing_mm: float = VOXEL_SPACING_MM,
    dose_percent: float = 3.0,
    distance_mm: float = 3.0,
) -> float:
    """
    3D gamma pass rate evaluated inside the body mask with a 10% low-dose cut-off.

    Returns the percentage of qualifying voxels with gamma ≤ 1.
    """
    pred_body   = np.where(body_mask > 0.5, pred_gy,  0.0)
    target_body = np.where(body_mask > 0.5, true_gy, 0.0)

    axes = tuple(np.arange(s) * voxel_spacing_mm for s in true_gy.shape)

    gamma_map = pymedphys.gamma(
        axes_reference            = axes,
        dose_reference            = target_body,
        axes_evaluation           = axes,
        dose_evaluation           = pred_body,
        dose_percent_threshold    = dose_percent,
        distance_mm_threshold     = distance_mm,
        lower_percent_dose_cutoff = 10.0,
        quiet                     = True,
    )

    valid = (body_mask > 0.5) & (target_body >= 0.1 * target_body.max())
    if valid.sum() == 0:
        return float("nan")
    return float((gamma_map[valid] <= 1.0).mean() * 100)


# ---------------------------------------------------------------------------
# Isodose Dice + HD95
# ---------------------------------------------------------------------------

def _hd95(pred_bin: np.ndarray, true_bin: np.ndarray,
          voxel_spacing_mm: float) -> float:
    from scipy.spatial import cKDTree
    struct  = generate_binary_structure(3, 1)
    p_surf  = pred_bin & ~binary_erosion(pred_bin, struct)
    t_surf  = true_bin & ~binary_erosion(true_bin, struct)
    p_pts   = np.argwhere(p_surf) * voxel_spacing_mm
    t_pts   = np.argwhere(t_surf) * voxel_spacing_mm
    if len(p_pts) == 0 or len(t_pts) == 0:
        return float("nan")
    d_pt = cKDTree(t_pts).query(p_pts)[0]
    d_tp = cKDTree(p_pts).query(t_pts)[0]
    return float(np.percentile(np.concatenate([d_pt, d_tp]), 95))


def compute_isodose_metrics(
    pred_gy: np.ndarray,
    true_gy: np.ndarray,
    ptv_mask: np.ndarray,
    body_mask: Optional[np.ndarray] = None,
    voxel_spacing_mm: float = VOXEL_SPACING_MM,
    isodose_levels: List[float] = [1.00, 0.95, 0.80, 0.50],
) -> Dict[str, float]:
    """
    Dice coefficient and HD95 at multiple isodose levels.

    Levels are expressed as fractions of the patient's prescription dose,
    defined as D95 of PTV from the ground-truth dose.

    When body_mask is provided both pred and true are zeroed outside the body
    before thresholding, preventing outside-body leakage from contaminating
    the spatial metrics.
    """
    if body_mask is not None:
        body = body_mask > 0.5
        pred_gy = np.where(body, pred_gy, 0.0)
        true_gy = np.where(body, true_gy, 0.0)
    prescription = compute_D95(true_gy, ptv_mask)
    if prescription < 1e-6:
        return {}
    metrics: Dict[str, float] = {}
    for level in isodose_levels:
        thresh     = level * prescription
        label      = f"{int(level * 100)}iso"
        pred_bin   = pred_gy >= thresh
        true_bin   = true_gy >= thresh
        inter      = (pred_bin & true_bin).sum()
        union      = pred_bin.sum() + true_bin.sum()
        metrics[f"Dice_{label}"]    = float(2 * inter / union) if union > 0 else 1.0
        metrics[f"HD95_{label}_mm"] = _hd95(pred_bin, true_bin, voxel_spacing_mm)
    return metrics


def compute_outside_body_leakage(
    pred_gy: np.ndarray,
    body_mask: np.ndarray,
    leakage_threshold_gy: float = 0.5,
) -> Dict[str, float]:
    """
    Quantifies unphysical dose predicted outside the body contour.

    Returns:
      leakage_mean_pred_Gy  : mean predicted dose in outside-body voxels
                              (should be ~0 for a physically correct model)
      leakage_vol_frac      : fraction of outside-body voxels with pred >= threshold
                              (default threshold = 0.5 Gy)
    """
    outside = body_mask < 0.5
    n_outside = outside.sum()
    if n_outside == 0:
        return {"leakage_mean_pred_Gy": 0.0, "leakage_vol_frac": 0.0}
    outside_pred = pred_gy[outside]
    return {
        "leakage_mean_pred_Gy": float(outside_pred.mean()),
        "leakage_vol_frac":     float((outside_pred >= leakage_threshold_gy).sum() / n_outside),
    }
