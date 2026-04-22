"""
preprocessing.py
================
SimpleITK-based preprocessing pipeline for LUND-PROBE NIfTI data.
Adapted from collaborator's preprocessingV5.py.

Produces a 9-channel input tensor per patient (channels 0–7 and 15 from
CHANNEL_MAP). Geometric channels 8–14 are skipped until V5geometric_channels.py
is available. Pickles are marked with `geometric_channels_pending=True` so a
separate script can append the missing channels without re-running this pipeline.

Pipeline per patient
--------------------
1. Load sCT + all structure masks + ground-truth dose
2. Resample everything to 1.5 mm isotropic spacing
3. Normalise sCT (HU window [-990, 2000] → z-score within body contour)
4. Normalise dose (divide by 50 Gy → range [0, 1])
5. Asymmetric PTV-centric fixed-size crop (128 × 256 × 320)
6. Stack 9 available channels into input tensor
7. Return dict ready for pickle serialisation

LUND-PROBE folder layout (per patient)
---------------------------------------
  <patient_id>/
    sCT/
      image_reg2MRI.nii.gz        ← sCT registered to MRI geometry (model input)
    MR_StorT2/
      dose_interpolated.nii.gz    ← ground-truth dose distribution
      mask_PTVT_427.nii.gz        ← Planning Target Volume
      mask_Rectum.nii.gz
      mask_Bladder.nii.gz
      mask_FemoralHead_L.nii.gz
      mask_FemoralHead_R.nii.gz
      mask_Genitalia.nii.gz       ← optional (missing for ~48 patients)
      mask_PenileBulb.nii.gz      ← optional (missing for ~48 patients)
      mask_BODY.nii.gz
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from config import PreprocessingConfig, CHANNEL_MAP, AVAILABLE_CHANNELS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level SimpleITK helpers
# ---------------------------------------------------------------------------

def load_nifti(path: Path) -> sitk.Image:
    """Load a NIfTI file (.nii or .nii.gz) and return a SimpleITK image."""
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    return sitk.ReadImage(str(path))


def resample_image(
    image: sitk.Image,
    target_spacing: Tuple[float, float, float],
    interpolator=sitk.sitkLinear,
    default_pixel_value: float = 0.0,
) -> sitk.Image:
    """
    Resample a SimpleITK image to the given isotropic voxel spacing.

    We compute the new grid size so that the physical extent of the volume
    is preserved: new_size = old_size * old_spacing / new_spacing.
    Linear interpolation is used for continuous-valued volumes (sCT, dose).
    Binary masks use nearest-neighbour instead — see resample_mask().
    """
    original_spacing = np.array(image.GetSpacing())
    original_size    = np.array(image.GetSize())

    new_size = (
        original_size * original_spacing / np.array(target_spacing)
    ).astype(int).tolist()

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetDefaultPixelValue(default_pixel_value)
    return resampler.Execute(image)


def resample_mask(
    mask: sitk.Image,
    reference: sitk.Image,
) -> sitk.Image:
    """
    Resample a binary structure mask to match the reference image grid.

    Nearest-neighbour interpolation is REQUIRED here. Linear interpolation
    would produce fractional values at mask boundaries (e.g. 0.3, 0.7),
    turning clean binary masks into blurry gradients. NN snaps each output
    voxel to the nearest input voxel value, preserving binary integrity.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)        # adopts spacing, size, origin, direction
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(mask)


def sitk_to_numpy(image: sitk.Image) -> np.ndarray:
    """
    Convert a SimpleITK image to a (D, H, W) float32 NumPy array.

    SimpleITK uses (x, y, z) axis order internally; GetArrayFromImage()
    transposes this to (z, y, x) == (D, H, W), which is the convention
    used throughout this pipeline and by PyTorch 3D convolutions.
    """
    return sitk.GetArrayFromImage(image).astype(np.float32)


# ---------------------------------------------------------------------------
# Intensity and dose normalisation
# ---------------------------------------------------------------------------

def normalise_sct(
    sct: np.ndarray,
    body_mask: np.ndarray,
    hu_window: Tuple[float, float] = (-990, 2000),
) -> np.ndarray:
    """
    Normalise a synthetic CT volume for use as a model input channel.

    sCTs express tissue density in Hounsfield Units (HU). Raw HU values
    vary slightly between acquisition groups (EDA confirmed: p98 HU 398.9
    newAcq vs 468.7 oldAcq), so we normalise to make values comparable
    across the cohort.

    Steps
    -----
    1. Isolate voxels inside the body contour (ignore background air).
    2. Clip to the approved HU window [-990, 2000] to suppress outliers.
    3. Z-score: subtract mean and divide by std, computed from body voxels.
    4. Set background (outside body) to 0.

    Z-score normalisation is computed inside the body contour only because
    background air voxels are not clinically meaningful and would distort
    the statistics.
    """
    lo, hi  = hu_window
    in_body = body_mask > 0.5

    # Compute statistics from clipped in-body voxels only
    clipped = np.clip(sct[in_body], lo, hi)
    mu      = clipped.mean()
    std     = clipped.std() + 1e-8   # avoid division by zero

    normalised          = np.zeros_like(sct, dtype=np.float32)
    normalised[in_body] = (np.clip(sct[in_body], lo, hi) - mu) / std

    log.debug(
        f"sCT normalisation: HU window [{lo}, {hi}] | "
        f"mu={mu:.1f}  std={std:.1f} | "
        f"output range [{normalised[in_body].min():.2f}, {normalised[in_body].max():.2f}]"
    )
    return normalised


def normalise_dose(
    dose: np.ndarray,
    dose_max_gy: float = 50.0,
) -> Tuple[np.ndarray, float]:
    """
    Normalise dose to [0, 1] using a fixed physical scale (Omar and Miguel approved).

    We divide by 50 Gy rather than by each patient's own max dose because:
    - The LUND-PROBE cohort uses a standardised 44.3 Gy prescription,
      so 50 Gy is a safe ceiling that keeps values in (0, 1).
    - Patient-specific normalisation would make predictions incomparable
      across patients (the model would need to learn a different scale
      for each patient rather than the actual dose distribution shape).

    Returns
    -------
    (normalised_dose, scale_factor) — multiply predictions by scale_factor
    to recover physical Gy values at evaluation time.
    """
    normalised = np.clip(dose / dose_max_gy, 0.0, 1.0).astype(np.float32)
    return normalised, float(dose_max_gy)


# ---------------------------------------------------------------------------
# Crop helpers
# ---------------------------------------------------------------------------

def _find_ptv_centroid_z(ptv_mask: np.ndarray) -> int:
    """
    Find the z-index (superior–inferior axis) of the PTV centroid.
    Used as the anchor for the asymmetric SI crop.
    """
    coords = np.argwhere(ptv_mask > 0.5)
    if len(coords) == 0:
        raise ValueError("PTV mask is empty — cannot find centroid z. Check this patient.")
    return int(coords[:, 0].mean())


def _find_ptv_centroid_x(ptv_mask: np.ndarray) -> int:
    """
    Find the x-index (left–right axis) of the PTV centroid.
    Used to centre the W crop on the prostate so it always appears at
    the horizontal midpoint of the tensor regardless of patient positioning.
    """
    coords = np.argwhere(ptv_mask > 0.5)
    if len(coords) == 0:
        raise ValueError("PTV mask is empty — cannot find centroid x. Check this patient.")
    return int(coords[:, 2].mean())


def _find_body_midpoint_y(body_mask_z_cropped: np.ndarray) -> int:
    """
    Find the midpoint of the body in the Y (anterior–posterior) axis,
    computed on the already Z-cropped subvolume.

    We use the body midpoint rather than the image centre because patients
    are not always centred on the scanner table — this makes the AP crop
    robust to positioning variation.
    """
    occupied_y = body_mask_z_cropped.any(axis=(0, 2))  # (H,) bool
    nonzero_y  = np.where(occupied_y)[0]

    if len(nonzero_y) == 0:
        # Should never happen with a valid BODY mask
        H = body_mask_z_cropped.shape[1]
        log.warning("BODY mask empty in Z-cropped subvolume — falling back to image centre for Y crop.")
        return H // 2

    return (int(nonzero_y.min()) + int(nonzero_y.max())) // 2


def fixed_size_crop(
    volumes: Dict[str, np.ndarray],
    ptv_mask: np.ndarray,
    body_mask: np.ndarray,
    crop_D: int,
    crop_H: int,
    crop_W: int,
    si_inferior: int,
    si_superior: int,
) -> Dict[str, np.ndarray]:
    """
    Crop all volumes to a fixed output shape (crop_D, crop_H, crop_W).

    Three-axis crop strategy (supervisor-approved)
    -----------------------------------------------
    Z (SI, depth=128) — asymmetric PTV-centric:
        40 slices below the PTV centroid (60 mm) and 88 above (132 mm).
        Asymmetric because the bladder dome extends far superior to the
        prostate; little dose-critical anatomy sits inferior to the apex.

    X (LR, width=320) — symmetric PTV-centric:
        Centred on the PTV centroid x so the prostate always sits at the
        horizontal midpoint of the tensor.

    Y (AP, height=256) — body midpoint on Z-cropped subvolume:
        Centred on the AP midpoint of the body contour within the Z window.
        Adapts to patient positioning rather than relying on image centre.

    Zero-padding is applied wherever the crop window falls outside the
    original volume. This is correct: background is air (HU ≈ -1000 →
    normalised ≈ 0), masks are 0, dose is 0.
    """
    assert si_inferior + si_superior == crop_D, (
        f"si_inferior ({si_inferior}) + si_superior ({si_superior}) "
        f"must equal crop_D ({crop_D})"
    )

    sample_vol = next(iter(volumes.values()))
    D, H, W = sample_vol.shape

    # Z axis: asymmetric PTV-centric crop
    ptv_z   = _find_ptv_centroid_z(ptv_mask)
    z_start = ptv_z - si_inferior
    z_end   = ptv_z + si_superior

    # X axis: symmetric PTV-centric crop
    ptv_x   = _find_ptv_centroid_x(ptv_mask)
    w_start = ptv_x - crop_W // 2
    w_end   = ptv_x + crop_W // 2

    # Y axis: body midpoint on Z-cropped subvolume
    sz0_body     = max(0, z_start)
    sz1_body     = min(D, z_end)
    body_z_crop  = body_mask[sz0_body:sz1_body, :, :]
    body_y_mid   = _find_body_midpoint_y(body_z_crop)
    h_start      = body_y_mid - crop_H // 2
    h_end        = h_start + crop_H

    log.info(
        f"Crop anchors — PTV z={ptv_z} x={ptv_x}  body_y_mid={body_y_mid} | "
        f"Z=[{z_start},{z_end}]  X=[{w_start},{w_end}]  Y=[{h_start},{h_end}]"
    )

    # Apply crop with zero-padding for out-of-bounds regions
    cropped = {}
    for name, vol in volumes.items():
        out = np.zeros((crop_D, crop_H, crop_W), dtype=np.float32)

        # Source (clamped to valid array range)
        sz0 = max(0, z_start);  sz1 = min(D, z_end)
        sh0 = max(0, h_start);  sh1 = min(H, h_end)
        sw0 = max(0, w_start);  sw1 = min(W, w_end)

        # Destination offsets inside the output array
        dz0 = sz0 - z_start;  dz1 = dz0 + (sz1 - sz0)
        dh0 = sh0 - h_start;  dh1 = dh0 + (sh1 - sh0)
        dw0 = sw0 - w_start;  dw1 = dw0 + (sw1 - sw0)

        out[dz0:dz1, dh0:dh1, dw0:dw1] = vol[sz0:sz1, sh0:sh1, sw0:sw1]
        cropped[name] = out

    cropped["__crop_offsets__"] = np.array([z_start, h_start, w_start], dtype=np.int32)
    return cropped


# ---------------------------------------------------------------------------
# Per-patient entry point
# ---------------------------------------------------------------------------

def preprocess_patient(
    patient_dir: Path,
    cfg_preprocessing: PreprocessingConfig,
) -> Dict:
    """
    Full preprocessing pipeline for one patient.

    Parameters
    ----------
    patient_dir      : path to the patient folder (e.g. basePart/newAcq_0a5468...)
    cfg_preprocessing: PreprocessingConfig instance from config.py

    Returns
    -------
    dict with keys:
      'input'                      : (9, D, H, W) float32  — 9 available channels
      'dose'                       : (D, H, W)    float32  — normalised dose [0, 1]
      'dose_scale'                 : float                  — multiply preds by this → Gy
      'ptv_mask'                   : (D, H, W)    bool
      'rectum_mask'                : (D, H, W)    bool
      'bladder_mask'               : (D, H, W)    bool
      'patient_id'                 : str
      'geometric_channels_pending' : True         — flag for the geometric channel top-up script
      'sct_sitk_ref'               : sitk.Image   — resampled sCT (for NIfTI export; removed by preprocess_all.py)
      'crop_offsets'               : (3,) int32   — z/h/w crop start indices (for NIfTI export metadata)
    """
    pcfg    = cfg_preprocessing
    spacing = (pcfg.target_spacing,) * 3   # (1.5, 1.5, 1.5) mm

    # ── Paths ──────────────────────────────────────────────────────────────
    # LUND-PROBE layout:
    #   patient_dir/sCT/image_reg2MRI.nii.gz      ← sCT in MRI geometry
    #   patient_dir/MR_StorT2/mask_*.nii.gz        ← structure masks
    #   patient_dir/MR_StorT2/dose_interpolated.nii.gz
    sct_dir = patient_dir / "sCT"
    mri_dir = patient_dir / "MR_StorT2"

    # ── Load and resample sCT ──────────────────────────────────────────────
    # The sCT is loaded first and used as the spatial reference for
    # resampling all masks (resample_mask aligns everything to this grid).
    sct_sitk = load_nifti(sct_dir / "image_reg2MRI.nii.gz")
    sct_sitk = resample_image(sct_sitk, spacing, sitk.sitkLinear)
    sct_np   = sitk_to_numpy(sct_sitk)

    # ── Load and resample structure masks ──────────────────────────────────
    # Channels 0–7 from CHANNEL_MAP. Optional structures get all-zero arrays.
    structure_names = [CHANNEL_MAP[i] for i in range(8)]   # ordered 0-7

    masks: Dict[str, np.ndarray] = {}
    for name in structure_names:
        mask_path = mri_dir / f"mask_{name}.nii.gz"
        if not mask_path.exists():
            if name in pcfg.optional_structures:
                # ~48 patients are missing Genitalia/PenileBulb; zero-fill the channel
                log.debug(f"Optional structure '{name}' missing → using zeros.")
                masks[name] = np.zeros_like(sct_np, dtype=np.float32)
            else:
                raise FileNotFoundError(f"Required structure missing: {mask_path}")
        else:
            m          = load_nifti(mask_path)
            m          = resample_mask(m, sct_sitk)    # nearest-neighbour; preserves binary values
            masks[name] = (sitk_to_numpy(m) > 0.5).astype(np.float32)

    # ── Load and resample ground-truth dose ────────────────────────────────
    dose_sitk = load_nifti(mri_dir / "dose_interpolated.nii.gz")
    dose_sitk = resample_image(dose_sitk, spacing, sitk.sitkLinear)
    dose_np   = sitk_to_numpy(dose_sitk)

    # ── Normalise sCT and dose ─────────────────────────────────────────────
    body_mask              = masks["BODY"]
    sct_norm               = normalise_sct(sct_np, body_mask, pcfg.sct_hu_window)
    dose_norm, dose_scale  = normalise_dose(dose_np, pcfg.dose_max_gy)

    # ── Assemble volumes dict for cropping ─────────────────────────────────
    all_volumes: Dict[str, np.ndarray] = {
        **masks,
        "sct_intensity": sct_norm,
        "dose":          dose_norm,
    }

    # ── Asymmetric PTV-centric fixed-size crop ─────────────────────────────
    crop = fixed_size_crop(
        volumes     = all_volumes,
        ptv_mask    = masks["PTVT_427"],
        body_mask   = masks["BODY"],
        crop_D      = pcfg.crop_size_D,
        crop_H      = pcfg.crop_size_H,
        crop_W      = pcfg.crop_size_W,
        si_inferior = pcfg.si_inferior_slices,
        si_superior = pcfg.si_superior_slices,
    )

    # ── Stack available channels into input tensor ─────────────────────────
    # Channels 0–7 (structure masks) + channel 15 (sCT intensity).
    # Channels 8–14 (geometric features) are omitted until
    # V5geometric_channels.py is available — see geometric_channels_pending flag.
    available_channel_names = [AVAILABLE_CHANNELS[i] for i in sorted(AVAILABLE_CHANNELS)]
    input_tensor = np.stack(
        [crop[ch] for ch in available_channel_names], axis=0
    ).astype(np.float32)   # shape: (9, 128, 256, 320)

    return {
        "input":                      input_tensor,               # (9, 128, 256, 320) float32
        "dose":                       crop["dose"],               # (128, 256, 320)    float32
        "dose_scale":                 dose_scale,                 # float — multiply predictions by this to get Gy
        "ptv_mask":                   crop["PTVT_427"].astype(bool),
        "rectum_mask":                crop["Rectum"].astype(bool),
        "bladder_mask":               crop["Bladder"].astype(bool),
        "patient_id":                 patient_dir.name,
        "geometric_channels_pending": True,                       # set to False once channels 8–14 are appended
        "sct_sitk_ref":               sct_sitk,                   # removed by preprocess_all.py before pickling
        "crop_offsets":               crop["__crop_offsets__"],   # needed for correct NIfTI export origin
    }