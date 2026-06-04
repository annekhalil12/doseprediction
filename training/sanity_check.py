"""
training/sanity_check.py
========================
Pretraining sanity checks 1–3. Run before any training job.

Check 1 — Shape verification
    For 10 random patients: baseline input (9,D,H,W), geom input (14,D,H,W),
    dose (1,D,H,W). Fails loudly on any mismatch.

Check 2 — Mask alignment
    Middle axial slice of one patient: sCT, dose, PTV, BODY, rectum, bladder
    overlaid as contours. Goal: confirm masks are spatially aligned with sCT/dose.

Check 3 — Geometric channel visualisation
    Middle axial slice of one patient: all 5 geom channels (dist_ptv,
    dist_body, dir_z, dir_y, dir_x). Goal: confirm channels are smooth
    and anatomically plausible.

Usage:
    PYTHONPATH=. python3 -m training.sanity_check

Output:
    outputs/sanity/check2_mask_alignment.png
    outputs/sanity/check3_geom_channels.png
"""

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import config_unet3d as cfg
from training.dataset import LUNDPROBEDataset

OUT_DIR = Path("outputs/sanity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_BASE_SHAPE = (9,  128, 256, 320)
EXPECTED_GEOM_SHAPE = (14, 128, 256, 320)
EXPECTED_DOSE_SHAPE = (1,  128, 256, 320)


# ---------------------------------------------------------------------------
# Check 1 — Shape verification
# ---------------------------------------------------------------------------
def check_shapes(n: int = 10) -> None:
    print("\n── Check 1: Dataset shapes ──────────────────────────────────────")

    ds_base = LUNDPROBEDataset(
        cfg.SPLIT_CSV, cfg.PICKLE_DIR,
        split="train", fold=0,
        use_geom_channels=False, use_flips=False,
    )
    ds_geom = LUNDPROBEDataset(
        cfg.SPLIT_CSV, cfg.PICKLE_DIR,
        split="train", fold=0,
        use_geom_channels=True, use_flips=False,
    )

    indices = random.sample(range(len(ds_base)), min(n, len(ds_base)))
    all_ok = True

    for i in indices:
        sb = ds_base[i]
        sg = ds_geom[i]

        base_ok = tuple(sb["input"].shape) == EXPECTED_BASE_SHAPE
        geom_ok = tuple(sg["input"].shape) == EXPECTED_GEOM_SHAPE
        dose_ok = tuple(sb["dose"].shape)  == EXPECTED_DOSE_SHAPE

        status = "OK" if (base_ok and geom_ok and dose_ok) else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(
            f"  [{status}] {sb['patient_id']}"
            f"  base={tuple(sb['input'].shape)}"
            f"  geom={tuple(sg['input'].shape)}"
            f"  dose={tuple(sb['dose'].shape)}"
        )

    if all_ok:
        print(f"  ✓ All {len(indices)} patients have correct shapes.")
    else:
        print("  ✗ Shape mismatches found — do NOT train.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Check 2 — Mask alignment
# ---------------------------------------------------------------------------
def check_mask_alignment(patient_idx: int = 0) -> None:
    print("\n── Check 2: Mask alignment ──────────────────────────────────────")

    ds = LUNDPROBEDataset(
        cfg.SPLIT_CSV, cfg.PICKLE_DIR,
        split="train", fold=0,
        use_geom_channels=False, use_flips=False,
    )
    sample = ds[patient_idx]
    pid = sample["patient_id"]

    inp  = sample["input"].numpy()    # (9, D, H, W)
    dose = sample["dose"].numpy()[0]  # (D, H, W)
    z    = inp.shape[1] // 2          # middle axial slice

    sct      = inp[8, z]              # sCT intensity
    ptv      = inp[0, z]              # PTV mask
    body     = inp[7, z]              # BODY mask
    rectum   = inp[1, z]              # Rectum mask
    bladder  = inp[2, z]              # Bladder mask
    dose_sl  = dose[z]

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    fig.suptitle(f"Check 2 — Mask alignment | {pid} | axial slice z={z}", fontsize=11)

    panels = [
        (sct,      "sCT",     "gray"),
        (dose_sl,  "Dose",    "hot"),
        (ptv,      "PTV",     "Reds"),
        (body,     "BODY",    "Blues"),
        (rectum,   "Rectum",  "Greens"),
        (bladder,  "Bladder", "Purples"),
    ]
    for ax, (arr, title, cmap) in zip(axes, panels):
        ax.imshow(arr, cmap=cmap, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Overlay mask contours on sCT
    axes[0].contour(ptv,    levels=[0.5], colors=["red"],    linewidths=0.8)
    axes[0].contour(body,   levels=[0.5], colors=["cyan"],   linewidths=0.8)
    axes[0].contour(rectum, levels=[0.5], colors=["lime"],   linewidths=0.8)
    axes[0].contour(bladder,levels=[0.5], colors=["yellow"], linewidths=0.8)

    plt.tight_layout()
    out = OUT_DIR / "check2_mask_alignment.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out}")


# ---------------------------------------------------------------------------
# Check 3 — Geometric channel visualisation
# ---------------------------------------------------------------------------
def check_geom_channels(patient_idx: int = 0) -> None:
    print("\n── Check 3: Geometric channels ──────────────────────────────────")

    ds = LUNDPROBEDataset(
        cfg.SPLIT_CSV, cfg.PICKLE_DIR,
        split="train", fold=0,
        use_geom_channels=True, use_flips=False,
    )
    sample = ds[patient_idx]
    pid = sample["patient_id"]

    inp = sample["input"].numpy()  # (14, D, H, W)
    z   = inp.shape[1] // 2

    geom_names = ["dist_to_ptv", "dist_to_body", "dir_z", "dir_y", "dir_x"]
    geom_channels = inp[9:14, z]   # channels 9–13

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f"Check 3 — Geometric channels | {pid} | axial slice z={z}", fontsize=11)

    for ax, arr, name in zip(axes, geom_channels, geom_names):
        im = ax.imshow(arr, cmap="RdBu_r", origin="lower",
                       vmin=arr.min(), vmax=arr.max())
        ax.set_title(f"{name}\n[{arr.min():.2f}, {arr.max():.2f}]", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = OUT_DIR / "check3_geom_channels.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    check_shapes(n=10)
    check_mask_alignment(patient_idx=0)
    check_geom_channels(patient_idx=0)
    print("\n✓ All sanity checks passed. Safe to train.")
