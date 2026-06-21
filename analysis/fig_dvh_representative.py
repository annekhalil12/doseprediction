"""
analysis/fig_dvh_representative.py
====================================
Thesis figure: dose-volume histograms (DVH) for a representative validation
patient.

Three panels — PTV / Rectum / Bladder — each showing:
  Ground truth  (solid black)
  U-Net         (blue dashed)
  DoseGAN       (red dash-dot)

Patient: median body_MAE_Gy from DoseGAN baseline fold-0 validation CSV
         (same selection as fig_dose_slices.py — identical patient guaranteed
         as long as the CSV is unchanged).

Output: results/figures/fig_dvh_representative.{pdf,png}
Usage (from project root):
    python3 -m analysis.fig_dvh_representative
    python3 -m analysis.fig_dvh_representative --fold 2 --patient-id <id>
"""

from __future__ import annotations

import argparse
import csv
import functools
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT        = Path(__file__).resolve().parents[1]
OUT_DIR     = ROOT / "results" / "figures"
EVAL_DIR    = ROOT / "outputs" / "evaluation"
DG_CKPT_DIR = ROOT / "outputs" / "checkpoints_dosegan"
UN_CKPT_DIR = ROOT / "outputs" / "checkpoints_unet3d"
PICKLE_DIR  = ROOT / "data" / "pickles"
DOSE_SCALE  = 50.0
DVH_GRID    = np.linspace(0, 65, 261)   # 0.25 Gy steps for smooth curves

OUT_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--fold",       type=int, default=0)
parser.add_argument("--patient-id", dest="patient_id", default=None)
args = parser.parse_args()
FOLD = args.fold


# ── patient selection ─────────────────────────────────────────────────────────

def select_patient() -> str:
    if args.patient_id:
        return args.patient_id
    csv_path = EVAL_DIR / f"dosegan_ngf32_sigmoid_snellius_fold{FOLD}_val.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"DoseGAN eval CSV not found: {csv_path}\n"
            "Run evaluation first: sbatch eval.sbatch MODEL=dosegan FOLD=0"
        )
    rows = sorted(csv.DictReader(open(csv_path)), key=lambda r: float(r["body_MAE_Gy"]))
    r = rows[len(rows) // 2]
    print(f"Patient (median body_MAE): {r['patient_id']}  MAE = {float(r['body_MAE_Gy']):.3f} Gy")
    return r["patient_id"]


# ── model loading ─────────────────────────────────────────────────────────────

def _uses_batchnorm(sd: dict) -> bool:
    return any("running_mean" in k for k in sd)


def load_dosegan(device) -> nn.Module:
    from configs import config_dosegan as cfg
    from models.dosegan import UnetGenerator3d
    ckpt_path = DG_CKPT_DIR / f"dosegan_ngf32_sigmoid_snellius_fold{FOLD}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    sd   = ckpt["generator"]
    norm = nn.BatchNorm3d if _uses_batchnorm(sd) else functools.partial(nn.InstanceNorm3d, affine=True)
    net  = UnetGenerator3d(input_nc=9, output_nc=cfg.OUTPUT_NC, ngf=cfg.NGF, norm_layer=norm).to(device)
    net.load_state_dict(sd, strict=False)
    net.eval()
    return net


def load_unet(device) -> nn.Module:
    from configs import config_unet3d as cfg
    from models.unet3d import UNet3d
    ckpt_path = UN_CKPT_DIR / f"unet3d_ch32_sigmoid_snellius_fold{FOLD}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    net  = UNet3d(
        in_channels=9, out_channels=cfg.OUTPUT_NC,
        channels=cfg.CHANNELS, strides=cfg.STRIDES,
        num_res_units=cfg.NUM_RES_UNITS,
        output_activation=cfg.OUTPUT_ACTIVATION,
    ).to(device)
    key = "model" if "model" in ckpt else "generator"
    net.load_state_dict(ckpt[key], strict=False)
    net.eval()
    return net


# ── DVH helper ────────────────────────────────────────────────────────────────

def cum_dvh(dose: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cumulative DVH: fraction of structure voxels receiving >= d Gy, for each d in DVH_GRID."""
    m = mask > 0.5
    if m.sum() == 0:
        return np.full_like(DVH_GRID, np.nan)
    vox = dose[m]
    return np.array([(vox >= d).mean() * 100.0 for d in DVH_GRID])


# ── main ─────────────────────────────────────────────────────────────────────

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

patient_id = select_patient()

with open(PICKLE_DIR / f"{patient_id}.pkl", "rb") as fh:
    data = pickle.load(fh)

inp_arr   = data["input"].astype(np.float32)          # (9, D, H, W)
true_dose = data["dose"].astype(np.float32) * DOSE_SCALE  # (D, H, W) Gy

inp_t = torch.from_numpy(inp_arr).unsqueeze(0).to(device)

print("Loading checkpoints:")
dg_net = load_dosegan(device)
un_net = load_unet(device)

print("Running inference...")
with torch.no_grad():
    dg_pred = dg_net(inp_t)[0, 0].cpu().numpy() * DOSE_SCALE
    un_pred = un_net(inp_t)[0, 0].cpu().numpy() * DOSE_SCALE

# structures: PTV=ch0, Rectum=ch1, Bladder=ch2  (from CLAUDE.md channel map)
structures = [
    ("PTV",     inp_arr[0]),
    ("Rectum",  inp_arr[1]),
    ("Bladder", inp_arr[2]),
]

# ── figure ────────────────────────────────────────────────────────────────────

plt.rcParams.update({"font.size": 9.5, "font.family": "sans-serif"})
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8),
                         gridspec_kw={"wspace": 0.30})

PLOT_STYLES = [
    ("Ground truth", true_dose, "#111111", "-",  2.0),
    ("U-Net",        un_pred,   "#2196F3", "--", 1.8),
    ("DoseGAN",      dg_pred,   "#E53935", "-.", 1.8),
]

for ax, (struct_name, mask) in zip(axes, structures):
    n_vox = int((mask > 0.5).sum())
    for label, dose, color, ls, lw in PLOT_STYLES:
        dvh = cum_dvh(dose, mask)
        ax.plot(DVH_GRID, dvh, color=color, linestyle=ls, linewidth=lw, label=label)
    ax.set_xlabel("Dose (Gy)", fontsize=9.5)
    ax.set_ylabel("Volume (%)", fontsize=9.5)
    ax.set_title(f"{struct_name}  ({n_vox:,} voxels)", fontsize=10.5)
    ax.set_xlim(0, 65)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.85)

acq = patient_id.split("_")[0]
fig.suptitle(
    f"Representative patient DVH — {patient_id} ({acq}) | fold {FOLD}",
    fontsize=9.5, y=1.02
)

out_stem = OUT_DIR / "fig_dvh_representative"
fig.savefig(str(out_stem) + ".pdf", bbox_inches="tight", facecolor="white")
fig.savefig(str(out_stem) + ".png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved: {out_stem}.pdf  +  {out_stem}.png")
