# analysis/plot_dose_with_masks.py
# Loads a checkpoint, runs inference on selected patient(s), and produces a
# 4-panel figure: sCT + contours | predicted dose | ground-truth dose | signed error.
# Structure contours (PTV, Rectum, Bladder) are overlaid on every panel.
#
# Run from project root (needs GPU or CPU):
#   python3 -m analysis.plot_dose_with_masks --model dosegan --fold 0
#   python3 -m analysis.plot_dose_with_masks --model unet3d  --fold 0
#   python3 -m analysis.plot_dose_with_masks --model dosegan --fold 0 --patient-id <id>
#   python3 -m analysis.plot_dose_with_masks --model dosegan --fold 0 --selector median worst best

from __future__ import annotations

import argparse
import importlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

# ── argument parsing before model imports ─────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["dosegan", "unet3d"], default="dosegan")
parser.add_argument("--fold",  type=int, default=0)
parser.add_argument("--run-name", dest="run_name", default=None,
                    help="Override RUN_NAME (default: baseline Sigmoid).")
parser.add_argument("--patient-id", dest="patient_id", default=None,
                    help="Specific patient. Overrides --selector.")
parser.add_argument("--selector", nargs="+",
                    choices=["median", "worst", "best"],
                    default=["median"],
                    help="Which patient(s) to visualise by MAE rank.")
args = parser.parse_args()

cfg = importlib.import_module(f"configs.config_{args.model}")

DEFAULTS: dict[str, str] = {
    "dosegan": "dosegan_ngf32_sigmoid_snellius",
    "unet3d":  "unet3d_ch32_sigmoid_snellius",
}
cfg.RUN_NAME = args.run_name if args.run_name else DEFAULTS[args.model]

if args.model == "dosegan":
    from models.dosegan import UnetGenerator3d as ModelClass
else:
    from models.unet3d import UNet3d as ModelClass

EVAL_DIR   = Path("outputs/evaluation")
PICKLE_DIR = Path("data/pickles")
OUT_DIR    = Path("outputs/analysis")
DOSE_SCALE = 50.0

STRUCT_STYLES: dict[str, dict] = {
    "ptv":     {"color": "#FF2222", "lw": 1.8, "label": "PTV"},
    "rectum":  {"color": "#22CC22", "lw": 1.4, "label": "Rectum"},
    "bladder": {"color": "#2288FF", "lw": 1.4, "label": "Bladder"},
}


def load_model(device: torch.device):
    ckpt_path = cfg.CKPT_DIR / f"{cfg.RUN_NAME}_fold{args.fold}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if args.model == "dosegan":
        net = ModelClass(
            input_nc=cfg.INPUT_NC, output_nc=cfg.OUTPUT_NC, ngf=cfg.NGF
        ).to(device)
        net.load_state_dict(ckpt["generator"])
    else:
        net = ModelClass(
            in_channels=cfg.INPUT_NC, out_channels=cfg.OUTPUT_NC,
            channels=cfg.CHANNELS, strides=cfg.STRIDES,
            num_res_units=cfg.NUM_RES_UNITS,
            output_activation=cfg.OUTPUT_ACTIVATION,
        ).to(device)
        key = "model" if "model" in ckpt else "generator"
        net.load_state_dict(ckpt[key])
    net.eval()
    print(f"Loaded: {ckpt_path} (epoch {ckpt['epoch']}, "
          f"best_val_L1={ckpt['best_val_loss']:.4f})")
    return net


def select_patients() -> list[tuple[str, str]]:
    """Returns list of (patient_id, selector_label) pairs."""
    if args.patient_id:
        return [(args.patient_id, "specified")]

    csv_path = EVAL_DIR / f"{cfg.RUN_NAME}_fold{args.fold}_val.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Eval CSV not found: {csv_path}\n"
            "Run eval.sbatch (with GEOM=0) first to generate it."
        )
    df = pd.read_csv(csv_path).sort_values("body_MAE_Gy").reset_index(drop=True)
    n = len(df)
    result = []
    for sel in args.selector:
        if sel == "median":
            idx = n // 2
        elif sel == "best":
            idx = 0
        else:  # worst
            idx = n - 1
        result.append((df.iloc[idx]["patient_id"], sel))
    return result


def ptv_centroid_z(ptv_mask: np.ndarray) -> int:
    vox = np.argwhere(ptv_mask > 0.5)
    return int(vox[:, 0].mean()) if len(vox) > 0 else ptv_mask.shape[0] // 2


def draw_contours(ax, masks: dict[str, np.ndarray], z: int) -> None:
    for struct, style in STRUCT_STYLES.items():
        m = masks.get(struct)
        if m is None or (m > 0.5).sum() == 0:
            continue
        slc = m[z, :, :]
        if (slc > 0.5).sum() > 0:
            ax.contour(slc, levels=[0.5], colors=[style["color"]], linewidths=style["lw"])


def make_figure(patient_id: str, selector: str,
                pred_gy: np.ndarray, true_gy: np.ndarray,
                sct: np.ndarray, masks: dict[str, np.ndarray],
                body_mae: float, out_path: Path) -> None:
    z = ptv_centroid_z(masks["ptv"])

    # scale doses to Gy for display
    vmax = float(np.nanpercentile(true_gy[masks.get("body", np.ones_like(true_gy)) > 0.5], 99.5))
    err  = pred_gy - true_gy
    emax = float(max(abs(np.nanpercentile(err, 1)), abs(np.nanpercentile(err, 99)), 2.0))

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(
        f"{args.model.upper()} — {cfg.RUN_NAME}  |  fold {args.fold}  |  "
        f"patient: {patient_id}  [{selector}]\n"
        f"body MAE = {body_mae:.3f} Gy  |  axial slice z = {z}",
        fontsize=11,
    )

    # Panel 1: sCT + contours
    axes[0].imshow(sct[z], cmap="gray", origin="lower", aspect="auto")
    draw_contours(axes[0], masks, z)
    axes[0].set_title("sCT + structure contours", fontsize=10)

    # Panel 2: predicted dose + contours
    im1 = axes[1].imshow(pred_gy[z], cmap="inferno", vmin=0, vmax=vmax,
                         origin="lower", aspect="auto")
    draw_contours(axes[1], masks, z)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Dose (Gy)")
    axes[1].set_title("Predicted dose", fontsize=10)

    # Panel 3: ground-truth dose + contours
    im2 = axes[2].imshow(true_gy[z], cmap="inferno", vmin=0, vmax=vmax,
                         origin="lower", aspect="auto")
    draw_contours(axes[2], masks, z)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Dose (Gy)")
    axes[2].set_title("Ground-truth dose", fontsize=10)

    # Panel 4: signed error + contours
    im3 = axes[3].imshow(err[z], cmap="RdBu_r", vmin=-emax, vmax=emax,
                         origin="lower", aspect="auto")
    draw_contours(axes[3], masks, z)
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, label="pred − true (Gy)")
    axes[3].set_title("Signed error (pred − true)", fontsize=10)

    for ax in axes:
        ax.axis("off")

    legend_handles = [
        Line2D([0], [0], color=s["color"], linewidth=s["lw"], label=s["label"])
        for s in STRUCT_STYLES.values()
    ]
    axes[0].legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.75)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | model: {args.model} | fold: {args.fold} | run: {cfg.RUN_NAME}")

    model  = load_model(device)
    patients = select_patients()
    print(f"Visualising {len(patients)} patient(s)")

    for pid, selector in patients:
        pkl_path = PICKLE_DIR / f"{pid}.pkl"
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        inp = torch.from_numpy(data["input"]).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_gy = model(inp)[0, 0].cpu().numpy() * DOSE_SCALE
        true_gy = data["dose"] * DOSE_SCALE
        sct     = data["input"][8]        # channel 8 = sCT
        body    = data["input"][7]        # channel 7 = BODY

        masks = {
            "ptv":     data.get("ptv_mask",     data["input"][0]),
            "rectum":  data.get("rectum_mask",  data["input"][1]),
            "bladder": data.get("bladder_mask", data["input"][2]),
            "body":    body,
        }

        body_vox = pred_gy[body > 0.5] - true_gy[body > 0.5]
        body_mae = float(np.abs(body_vox).mean())

        out_path = OUT_DIR / (
            f"dose_with_masks_{args.model}_fold{args.fold}_{selector}_{pid}.png"
        )
        make_figure(pid, selector, pred_gy, true_gy, sct, masks, body_mae, out_path)


if __name__ == "__main__":
    main()
