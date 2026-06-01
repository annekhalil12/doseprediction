# analysis/inv2_worst_patient_dvh_maps.py
# Investigation 2: for the 3 worst-MAE patients in each DoseGAN fold,
# regenerate predictions and produce DVH curves + spatial error heatmaps.
#
# Reads:  outputs/evaluation/{run_name}_fold{0..4}_val.csv
#         outputs/checkpoints_dosegan/{run_name}_fold{0..4}_best.pt
#         data/pickles/<patient_id>.pkl
# Writes: outputs/analysis/inv2_worst_<rank>_fold{F}_<patient>.png  (15 figures)
#
# Run after eval_dosegan.sbatch finishes:
#   python3 -m analysis.inv2_worst_patient_dvh_maps

import argparse
import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["dosegan", "unet3d"], default="dosegan",
                    help="Which model's checkpoints to load.")
parser.add_argument("--activation", choices=["sigmoid", "tanh"], default=None,
                    help="(unet3d only) override the activation token in RUN_NAME.")
parser.add_argument("--run-name", dest="run_name", type=str, default=None,
                    help="Override cfg.RUN_NAME directly.")
parser.add_argument("--fold", type=int, default=None,
                    help="Only process this fold (default: all 5).")
parser.add_argument("--n-worst", dest="n_worst", type=int, default=None,
                    help="How many worst patients per fold (default: N_WORST=3).")
args, _ = parser.parse_known_args()
cfg = importlib.import_module(f"configs.config_{args.model}")
if args.model == "unet3d" and args.activation is not None:
    cfg.RUN_NAME = cfg.RUN_NAME.replace(cfg.OUTPUT_ACTIVATION, args.activation)
    cfg.OUTPUT_ACTIVATION = args.activation
if args.run_name is not None:
    cfg.RUN_NAME = args.run_name
if args.model == "dosegan":
    from models.dosegan import UnetGenerator3d as _ModelClass
else:
    from models.unet3d import UNet3d as _ModelClass

EVAL_DIR    = Path("outputs/evaluation")
OUT_DIR     = Path("outputs/analysis")
CKPT_DIR    = cfg.CKPT_DIR
PICKLE_DIR  = cfg.PICKLE_DIR
DOSE_SCALE  = 50.0
N_WORST     = 3
DVH_GRID    = np.linspace(0, 65, 201)   # Gy axis for cumulative DVH


def worst_per_fold() -> pd.DataFrame:
    n = args.n_worst if args.n_worst is not None else N_WORST
    folds = [args.fold] if args.fold is not None else range(5)
    rows = []
    for fold in folds:
        path = EVAL_DIR / f"{cfg.RUN_NAME}_fold{fold}_val.csv"
        if not path.exists():
            sys.exit(f"missing eval CSV: {path}. Run eval_dosegan.sbatch first.")
        df = pd.read_csv(path).sort_values("body_MAE_Gy", ascending=False)
        df["rank"] = np.arange(1, len(df) + 1)
        rows.append(df.head(n))
    return pd.concat(rows, ignore_index=True)


def load_generator(fold: int, device: torch.device):
    ckpt_path = CKPT_DIR / f"{cfg.RUN_NAME}_fold{fold}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    if args.model == "dosegan":
        net = _ModelClass(input_nc=cfg.INPUT_NC, output_nc=cfg.OUTPUT_NC, ngf=cfg.NGF).to(device)
        net.load_state_dict(ckpt["generator"])
    else:
        net = _ModelClass(
            in_channels       = cfg.INPUT_NC,
            out_channels      = cfg.OUTPUT_NC,
            channels          = cfg.CHANNELS,
            strides           = cfg.STRIDES,
            num_res_units     = cfg.NUM_RES_UNITS,
            output_activation = cfg.OUTPUT_ACTIVATION,
        ).to(device)
        state_key = "model" if "model" in ckpt else "generator"
        net.load_state_dict(ckpt[state_key])
    net.eval()
    return net


def cumulative_dvh(dose: np.ndarray, mask: np.ndarray, grid: np.ndarray) -> np.ndarray:
    # Fraction of structure voxels with dose ≥ d, for d in grid.
    m = mask > 0.5
    if m.sum() == 0:
        return np.full_like(grid, np.nan, dtype=float)
    vox = dose[m]
    return np.array([(vox >= d).mean() for d in grid]) * 100.0  # percent


def _ptv_centroid(ptv: np.ndarray) -> tuple:
    vox = np.argwhere(ptv > 0.5)
    if len(vox) == 0:
        D, H, W = ptv.shape
        return D // 2, H // 2, W // 2
    return int(vox[:, 0].mean()), int(vox[:, 1].mean()), int(vox[:, 2].mean())


def _draw_contours(ax, ptv: np.ndarray, rectum: np.ndarray, bladder: np.ndarray,
                   axis: str, idx: int) -> None:
    styles = [
        (ptv,     "#FF2222", 1.6, "PTV"),
        (rectum,  "#22CC22", 1.3, "Rectum"),
        (bladder, "#2288FF", 1.3, "Bladder"),
    ]
    for mask, color, lw, _ in styles:
        if (mask > 0.5).sum() == 0:
            continue
        if axis == "axial":
            slc = mask[idx, :, :]
        elif axis == "coronal":
            slc = mask[:, idx, :]
        else:
            slc = mask[:, :, idx]
        if (slc > 0.5).sum() > 0:
            ax.contour(slc, levels=[0.5], colors=[color], linewidths=lw)


def make_figure(patient_id: str, fold: int, rank: int, body_mae: float,
                pred_gy: np.ndarray, true_gy: np.ndarray,
                ptv: np.ndarray, rectum: np.ndarray, bladder: np.ndarray,
                body: np.ndarray, out_path: Path) -> None:
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Row 1 — DVH curves (predicted vs true) for three structures
    structures = [
        ("PTV",     ptv,     "tab:blue"),
        ("Rectum",  rectum,  "tab:green"),
        ("Bladder", bladder, "tab:orange"),
    ]
    for ax, (name, mask, color) in zip(axes[0], structures):
        pred_dvh = cumulative_dvh(pred_gy, mask, DVH_GRID)
        true_dvh = cumulative_dvh(true_gy, mask, DVH_GRID)
        ax.plot(DVH_GRID, true_dvh, color=color, linestyle="-",  label="ground truth", linewidth=2)
        ax.plot(DVH_GRID, pred_dvh, color=color, linestyle="--", label="predicted",    linewidth=2)
        ax.set_xlabel("Dose (Gy)")
        ax.set_ylabel("Volume (%)")
        ax.set_title(f"{name} DVH")
        ax.set_xlim(0, 65)
        ax.set_ylim(0, 105)
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(alpha=0.3)

    # Row 2 — error maps centred on PTV, with structure contours overlaid
    err = (pred_gy - true_gy) * (body > 0.5)
    body_vox = err[body > 0.5]
    vmax = float(max(abs(np.percentile(body_vox, 1)),
                     abs(np.percentile(body_vox, 99)), 1.0))

    z_c, y_c, x_c = _ptv_centroid(ptv)
    views = [
        ("axial",   z_c, err[z_c, :, :],  f"axial  (z={z_c})"),
        ("coronal", y_c, err[:, y_c, :],  f"coronal (y={y_c})"),
        ("sagittal",x_c, err[:, :, x_c],  f"sagittal (x={x_c})"),
    ]

    for ax, (axis, idx, slc, title) in zip(axes[1], views):
        im = ax.imshow(slc, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        _draw_contours(ax, ptv, rectum, bladder, axis, idx)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="pred − true (Gy)")

    legend_handles = [
        Line2D([0], [0], color="#FF2222", linewidth=1.6, label="PTV"),
        Line2D([0], [0], color="#22CC22", linewidth=1.3, label="Rectum"),
        Line2D([0], [0], color="#2288FF", linewidth=1.3, label="Bladder"),
    ]
    axes[1][0].legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.7)

    fig.suptitle(
        f"Fold {fold} rank #{rank} (worst-MAE)  |  {patient_id}  |  "
        f"body MAE = {body_mae:.2f} Gy",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    worst = worst_per_fold()
    print(f"selected {len(worst)} worst-MAE patients across 5 folds")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    for fold in sorted(worst["fold"].unique()):
        gen = load_generator(fold, device)
        sub = worst[worst.fold == fold].sort_values("rank")
        for _, row in sub.iterrows():
            patient_id = row["patient_id"]
            pickle_path = PICKLE_DIR / f"{patient_id}.pkl"
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)

            inp = torch.from_numpy(data["input"]).unsqueeze(0).to(device)   # (1,9,D,H,W)
            with torch.no_grad():
                pred = gen(inp)[0, 0].cpu().numpy() * DOSE_SCALE
            true = data["dose"] * DOSE_SCALE                                # (D,H,W)
            body = data["input"][7]                                         # (D,H,W)

            out_path = OUT_DIR / f"inv2_{cfg.RUN_NAME}_worst_{int(row['rank'])}_fold{fold}_{patient_id}.png"
            make_figure(
                patient_id=patient_id, fold=fold, rank=int(row["rank"]),
                body_mae=float(row["body_MAE_Gy"]),
                pred_gy=pred, true_gy=true,
                ptv=data["ptv_mask"], rectum=data["rectum_mask"], bladder=data["bladder_mask"],
                body=body, out_path=out_path,
            )


if __name__ == "__main__":
    main()
