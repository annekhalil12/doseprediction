"""
training/manifest.py
====================
Write and update a per-run JSON manifest for reproducibility.

Each training run produces:
    outputs/runs/<run_name>_fold<N>_manifest.json

The manifest is written twice:
    1. At training start  — captures config, code version, data hash.
    2. At training end    — fills in checkpoint path, W&B run ID, best metrics,
                           training duration, and final status.

Import and call from training scripts only — not a standalone script.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

MANIFEST_DIR = Path("outputs/runs")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return bool(result)
    except Exception:
        return False


def write_start(
    run_name:         str,
    fold:             int,
    model_type:       str,
    cfg:              Any,
    config_file:      str,
    checkpoint_path:  Path,
    eval_csv_path:    Path,
    seed:             Optional[int] = None,
) -> Path:
    """
    Write the initial manifest at training start.
    Returns the manifest path so the caller can pass it to write_end().
    """
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"{run_name}_fold{fold}_manifest.json"

    split_csv = Path(cfg.SPLIT_CSV)

    doc = {
        # ── Identity ──────────────────────────────────────────────────────
        "run_name":            run_name,
        "fold":                fold,
        "model":               model_type,
        "status":              "running",

        # ── Config snapshot ───────────────────────────────────────────────
        "config_file":         config_file,
        "use_geom_channels":   bool(cfg.USE_GEOM_CHANNELS),
        "use_flips":           bool(cfg.USE_FLIPS),
        "input_nc":            int(cfg.INPUT_NC),
        "output_nc":           int(cfg.OUTPUT_NC),
        "output_activation":   str(cfg.OUTPUT_ACTIVATION),
        "epochs":              int(cfg.EPOCHS),
        "batch_size":          int(cfg.BATCH_SIZE),
        "early_stop_patience": int(cfg.EARLY_STOP_PATIENCE),
        "lambda_dvh":          float(cfg.LAMBDA_DVH),
        "lambda_grad":         float(getattr(cfg, "LAMBDA_GRAD", 0.0)),
        "seed":                seed,

        # ── Data provenance ───────────────────────────────────────────────
        "split_csv":           str(split_csv),
        "split_csv_sha256":    _sha256(split_csv) if split_csv.exists() else "missing",
        "pickle_dir":          str(cfg.PICKLE_DIR),

        # ── Code version ──────────────────────────────────────────────────
        "git_commit":          _git_commit(),
        "git_dirty":           _git_dirty(),

        # ── Output paths ──────────────────────────────────────────────────
        "checkpoint_path":     str(checkpoint_path),
        "eval_csv_path":       str(eval_csv_path),

        # ── Filled at end of training ─────────────────────────────────────
        "wandb_run_id":        None,
        "best_val_loss":       None,
        "best_val_dvh_score":  None,
        "epochs_trained":      None,
        "training_start_utc":  datetime.now(timezone.utc).isoformat(),
        "training_end_utc":    None,
        "training_duration_h": None,
    }

    with open(manifest_path, "w") as f:
        json.dump(doc, f, indent=2)

    return manifest_path


def write_end(
    manifest_path:    Path,
    wandb_run_id:     Optional[str],
    best_val_loss:    float,
    best_dvh_score:   Optional[float],
    epochs_trained:   int,
    start_utc:        str,
    status:           str = "completed",
) -> None:
    """Update the manifest at training end with final metrics and W&B run ID."""
    if not manifest_path.exists():
        return

    with open(manifest_path) as f:
        doc = json.load(f)

    end_utc = datetime.now(timezone.utc)
    try:
        start_dt = datetime.fromisoformat(start_utc)
        duration_h = (end_utc - start_dt).total_seconds() / 3600
    except Exception:
        duration_h = None

    ckpt_path = Path(doc.get("checkpoint_path", ""))
    ckpt_sha  = _sha256(ckpt_path) if ckpt_path.exists() else "missing"

    doc.update({
        "status":              status,
        "wandb_run_id":        wandb_run_id,
        "best_val_loss":       round(best_val_loss, 6) if best_val_loss is not None else None,
        "best_val_dvh_score":  round(best_dvh_score, 6) if best_dvh_score is not None else None,
        "epochs_trained":      epochs_trained,
        "checkpoint_sha256":   ckpt_sha,
        "training_end_utc":    end_utc.isoformat(),
        "training_duration_h": round(duration_h, 3) if duration_h is not None else None,
    })

    with open(manifest_path, "w") as f:
        json.dump(doc, f, indent=2)
