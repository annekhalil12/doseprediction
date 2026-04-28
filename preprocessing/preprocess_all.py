"""
preprocess_all.py
=================
Batch preprocessing runner for all 432 LUND-PROBE patients.

Loops over every patient directory in DATA_ROOT, calls preprocess_patient(),
and saves the result as a .pkl file in OUTPUT_DIR. Already-cached patients
are skipped automatically, so the job can be safely interrupted and resumed.

Run from the preprocessing/ directory on OneView:
    python preprocess_all.py

This script runs on CPU — no GPU is required for preprocessing.
SimpleITK resampling is CPU-bound and runs fine on OneView's CPU cores.
Expect roughly 15–30 seconds per patient; ~432 patients ≈ 2–4 hours total.

Output
------
One .pkl file per patient in OUTPUT_DIR, e.g.:
    <OUTPUT_DIR>/newAcq_01d2150e9b50efa1.pkl
    <OUTPUT_DIR>/oldAcq_003f2a...pkl
    ...

A summary CSV is also written to OUTPUT_DIR/preprocessing_summary.csv,
recording the outcome (success / skipped / failed) for every patient.
This is your audit trail — keep it alongside the pickles.
"""

import csv
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_ROOT, OUTPUT_DIR, PreprocessingConfig
from preprocessing import preprocess_patient

# ── Logging ───────────────────────────────────────────────────────────────────
# Logs go to both the terminal and a log file so you have a record even
# if the terminal session closes before the job finishes.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log_path = OUTPUT_DIR / "preprocess_all.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="a"),   # append so reruns don't wipe history
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
cfg = PreprocessingConfig()

# ── Discover all patient directories ─────────────────────────────────────────
patient_dirs = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])
n_total = len(patient_dirs)

if n_total == 0:
    raise RuntimeError(
        f"No patient directories found under:\n  {DATA_ROOT}\n"
        "Check your DATA_ROOT path in config.py."
    )

log.info(f"Found {n_total} patient directories under {DATA_ROOT}")
log.info(f"Cache directory: {OUTPUT_DIR.resolve()}")
log.info(f"Log file: {log_path.resolve()}")
log.info("-" * 60)

# ── Counters and audit trail ──────────────────────────────────────────────────
results_log = []   # list of dicts — written to CSV at the end
n_success   = 0
n_skipped   = 0
n_failed    = 0
t_run_start = time.time()

# ── Main loop ─────────────────────────────────────────────────────────────────
for i, patient_dir in enumerate(patient_dirs):
    patient_id = patient_dir.name
    cache_path = OUTPUT_DIR / f"{patient_id}.pkl"

    # ── Skip already-cached patients ─────────────────────────────────────────
    # This means you can safely Ctrl+C and rerun — completed patients are not
    # reprocessed. The cache file is only written after successful completion,
    # so a crash mid-patient does not leave a corrupt pickle.
    if cache_path.exists():
        n_skipped += 1
        log.info(f"[{i+1:3d}/{n_total}] {patient_id}: already cached — skipping")
        results_log.append({"patient_id": patient_id, "status": "skipped", "error": ""})
        continue

    # ── Preprocess ────────────────────────────────────────────────────────────
    try:
        t0     = time.time()
        result = preprocess_patient(patient_dir, cfg)

        # Remove items not needed for training before pickling.
        # sct_sitk_ref is a SimpleITK object (large, not serialisable reliably).
        # crop_offsets are kept separately if you ever need NIfTI export later.
        result.pop("sct_sitk_ref", None)
        result.pop("crop_offsets",  None)

        # ── Write pickle atomically ───────────────────────────────────────────
        # We write to a temporary file first, then rename. This way, if the
        # process crashes mid-write, you never end up with a partial/corrupt
        # pickle that would silently pass the "already cached" check above.
        tmp_path = cache_path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(result, f)
        tmp_path.rename(cache_path)   # atomic on the same filesystem

        elapsed   = time.time() - t0
        n_success += 1

        # Estimate remaining time based on average time per completed patient
        n_done    = n_success + n_skipped
        avg_time  = (time.time() - t_run_start) / n_done
        remaining = avg_time * (n_total - i - 1)

        log.info(
            f"[{i+1:3d}/{n_total}] {patient_id}: OK  "
            f"shape={result['input'].shape}  "
            f"dose_max={result['dose'].max():.3f}  "
            f"time={elapsed:.1f}s  "
            f"remaining≈{remaining/60:.0f}min"
        )
        results_log.append({"patient_id": patient_id, "status": "success", "error": ""})

    except Exception as e:
        # Log the failure but continue with the next patient.
        # Failures are collected and summarised at the end.
        n_failed += 1
        log.error(f"[{i+1:3d}/{n_total}] {patient_id}: FAILED — {e}")
        results_log.append({"patient_id": patient_id, "status": "failed", "error": str(e)})


# ── Write audit CSV ───────────────────────────────────────────────────────────
# This CSV is your record of which patients were processed successfully.
# You'll reference it when building the train/val/test split — any patients
# in the "failed" column should be investigated before being included in training.
summary_path = OUTPUT_DIR / "preprocessing_summary.csv"
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["patient_id", "status", "error"])
    writer.writeheader()
    writer.writerows(results_log)

# ── Final summary ─────────────────────────────────────────────────────────────
total_time = time.time() - t_run_start
log.info("")
log.info("=" * 60)
log.info(f"Finished in {total_time / 60:.1f} minutes")
log.info(f"  Successful : {n_success}")
log.info(f"  Skipped    : {n_skipped}  (already cached from a previous run)")
log.info(f"  Failed     : {n_failed}")
log.info(f"  Summary CSV: {summary_path.resolve()}")

if n_failed > 0:
    log.info("")
    log.info("Failed patients (investigate before training):")
    for row in results_log:
        if row["status"] == "failed":
            log.info(f"  {row['patient_id']}: {row['error']}")