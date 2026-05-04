"""
create_split.py
===============
Generate a reproducible, stratified k-fold split for the
LUND-PROBE cohort and save it as a CSV artefact.

Run from the preprocessing/ directory:
    python create_split.py

Why this script exists as a separate step
-----------------------------------------
The split is created ONCE and saved to disk before any model code is written.
This matters for two reasons:

  1. Scientific integrity: if the split were generated inside the training
     script, a different random seed or a code change could silently reassign
     patients across sets. Saving it as a fixed file means the boundary between
     training and evaluation data is locked and auditable.

  2. Reproducibility: any collaborator (or examiner) can re-run training from
     scratch and be guaranteed to use exactly the same patients in each set,
     because the split CSV is committed to the repository.

Stratification by acquisition group
-------------------------------------
LUND-PROBE contains two acquisition groups (oldAcq, newAcq) in a ~75/25
ratio. A random split without stratification risks skewing this ratio in
any individual fold, which would mean validation performance in that fold
is not representative of the full cohort.

This script uses a two-stage stratified approach:

  1. A fixed 15% held-out test set is carved out first. It is stratified
     by acquisition group and never used during cross-validation. This
     preserves an honest, unbiased final evaluation.

  2. The remaining 85% of patients are split into 5 folds using
     StratifiedKFold, which ensures every fold maintains the cohort-level
     ~75/25 oldAcq/newAcq ratio. During training, one fold at a time serves
     as the validation set while the other four are used for training.

Output
------
  <OUTPUT_DIR>/split.csv   — one row per patient with columns:
      patient_id, acquisition_group, is_test (bool), fold (0–4 or empty for test)

  Test patients have is_test=True and fold is empty.
  Non-test patients have is_test=False and fold in {0, 1, 2, 3, 4}.

Commit this file to the repository so the split is version-controlled.
"""

import sys
import csv
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from configs.config_preprocessing_shared import OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Split parameters
# ---------------------------------------------------------------------------
TEST_FRAC = 0.15   # 15% of all patients held out as a fixed test set
N_FOLDS   = 5      # number of cross-validation folds on the remaining 85%
SEED      = 42     # fixed seed → reproducible splits every run

# ---------------------------------------------------------------------------
# Load the preprocessing summary to get the list of successful patients
# ---------------------------------------------------------------------------
# We only include patients that preprocessed successfully. Any failed patients
# recorded in preprocessing_summary.csv are excluded — you don't want to
# silently try to load a missing pickle during training.
summary_path = OUTPUT_DIR / "preprocessing_summary.csv"
if not summary_path.exists():
    raise FileNotFoundError(
        f"preprocessing_summary.csv not found at {summary_path}.\n"
        "Run preprocess_all.py first."
    )

successful_patients = []
with open(summary_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["status"] == "success":
            successful_patients.append(row["patient_id"])

log.info(f"Patients with successful preprocessing: {len(successful_patients)}")

# ---------------------------------------------------------------------------
# Infer acquisition group from patient ID prefix
# ---------------------------------------------------------------------------
# Every LUND-PROBE patient folder is named either oldAcq_<hash> or newAcq_<hash>.
# We extract the group label directly from the folder name rather than relying
# on an external metadata file, which keeps this script self-contained.
def get_acquisition_group(patient_id: str) -> str:
    if patient_id.startswith("oldAcq_"):
        return "oldAcq"
    elif patient_id.startswith("newAcq_"):
        return "newAcq"
    else:
        raise ValueError(
            f"Cannot infer acquisition group from patient ID: '{patient_id}'. "
            "Expected prefix 'oldAcq_' or 'newAcq_'."
        )

groups = {pid: get_acquisition_group(pid) for pid in successful_patients}

old_acq_patients = [pid for pid, g in groups.items() if g == "oldAcq"]
new_acq_patients = [pid for pid, g in groups.items() if g == "newAcq"]

log.info(f"  oldAcq patients: {len(old_acq_patients)}")
log.info(f"  newAcq patients: {len(new_acq_patients)}")

# ---------------------------------------------------------------------------
# Stage 1: Carve out the fixed held-out test set
# ---------------------------------------------------------------------------
# The test set is separated BEFORE any cross-validation fold assignment.
# This ensures test patients are never seen during training or fold-level
# validation — preserving an honest final evaluation.
# stratify= ensures the 75/25 oldAcq/newAcq ratio is maintained in the test set.
all_patient_ids = list(groups.keys())
acq_labels      = [groups[pid] for pid in all_patient_ids]

train_val_ids, test_ids = train_test_split(
    all_patient_ids,
    test_size=TEST_FRAC,
    stratify=acq_labels,
    random_state=SEED,
)

log.info("")
log.info(f"Test set (held out): {len(test_ids)} patients ({len(test_ids)/len(all_patient_ids)*100:.1f}%)")
log.info(f"Train/val pool:      {len(train_val_ids)} patients ({len(train_val_ids)/len(all_patient_ids)*100:.1f}%)")

# ---------------------------------------------------------------------------
# Stage 2: Assign k-fold numbers to the train/val pool
# ---------------------------------------------------------------------------
# StratifiedKFold divides the train/val pool into N_FOLDS chunks, each
# maintaining the cohort-level acquisition group ratio. The fold number
# tells downstream code which patients form the validation set for that
# fold — everything else in the pool is training data for that fold.
train_val_acq_labels = [groups[pid] for pid in train_val_ids]

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# fold_map stores: patient_id -> fold index (0 through N_FOLDS-1)
fold_map = {}
for fold_idx, (_, val_indices) in enumerate(
    skf.split(train_val_ids, train_val_acq_labels)
):
    # val_indices are positional indices into train_val_ids
    for i in val_indices:
        fold_map[train_val_ids[i]] = fold_idx

log.info("")
log.info("Fold composition (train/val pool only):")
for fold_idx in range(N_FOLDS):
    fold_patients = [pid for pid, f in fold_map.items() if f == fold_idx]
    n_old = sum(1 for pid in fold_patients if groups[pid] == "oldAcq")
    n_new = sum(1 for pid in fold_patients if groups[pid] == "newAcq")
    log.info(
        f"  Fold {fold_idx}: {len(fold_patients)} patients | "
        f"oldAcq={n_old} ({n_old/len(fold_patients)*100:.1f}%)  "
        f"newAcq={n_new} ({n_new/len(fold_patients)*100:.1f}%)"
    )

log.info("")
log.info("Test set acquisition breakdown:")
n_old = sum(1 for pid in test_ids if groups[pid] == "oldAcq")
n_new = sum(1 for pid in test_ids if groups[pid] == "newAcq")
log.info(
    f"  oldAcq={n_old} ({n_old/len(test_ids)*100:.1f}%)  "
    f"newAcq={n_new} ({n_new/len(test_ids)*100:.1f}%)"
)

# ---------------------------------------------------------------------------
# Save the split CSV
# ---------------------------------------------------------------------------
# Sort by patient ID for a deterministic, human-readable file.
rows = []
for pid in sorted(successful_patients):
    is_test = pid in set(test_ids)
    rows.append({
        "patient_id":        pid,
        "acquisition_group": groups[pid],
        "is_test":           is_test,
        "fold":              "" if is_test else fold_map[pid],
    })

split_path = OUTPUT_DIR / "split.csv"
with open(split_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["patient_id", "acquisition_group", "is_test", "fold"]
    )
    writer.writeheader()
    writer.writerows(rows)

log.info("")
log.info(f"Split saved to: {split_path.resolve()}")
log.info("Commit this file to your repository — it is the permanent record of your data split.")