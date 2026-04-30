"""
create_split.py
===============
Generate a reproducible, stratified train / val / test split for the
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
the test set, which would mean evaluation results are not representative
of the full cohort. Stratification applies the 70/15/15 split independently
within each group so that each set inherits the cohort-level ratio.

Output
------
  <OUTPUT_DIR>/split.csv   — one row per patient with columns:
      patient_id, acquisition_group, split (train / val / test)

Commit this file to the repository so the split is version-controlled.
"""

import sys
import csv
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing_config import OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Split parameters
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.70   # 70% training
VAL_FRAC   = 0.15   # 15% validation
TEST_FRAC  = 0.15   # 15% test  (= 1 - TRAIN_FRAC - VAL_FRAC)
SEED       = 42     # fixed seed → reproducible splits every run

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
# Stratified split — applied independently within each acquisition group
# ---------------------------------------------------------------------------
# We perform the split in two stages:
#   Stage 1: separate train from (val + test) using TRAIN_FRAC
#   Stage 2: split the remaining pool 50/50 into val and test
#            (since VAL_FRAC == TEST_FRAC == 0.15, the remainder is 0.30,
#            and 0.15 / 0.30 = 0.50)
# This is applied identically to oldAcq and newAcq, then combined.

val_of_remainder = VAL_FRAC / (VAL_FRAC + TEST_FRAC)  # = 0.5

split_assignments = {}   # patient_id -> "train" | "val" | "test"

for group_name, patient_list in [("oldAcq", old_acq_patients),
                                   ("newAcq", new_acq_patients)]:

    # Stage 1: train vs remainder
    train_ids, remainder_ids = train_test_split(
        patient_list,
        test_size=(1.0 - TRAIN_FRAC),
        random_state=SEED,
    )

    # Stage 2: val vs test from the remainder
    val_ids, test_ids = train_test_split(
        remainder_ids,
        test_size=(1.0 - val_of_remainder),
        random_state=SEED,
    )

    for pid in train_ids:
        split_assignments[pid] = "train"
    for pid in val_ids:
        split_assignments[pid] = "val"
    for pid in test_ids:
        split_assignments[pid] = "test"

    log.info(
        f"  {group_name}: {len(train_ids)} train | "
        f"{len(val_ids)} val | {len(test_ids)} test"
    )

# ---------------------------------------------------------------------------
# Summary counts and ratio check
# ---------------------------------------------------------------------------
train_pids = [pid for pid, s in split_assignments.items() if s == "train"]
val_pids   = [pid for pid, s in split_assignments.items() if s == "val"]
test_pids  = [pid for pid, s in split_assignments.items() if s == "test"]

log.info("")
log.info("Overall split:")
log.info(f"  Train : {len(train_pids):3d} patients ({len(train_pids)/len(successful_patients)*100:.1f}%)")
log.info(f"  Val   : {len(val_pids):3d} patients ({len(val_pids)/len(successful_patients)*100:.1f}%)")
log.info(f"  Test  : {len(test_pids):3d} patients ({len(test_pids)/len(successful_patients)*100:.1f}%)")
log.info("")

# Verify acquisition group ratios are preserved across sets
for set_name, pid_list in [("Train", train_pids), ("Val", val_pids), ("Test", test_pids)]:
    n_old = sum(1 for pid in pid_list if groups[pid] == "oldAcq")
    n_new = sum(1 for pid in pid_list if groups[pid] == "newAcq")
    log.info(
        f"  {set_name:5s} acquisition breakdown: "
        f"oldAcq={n_old} ({n_old/len(pid_list)*100:.1f}%)  "
        f"newAcq={n_new} ({n_new/len(pid_list)*100:.1f}%)"
    )

# ---------------------------------------------------------------------------
# Save the split CSV
# ---------------------------------------------------------------------------
# Sort by patient ID for a deterministic, human-readable file.
split_path = OUTPUT_DIR / "split.csv"
rows = sorted(
    [
        {
            "patient_id":        pid,
            "acquisition_group": groups[pid],
            "split":             split_assignments[pid],
        }
        for pid in successful_patients
    ],
    key=lambda r: r["patient_id"],
)

with open(split_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["patient_id", "acquisition_group", "split"])
    writer.writeheader()
    writer.writerows(rows)

log.info("")
log.info(f"Split saved to: {split_path.resolve()}")
log.info("Commit this file to your repository — it is the permanent record of your data split.")