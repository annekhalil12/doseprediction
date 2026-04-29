"""
test_single_patient.py
======================
Sanity-check the preprocessing pipeline on a single patient before
running the full 432-patient batch.

Run from the preprocessing/ directory:
    python test_single_patient.py

What this script checks
-----------------------
1. The pipeline runs without errors on one patient.
2. The output tensor has the expected shape (9, 128, 256, 320).
3. The dose array is in [0, 1] and has the expected shape.
4. Each structure mask is binary (only 0s and 1s).
5. The sCT channel has a sensible z-score distribution (mean ≈ 0 inside body).
6. Optional structures (Genitalia, PenileBulb) are handled gracefully.
7. Saves the result as a .pkl and reloads it to confirm serialisation works.

This script runs entirely on your LAPTOP — no GPU needed.
SimpleITK does CPU-based resampling; the output is plain NumPy arrays.
"""

import pickle
import logging
import sys
from pathlib import Path

import numpy as np

# Add the preprocessing folder to the Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.preprocessing_config import DATA_ROOT, OUTPUT_DIR, PreprocessingConfig, CHANNEL_MAP, AVAILABLE_CHANNELS
from preprocessing import preprocess_patient

# ── Logging setup ─────────────────────────────────────────────────────────────
# INFO level shows crop anchor positions and normalisation stats per patient.
# Change to logging.DEBUG for more detail.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Pick one patient to test on ───────────────────────────────────────────────
# We sort the patient directories and take the first one. This is deterministic
# so you'll always get the same patient when you re-run the script.
patient_dirs = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])
if not patient_dirs:
    raise RuntimeError(f"No patient directories found under {DATA_ROOT}. Check your DATA_ROOT path.")

test_patient = patient_dirs[0]
log.info(f"Testing on patient: {test_patient.name}")
log.info(f"Full path: {test_patient}")


# ── Run the preprocessing pipeline ───────────────────────────────────────────
cfg = PreprocessingConfig()
log.info("Running preprocess_patient() — this takes ~10–30 seconds on a laptop...")

result = preprocess_patient(test_patient, cfg)

log.info("Preprocessing complete.")


# ── Sanity checks ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)

# 1. Input tensor shape
input_tensor = result["input"]
expected_shape = (9, 128, 256, 320)
shape_ok = input_tensor.shape == expected_shape
print(f"\n[{'OK' if shape_ok else 'FAIL'}] Input tensor shape: {input_tensor.shape}  (expected {expected_shape})")

# 2. Input tensor dtype
dtype_ok = input_tensor.dtype == np.float32
print(f"[{'OK' if dtype_ok else 'FAIL'}] Input tensor dtype: {input_tensor.dtype}  (expected float32)")

# 3. Dose shape and range
dose = result["dose"]
dose_shape_ok = dose.shape == (128, 256, 320)
dose_range_ok = dose.min() >= 0.0 and dose.max() <= 1.0
print(f"\n[{'OK' if dose_shape_ok else 'FAIL'}] Dose shape: {dose.shape}  (expected (128, 256, 320))")
print(f"[{'OK' if dose_range_ok else 'FAIL'}] Dose range: [{dose.min():.4f}, {dose.max():.4f}]  (expected [0, 1])")
print(f"      Dose scale factor: {result['dose_scale']} Gy  (multiply predictions by this to recover Gy values)")

# 4. Structure mask channels — should be binary (only 0.0 and 1.0)
print("\n  Structure mask channels (channels 0–7):")
for ch_idx in range(8):
    ch_name  = CHANNEL_MAP[ch_idx]
    ch_data  = input_tensor[ch_idx]
    unique   = np.unique(ch_data)
    is_binary = set(unique).issubset({0.0, 1.0})
    voxels_on = int(ch_data.sum())
    # For optional structures, zero-fill is expected and correct
    if voxels_on == 0 and ch_name in cfg.optional_structures:
        status = "OK (zero-filled — optional structure missing)"
    elif is_binary:
        status = f"OK  ({voxels_on:,} voxels active)"
    else:
        status = f"FAIL — non-binary values found: {unique[:6]}"
    print(f"  [{ch_idx:2d}] {ch_name:<20s}: {status}")

# 5. sCT channel (channel index 8 in the 9-channel tensor, i.e. AVAILABLE_CHANNELS index)
# In CHANNEL_MAP this is channel 15, but in our 9-channel tensor it's the last (index 8).
sct_channel = input_tensor[8]
body_mask   = result["ptv_mask"]   # use PTV as a rough proxy to confirm non-trivial values
sct_nonzero = sct_channel[sct_channel != 0]
print(f"\n  sCT intensity channel (index 8 in 9-ch tensor, channel 15 in full CHANNEL_MAP):")
print(f"    Non-zero voxel count : {len(sct_nonzero):,}")
print(f"    Mean (non-zero)      : {sct_nonzero.mean():.3f}  (z-scored inside body, so ≈ 0 expected)")
print(f"    Std  (non-zero)      : {sct_nonzero.std():.3f}   (should be close to 1.0)")
print(f"    Range                : [{sct_nonzero.min():.2f}, {sct_nonzero.max():.2f}]")

# 6. Geometric channels pending flag
geom_flag = result.get("geometric_channels_pending", False)
print(f"\n[{'OK' if geom_flag else 'NOTE'}] geometric_channels_pending = {geom_flag}")
print(f"      Channels 8–14 will be appended once V5geometric_channels.py is available.")

# 7. Patient ID preserved
print(f"\n  Patient ID in result : '{result['patient_id']}'")
print(f"  Expected             : '{test_patient.name}'")


# ── Save and reload pickle ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PICKLE SERIALISATION TEST")
print("=" * 60)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
cache_path = OUTPUT_DIR / f"{test_patient.name}.pkl"

# Remove the SimpleITK reference object before pickling — sitk.Image objects
# are not needed for training and significantly increase file size.
result_to_save = {k: v for k, v in result.items() if k not in ("sct_sitk_ref", "crop_offsets")}

with open(cache_path, "wb") as f:
    pickle.dump(result_to_save, f)

file_size_mb = cache_path.stat().st_size / (1024 ** 2)
print(f"\nSaved to : {cache_path}")
print(f"File size: {file_size_mb:.1f} MB")

# Reload and confirm the tensor survives the round-trip intact
with open(cache_path, "rb") as f:
    reloaded = pickle.load(f)

arrays_match = np.array_equal(reloaded["input"], result["input"])
print(f"[{'OK' if arrays_match else 'FAIL'}] Reloaded tensor matches original: {arrays_match}")

print("\n" + "=" * 60)
print("Done. If all checks show [OK], the pipeline is working correctly.")
print("Next step: run preprocess_all.py on the full cohort (on OneView).")
print("=" * 60)