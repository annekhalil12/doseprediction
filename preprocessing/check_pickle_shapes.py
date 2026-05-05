# preprocessing/check_pickle_shapes.py
# Checks ALL pickles for correct shape before training.
# Usage: python -m preprocessing.check_pickle_shapes

import pickle
from pathlib import Path

pickle_dir = Path("outputs/pickles")
pickles    = sorted(pickle_dir.glob("*.pkl"))

print(f"Total pickles found: {len(pickles)}\n")

all_ok    = True
bad_files = []

for path in pickles:
    with open(path, "rb") as f:
        data = pickle.load(f)

    input_shape = data["input"].shape
    dose_shape  = data["dose"].shape
    ok          = input_shape == (9, 128, 256, 320) and dose_shape == (128, 256, 320)

    if not ok:
        all_ok = False
        bad_files.append((path.name, input_shape, dose_shape))
        print(f"✗  {path.name}  input:{input_shape}  dose:{dose_shape}")

if all_ok:
    print(f"All {len(pickles)} pickles have correct shape — safe to train.")
else:
    print(f"\n{len(bad_files)} pickle(s) have wrong shape — do NOT train yet.")