# preprocessing/check_pickle_shapes.py
# Checks ALL pickles for correct shape before training.
# Usage: python -m preprocessing.check_pickle_shapes

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config_preprocessing_shared import OUTPUT_DIR

EXPECTED_INPUT = (9, 128, 256, 320)
EXPECTED_DOSE  = (128, 256, 320)
EXPECTED_GEOM  = (5, 128, 256, 320)


def main():
    pickles = sorted(OUTPUT_DIR.glob("*.pkl"))
    print(f"Total pickles found: {len(pickles)}\n")

    all_ok      = True
    bad_files   = []
    n_with_geom = 0
    n_bad_geom  = 0

    for path in pickles:
        with open(path, "rb") as f:
            data = pickle.load(f)

        input_shape = data["input"].shape
        dose_shape  = data["dose"].shape
        base_ok     = input_shape == EXPECTED_INPUT and dose_shape == EXPECTED_DOSE

        geom_shape = data["geom_channels"].shape if "geom_channels" in data else None
        geom_ok    = geom_shape == EXPECTED_GEOM if geom_shape is not None else True

        if geom_shape is not None:
            n_with_geom += 1
            if not geom_ok:
                n_bad_geom += 1

        if not base_ok or not geom_ok:
            all_ok = False
            bad_files.append(path.name)
            print(
                f"✗  {path.name}"
                f"  input:{input_shape}"
                f"  dose:{dose_shape}"
                + (f"  geom:{geom_shape}" if geom_shape is not None else "  geom:missing")
            )

    print(f"\nBaseline channels (9):  {len(pickles)} / {len(pickles)}")
    print(f"Geom channels (5):      {n_with_geom} / {len(pickles)}")
    if n_bad_geom:
        print(f"Bad geom shapes:        {n_bad_geom}")

    if all_ok:
        print(f"\nAll {len(pickles)} pickles have correct shape — safe to train.")
    else:
        print(f"\n{len(bad_files)} pickle(s) have wrong shape — do NOT train yet.")
        sys.exit(1)


if __name__ == "__main__":
    main()
