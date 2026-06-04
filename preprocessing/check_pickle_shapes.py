# preprocessing/check_pickle_shapes.py
# Checks ALL pickles for correct shape before training.
#
# Usage:
#   python -m preprocessing.check_pickle_shapes              # base shapes only
#   python -m preprocessing.check_pickle_shapes --require-geom  # also require geom_channels

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config_preprocessing_shared import OUTPUT_DIR

EXPECTED_INPUT = (9, 128, 256, 320)
EXPECTED_DOSE  = (128, 256, 320)
EXPECTED_GEOM  = (5, 128, 256, 320)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-geom", dest="require_geom", action="store_true",
        help="Fail if any pickle is missing geom_channels or has wrong geom shape.",
    )
    args = parser.parse_args()

    pickles = sorted(OUTPUT_DIR.glob("*.pkl"))
    print(f"Total pickles found: {len(pickles)}")
    if args.require_geom:
        print("Mode: base + geom required\n")
    else:
        print("Mode: base only (pass --require-geom to also check geom_channels)\n")

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

        if geom_shape is not None:
            n_with_geom += 1
            geom_ok = geom_shape == EXPECTED_GEOM
            if not geom_ok:
                n_bad_geom += 1
        else:
            # Missing geom_channels is only a failure when --require-geom is set.
            geom_ok = not args.require_geom

        if not base_ok or not geom_ok:
            all_ok = False
            bad_files.append(path.name)
            geom_note = f"geom:{geom_shape}" if geom_shape is not None else "geom:MISSING"
            print(f"FAIL  {path.name}  input:{input_shape}  dose:{dose_shape}  {geom_note}")

    print(f"\nBaseline channels (9):  {len(pickles)} / {len(pickles)}")
    print(f"Geom channels (5):      {n_with_geom} / {len(pickles)}"
          + (" (required)" if args.require_geom else ""))
    if n_bad_geom:
        print(f"Bad geom shapes:        {n_bad_geom}")

    if all_ok:
        print(f"\nAll {len(pickles)} pickles OK — safe to train.")
    else:
        print(f"\n{len(bad_files)} pickle(s) failed — do NOT train yet.")
        sys.exit(1)


if __name__ == "__main__":
    main()
