
from pathlib import Path
import json

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "models" / "preprocessing"

FEATURES_TO_CHECK = [
    " min_seg_size_forward",
    " Bwd Header Length",
]


def main():
    with open(
        ARTIFACT_DIR / "feature_columns.json",
        "r",
        encoding="utf-8",
    ) as file:
        feature_columns = json.load(file)

    with open(
        ARTIFACT_DIR / "imputation_medians.json",
        "r",
        encoding="utf-8",
    ) as file:
        medians = json.load(file)

    scaler = joblib.load(
        ARTIFACT_DIR / "standard_scaler.joblib"
    )

    for feature in FEATURES_TO_CHECK:
        if feature not in feature_columns:
            print(f"\nFeature not found: {feature}")
            continue

        index = feature_columns.index(feature)

        print(f"\n=== {feature} ===")
        print("Training median:", medians[feature])
        print("Training mean:", float(scaler.mean_[index]))
        print("Training standard deviation:", float(scaler.scale_[index]))
        print(
            "Training variance:",
            float(scaler.var_[index]),
        )

        if scaler.scale_[index] == 0:
            print("WARNING: Scaler has zero scale.")
        elif not np.isfinite(scaler.scale_[index]):
            print("WARNING: Scaler scale is not finite.")
        else:
            print("Scaler statistics are finite and nonzero.")

    print("\nInspection finished.")


if __name__ == "__main__":
    main()
