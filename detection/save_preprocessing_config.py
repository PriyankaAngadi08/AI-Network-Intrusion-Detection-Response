
from pathlib import Path
import json

import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "models" / "preprocessing"

SCALER_FILE = ARTIFACT_DIR / "standard_scaler.joblib"
MEDIANS_FILE = ARTIFACT_DIR / "imputation_medians.json"
FEATURES_FILE = ARTIFACT_DIR / "feature_columns.json"
CONFIG_FILE = ARTIFACT_DIR / "preprocessing_config.json"


def main():
    print("=== Task 18: Save Preprocessing Configuration ===")

    # Confirm all previously saved artifacts exist.
    for path in [SCALER_FILE, MEDIANS_FILE, FEATURES_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required preprocessing artifact not found: {path}"
            )

    with open(FEATURES_FILE, "r", encoding="utf-8") as file:
        features = json.load(file)

    with open(MEDIANS_FILE, "r", encoding="utf-8") as file:
        medians = json.load(file)

    scaler = joblib.load(SCALER_FILE)

    # Verify consistency between the saved artifacts.
    if len(features) != 70:
        raise ValueError(
            f"Expected 70 features, but found {len(features)}."
        )

    if len(set(features)) != len(features):
        raise ValueError("Duplicate feature names found.")

    if set(features) != set(medians):
        raise ValueError(
            "The saved feature list and median keys do not match."
        )

    if getattr(scaler, "n_features_in_", None) != len(features):
        raise ValueError(
            "Scaler feature count does not match the feature list."
        )

    if not all(
        isinstance(value, (int, float)) for value in medians.values()
    ):
        raise ValueError("Invalid median values found.")

    config = {
        "dataset": "CIC-IDS2017",
        "target_column": " Label",
        "original_feature_count": 78,
        "constant_features_removed": 8,
        "selected_feature_count": len(features),
        "selected_features": features,
        "missing_value_strategy": "Training-set median imputation",
        "infinite_value_strategy": "Replace positive and negative infinity with missing values",
        "invalid_negative_value_strategy": (
        "Replace negative values in Flow Duration,   Flow Packets/s, "
        "Flow Bytes/s, min_seg_size_forward, and Bwd Header Length "
        "with missing values; then impute using training-set medians"
        ),
        "scaling_method": "StandardScaler",
        "scaler_fitted_on": "Training split only",
        "imputation_medians_fitted_on": "Training split only",
        "split_ratios": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15
        },
        "chunk_size": 10000,
        "artifacts": {
            "scaler": "standard_scaler.joblib",
            "medians": "imputation_medians.json",
            "features": "feature_columns.json"
        }
    }

    with open(CONFIG_FILE, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print(f"Selected features verified: {len(features)}")
    print(f"Median values verified: {len(medians)}")
    print(f"Scaler input features: {scaler.n_features_in_}")
    print(f"Configuration saved to: {CONFIG_FILE}")
    print("Task 18 configuration saved successfully.")


if __name__ == "__main__":
    main()
