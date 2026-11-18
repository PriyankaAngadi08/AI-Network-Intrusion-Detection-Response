
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
CONFIG_DIR = BASE_DIR / "models" / "preprocessing"

files = {
    "training": DATA_DIR / "training_processed.csv",
    "validation": DATA_DIR / "validation_processed.csv",
    "test": DATA_DIR / "test_processed.csv",
}

with open(CONFIG_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
    saved_features = json.load(f)

if isinstance(saved_features, dict):
    saved_features = saved_features.get(
        "feature_columns",
        saved_features.get("features")
    )

if not isinstance(saved_features, list):
    raise ValueError(
        "Could not identify the feature list in feature_columns.json. "
        "Inspect the JSON structure before continuing."
    )

reference_columns = None

for split_name, file_path in files.items():
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    # Read only the CSV header; do not load the dataset.
    columns = pd.read_csv(file_path, nrows=0).columns.tolist()

    print(f"\n--- {split_name.upper()} ---")
    print("Total columns:", len(columns))
    print("Last column:", repr(columns[-1]))

    target_candidates = [
        col for col in columns
        if col.strip() == "Label"
    ]

    if len(target_candidates) != 1:
        raise ValueError(
            f"Expected exactly one target column, found: "
            f"{target_candidates}"
        )

    target = target_candidates[0]
    feature_columns = [col for col in columns if col != target]

    print("Target column:", repr(target))
    print("Feature count:", len(feature_columns))
    print("Matches saved feature count:", len(feature_columns) == len(saved_features))
    print("Feature order matches saved list:", feature_columns == saved_features)

    if len(feature_columns) != len(saved_features):
        raise ValueError(f"{split_name}: feature count mismatch")

    if feature_columns != saved_features:
        raise ValueError(f"{split_name}: feature order mismatch")

    if reference_columns is None:
        reference_columns = columns
    elif columns != reference_columns:
        raise ValueError(f"{split_name}: columns differ from training")

print("\nSUCCESS: All split headers match the saved feature list.")
