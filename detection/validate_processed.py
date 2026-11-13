
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "processed"
)

TARGET_COLUMN = " Label"
CHUNK_SIZE = 10_000

files = {
    "validation_processed.csv": 424611,
    "test_processed.csv": 424612,
}

for filename, expected_rows in files.items():
    file_path = DATA_DIR / filename

    print(f"\n=== Checking {filename} ===")

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    total_rows = 0
    missing_features = 0
    infinite_features = 0
    missing_targets = 0
    feature_count = None
    labels = set()

    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
        if TARGET_COLUMN not in chunk.columns:
            raise ValueError("Target column is missing.")

        features = chunk.drop(columns=[TARGET_COLUMN])

        if feature_count is None:
            feature_count = len(features.columns)

        # Count missing feature values.
        missing_features += int(features.isna().sum().sum())

        # Check for positive or negative infinity.
        values = features.to_numpy(dtype=np.float64)
        infinite_features += int(np.isinf(values).sum())

        # Check for missing target labels.
        missing_targets += int(chunk[TARGET_COLUMN].isna().sum())

        labels.update(chunk[TARGET_COLUMN].dropna().unique())

        total_rows += len(chunk)

    print(f"Expected rows: {expected_rows:,}")
    print(f"Actual rows: {total_rows:,}")
    print(f"Feature columns: {feature_count}")
    print(f"Total columns: {feature_count + 1}")
    print(f"Missing feature values: {missing_features:,}")
    print(f"Infinite feature values: {infinite_features:,}")
    print(f"Missing target labels: {missing_targets:,}")
    print(f"Distinct target labels: {len(labels)}")

    if total_rows != expected_rows:
        print("RESULT: FAIL - Row count mismatch.")
    elif feature_count != 70:
        print("RESULT: FAIL - Feature count mismatch.")
    elif missing_features != 0 or infinite_features != 0:
        print("RESULT: FAIL - Invalid feature values found.")
    elif missing_targets != 0:
        print("RESULT: FAIL - Missing target labels found.")
    else:
        print("RESULT: PASS - All checks passed.")

print("\nFull dataset validation finished.")
