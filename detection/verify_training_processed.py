
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "training_processed.csv"
ARTIFACT_FILE = ROOT / "models" / "preprocessing" / "feature_columns.json"

EXPECTED_ROWS = 1_981_520
TARGET = " Label"
CHUNK_SIZE = 10_000


def main():
    with open(ARTIFACT_FILE, encoding="utf-8") as file:
        features = json.load(file)

    expected_columns = features + [TARGET]

    total_rows = 0
    missing_features = 0
    infinite_features = 0
    missing_labels = 0
    label_counts = {}

    print("=== Verify Saved Training Dataset ===")

    for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
        if chunk.columns.tolist() != expected_columns:
            raise ValueError("Feature order or columns do not match.")

        X = chunk[features].to_numpy(dtype=np.float64)

        missing_features += int(np.isnan(X).sum())
        infinite_features += int(np.isinf(X).sum())
        missing_labels += int(chunk[TARGET].isna().sum())

        label_counts_chunk = chunk[TARGET].value_counts()
        for label, count in label_counts_chunk.items():
            label_counts[label] = (
                label_counts.get(label, 0) + int(count)
            )

        total_rows += len(chunk)

    print(f"Rows: {total_rows:,}")
    print(f"Feature columns: {len(features)}")
    print(f"Total columns: {len(expected_columns)}")
    print(f"Distinct target labels: {len(label_counts)}")
    print(f"Missing feature values: {missing_features}")
    print(f"Infinite feature values: {infinite_features}")
    print(f"Missing target labels: {missing_labels}")

    print("\nTarget label counts:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count:,}")

    checks_passed = (
        total_rows == EXPECTED_ROWS
        and missing_features == 0
        and infinite_features == 0
        and missing_labels == 0
        and len(label_counts) == 15
    )

    if checks_passed:
        print("\nRESULT: PASS - Training dataset verified.")
    else:
        print("\nRESULT: FAIL - Review the checks above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
