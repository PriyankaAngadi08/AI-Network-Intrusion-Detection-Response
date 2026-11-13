
from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "preprocessing"

TARGET_COLUMN = " Label"
CHUNK_SIZE = 100_000

FILES = [
    "validation_processed.csv",
    "test_processed.csv",
]


def main():
    # Load the feature order saved during preprocessing.
    with open(
        ARTIFACT_DIR / "feature_columns.json",
        "r",
        encoding="utf-8",
    ) as file:
        expected_features = json.load(file)

    for filename in FILES:
        file_path = DATA_DIR / filename

        print(f"\n=== Checking {filename} ===")

        # Read only the header to check column order.
        actual_columns = pd.read_csv(
            file_path,
            nrows=0,
        ).columns.tolist()

        actual_features = [
            column for column in actual_columns
            if column != TARGET_COLUMN
        ]

        print("Feature count:", len(actual_features))
        print("Target column present:", TARGET_COLUMN in actual_columns)
        print(
            "Feature order matches:",
            actual_features == expected_features,
        )

        assert TARGET_COLUMN in actual_columns
        assert actual_features == expected_features

        # Count labels using chunks to conserve memory.
        label_counts = {}

        for chunk in pd.read_csv(
            file_path,
            usecols=[TARGET_COLUMN],
            chunksize=CHUNK_SIZE,
        ):
            counts = chunk[TARGET_COLUMN].value_counts()

            for label, count in counts.items():
                label_counts[label] = (
                    label_counts.get(label, 0) + int(count)
                )

        print("Distinct target labels:", len(label_counts))
        print("Total target records:", sum(label_counts.values()))

        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count:,}")

        assert len(label_counts) == 15, (
            f"Expected 15 classes, found {len(label_counts)}."
        )

        print("RESULT: PASS")

    print("\nFeature order and target-label checks passed.")


if __name__ == "__main__":
    main()
