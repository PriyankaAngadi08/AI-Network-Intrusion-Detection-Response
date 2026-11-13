
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ==================================================
# 1. Paths and configuration
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = Path(
    r"D:\Downloads\MachineLearningCSV\MachineLearningCVE"
)

ASSIGNMENTS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "split_assignments.csv"
)

ARTIFACT_DIR = PROJECT_ROOT / "models" / "preprocessing"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COLUMN = " Label"
CHUNK_SIZE = 10_000

NONNEGATIVE_FEATURES = [
    " Flow Duration",
    " Flow Packets/s",
    "Flow Bytes/s",
    " min_seg_size_forward",
    " Bwd Header Length",
]


# ==================================================
# 2. Load the training-only artifacts
# ==================================================

def main():
    print("=== Task 16: Transform Validation and Test Data ===")

    assignments = pd.read_csv(ASSIGNMENTS_FILE)

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

    if len(feature_columns) != len(medians):
        raise ValueError("Feature and median counts do not match.")

    print(f"Selected features: {len(feature_columns)}")
    print("Loaded saved medians and fitted scaler.")

    # Store transformed features and labels separately
    # for validation and test.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_files = {
        "validation": OUTPUT_DIR / "validation_processed.csv",
        "test": OUTPUT_DIR / "test_processed.csv",
    }

    for path in output_files.values():
        if path.exists():
            path.unlink()

    # Preserve the same global row order used when
    # creating split_assignments.csv.
    filenames = sorted(
        path.name for path in DATASET_DIR.glob("*.csv")
    )

    expected_files = sorted(assignments["source_file"].unique())

    if filenames != expected_files:
        raise ValueError(
            "Dataset filenames do not match split assignments."
        )

    global_offset = 0
    transformed_counts = {"validation": 0, "test": 0}

    for filename in filenames:
        print(f"Processing: {filename}")

        file_path = DATASET_DIR / filename

        file_assignments = assignments.loc[
            assignments["source_file"] == filename
        ].sort_values("global_row_index")

        expected_indices = np.arange(
            global_offset,
            global_offset + len(file_assignments),
        )

        if not np.array_equal(
            file_assignments["global_row_index"].to_numpy(),
            expected_indices,
        ):
            raise ValueError(
                f"Row alignment mismatch in {filename}"
            )

        local_position = 0

        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
        ):
            start = local_position
            end = start + len(chunk)

            split_values = file_assignments["split"].iloc[
                start:end
            ].to_numpy()

            if len(split_values) != len(chunk):
                raise ValueError(
                    f"Assignment count mismatch in {filename}"
                )

            local_position = end

            for split_name in ("validation", "test"):
                mask = split_values == split_name

                if not mask.any():
                    continue

                selected = chunk.loc[mask]

                X = selected[feature_columns].copy()

                # Convert infinite values to missing values.
                X = X.replace([np.inf, -np.inf], np.nan)

                # Apply the same invalid-value rules as Task 15.
                for column in NONNEGATIVE_FEATURES:
                    if column in X.columns:
                        X.loc[X[column] < 0, column] = np.nan

                # Use training medians; do not recalculate them.
                X = X.fillna(medians)

                if X.isna().any().any():
                    raise ValueError(
                        f"Missing values remain in {split_name}."
                    )

                values = X.to_numpy(dtype=np.float64)

                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Invalid values remain in {split_name}."
                    )

                # Transform only. Never fit on validation/test.
                scaled_values = scaler.transform(values)

                if not np.isfinite(scaled_values).all():
                    raise ValueError(
                        f"Non-finite scaled values in {split_name}."
                    )

                processed = pd.DataFrame(
                    scaled_values,
                    columns=feature_columns,
                )

                # Retain the original class labels for evaluation.
                processed[TARGET_COLUMN] = (
                    selected[TARGET_COLUMN].to_numpy()
                )

                processed.to_csv(
                    output_files[split_name],
                    mode="a",
                    header=not output_files[split_name].exists()
                    or output_files[split_name].stat().st_size == 0,
                    index=False,
                )

                transformed_counts[split_name] += len(processed)

        if local_position != len(file_assignments):
            raise ValueError(
                f"Not all rows processed in {filename}"
            )

        global_offset += len(file_assignments)

    print("\n=== Task 16 Results ===")
    for split_name, count in transformed_counts.items():
        print(f"{split_name.title()} rows transformed: {count:,}")
        print(f"Saved to: {output_files[split_name]}")

    print("\nTraining scaler and medians were reused.")
    print("Validation and test data were not used for fitting.")
    print("Original dataset CSV files were not modified.")


if __name__ == "__main__":
    main()
