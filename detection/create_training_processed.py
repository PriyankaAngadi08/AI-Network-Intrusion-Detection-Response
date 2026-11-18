
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

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "preprocessing"

ASSIGNMENTS_FILE = PROCESSED_DIR / "split_assignments.csv"
OUTPUT_FILE = PROCESSED_DIR / "training_processed.csv"

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
# 2. Load saved training-only artifacts
# ==================================================

def main():
    print("=== Task 21: Save Processed Training Data ===")

    with open(
        ARTIFACT_DIR / "feature_columns.json",
        encoding="utf-8",
    ) as file:
        feature_columns = json.load(file)

    with open(
        ARTIFACT_DIR / "imputation_medians.json",
        encoding="utf-8",
    ) as file:
        medians = json.load(file)

    scaler = joblib.load(
        ARTIFACT_DIR / "standard_scaler.joblib"
    )

    if feature_columns != list(medians.keys()):
        raise ValueError(
            "Feature order does not match the saved medians."
        )

    if scaler.n_features_in_ != len(feature_columns):
        raise ValueError(
            "Saved scaler feature count does not match."
        )

    if not DATASET_DIR.exists():
        raise FileNotFoundError(DATASET_DIR)

    assignments = pd.read_csv(ASSIGNMENTS_FILE)

    required = {
        "global_row_index",
        "source_file",
        "split",
    }

    if not required.issubset(assignments.columns):
        raise ValueError("Split assignment columns are missing.")

    if assignments[
        ["global_row_index", "source_file", "split"]
    ].isnull().any().any():
        raise ValueError("Missing values in split assignments.")

    if assignments["global_row_index"].duplicated().any():
        raise ValueError("Duplicate global row indices.")

    assignments = assignments.sort_values(
        "global_row_index"
    ).reset_index(drop=True)

    if assignments["global_row_index"].tolist() != list(
        range(len(assignments))
    ):
        raise ValueError("Global row indices are incomplete.")

    train_count = int(
        (assignments["split"] == "train").sum()
    )

    filenames = sorted(
        path.name for path in DATASET_DIR.glob("*.csv")
    )

    if sorted(assignments["source_file"].unique()) != filenames:
        raise ValueError(
            "Source files do not match split assignments."
        )

    # Verify feature columns and target exist in every file.
    for filename in filenames:
        columns = pd.read_csv(
            DATASET_DIR / filename,
            nrows=0,
        ).columns.tolist()

        if TARGET_COLUMN not in columns:
            raise ValueError(
                f"Target column missing in {filename}"
            )

        missing = set(feature_columns) - set(columns)

        if missing:
            raise ValueError(
                f"Features missing in {filename}: {missing}"
            )

    print(f"Training records expected: {train_count:,}")
    print(f"Features: {len(feature_columns)}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    # Do not silently overwrite an existing output.
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"{OUTPUT_FILE} already exists. "
            "Rename or remove it only if you intend to recreate it."
        )

    rows_written = 0
    global_offset = 0

    try:
        # Write incrementally to avoid loading the full dataset.
        for filename in filenames:
            print(f"Processing: {filename}")

            file_assignments = assignments.loc[
                assignments["source_file"] == filename
            ].sort_values("global_row_index")

            expected_indices = np.arange(
                global_offset,
                global_offset + len(file_assignments),
            )

            if not np.array_equal(
                file_assignments[
                    "global_row_index"
                ].to_numpy(),
                expected_indices,
            ):
                raise ValueError(
                    f"Row alignment mismatch in {filename}"
                )

            local_position = 0
            file_path = DATASET_DIR / filename

            for chunk in pd.read_csv(
                file_path,
                chunksize=CHUNK_SIZE,
            ):
                start = local_position
                end = start + len(chunk)

                split_values = file_assignments[
                    "split"
                ].iloc[start:end].to_numpy()

                if len(split_values) != len(chunk):
                    raise ValueError(
                        f"Assignment count mismatch in {filename}"
                    )

                local_position = end
                train_mask = split_values == "train"

                if not train_mask.any():
                    continue

                selected = chunk.loc[train_mask]
                X = selected[feature_columns].copy()

                # Apply the same cleaning rules as training.
                X = X.replace(
                    [np.inf, -np.inf],
                    np.nan,
                )

                for column in NONNEGATIVE_FEATURES:
                    if column in X.columns:
                        X.loc[X[column] < 0, column] = np.nan

                # Reuse training medians; never refit here.
                X = X.fillna(medians)

                if X.isna().any().any():
                    raise ValueError(
                        "Missing feature values remain."
                    )

                values = X.to_numpy(dtype=np.float64)

                if not np.isfinite(values).all():
                    raise ValueError(
                        "Non-finite feature values found."
                    )

                # Transform only; do not fit the scaler.
                scaled_values = scaler.transform(values)

                if not np.isfinite(scaled_values).all():
                    raise ValueError(
                        "Non-finite scaled values found."
                    )

                output_chunk = pd.DataFrame(
                    scaled_values,
                    columns=feature_columns,
                )

                # Preserve original target labels as text.
                output_chunk[TARGET_COLUMN] = (
                    selected[TARGET_COLUMN].to_numpy()
                )

                output_chunk.to_csv(
                    OUTPUT_FILE,
                    mode="a",
                    header=(rows_written == 0),
                    index=False,
                )

                rows_written += len(output_chunk)

            if local_position != len(file_assignments):
                raise ValueError(
                    f"Not all rows processed in {filename}"
                )

            global_offset += len(file_assignments)

        if rows_written != train_count:
            raise ValueError(
                f"Expected {train_count:,} rows, "
                f"but wrote {rows_written:,}."
            )

        print("\n=== Task 21 Results ===")
        print(f"Training rows saved: {rows_written:,}")
        print(f"Features saved: {len(feature_columns)}")
        print(f"Total columns: {len(feature_columns) + 1}")
        print(f"Output: {OUTPUT_FILE}")
        print("Existing training artifacts were reused.")
        print("Raw dataset files were not modified.")

    except Exception:
        # Remove incomplete output so it cannot be mistaken
        # for a complete training dataset.
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()
        raise


if __name__ == "__main__":
    main()
