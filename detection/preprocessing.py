
from pathlib import Path
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "preprocessing"

TEMP_MATRIX = PROCESSED_DIR / "training_features_temp.dat"

TARGET_COLUMN = " Label"
CHUNK_SIZE = 10_000

CONSTANT_FEATURES = [
    " Bwd PSH Flags",
    " Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    " Fwd Avg Packets/Bulk",
    " Fwd Avg Bulk Rate",
    " Bwd Avg Bytes/Bulk",
    " Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]

NONNEGATIVE_FEATURES = [
    " Flow Duration",
    " Flow Packets/s",
    "Flow Bytes/s",
    " min_seg_size_forward",
    " Bwd Header Length",
]


for folder in [PROCESSED_DIR, ARTIFACT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. Validate saved split assignments
# ==================================================

def load_assignments():
    if not ASSIGNMENTS_FILE.exists():
        raise FileNotFoundError(ASSIGNMENTS_FILE)

    assignments = pd.read_csv(ASSIGNMENTS_FILE)

    required = {"global_row_index", "source_file", "split"}

    if not required.issubset(assignments.columns):
        raise ValueError(
            f"Missing assignment columns: "
            f"{required - set(assignments.columns)}"
        )

    if assignments[
        ["global_row_index", "source_file", "split"]
    ].isnull().any().any():
        raise ValueError("Missing values in split assignments.")

    if assignments["global_row_index"].duplicated().any():
        raise ValueError("Duplicate global row indices.")

    if set(assignments["split"].unique()) != {
        "train", "validation", "test"
    }:
        raise ValueError("Unexpected or missing split names.")

    assignments = assignments.sort_values("global_row_index")

    if assignments["global_row_index"].tolist() != list(
        range(len(assignments))
    ):
        raise ValueError("Global row indices are incomplete.")

    filenames = sorted(
        path.name for path in DATASET_DIR.glob("*.csv")
    )

    if sorted(assignments["source_file"].unique()) != filenames:
        raise ValueError("Source files do not match assignments.")

    return assignments, filenames


# ==================================================
# 3. Read numeric features and clean invalid values
# ==================================================

def clean_features(frame, feature_columns):
    X = frame[feature_columns].copy()

    # Infinite values are treated as missing.
    X = X.replace([np.inf, -np.inf], np.nan)

    # Negative duration/rate values are invalid.
    for column in NONNEGATIVE_FEATURES:
        if column in X.columns:
            X.loc[X[column] < 0, column] = np.nan

    return X


# ==================================================
# 4. Fit using training records only
# ==================================================

def main():
    print("=== Task 15: Training-Only Preprocessing Fit ===")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(DATASET_DIR)

    assignments, filenames = load_assignments()

    first_file = DATASET_DIR / filenames[0]
    original_columns = pd.read_csv(
        first_file, nrows=0
    ).columns.tolist()

    if TARGET_COLUMN not in original_columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    original_features = [
        column for column in original_columns
        if column != TARGET_COLUMN
    ]

    # Check all files have the same column layout.
    for filename in filenames:
        columns = pd.read_csv(
            DATASET_DIR / filename, nrows=0
        ).columns.tolist()

        if columns != original_columns:
            raise ValueError(
                f"Column mismatch in {filename}"
            )

    missing_constants = (
        set(CONSTANT_FEATURES) - set(original_features)
    )

    if missing_constants:
        raise ValueError(
            f"Constant features not found: {missing_constants}"
        )

    feature_columns = [
        column for column in original_features
        if column not in CONSTANT_FEATURES
    ]

    train_count = int((assignments["split"] == "train").sum())
    feature_count = len(feature_columns)

    estimated_bytes = train_count * feature_count * 8
    free_bytes = shutil.disk_usage(PROCESSED_DIR).free

    print(f"Training records: {train_count:,}")
    print(f"Selected features: {feature_count}")
    print(f"Temporary matrix size: {estimated_bytes / 1e9:.2f} GB")

    if free_bytes < estimated_bytes * 1.2:
        raise OSError(
            "Insufficient free space for the temporary training matrix. "
            "Free additional disk space before continuing."
        )

    # Remove only a stale temporary file from a previous run.
    if TEMP_MATRIX.exists():
        TEMP_MATRIX.unlink()

    # A disk-backed matrix avoids holding all training rows in RAM.
    matrix = np.memmap(
        TEMP_MATRIX,
        dtype="float64",
        mode="w+",
        shape=(train_count, feature_count),
    )

    train_position = 0
    global_offset = 0

    try:
        # Pass 1: copy cleaned training features to disk.
        for filename in filenames:
            print(f"Reading training rows: {filename}")

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
            file_path = DATASET_DIR / filename

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
                train_mask = split_values == "train"

                if train_mask.any():
                    X_train = clean_features(
                        chunk.loc[train_mask],
                        feature_columns,
                    )

                    values = X_train.to_numpy(
                        dtype=np.float64,
                        copy=True,
                    )

                    rows = len(values)

                    matrix[
                        train_position:train_position + rows
                    ] = values

                    train_position += rows

            if local_position != len(file_assignments):
                raise ValueError(
                    f"Not all rows processed in {filename}"
                )

            global_offset += len(file_assignments)

        if train_position != train_count:
            raise ValueError(
                f"Expected {train_count} training rows, "
                f"wrote {train_position}"
            )

        matrix.flush()

        # Exact median of each training feature.
        # Only one feature column is processed at a time.
        medians = {}

        print("\nCalculating exact training-set medians...")

        for index, column in enumerate(feature_columns):
            column_values = np.asarray(matrix[:, index])

            if np.isnan(column_values).all():
                raise ValueError(
                    f"Feature contains no valid training values: {column}"
                )

            median = float(np.nanmedian(column_values))
            medians[column] = median

            # Impute the disk-backed matrix in place.
            missing_mask = np.isnan(column_values)

            if missing_mask.any():
                matrix[missing_mask, index] = median

        matrix.flush()

        # Pass 2: fit StandardScaler only on training data.
        scaler = StandardScaler()

        print("Fitting StandardScaler on training data only...")

        for start in range(0, train_count, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, train_count)

            scaler.partial_fit(
                np.asarray(matrix[start:end])
            )

        if not np.isfinite(scaler.mean_).all():
            raise ValueError("Non-finite scaler means detected.")

        if not np.isfinite(scaler.scale_).all():
            raise ValueError("Non-finite scaler scales detected.")

        # Save artifacts only after all checks pass.
        joblib.dump(
            scaler,
            ARTIFACT_DIR / "standard_scaler.joblib",
        )

        with open(
            ARTIFACT_DIR / "imputation_medians.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(medians, file, indent=2)

        with open(
            ARTIFACT_DIR / "feature_columns.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(feature_columns, file, indent=2)

        print("\n=== Task 15 Results ===")
        print(f"Training rows used: {train_count:,}")
        print(f"Features retained: {feature_count}")
        print(f"Features removed as constant: {len(CONSTANT_FEATURES)}")
        print(f"Imputation medians calculated: {len(medians)}")
        print(f"Scaler samples seen: {scaler.n_samples_seen_}")
        print(f"Artifacts saved in: {ARTIFACT_DIR}")
        print("Original CSV files were not modified.")

    
    finally:
        # Release the memory map and its underlying file handle.
        if "matrix" in locals():
            matrix.flush()
            matrix._mmap.close()
            del matrix

        if TEMP_MATRIX.exists():
            try:
                TEMP_MATRIX.unlink()
                print("Temporary training matrix cleaned up.")
            except PermissionError:
                print(
                    "Warning: Windows still has the temporary file open. "
                    "Close any process using it and delete it later."
                )



if __name__ == "__main__":
    main()
