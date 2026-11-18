
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

FILES = [
    "validation_processed.csv",
    "test_processed.csv",
]

TARGET_COLUMN = " Label"
CHUNK_SIZE = 100_000


def main():
    for filename in FILES:
        file_path = DATA_DIR / filename

        print(f"\n=== Scaled range check: {filename} ===")

        feature_min = None
        feature_max = None
        max_absolute_value = 0.0
        max_feature_name = None
        max_scaled_value = None
        total_rows = 0

        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
        ):
            features = chunk.drop(columns=[TARGET_COLUMN])
            values = features.to_numpy(dtype=np.float64)

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Non-finite values found in {filename}."
                )

            chunk_min = values.min(axis=0)
            chunk_max = values.max(axis=0)

            if feature_min is None:
                feature_min = chunk_min
                feature_max = chunk_max
            else:
                feature_min = np.minimum(feature_min, chunk_min)
                feature_max = np.maximum(feature_max, chunk_max)

            # Find the largest absolute value in this chunk.
            absolute_values = np.abs(values)

            position = np.unravel_index(
                np.argmax(absolute_values),
                absolute_values.shape,
            )

            chunk_max_value = float(absolute_values[position])

            # Update the overall maximum if necessary.
            if chunk_max_value > max_absolute_value:
                max_absolute_value = chunk_max_value

                row_index, column_index = position
                max_feature_name = features.columns[column_index]
                max_scaled_value = float(
                    values[row_index, column_index]
                )

            # Count every row, regardless of the maximum.
            total_rows += len(chunk)

        print("Rows checked:", f"{total_rows:,}")
        print("Largest absolute scaled value:", max_absolute_value)
        print("Feature with largest absolute value:", max_feature_name)
        print("Actual scaled value:", max_scaled_value)
        print("Smallest feature minimum:", float(feature_min.min()))
        print("Largest feature maximum:", float(feature_max.max()))
        print("All values finite: True")

    print("\nScaled range inspection finished.")


if __name__ == "__main__":
    main()
