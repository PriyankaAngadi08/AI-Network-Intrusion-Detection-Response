
import pandas as pd

from preprocessing_utils import (
    load_preprocessing_artifacts,
    preprocess_network_traffic,
)


# Load the selected feature names.
feature_columns, medians, scaler = load_preprocessing_artifacts()

# Create three sample records using saved training medians.
sample = pd.DataFrame(
    [
        {column: medians[column] for column in feature_columns},
        {column: medians[column] for column in feature_columns},
        {column: medians[column] for column in feature_columns},
    ]
)

# Introduce examples of invalid values to test cleaning.
sample.loc[0, "Flow Bytes/s"] = float("inf")
sample.loc[1, "Flow Bytes/s"] = -100.0

# Apply the reusable function.
processed = preprocess_network_traffic(sample)

print("Input shape:", sample.shape)
print("Output shape:", processed.shape)
print("Missing values:", int(processed.isna().sum().sum()))
print("All values finite:", bool(processed.map(
    lambda value: abs(value) != float("inf")
).all().all()))

print("\nFirst two transformed records:")
print(processed.head(2))

assert processed.shape == (3, 70)
assert processed.isna().sum().sum() == 0
assert processed.to_numpy().dtype == "float64"

print("\nTask 19 basic test passed.")
