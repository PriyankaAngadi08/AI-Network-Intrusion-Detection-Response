
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# Locate the saved preprocessing artifacts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "models" / "preprocessing"


NONNEGATIVE_FEATURES = [
    " Flow Duration",
    " Flow Packets/s",
    "Flow Bytes/s",
    " min_seg_size_forward",
    " Bwd Header Length",
]



def load_preprocessing_artifacts():
    """Load the feature list, training medians, and fitted scaler."""

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

    if len(feature_columns) != 70:
        raise ValueError("Expected exactly 70 selected features.")

    if set(feature_columns) != set(medians):
        raise ValueError("Feature names and imputation medians do not match.")

    if scaler.n_features_in_ != len(feature_columns):
        raise ValueError("Scaler and feature counts do not match.")

    return feature_columns, medians, scaler


def preprocess_network_traffic(data):
    """
    Clean and scale network-traffic records.

    Input:
        A pandas DataFrame containing the original CIC-IDS2017 features.

    Output:
        A DataFrame containing 70 cleaned and scaled features.

    This function transforms data only. It never fits a new scaler
    or recalculates imputation medians.
    """

    feature_columns, medians, scaler = load_preprocessing_artifacts()

    missing_columns = set(feature_columns) - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Input is missing required features: {sorted(missing_columns)}"
        )

    # Select features in the exact training order.
    X = data[feature_columns].copy()

    # Convert infinite values to missing values.
    X = X.replace([np.inf, -np.inf], np.nan)

    # Apply the same invalid-value rules as training.
    for column in NONNEGATIVE_FEATURES:
        if column in X.columns:
            X.loc[X[column] < 0, column] = np.nan

    # Use the medians saved from the training set.
    X = X.fillna(medians)

    if X.isna().any().any():
        raise ValueError("Missing values remain after imputation.")

    values = X.to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError("Invalid numerical values remain after cleaning.")

    # Transform using the already-fitted training scaler.
    scaled_values = scaler.transform(values)

    if not np.isfinite(scaled_values).all():
        raise ValueError("Non-finite values found after scaling.")

    return pd.DataFrame(
        scaled_values,
        columns=feature_columns,
        index=data.index,
    )
