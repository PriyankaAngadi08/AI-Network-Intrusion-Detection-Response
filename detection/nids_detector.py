
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import csv
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "baselines" / "decision_tree.joblib"
PREPROCESSING_DIR = PROJECT_ROOT / "models" / "preprocessing"

FEATURE_PATH = PREPROCESSING_DIR / "feature_columns.json"
MEDIANS_PATH = PREPROCESSING_DIR / "imputation_medians.json"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "detection_log.csv"
ERROR_LOG_FILE = LOG_DIR / "detection_errors.csv"
INVALID_NEGATIVE_FEATURES = [
    " Flow Duration",
    " Flow Packets/s",
    "Flow Bytes/s",
    " min_seg_size_forward",
    " Bwd Header Length"
]


class NIDSDetector:

    def __init__(self):

        # Load the trained model once.
        self.model = joblib.load(MODEL_PATH)

        # Load the exact feature order used during training.
        with open(FEATURE_PATH, "r", encoding="utf-8") as file:
            self.feature_columns = json.load(file)

        # Load training-set imputation medians.
        with open(MEDIANS_PATH, "r", encoding="utf-8") as file:
            self.medians = json.load(file)

        # Verify model and artifact compatibility.
        if self.model.n_features_in_ != len(self.feature_columns):
            raise ValueError("Model and feature count mismatch.")

        if len(self.feature_columns) != len(self.medians):
            raise ValueError("Feature and median count mismatch.")

        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Model does not support probability prediction.")

        print("NIDS Detector initialized successfully.")
        print("Model: Decision Tree")
        print("Features:", len(self.feature_columns))
        print("Classes:", len(self.model.classes_))

    def validate_input(self, data):

        if not isinstance(data, (dict, pd.DataFrame)):
            raise TypeError("Input must be a dictionary or pandas DataFrame.")

        if isinstance(data, dict):
            data = pd.DataFrame([data])

        if data.empty:
            raise ValueError("Input contains no records.")

        missing_features = [
            feature for feature in self.feature_columns
            if feature not in data.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}"
            )

        # Ignore additional fields and preserve training feature order.
        data = data[self.feature_columns].copy()

        # Convert all features to numeric values.
        for feature in self.feature_columns:
            data[feature] = pd.to_numeric(
                data[feature], errors="coerce"
            )

        # Replace infinite values with missing values.
        data = data.replace([np.inf, -np.inf], np.nan)

        # Replace invalid negative values with missing values.
        for feature in INVALID_NEGATIVE_FEATURES:
            if feature in data.columns:
                data.loc[data[feature] < 0, feature] = np.nan

        # Impute missing and invalid values using training medians.
        for feature in self.feature_columns:
            data[feature] = data[feature].fillna(
                self.medians[feature]
            )

        # Reject records that still contain invalid values.
        if not np.isfinite(data.to_numpy(dtype=float)).all():
            raise ValueError("Input contains unresolved invalid values.")

        return data

    def detect(self, data):

        try:
            features = self.validate_input(data)

            predictions = self.model.predict(features)
            probabilities = self.model.predict_proba(features)

            results = []

            for index, prediction in enumerate(predictions):

                confidence = float(np.max(probabilities[index]))

                if prediction == "BENIGN":
                    status = "Normal"
                    attack_type = None
                else:
                    status = "Attack"
                    attack_type = str(prediction)

                result = {
                    "status": status,
                    "prediction": str(prediction),
                    "attack_type": attack_type,
                    "confidence": round(confidence, 6)
                }

                self.log_detection(result)
                results.append(result)

            return results

        except Exception as error:
            self.log_error(error, type(error).__name__)
            raise

    def log_error(self, error, error_type):
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        file_exists = ERROR_LOG_FILE.exists()

        with open(ERROR_LOG_FILE, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["timestamp", "error_type", "error_message"]
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "error_type": error_type,
                "error_message": str(error)
            })
    

    def log_detection(self, result):

        LOG_DIR.mkdir(parents=True, exist_ok=True)

        file_exists = LOG_FILE.exists()

        with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "timestamp",
                    "status",
                    "prediction",
                    "attack_type",
                    "confidence"
                ]
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "status": result["status"],
                "prediction": result["prediction"],
                "attack_type": result["attack_type"],
                "confidence": result["confidence"]
            })


if __name__ == "__main__":

    detector = NIDSDetector()

    print("\nDetector is ready for flow-level traffic records.")
