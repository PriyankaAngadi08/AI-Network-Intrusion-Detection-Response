
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALIDATION_FILE = (
    PROJECT_ROOT / "data" / "processed" / "validation_processed.csv"
)
MODEL_FILE = PROJECT_ROOT / "models" / "baselines" / "random_forest.joblib"
RESULTS_DIR = PROJECT_ROOT / "reports" / "sprint4"

TARGET_COLUMN = " Label"
CHUNK_SIZE = 50_000


def main():
    print("=== Random Forest Validation Evaluation ===")

    model = joblib.load(MODEL_FILE)
    labels = model.classes_

    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    total_rows = 0

    print("\nLoading validation data in chunks...")

    for chunk in pd.read_csv(VALIDATION_FILE, chunksize=CHUNK_SIZE):
        y_true = chunk[TARGET_COLUMN]
        X = chunk.drop(columns=[TARGET_COLUMN])
        y_pred = model.predict(X)

        matrix += confusion_matrix(
            y_true,
            y_pred,
            labels=labels
        )

        total_rows += len(chunk)
        print(f"Validation rows processed: {total_rows:,}")

    true_positives = np.diag(matrix)
    actual_counts = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)

    precision = np.divide(
        true_positives,
        predicted_counts,
        out=np.zeros(len(labels), dtype=float),
        where=predicted_counts != 0
    )

    recall = np.divide(
        true_positives,
        actual_counts,
        out=np.zeros(len(labels), dtype=float),
        where=actual_counts != 0
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(labels), dtype=float),
        where=(precision + recall) != 0
    )

    total = matrix.sum()
    accuracy = true_positives.sum() / total if total else 0.0
    supports = actual_counts

    weighted_f1 = (
        np.sum(f1 * supports) / supports.sum()
        if supports.sum() else 0.0
    )

    false_positives = predicted_counts - true_positives
    negatives = total - actual_counts
    false_positive_rates = np.divide(
        false_positives,
        negatives,
        out=np.zeros(len(labels), dtype=float),
        where=negatives != 0
    )

    metrics = {
        "validation_rows": int(total_rows),
        "accuracy": float(accuracy),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(weighted_f1),
        "macro_false_positive_rate": float(false_positive_rates.mean())
    }

    per_class = pd.DataFrame({
        "Class": labels,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Support": supports
    })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_file = RESULTS_DIR / "random_forest_validation_metrics.json"
    per_class_file = RESULTS_DIR / "random_forest_validation_per_class.csv"
    matrix_file = RESULTS_DIR / "random_forest_validation_confusion_matrix.csv"

    with open(metrics_file, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    per_class.to_csv(per_class_file, index=False)

    pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    ).to_csv(matrix_file, index_label="Actual")

    print("\nRandom Forest Validation Results")
    print("--------------------------------")
    print(f"Validation rows: {total_rows:,}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision.mean():.4f}")
    print(f"Macro Recall: {recall.mean():.4f}")
    print(f"Macro F1: {f1.mean():.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(
        "Macro False-Positive Rate: "
        f"{false_positive_rates.mean():.4f}"
    )

    print("\nPer-class results:")
    print(per_class.to_string(index=False))

    print("\nSaved evaluation reports:")
    print(metrics_file)
    print(per_class_file)
    print(matrix_file)
    print("\nRandom Forest evaluation completed.")


if __name__ == "__main__":
    main()
