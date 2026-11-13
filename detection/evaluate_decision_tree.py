
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALIDATION_FILE = (
    PROJECT_ROOT / "data" / "processed" / "validation_processed.csv"
)
MODEL_FILE = PROJECT_ROOT / "models" / "baselines" / "decision_tree.joblib"
FEATURE_FILE = (
    PROJECT_ROOT / "models" / "preprocessing" / "feature_columns.json"
)
RESULTS_DIR = PROJECT_ROOT / "reports" / "sprint4"

TARGET_COLUMN = " Label"
CHUNK_SIZE = 50_000


def main():
    print("=== Decision Tree Validation Evaluation ===")

    # Load model and saved feature order.
    model = joblib.load(MODEL_FILE)

    with open(FEATURE_FILE, "r", encoding="utf-8") as file:
        feature_columns = json.load(file)

    labels = list(model.classes_)
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    label_to_index = {label: i for i, label in enumerate(labels)}

    total_rows = 0

    # Evaluate in chunks to limit memory use.
    for chunk in pd.read_csv(
        VALIDATION_FILE,
        chunksize=CHUNK_SIZE
    ):
        X = chunk[feature_columns]
        y_true = chunk[TARGET_COLUMN].astype(str)
        y_pred = model.predict(X)

        for actual, predicted in zip(y_true, y_pred):
            if actual in label_to_index and predicted in label_to_index:
                matrix[
                    label_to_index[actual],
                    label_to_index[predicted]
                ] += 1

        total_rows += len(chunk)

        if total_rows % 200_000 < len(chunk):
            print(f"Validation rows processed: {total_rows:,}")

    # Calculate metrics from the accumulated confusion matrix.
    true_counts = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)
    true_positives = np.diag(matrix)

    precision_per_class = np.divide(
        true_positives,
        predicted_counts,
        out=np.zeros(len(labels), dtype=float),
        where=predicted_counts != 0
    )

    recall_per_class = np.divide(
        true_positives,
        true_counts,
        out=np.zeros(len(labels), dtype=float),
        where=true_counts != 0
    )

    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        out=np.zeros(len(labels), dtype=float),
        where=(precision_per_class + recall_per_class) != 0
    )

    accuracy = accuracy_score(
        np.repeat(labels, true_counts),
        np.concatenate([
            np.repeat(labels[i], matrix[i, j])
            for i in range(len(labels))
            for j in range(len(labels))
        ]) if False else np.array([])
    ) if False else true_positives.sum() / max(matrix.sum(), 1)

    macro_precision = precision_per_class.mean()
    macro_recall = recall_per_class.mean()
    macro_f1 = f1_per_class.mean()

    total_support = true_counts.sum()
    weighted_f1 = (
        np.sum(f1_per_class * true_counts) / max(total_support, 1)
    )

    # Multiclass one-vs-rest false-positive rate, macro averaged.
    false_positives = predicted_counts - true_positives
    true_negatives = total_support - true_counts - false_positives

    fpr_per_class = np.divide(
        false_positives,
        false_positives + true_negatives,
        out=np.zeros(len(labels), dtype=float),
        where=(false_positives + true_negatives) != 0
    )

    print("\nValidation Results")
    print("------------------")
    print(f"Validation rows: {total_rows:,}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Macro False-Positive Rate: {fpr_per_class.mean():.4f}")

    report = pd.DataFrame({
        "Class": labels,
        "Precision": precision_per_class,
        "Recall": recall_per_class,
        "F1": f1_per_class,
        "Support": true_counts
    })

    print("\nPer-class results:")
    print(report.to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report.to_csv(
        RESULTS_DIR / "decision_tree_validation_per_class.csv",
        index=False
    )

    pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    ).to_csv(
        RESULTS_DIR / "decision_tree_validation_confusion_matrix.csv",
        index_label="Actual"
    )

    metrics = {
        "model": "Decision Tree",
        "validation_rows": int(total_rows),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_false_positive_rate": float(fpr_per_class.mean())
    }

    with open(
        RESULTS_DIR / "decision_tree_validation_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\nSaved validation metrics, per-class results, and confusion matrix.")
    print("Decision Tree validation evaluation completed.")


if __name__ == "__main__":
    main()
