import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "sprint7"
    / "cnn_bilstm_test_confusion_matrix.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "sprint7"

def calculate_metrics():
    # Load confusion matrix
    confusion_df = pd.read_csv(INPUT_FILE, index_col=0)
    confusion = confusion_df.values.astype(np.int64)

    class_names = confusion_df.index.tolist()

    # Basic values
    true_positives = np.diag(confusion)
    actual_counts = confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)

    # Precision
    precision = np.divide(
        true_positives,
        predicted_counts,
        out=np.zeros(len(class_names), dtype=float),
        where=predicted_counts != 0,
    )

    # Recall
    recall = np.divide(
        true_positives,
        actual_counts,
        out=np.zeros(len(class_names), dtype=float),
        where=actual_counts != 0,
    )

    # F1-score
    f1_score = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(class_names), dtype=float),
        where=(precision + recall) != 0,
    )

    support = actual_counts

    # Per-class report
    report_df = pd.DataFrame({
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
        "Support": support,
    })

    report_df.to_csv(
        REPORT_DIR / "cnn_bilstm_test_classification_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

        # Overall metrics
    total_samples = confusion.sum()
    correct_predictions = true_positives.sum()

    accuracy = correct_predictions / total_samples

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1_score.mean()

    weighted_precision = np.average(
        precision,
        weights=support
    )

    weighted_recall = np.average(
        recall,
        weights=support
    )

    weighted_f1 = np.average(
        f1_score,
        weights=support
    )

    overall_metrics = {
        "Test Samples": int(total_samples),
        "Correct Predictions": int(correct_predictions),
        "Test Accuracy": float(accuracy),
        "Macro Precision": float(macro_precision),
        "Macro Recall": float(macro_recall),
        "Macro F1": float(macro_f1),
        "Weighted Precision": float(weighted_precision),
        "Weighted Recall": float(weighted_recall),
        "Weighted F1": float(weighted_f1),
    }

    with open(
        REPORT_DIR / "cnn_bilstm_test_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(overall_metrics, file, indent=4)

    print("\n========== CNN-BiLSTM TEST METRICS ==========")

    for metric, value in overall_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value:,}")

    print("\n========== PER-CLASS METRICS ==========")
    print(
        report_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nMetrics saved successfully.")


if __name__ == "__main__":
    calculate_metrics()