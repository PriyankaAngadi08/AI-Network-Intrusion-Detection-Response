
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test_processed.csv"
FEATURE_FILE = PROJECT_ROOT / "models" / "preprocessing" / "feature_columns.json"
MODEL_DIR = PROJECT_ROOT / "models" / "baselines"
RESULTS_DIR = PROJECT_ROOT / "reports" / "sprint4"

TARGET_COLUMN = " Label"
CHUNK_SIZE = 50_000

MODELS = {
    "SGD Logistic Regression": "logistic_regression_sgd.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
}


def main():
    print("=== Baseline Models: Final Test Evaluation ===")

    with open(FEATURE_FILE, "r", encoding="utf-8") as file:
        feature_columns = json.load(file)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, model_filename in MODELS.items():
        model_path = MODEL_DIR / model_filename
        model = joblib.load(model_path)

        labels = list(model.classes_)
        label_to_index = {label: i for i, label in enumerate(labels)}
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)

        total_rows = 0
        prediction_file = RESULTS_DIR / (
            model_filename.replace(".joblib", "_test_predictions.csv")
        )

        print(f"\nEvaluating {model_name}...")
        first_chunk = True

        for chunk in pd.read_csv(TEST_FILE, chunksize=CHUNK_SIZE):
            X = chunk[feature_columns]
            y_true = chunk[TARGET_COLUMN].astype(str)
            y_pred = model.predict(X)

            for actual, predicted in zip(y_true, y_pred):
                if actual in label_to_index and predicted in label_to_index:
                    matrix[
                        label_to_index[actual],
                        label_to_index[predicted]
                    ] += 1

            predictions = pd.DataFrame({
                "Actual": y_true,
                "Predicted": y_pred
            })

            predictions.to_csv(
                prediction_file,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )
            first_chunk = False

            total_rows += len(chunk)
            print(f"Test rows processed: {total_rows:,}", end="\r")

        tp = np.diag(matrix)
        actual_counts = matrix.sum(axis=1)
        predicted_counts = matrix.sum(axis=0)

        precision = np.divide(
            tp, predicted_counts,
            out=np.zeros(len(labels), dtype=float),
            where=predicted_counts != 0
        )

        recall = np.divide(
            tp, actual_counts,
            out=np.zeros(len(labels), dtype=float),
            where=actual_counts != 0
        )

        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros(len(labels), dtype=float),
            where=(precision + recall) != 0
        )

        accuracy = tp.sum() / max(matrix.sum(), 1)
        weighted_f1 = (
            np.sum(f1 * actual_counts) / max(actual_counts.sum(), 1)
        )

        false_positives = predicted_counts - tp
        negatives = matrix.sum() - actual_counts

        fpr = np.divide(
            false_positives, negatives,
            out=np.zeros(len(labels), dtype=float),
            where=negatives != 0
        )

        metrics = {
            "model": model_name,
            "test_rows": int(total_rows),
            "accuracy": float(accuracy),
            "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()),
            "weighted_f1": float(weighted_f1),
            "macro_false_positive_rate": float(fpr.mean())
        }

        per_class = pd.DataFrame({
            "Class": labels,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": actual_counts
        })

        safe_name = model_filename.replace(".joblib", "")
        metrics_file = RESULTS_DIR / f"{safe_name}_test_metrics.json"
        per_class_file = RESULTS_DIR / f"{safe_name}_test_per_class.csv"
        matrix_file = RESULTS_DIR / f"{safe_name}_test_confusion_matrix.csv"

        with open(metrics_file, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

        per_class.to_csv(per_class_file, index=False)

        pd.DataFrame(
            matrix,
            index=labels,
            columns=labels
        ).to_csv(matrix_file, index_label="Actual")

        print(f"\n{model_name} test results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro Precision: {precision.mean():.4f}")
        print(f"Macro Recall: {recall.mean():.4f}")
        print(f"Macro F1: {f1.mean():.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print(f"Macro False-Positive Rate: {fpr.mean():.4f}")
        print(f"Predictions saved to: {prediction_file}")

    print("\nFinal test evaluation completed.")


if __name__ == "__main__":
    main()
