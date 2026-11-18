
from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports" / "sprint4"

MODEL_FILES = {
    "SGD Logistic Regression": "logistic_regression_sgd_test_metrics.json",
    "Decision Tree": "decision_tree_test_metrics.json",
    "Random Forest": "random_forest_test_metrics.json",
}


def main():
    results = []

    for model_name, filename in MODEL_FILES.items():
        metrics_path = REPORT_DIR / filename

        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        results.append({
            "Model": model_name,
            "Test Rows": metrics["test_rows"],
            "Accuracy": metrics["accuracy"],
            "Macro Precision": metrics["macro_precision"],
            "Macro Recall": metrics["macro_recall"],
            "Macro F1": metrics["macro_f1"],
            "Weighted F1": metrics["weighted_f1"],
            "Macro FPR": metrics["macro_false_positive_rate"],
        })

    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values(
        by="Macro F1", ascending=False
    )

    csv_path = REPORT_DIR / "baseline_test_comparison.csv"
    json_path = REPORT_DIR / "baseline_test_comparison.json"

    comparison.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(comparison.to_dict(orient="records"), file, indent=4)

    print("\n=== Final Baseline Test Comparison ===")
    print(comparison.to_string(index=False))

    print("\nSaved:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
