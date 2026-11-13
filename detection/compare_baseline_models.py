
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports" / "sprint4"

MODEL_FILES = {
    "SGD Logistic Regression": "logistic_regression_validation_metrics.json",
    "Decision Tree": "decision_tree_validation_metrics.json",
    "Random Forest": "random_forest_validation_metrics.json",
}

# Previously recorded Logistic Regression validation metrics.
# These are rounded values, not reloaded from a saved metrics file.
RECORDED_LOGISTIC_METRICS = {
    "accuracy": 0.8206,
    "macro_precision": 0.1610,
    "macro_recall": 0.2537,
    "macro_f1": 0.1416,
    "weighted_f1": 0.7968,
}


def main():
    results = []

    for model_name, filename in MODEL_FILES.items():
        metrics_path = REPORT_DIR / filename

        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as file:
                metrics = json.load(file)
            source = "Saved JSON"
        elif model_name == "SGD Logistic Regression":
            metrics = RECORDED_LOGISTIC_METRICS.copy()
            source = "Previously recorded, rounded values"
            print(
                "Note: Logistic Regression metrics are using "
                "previously recorded rounded values."
            )
        else:
            raise FileNotFoundError(
                f"Metrics file not found: {metrics_path}"
            )

        results.append({
            "Model": model_name,
            "Accuracy": metrics.get("accuracy"),
            "Macro Precision": metrics.get("macro_precision"),
            "Macro Recall": metrics.get("macro_recall"),
            "Macro F1": metrics.get("macro_f1"),
            "Weighted F1": metrics.get("weighted_f1"),
            "Macro FPR": metrics.get("macro_false_positive_rate"),
            "Metrics Source": source,
        })

    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values(
        by="Macro F1", ascending=False
    )

    csv_path = REPORT_DIR / "baseline_model_comparison.csv"
    json_path = REPORT_DIR / "baseline_model_comparison.json"

    comparison.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            comparison.to_dict(orient="records"),
            file,
            indent=4,
        )

    print("\n=== Baseline Model Comparison ===")
    print(comparison.to_string(index=False))

    print("\nComparison files saved:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
