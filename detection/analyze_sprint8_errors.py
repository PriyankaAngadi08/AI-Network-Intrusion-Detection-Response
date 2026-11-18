from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "sprint8" / "error_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFUSION_FILES = {
    "Logistic Regression": ROOT / "reports/sprint4/logistic_regression_sgd_test_confusion_matrix.csv",
    "Decision Tree": ROOT / "reports/sprint4/decision_tree_test_confusion_matrix.csv",
    "Random Forest": ROOT / "reports/sprint4/random_forest_test_confusion_matrix.csv",
    "CNN": ROOT / "reports/sprint5/cnn_test_confusion_matrix.csv",
    "BiLSTM": ROOT / "reports/sprint6/bilstm_test_confusion_matrix_unweighted.csv",
    "CNN-BiLSTM": ROOT / "reports/sprint7/cnn_bilstm_test_confusion_matrix.csv",
}

class_wise_results = []
top_errors = []
overall_results = []

canonical_classes = None

print("\n========== SPRINT 8 ERROR ANALYSIS ==========\n")

for model_name, file_path in CONFUSION_FILES.items():

    if not file_path.exists():
        raise FileNotFoundError(f"Missing confusion matrix: {file_path}")

    df = pd.read_csv(file_path, index_col=0)

    # Use matrix position rather than text labels because some
    # files contain different Unicode replacement characters.
    matrix = df.to_numpy(dtype="int64")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{model_name}: Confusion matrix is not square.")

    if canonical_classes is None:
        canonical_classes = [str(x) for x in df.index]

    if matrix.shape != (len(canonical_classes), len(canonical_classes)):
        raise ValueError(f"{model_name}: Unexpected matrix dimensions.")

    row_totals = matrix.sum(axis=1)
    column_totals = matrix.sum(axis=0)
    diagonal = matrix.diagonal()

    total = int(matrix.sum())
    correct = int(diagonal.sum())
    errors = total - correct

    # Class-wise false positives, false negatives and error rates.
    for i, class_name in enumerate(canonical_classes):
        support = int(row_totals[i])
        tp = int(diagonal[i])
        fn = support - tp
        fp = int(column_totals[i]) - tp

        class_wise_results.append({
            "Model": model_name,
            "Class": class_name,
            "Actual_Samples": support,
            "Correct_Predictions": tp,
            "False_Negatives": fn,
            "False_Positives": fp,
            "Misclassification_Rate": fn / support if support else 0.0,
        })

    # Collect all off-diagonal actual-to-predicted errors.
    model_pairs = []

    for actual_index in range(len(canonical_classes)):
        for predicted_index in range(len(canonical_classes)):
            count = int(matrix[actual_index, predicted_index])

            if actual_index != predicted_index and count > 0:
                model_pairs.append({
                    "Model": model_name,
                    "Actual_Class": canonical_classes[actual_index],
                    "Predicted_Class": canonical_classes[predicted_index],
                    "Misclassified_Samples": count,
                })

    model_pairs.sort(
        key=lambda item: item["Misclassified_Samples"],
        reverse=True
    )

    top_errors.extend(model_pairs[:10])

    overall_results.append({
        "Model": model_name,
        "Total_Test_Samples": total,
        "Correct_Predictions": correct,
        "Total_Errors": errors,
        "False_Positives_Total": errors,
        "False_Negatives_Total": errors,
        "Accuracy_From_Matrix": correct / total if total else 0.0,
        "Error_Rate": errors / total if total else 0.0,
    })

    print(f"{model_name}:")
    print(f"  Test samples: {total:,}")
    print(f"  Correct predictions: {correct:,}")
    print(f"  Misclassified predictions: {errors:,}")
    print(f"  Accuracy: {correct / total:.6f}")
    print()

# Save all reports.
class_wise_df = pd.DataFrame(class_wise_results)
top_errors_df = pd.DataFrame(top_errors)
overall_df = pd.DataFrame(overall_results)

class_wise_df.to_csv(
    OUTPUT_DIR / "error_analysis_class_wise.csv",
    index=False
)

top_errors_df.to_csv(
    OUTPUT_DIR / "top_misclassification_pairs.csv",
    index=False
)

overall_df.to_csv(
    OUTPUT_DIR / "overall_error_comparison.csv",
    index=False
)

print("========== OVERALL ERROR COMPARISON ==========")
print(overall_df.to_string(index=False))

print("\n========== TOP MISCLASSIFICATIONS ==========")

for model_name in CONFUSION_FILES:
    print(f"\n{model_name}:")
    model_errors = top_errors_df[
        top_errors_df["Model"] == model_name
    ].head(5)

    if model_errors.empty:
        print("  No misclassifications.")
    else:
        print(model_errors[
            ["Actual_Class", "Predicted_Class", "Misclassified_Samples"]
        ].to_string(index=False))

print("\n========== FILES SAVED ==========")
for file_path in sorted(OUTPUT_DIR.glob("*.csv")):
    print(f"{file_path.name} ({file_path.stat().st_size:,} bytes)")

print("\nError analysis completed successfully.")
