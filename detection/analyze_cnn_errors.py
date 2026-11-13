
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = PROJECT_ROOT / "reports" / "sprint5"

CONFUSION_FILE = REPORT_DIR / "cnn_test_confusion_matrix.csv"


def analyze_errors():
    # Load the confusion matrix
    confusion_df = pd.read_csv(
        CONFUSION_FILE,
        index_col=0,
    )

    confusion = confusion_df.values
    class_names = confusion_df.index.tolist()

    # Calculate class-wise errors
    results = []

    for i, class_name in enumerate(class_names):
        true_positives = int(confusion[i, i])
        actual_count = int(confusion[i, :].sum())
        predicted_count = int(confusion[:, i].sum())

        false_negatives = actual_count - true_positives
        false_positives = predicted_count - true_positives

        error_rate = (
            false_negatives / actual_count
            if actual_count > 0 else 0
        )

        results.append({
            "Class": class_name,
            "Actual Samples": actual_count,
            "Correct Predictions": true_positives,
            "False Negatives": false_negatives,
            "False Positives": false_positives,
            "Misclassification Rate": error_rate,
        })

    error_df = pd.DataFrame(results)

    # Find the most frequent incorrect prediction pairs
    misclassifications = []

    for actual_id in range(len(class_names)):
        for predicted_id in range(len(class_names)):

            if actual_id != predicted_id:
                count = int(confusion[actual_id, predicted_id])

                if count > 0:
                    misclassifications.append({
                        "Actual Class": class_names[actual_id],
                        "Predicted Class": class_names[predicted_id],
                        "Misclassified Samples": count,
                    })

    misclassification_df = pd.DataFrame(misclassifications)

    if not misclassification_df.empty:
        misclassification_df = misclassification_df.sort_values(
            by="Misclassified Samples",
            ascending=False,
        )

    # Save reports
    error_df.to_csv(
        REPORT_DIR / "cnn_class_wise_error_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )

    misclassification_df.to_csv(
        REPORT_DIR / "cnn_top_misclassifications.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Display results
    print("\n========== CLASS-WISE ERROR ANALYSIS ==========")
    print(
        error_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n========== TOP 15 MISCLASSIFICATIONS ==========")
    print(
        misclassification_df.head(15).to_string(index=False)
    )

    print("\nError analysis reports saved successfully.")


if __name__ == "__main__":
    analyze_errors()
