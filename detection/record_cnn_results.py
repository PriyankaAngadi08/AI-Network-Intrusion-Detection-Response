
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports" / "sprint5"

# Compare the two CNN experiments
experiments = [
    {
        "Experiment": "CNN Experiment 1 - Class Weighted",
        "Training Strategy": "Full dataset with class-weighted CrossEntropyLoss",
        "Training Samples": 1981520,
        "Training Loss": 0.0647,
        "Validation Loss": 3.0529,
        "Validation Accuracy": 0.8075,
        "Validation Macro F1": 0.0955,
        "Selected Model": "No",
    },
    {
        "Experiment": "CNN Experiment 2 - Balanced Sampling",
        "Training Strategy": "Class-aware reservoir sampling with standard CrossEntropyLoss",
        "Training Samples": 44969,
        "Training Loss": 0.4776,
        "Validation Loss": 0.2826,
        "Validation Accuracy": 0.9110,
        "Validation Macro F1": 0.4354,
        "Selected Model": "Yes",
    },
]

comparison_df = pd.DataFrame(experiments)

comparison_df.to_csv(
    REPORT_DIR / "cnn_experiment_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

# Record final selected model's test results
final_results = {
    "Selected Experiment": "CNN Experiment 2 - Balanced Sampling",
    "Model Checkpoint": "models/cnn/best_cnn_model_balanced.pt",
    "Test Samples": 424612,
    "Correct Predictions": 386680,
    "Test Loss": 0.2819,
    "Test Accuracy": 0.9107,
    "Macro Precision": 0.3581,
    "Macro Recall": 0.7099,
    "Macro F1": 0.4216,
    "Weighted Precision": 0.9620,
    "Weighted Recall": 0.9107,
    "Weighted F1": 0.9312,
    "Selection Reason": (
        "The balanced experiment achieved substantially better "
        "validation accuracy and macro F1 than the first experiment."
    ),
    "Important Limitation": (
        "Performance remains weak for several extremely rare attack classes."
    ),
}

with open(
    REPORT_DIR / "cnn_final_results.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(final_results, file, indent=4)

# Display comparison
print("\n========== CNN EXPERIMENT COMPARISON ==========")
print(comparison_df.to_string(index=False))

print("\n========== FINAL SELECTED CNN TEST RESULTS ==========")

for metric, value in final_results.items():
    print(f"{metric}: {value}")

print("\nResults recorded successfully.")
