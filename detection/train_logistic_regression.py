
from pathlib import Path
import json
import pandas as pd
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input datasets
TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "training_processed.csv"
VALIDATION_FILE = PROJECT_ROOT / "data" / "processed" / "validation_processed.csv"

# Output directory for baseline results
MODEL_DIR = PROJECT_ROOT / "models" / "baselines"
RESULTS_DIR = PROJECT_ROOT / "reports" / "sprint4"

# Training configuration
RANDOM_STATE = 42
CHUNK_SIZE = 50000
EPOCHS = 1


def calculate_class_weights():
    """Calculate class weights using labels from the training dataset."""

    print("Reading training labels to calculate class weights...")

    labels = pd.read_csv(
        TRAIN_FILE,
        usecols=[" Label"]
    )[" Label"].astype(str)

    classes = np.sort(labels.unique())

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    MAX_CLASS_WEIGHT = 100.0

    class_weights = {
        label: float(min(weight, MAX_CLASS_WEIGHT))
        for label, weight in zip(classes, weights)
        }   

    print("\nNumber of classes:", len(classes))
    print("Class weights:")

    for label, weight in class_weights.items():
        print(f"{label}: {weight:.4f}")

    return classes, class_weights



def create_model(class_weights):
    """Create an incremental logistic regression classifier."""

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        learning_rate="optimal",
        class_weight=class_weights,
        random_state=RANDOM_STATE
    )

    print("\nLogistic Regression baseline initialized.")
    print("Training method: Incremental SGD")
    print("Loss function: Log loss")
    print("Regularization: L2")

    return model



def train_model(model, classes):
    """Train the model using chunks of the training CSV."""

    for epoch in range(EPOCHS):
        print(f"\nStarting epoch {epoch + 1}/{EPOCHS}")
        rows_seen = 0

        for chunk in pd.read_csv(TRAIN_FILE, chunksize=CHUNK_SIZE):
            # Shuffle rows within this chunk reproducibly.
            chunk = chunk.sample(
            frac=1,
            random_state=RANDOM_STATE + epoch + rows_seen
            ).reset_index(drop=True)

            y = chunk[" Label"].astype(str)
            X = chunk.drop(columns=[" Label"])

            # Train incrementally; supply all classes on the first chunk.
            if rows_seen == 0:
                model.partial_fit(X, y, classes=classes)
            else:
                model.partial_fit(X, y)

            rows_seen += len(chunk)

            if rows_seen % 250000 < CHUNK_SIZE:
                print(f"Rows processed in this epoch: {rows_seen:,}")

        print(f"Epoch {epoch + 1} completed. Rows seen: {rows_seen:,}")

    return model




def evaluate_model(model):
    """Evaluate the model using a chunk-wise confusion matrix."""

    print("\nEvaluating on validation data...")

    labels = model.classes_
    matrix = np.zeros(
        (len(labels), len(labels)),
        dtype=np.int64
    )

    label_to_index = {
        label: index
        for index, label in enumerate(labels)
    }

    for chunk in pd.read_csv(
        VALIDATION_FILE,
        chunksize=CHUNK_SIZE
    ):
        y_true = chunk[" Label"].astype(str)
        X = chunk.drop(columns=[" Label"])
        y_pred = model.predict(X)

        for actual, predicted in zip(y_true, y_pred):
            if actual in label_to_index and predicted in label_to_index:
                matrix[
                    label_to_index[actual],
                    label_to_index[predicted]
                ] += 1

    total = matrix.sum()
    correct = np.trace(matrix)

    class_f1 = []
    class_precision = []
    class_recall = []
    class_support = matrix.sum(axis=1)

    for i in range(len(labels)):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        class_precision.append(precision)
        class_recall.append(recall)
        class_f1.append(f1)

    accuracy = correct / total if total else 0.0
    macro_precision = float(np.mean(class_precision))
    macro_recall = float(np.mean(class_recall))
    macro_f1 = float(np.mean(class_f1))

    weighted_f1 = (
        float(np.average(class_f1, weights=class_support))
        if total else 0.0
    )

    print("\nValidation Results")
    print("------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nPer-class results:")
    for i, label in enumerate(labels):
        print(
            f"{label}: "
            f"Precision={class_precision[i]:.4f}, "
            f"Recall={class_recall[i]:.4f}, "
            f"F1={class_f1[i]:.4f}, "
            f"Support={class_support[i]}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    ).to_csv(
        RESULTS_DIR / "logistic_regression_validation_confusion_matrix.csv"
    )

    print("\nConfusion matrix saved.")


def save_model(model):
    """Save the trained model for future use."""

    import joblib

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "logistic_regression_sgd.joblib"
    joblib.dump(model, model_path)

    print("\nModel saved to:", model_path)


def main():
    """Run the complete logistic regression baseline workflow."""

    print("=== Logistic Regression Baseline ===")

    classes, class_weights = calculate_class_weights()

    model = create_model(class_weights)

    model = train_model(model, classes)

    save_model(model)

    evaluate_model(model)

    print("\nLogistic Regression baseline completed.")


if __name__ == "__main__":
    main()
