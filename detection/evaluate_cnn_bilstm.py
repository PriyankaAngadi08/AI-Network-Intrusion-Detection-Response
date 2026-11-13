import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report

from cnn_bilstm_model import CNNBiLSTM


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_processed.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "cnn_bilstm"
    / "best_cnn_bilstm.pt"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "sprint7"


# Evaluation settings
TARGET_COLUMN = " Label"
CHUNK_SIZE = 500
BATCH_SIZE = 8
CPU_THREADS = 2

torch.set_num_threads(CPU_THREADS)

def load_trained_model():
    """Load the best saved CNN-BiLSTM checkpoint."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_FILE}"
        )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=False
    )

    class_names = checkpoint["class_names"]
    label_to_id = checkpoint["label_to_id"]

    model = CNNBiLSTM(
        num_classes=checkpoint["num_classes"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\n========== MODEL LOADED ==========")
    print("Checkpoint:", MODEL_FILE)
    print("Best epoch:", checkpoint["epoch"])
    print(
        "Best validation macro F1:",
        round(checkpoint["validation_macro_f1"], 4)
    )
    print("Number of features:", checkpoint["num_features"])
    print("Number of classes:", checkpoint["num_classes"])

    return model, class_names, label_to_id

def evaluate_test_dataset(model, label_to_id, class_names):
    """Generate predictions for the test dataset in chunks."""

    all_actual = []
    all_predicted = []

    total_rows = 0
    processed_chunks = 0

    print("\n========== TEST EVALUATION ==========")

    for chunk in pd.read_csv(
        TEST_FILE,
        chunksize=CHUNK_SIZE
    ):
        features = chunk.drop(
            columns=[TARGET_COLUMN]
        ).to_numpy(
            dtype=np.float32,
            copy=True
        )

        labels = (
            chunk[TARGET_COLUMN]
            .astype(str)
            .map(label_to_id)
            .to_numpy(dtype=np.int64)
        )

        if not np.isfinite(features).all():
            raise ValueError("Invalid feature values found.")

        if (pd.isna(labels)).any():
            raise ValueError("Unknown labels found in test dataset.")

        predictions = []

        with torch.no_grad():
            for start in range(0, len(labels), BATCH_SIZE):

                end = start + BATCH_SIZE

                batch_features = torch.from_numpy(
                    features[start:end].copy()
                ).unsqueeze(1)

                outputs = model(batch_features)

                batch_predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                predictions.extend(
                    batch_predictions.numpy().tolist()
                )

        all_actual.extend(labels.tolist())
        all_predicted.extend(predictions)

        total_rows += len(labels)
        processed_chunks += 1

        if processed_chunks % 100 == 0:
            print(
                f"Processed chunks: {processed_chunks}"
            )

        del features, labels, predictions
        gc.collect()

    print("\nTest evaluation completed.")
    print("Total test samples:", f"{total_rows:,}")

    return (
        np.array(all_actual),
        np.array(all_predicted),
        class_names
    )


def save_confusion_matrix(y_actual, y_predicted, class_names):
    """Generate and save the test confusion matrix."""

    matrix = confusion_matrix(
        y_actual,
        y_predicted,
        labels=list(range(len(class_names)))
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names
    )

    output_file = REPORT_DIR / "cnn_bilstm_test_confusion_matrix.csv"

    confusion_df.to_csv(
        output_file,
        encoding="utf-8-sig"
    )

    print("\n========== CONFUSION MATRIX ==========")
    print(confusion_df.to_string())

    print("\nConfusion matrix saved:", output_file)

    return matrix

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load best trained model
    model, class_names, label_to_id = load_trained_model()

    # Evaluate using test dataset
    y_actual, y_predicted, class_names = evaluate_test_dataset(
        model,
        label_to_id,
        class_names
    )

    # Generate confusion matrix
    save_confusion_matrix(
        y_actual,
        y_predicted,
        class_names
    )

    print("\nCNN-BiLSTM test evaluation completed successfully.")


if __name__ == "__main__":
    main()