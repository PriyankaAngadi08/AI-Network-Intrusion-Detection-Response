from pathlib import Path
import json
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report

from cnn_bilstm_model import CNNBiLSTM


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "training_processed.csv"
VALIDATION_FILE = PROJECT_ROOT / "data" / "processed" / "validation_processed.csv"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test_processed.csv"

FEATURE_FILE = PROJECT_ROOT / "models" / "preprocessing" / "feature_columns.json"

MODEL_DIR = PROJECT_ROOT / "models" / "cnn_bilstm"
REPORT_DIR = PROJECT_ROOT / "reports" / "sprint7"


# Training configuration
RANDOM_STATE = 42

NUM_CLASSES = 15
NUM_FEATURES = 70
TARGET_COLUMN = " Label"

MAX_PER_CLASS = 500
BATCH_SIZE = 8
CHUNK_SIZE = 500
EPOCHS = 2

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
CPU_THREADS = 2

PROGRESS_EVERY_CHUNKS = 100


# Reproducibility and CPU memory management
torch.set_num_threads(CPU_THREADS)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def verify_feature_columns():
    """Verify that all datasets have the expected 70 features."""

    with open(FEATURE_FILE, "r", encoding="utf-8") as file:
        expected_features = json.load(file)

    if len(expected_features) != NUM_FEATURES:
        raise ValueError("Feature count does not match 70.")

    for csv_file in (TRAIN_FILE, VALIDATION_FILE, TEST_FILE):
        actual_columns = pd.read_csv(csv_file, nrows=0).columns.tolist()

        actual_features = [
            column for column in actual_columns
            if column != TARGET_COLUMN
        ]

        if actual_features != expected_features:
            raise ValueError(
                f"Feature order mismatch in {csv_file.name}"
            )

        print(
            f"{csv_file.name}: {len(actual_features)} features verified."
        )

    print("Feature verification passed.")


def prepare_label_mapping():
    """Count training labels using chunked CSV reading."""

    print("\nCounting training labels...")

    counts = {}

    for chunk in pd.read_csv(
        TRAIN_FILE,
        usecols=[TARGET_COLUMN],
        chunksize=10000
    ):
        chunk_counts = chunk[TARGET_COLUMN].astype(str).value_counts()

        for label, count in chunk_counts.items():
            counts[label] = counts.get(label, 0) + int(count)

        del chunk

    class_names = sorted(counts.keys())

    if len(class_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, found {len(class_names)}."
        )

    label_to_id = {
        label: index
        for index, label in enumerate(class_names)
    }

    print("Total training rows:", f"{sum(counts.values()):,}")
    print("Number of classes:", len(class_names))

    for label, class_id in label_to_id.items():
        print(
            f"ID {class_id}: {label} "
            f"({counts[label]:,} rows)"
        )

    return label_to_id, class_names

def select_training_sample(label_to_id):
    """Select a balanced training sample using bounded-memory reservoir sampling."""

    print("\nSelecting training samples in chunks...")

    rng = np.random.default_rng(RANDOM_STATE)

    feature_reservoirs = np.empty(
        (NUM_CLASSES, MAX_PER_CLASS, NUM_FEATURES),
        dtype=np.float32
    )

    selected_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    seen_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for chunk_number, chunk in enumerate(
        pd.read_csv(TRAIN_FILE, chunksize=CHUNK_SIZE),
        start=1
    ):

        features = chunk.drop(columns=[TARGET_COLUMN]).to_numpy(
            dtype=np.float32,
            copy=True
        )

        labels = (
            chunk[TARGET_COLUMN]
            .astype(str)
            .map(label_to_id)
            .to_numpy(dtype=np.int64)
        )

        if not np.isfinite(features).all() or (labels < 0).any():
            raise ValueError("Invalid features or labels found.")

        for row, class_id_value in zip(features, labels):

            class_id = int(class_id_value)
            seen_counts[class_id] += 1

            current_count = selected_counts[class_id]

            if current_count < MAX_PER_CLASS:
                feature_reservoirs[class_id, current_count] = row
                selected_counts[class_id] += 1

            else:
                replacement = rng.integers(0, seen_counts[class_id])

                if replacement < MAX_PER_CLASS:
                    feature_reservoirs[class_id, replacement] = row

        del chunk, features, labels

        if chunk_number % PROGRESS_EVERY_CHUNKS == 0:
            print(f"Processed chunks: {chunk_number}", flush=True)

    sample_size = int(selected_counts.sum())

    X = np.empty(
        (sample_size, NUM_FEATURES),
        dtype=np.float32
    )

    y = np.empty(sample_size, dtype=np.int64)

    position = 0

    for class_id in range(NUM_CLASSES):

        count = int(selected_counts[class_id])

        X[position:position + count] = (
            feature_reservoirs[class_id, :count]
        )

        y[position:position + count] = class_id

        position += count

        print(
            f"Class {class_id}: selected {count:,} "
            f"of {seen_counts[class_id]:,}"
        )

    del feature_reservoirs
    gc.collect()

    order = rng.permutation(sample_size)

    X = X[order]
    y = y[order]

    del order

    print("\nTotal selected training samples:", f"{sample_size:,}")

    return X, y


def train_one_epoch(model, optimizer, criterion, X, y):
    """Train the hybrid model for one epoch."""

    model.train()

    total_loss = 0.0
    total_samples = 0

    for start in range(0, len(y), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(y))

        batch_features = torch.from_numpy(
            X[start:end]
        ).unsqueeze(1)

        batch_labels = torch.from_numpy(
            y[start:end]
        ).long()

        optimizer.zero_grad(set_to_none=True)

        outputs = model(batch_features)

        loss = criterion(outputs, batch_labels)

        loss.backward()

        optimizer.step()

        batch_size = len(batch_labels)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    average_loss = total_loss / total_samples

    return average_loss


def validate_model(model, criterion, label_to_id):
    """Evaluate model using chunked validation CSV reading."""

    model.eval()

    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    with torch.no_grad():

        for chunk_number, chunk in enumerate(
            pd.read_csv(VALIDATION_FILE, chunksize=CHUNK_SIZE),
            start=1
        ):

            features = chunk.drop(
                columns=[TARGET_COLUMN]
            ).to_numpy(dtype=np.float32, copy=True)

            labels = (
                chunk[TARGET_COLUMN]
                .astype(str)
                .map(label_to_id)
                .to_numpy(dtype=np.int64)
            )

            if not np.isfinite(features).all() or (labels < 0).any():
                raise ValueError("Invalid validation data found.")

            for start in range(0, len(labels), BATCH_SIZE):

                end = min(start + BATCH_SIZE, len(labels))

                batch_features = torch.from_numpy(
                    features[start:end]
                ).unsqueeze(1)

                batch_labels = torch.from_numpy(
                    labels[start:end]
                ).long()

                outputs = model(batch_features)

                loss = criterion(outputs, batch_labels)

                predictions = outputs.argmax(dim=1)

                batch_size = len(batch_labels)

                total_loss += loss.item() * batch_size
                total_samples += batch_size

                correct_predictions += (
                    predictions == batch_labels
                ).sum().item()

                batch_confusion = confusion_matrix(
                    batch_labels.numpy(),
                    predictions.numpy(),
                    labels=list(range(NUM_CLASSES))
                )

                confusion += batch_confusion

            del chunk, features, labels

            if chunk_number % PROGRESS_EVERY_CHUNKS == 0:
                print(
                    f"Validation chunks processed: {chunk_number}",
                    flush=True
                )

            gc.collect()

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    true_positives = np.diag(confusion).astype(np.float64)
    actual = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)

    denominator = actual + predicted

    class_f1 = np.divide(
        2 * true_positives,
        denominator,
        out=np.zeros(NUM_CLASSES),
        where=denominator != 0
    )

    macro_f1 = float(class_f1.mean())

    return average_loss, accuracy, macro_f1, confusion


def main():

    print("========== CNN-BiLSTM TRAINING ==========")

    # Verify dataset features
    verify_feature_columns()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare class mapping
    label_to_id, class_names = prepare_label_mapping()

    # Select bounded training sample
    X, y = select_training_sample(label_to_id)

    # Initialize hybrid model
    model = CNNBiLSTM(num_classes=NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_macro_f1 = -1.0

    history = {
        "epoch": [],
        "training_loss": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "validation_macro_f1": []
    }

    best_model_path = MODEL_DIR / "best_cnn_bilstm.pt"

    print("\nTraining configuration:")
    print("Training samples:", len(y))
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("CPU threads:", CPU_THREADS)

    # Training loop
    for epoch in range(1, EPOCHS + 1):

        print(f"\n========== EPOCH {epoch}/{EPOCHS} ==========")

        train_loss = train_one_epoch(
            model,
            optimizer,
            criterion,
            X,
            y
        )

        print(f"Training loss: {train_loss:.4f}")

        val_loss, val_accuracy, val_macro_f1, confusion = (
            validate_model(model, criterion, label_to_id)
        )

        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Validation macro F1: {val_macro_f1:.4f}")

        history["epoch"].append(epoch)
        history["training_loss"].append(train_loss)
        history["validation_loss"].append(val_loss)
        history["validation_accuracy"].append(val_accuracy)
        history["validation_macro_f1"].append(val_macro_f1)

        # Save best model
        if val_macro_f1 > best_macro_f1:

            best_macro_f1 = val_macro_f1

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "label_to_id": label_to_id,
                    "num_features": NUM_FEATURES,
                    "num_classes": NUM_CLASSES,
                    "epoch": epoch,
                    "validation_macro_f1": val_macro_f1
                },
                best_model_path
            )

            print("Best model checkpoint saved.")

        gc.collect()

    # Save training history
    history_path = REPORT_DIR / "cnn_bilstm_training_history.json"

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    print("\n========== TRAINING COMPLETED ==========")
    print("Best validation macro F1:", round(best_macro_f1, 4))
    print("Model saved:", best_model_path)
    print("History saved:", history_path)


if __name__ == "__main__":
    main()