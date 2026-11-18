
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix


# Project paths and configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "training_processed.csv"
VALIDATION_FILE = PROJECT_ROOT / "data" / "processed" / "validation_processed.csv"
FEATURE_FILE = PROJECT_ROOT / "models" / "preprocessing" / "feature_columns.json"
MODEL_DIR = PROJECT_ROOT / "models" / "cnn"

RANDOM_STATE = 42
BATCH_SIZE = 64
CHUNK_SIZE = 2000
EPOCHS = 1
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MAX_CLASS_WEIGHT = 100.0
CPU_THREADS = 2
PROGRESS_EVERY_CHUNKS = 25
MAX_TRAIN_ROWS = 100_000

TARGET_COLUMN = " Label"
NUM_CLASSES = 15
NUM_FEATURES = 70

torch.set_num_threads(CPU_THREADS)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class IntrusionCNN(nn.Module):
    """Small 1D CNN for classifying network traffic."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Feature length: 70 -> 35 -> 17.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 17, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def verify_feature_columns():
    """Verify the training and validation feature order."""

    with open(FEATURE_FILE, "r", encoding="utf-8") as file:
        expected_features = json.load(file)

    if len(expected_features) != NUM_FEATURES:
        raise ValueError(
            f"Expected {NUM_FEATURES} saved features, "
            f"found {len(expected_features)}."
        )

    for csv_file in (TRAIN_FILE, VALIDATION_FILE):
        actual_columns = pd.read_csv(csv_file, nrows=0).columns.tolist()

        if TARGET_COLUMN not in actual_columns:
            raise ValueError(
                f"Target column {TARGET_COLUMN!r} "
                f"not found in {csv_file.name}."
            )

        actual_features = [
            column for column in actual_columns
            if column != TARGET_COLUMN
        ]

        if actual_features != expected_features:
            raise ValueError(
                f"Feature order mismatch in {csv_file.name}."
            )

        print(
            f"{csv_file.name}: {len(actual_features)} features verified.",
            flush=True,
        )

    print("Feature order verification passed.", flush=True)


def prepare_labels_and_weights():
    """Count labels in chunks and calculate capped class weights."""

    print("Reading training labels in chunks...", flush=True)
    counts = {}

    for chunk in pd.read_csv(
        TRAIN_FILE,
        usecols=[TARGET_COLUMN],
        chunksize=10000,
    ):
        chunk_counts = chunk[TARGET_COLUMN].astype(str).value_counts()

        for label, count in chunk_counts.items():
            counts[label] = counts.get(label, 0) + int(count)

    class_names = sorted(counts.keys())

    if len(class_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, found {len(class_names)}."
        )

    label_to_id = {
        label: index for index, label in enumerate(class_names)
    }

    total_rows = sum(counts.values())
    class_weights = []

    for label in class_names:
        weight = total_rows / (NUM_CLASSES * counts[label])
        class_weights.append(min(weight, MAX_CLASS_WEIGHT))

    weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
    )

    print("Training rows:", f"{total_rows:,}", flush=True)
    print("Number of classes:", len(class_names), flush=True)
    print("Number of features:", NUM_FEATURES, flush=True)
    print("\nClass mapping and weights:", flush=True)

    for label, weight in zip(class_names, class_weights):
        print(
            f"ID {label_to_id[label]:2d}: {label!r} | "
            f"weight={weight:.4f} | count={counts[label]:,}",
            flush=True,
        )

    return label_to_id, class_names, weights_tensor


def load_batch(chunk, label_to_id):
    """Convert a processed CSV chunk into CNN-ready tensors."""

    features = chunk.drop(columns=[TARGET_COLUMN]).to_numpy(
        dtype=np.float32,
        copy=True,
    )

    labels = (
        chunk[TARGET_COLUMN]
        .astype(str)
        .map(label_to_id)
        .to_numpy(dtype=np.int64, copy=True)
    )

    if not np.isfinite(features).all():
        raise ValueError(
            "Non-finite feature values found in a processed chunk."
        )

    if pd.isna(labels).any():
        raise ValueError(
            "A target label was not found in the class mapping."
        )

    features_tensor = torch.from_numpy(features).unsqueeze(1)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return features_tensor, labels_tensor





def train_one_epoch(model, optimizer, criterion, label_to_id):
    """Train on all rows, processing chunks in shuffled order."""
    model.train()

    total_loss = 0.0
    total_samples = 0

    print("\nCounting training chunks...", flush=True)

    total_chunks = sum(
        1
        for _ in pd.read_csv(
            TRAIN_FILE,
            usecols=[TARGET_COLUMN],
            chunksize=CHUNK_SIZE,
        )
    )

    chunk_order = list(range(total_chunks))
    random.shuffle(chunk_order)

    print(
        f"Training on {total_chunks} chunks in shuffled order",
        flush=True,
    )

    for chunk_number, chunk_index in enumerate(chunk_order, start=1):
        chunk = pd.read_csv(
            TRAIN_FILE,
            skiprows=range(1, chunk_index * CHUNK_SIZE + 1),
            nrows=CHUNK_SIZE,
        )

        features_tensor, labels_tensor = load_batch(
            chunk,
            label_to_id,
        )

        permutation = torch.randperm(len(labels_tensor))
        features_tensor = features_tensor[permutation]
        labels_tensor = labels_tensor[permutation]

        for start in range(0, len(labels_tensor), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(labels_tensor))

            batch_features = features_tensor[start:end]
            batch_labels = labels_tensor[start:end]

            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            batch_size = len(batch_labels)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        if (
            chunk_number == 1
            or chunk_number % PROGRESS_EVERY_CHUNKS == 0
            or chunk_number == total_chunks
        ):
            print(
                f"Training progress: {chunk_number}/{total_chunks} chunks",
                flush=True,
            )

        del chunk, features_tensor, labels_tensor, permutation

    if total_samples == 0:
        raise ValueError("No training rows were processed.")

    print(f"Rows used for training: {total_samples:,}", flush=True)

    return total_loss / total_samples




def validate_model(model, criterion, label_to_id):
    """Evaluate loss, accuracy, and macro F1 using a confusion matrix."""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )

    with torch.no_grad():
        for chunk_number, chunk in enumerate(
            pd.read_csv(VALIDATION_FILE, chunksize=CHUNK_SIZE),
            start=1,
        ):
            if (
                chunk_number == 1
                or chunk_number % PROGRESS_EVERY_CHUNKS == 0
            ):
                print(
                    f"Validation progress: chunk {chunk_number}",
                    flush=True,
                )

            features_tensor, labels_tensor = load_batch(
                chunk,
                label_to_id,
            )

            for start in range(0, len(labels_tensor), BATCH_SIZE):
                end = start + BATCH_SIZE

                batch_features = features_tensor[start:end]
                batch_labels = labels_tensor[start:end]

                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                predictions = outputs.argmax(dim=1)

                batch_size = len(batch_labels)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                correct_predictions += (
                    predictions == batch_labels
                ).sum().item()

                confusion += confusion_matrix(
                    batch_labels.numpy(),
                    predictions.numpy(),
                    labels=list(range(NUM_CLASSES)),
                )

            del chunk, features_tensor, labels_tensor

    if total_samples == 0:
        raise ValueError("No validation rows were processed.")

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    true_positives = np.diag(confusion).astype(np.float64)
    actual_counts = confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)
    denominators = actual_counts + predicted_counts

    class_f1 = np.divide(
        2.0 * true_positives,
        denominators,
        out=np.zeros(NUM_CLASSES, dtype=np.float64),
        where=denominators != 0,
    )

    macro_f1 = float(class_f1.mean())

    print("\nPer-class validation F1:", flush=True)

    for class_name, class_id in sorted(
        label_to_id.items(),
        key=lambda item: item[1],
    ):
        print(
            f"{class_name}: F1 = {class_f1[class_id]:.4f}",
            flush=True,
        )

    
    print("\nActual vs Predicted class counts:", flush=True)
    print(f"{'Class':30} {'Actual':>10} {'Predicted':>10}", flush=True)

    for class_name, class_id in sorted(
        label_to_id.items(),
        key=lambda item: item[1],
    ):
        actual = int(confusion[class_id, :].sum())
        predicted = int(confusion[:, class_id].sum())

        print(
            f"{class_name[:30]:30} {actual:10,} {predicted:10,}",
            flush=True,
        )

    return average_loss, accuracy, macro_f1



def main():
    """Verify inputs, train the CNN, and save the best checkpoint."""

    verify_feature_columns()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    label_to_id, class_names, class_weights = (
        prepare_labels_and_weights()
    )

    model = IntrusionCNN()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_macro_f1 = -1.0
    best_model_path = MODEL_DIR / "best_cnn_model.pt"

    print("\nStarting CNN training...", flush=True)
    print("Epochs:", EPOCHS, flush=True)
    print("Batch size:", BATCH_SIZE, flush=True)
    print("Chunk size:", CHUNK_SIZE, flush=True)
    print("CPU threads:", CPU_THREADS, flush=True)

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}", flush=True)

        train_loss = train_one_epoch(
            model,
            optimizer,
            criterion,
            label_to_id,
        )

        print(f"Training loss: {train_loss:.4f}", flush=True)

        val_loss, val_accuracy, val_macro_f1 = validate_model(
            model,
            criterion,
            label_to_id,
        )

        print(f"Validation loss: {val_loss:.4f}", flush=True)
        print(f"Validation accuracy: {val_accuracy:.4f}", flush=True)
        print(f"Validation macro F1: {val_macro_f1:.4f}", flush=True)

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
                    "validation_macro_f1": val_macro_f1,
                },
                best_model_path,
            )

            print(
                f"Saved new best model: {best_model_path}",
                flush=True,
            )

    print("\nCNN training completed.", flush=True)
    print(
        f"Best validation macro F1: {best_macro_f1:.4f}",
        flush=True,
    )
    print(f"Best model path: {best_model_path}", flush=True)


def evaluate_saved_model():
    """Evaluate the saved CNN without training again."""

    verify_feature_columns()

    model_path = MODEL_DIR / "best_cnn_model.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved CNN model not found: {model_path}"
        )

    print("Loading saved CNN model...", flush=True)

    label_to_id, class_names, class_weights = (
        prepare_labels_and_weights()
    )

    model = IntrusionCNN()

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("Evaluating saved model only...", flush=True)

    loss, accuracy, macro_f1 = validate_model(
        model,
        criterion,
        label_to_id,
    )

    print(f"\nValidation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation macro F1: {macro_f1:.4f}")




def inspect_training_chunk_labels():
    print("\nInspecting training CSV label order:")

    total_chunks = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            TRAIN_FILE,
            usecols=[TARGET_COLUMN],
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        total_chunks = chunk_number
        labels = chunk[TARGET_COLUMN]

        if (
            chunk_number <= 5
            or chunk_number % 100 == 0
        ):
            print(
                f"\nChunk {chunk_number}: "
                f"first={labels.iloc[0]!r}, "
                f"last={labels.iloc[-1]!r}"
            )
            print(labels.value_counts().to_string())

    print(f"\nTotal chunks inspected: {total_chunks}")




if __name__ == "__main__":
    inspect_training_chunk_labels()

