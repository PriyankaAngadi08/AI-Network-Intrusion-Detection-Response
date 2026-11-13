from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix


# ============================================================
# Sprint 6: BiLSTM Intrusion Detection
# This script is independent of train_cnn.py.
# It uses the existing processed CIC-IDS2017 splits and
# preprocessing artifacts. It does not fit a new scaler.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "training_processed.csv"
VALIDATION_FILE = PROJECT_ROOT / "data" / "processed" / "validation_processed.csv"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test_processed.csv"

FEATURE_FILE = PROJECT_ROOT / "models" / "preprocessing" / "feature_columns.json"
PREPROCESSING_CONFIG_FILE = (
    PROJECT_ROOT / "models" / "preprocessing" / "preprocessing_config.json"
)

MODEL_DIR = PROJECT_ROOT / "models" / "bilstm"
REPORT_DIR = PROJECT_ROOT / "reports" / "sprint6"

MODEL_PATH = MODEL_DIR / "best_bilstm_model_unweighted.pt"
CONFIG_PATH = REPORT_DIR / "bilstm_experiment_config.json"
RESULTS_PATH = REPORT_DIR / "bilstm_results.json"
CONFUSION_MATRIX_PATH = REPORT_DIR / "bilstm_test_confusion_matrix.csv"
CLASSIFICATION_REPORT_PATH = REPORT_DIR / "bilstm_test_classification_report.txt"

RANDOM_STATE = 42
CHUNK_SIZE = 500
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.2
CPU_THREADS = 2
MAX_SAMPLES_PER_CLASS = 500
VALIDATION_SAMPLES_PER_CLASS = 200
MAX_CLASS_WEIGHT = 100.0
PROGRESS_EVERY_CHUNKS = 100

TARGET_COLUMN = " Label"
NUM_CLASSES = 15
NUM_FEATURES = 70

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.set_num_threads(CPU_THREADS)


class IntrusionBiLSTM(nn.Module):
    """Small BiLSTM over the ordered 70 feature positions."""

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,  # PyTorch LSTM dropout applies only with >1 layer.
        )

        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(HIDDEN_SIZE * 2, NUM_CLASSES)

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        pooled = sequence_output.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def load_feature_columns():
    with open(FEATURE_FILE, "r", encoding="utf-8") as file:
        feature_columns = json.load(file)

    if len(feature_columns) != NUM_FEATURES:
        raise ValueError(
            f"Expected {NUM_FEATURES} features, found {len(feature_columns)}."
        )

    return feature_columns


def verify_inputs(feature_columns):
    """Verify headers and exact feature order without loading CSV data."""

    for csv_file in (TRAIN_FILE, VALIDATION_FILE, TEST_FILE):
        if not csv_file.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_file}")

        actual_columns = pd.read_csv(csv_file, nrows=0).columns.tolist()

        if TARGET_COLUMN not in actual_columns:
            raise ValueError(
                f"Target column {TARGET_COLUMN!r} missing from {csv_file.name}."
            )

        actual_features = [
            column for column in actual_columns if column != TARGET_COLUMN
        ]

        if actual_features != feature_columns:
            raise ValueError(
                f"Feature order mismatch in {csv_file.name}. "
                "The saved preprocessing feature order must be preserved."
            )

        print(
            f"{csv_file.name}: {len(actual_features)} features verified.",
            flush=True,
        )

    print("All input headers and feature orders verified.", flush=True)


def collect_training_labels(feature_columns):
    """Count training labels in chunks and build CNN-compatible mapping."""

    counts = {}

    for chunk in pd.read_csv(
        TRAIN_FILE,
        usecols=[TARGET_COLUMN],
        chunksize=CHUNK_SIZE,
    ):
        values = chunk[TARGET_COLUMN].astype(str).value_counts()

        for label, count in values.items():
            counts[label] = counts.get(label, 0) + int(count)

    class_names = sorted(counts)

    if len(class_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} training classes, found {len(class_names)}."
        )

    label_to_id = {
        label: index for index, label in enumerate(class_names)
    }

    total_rows = sum(counts.values())
    class_weights = []

    for label in class_names:
        weight = total_rows / (NUM_CLASSES * counts[label])
        class_weights.append(min(weight, MAX_CLASS_WEIGHT))

    print(f"Training rows counted: {total_rows:,}", flush=True)
    print("Class mapping (same alphabetical strategy as Sprint 5 CNN):")
    for label, weight in zip(class_names, class_weights):
        print(
            f"  ID {label_to_id[label]:2d}: {label!r} | "
            f"count={counts[label]:,} | weight={weight:.4f}",
            flush=True,
        )

    return label_to_id, class_names, counts, class_weights


def make_class_aware_sample(feature_columns, label_to_id):
    """
    Build a bounded, class-aware reservoir sample.
    Only up to MAX_SAMPLES_PER_CLASS rows per class are retained.
    """

    rng = np.random.default_rng(RANDOM_STATE)
    feature_buffers = {
        class_id: np.empty(
            (MAX_SAMPLES_PER_CLASS, NUM_FEATURES), dtype=np.float32
        )
        for class_id in range(NUM_CLASSES)
    }
    seen = np.zeros(NUM_CLASSES, dtype=np.int64)
    kept = np.zeros(NUM_CLASSES, dtype=np.int64)

    use_columns = feature_columns + [TARGET_COLUMN]

    print("\nCreating bounded class-aware training sample...", flush=True)

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            TRAIN_FILE,
            usecols=use_columns,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        labels = chunk[TARGET_COLUMN].astype(str).map(label_to_id).to_numpy()
        features = chunk[feature_columns].to_numpy(dtype=np.float32, copy=True)

        if not np.isfinite(features).all():
            raise ValueError(
                f"Non-finite feature value found in training chunk {chunk_number}."
            )

        for row, class_id_value in zip(features, labels):
            class_id = int(class_id_value)
            seen[class_id] += 1

            if kept[class_id] < MAX_SAMPLES_PER_CLASS:
                feature_buffers[class_id][kept[class_id]] = row
                kept[class_id] += 1
            else:
                replacement = rng.integers(0, seen[class_id])
                if replacement < MAX_SAMPLES_PER_CLASS:
                    feature_buffers[class_id][replacement] = row

        if chunk_number % PROGRESS_EVERY_CHUNKS == 0:
            print(
                f"  Scanned {chunk_number} chunks; "
                f"rows seen: {int(seen.sum()):,}",
                flush=True,
            )

        del chunk, labels, features

    x_parts = []
    y_parts = []

    for class_id in range(NUM_CLASSES):
        count = int(kept[class_id])
        if count == 0:
            raise ValueError(
                f"No training examples collected for class ID {class_id}."
            )
        x_parts.append(feature_buffers[class_id][:count].copy())
        y_parts.append(np.full(count, class_id, dtype=np.int64))

    x_sample = np.concatenate(x_parts, axis=0)
    y_sample = np.concatenate(y_parts, axis=0)

    permutation = rng.permutation(len(y_sample))
    x_sample = x_sample[permutation]
    y_sample = y_sample[permutation]

    print(f"Class-aware sample size: {len(y_sample):,}", flush=True)
    for class_id, class_name in enumerate(sorted(label_to_id, key=label_to_id.get)):
        print(
            f"  {class_name}: {int(np.sum(y_sample == class_id)):,}",
            flush=True,
        )

    return x_sample, y_sample, kept.tolist()



def make_validation_sample(feature_columns, label_to_id):
    """Create a small class-aware validation sample without loading the CSV fully."""

    rng = np.random.default_rng(RANDOM_STATE + 1)
    buffers = {
        class_id: np.empty(
            (VALIDATION_SAMPLES_PER_CLASS, NUM_FEATURES), dtype=np.float32
        )
        for class_id in range(NUM_CLASSES)
    }
    seen = np.zeros(NUM_CLASSES, dtype=np.int64)
    kept = np.zeros(NUM_CLASSES, dtype=np.int64)
    use_columns = feature_columns + [TARGET_COLUMN]

    print("\nCreating bounded class-aware validation sample...", flush=True)

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            VALIDATION_FILE,
            usecols=use_columns,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        labels = chunk[TARGET_COLUMN].astype(str).map(label_to_id).to_numpy()
        features = chunk[feature_columns].to_numpy(dtype=np.float32, copy=True)

        if not np.isfinite(features).all() or (labels < 0).any():
            raise ValueError(
                f"Invalid feature or unknown label in validation chunk {chunk_number}."
            )

        for row, class_id_value in zip(features, labels):
            class_id = int(class_id_value)
            seen[class_id] += 1

            if kept[class_id] < VALIDATION_SAMPLES_PER_CLASS:
                buffers[class_id][kept[class_id]] = row
                kept[class_id] += 1
            else:
                replacement = rng.integers(0, seen[class_id])
                if replacement < VALIDATION_SAMPLES_PER_CLASS:
                    buffers[class_id][replacement] = row

        del chunk, labels, features

    x_parts = []
    y_parts = []
    for class_id in range(NUM_CLASSES):
        count = int(kept[class_id])
        if count == 0:
            raise ValueError(
                f"No validation examples collected for class ID {class_id}."
            )
        x_parts.append(buffers[class_id][:count].copy())
        y_parts.append(np.full(count, class_id, dtype=np.int64))

    x_validation = np.concatenate(x_parts, axis=0)
    y_validation = np.concatenate(y_parts, axis=0)
    permutation = rng.permutation(len(y_validation))

    x_validation = x_validation[permutation]
    y_validation = y_validation[permutation]

    print(
        f"Validation sample size: {len(y_validation):,} "
        f"(maximum {VALIDATION_SAMPLES_PER_CLASS} per class)",
        flush=True,
    )
    return x_validation, y_validation


def evaluate_sample(model, x_sample, y_sample, criterion, device):
    """Evaluate a bounded in-memory validation sample in mini-batches."""

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for start in range(0, len(y_sample), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_x = torch.from_numpy(x_sample[start:end]).to(device)
            batch_x = batch_x.unsqueeze(-1)
            batch_y = torch.from_numpy(y_sample[start:end]).long().to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            count = len(batch_y)

            total_loss += loss.item() * count
            total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total_seen += count

            del batch_x, batch_y, logits, loss

    return total_loss / total_seen, total_correct / total_seen


def train_epoch(model, optimizer, criterion, x_sample, y_sample, device):
    model.train()
    permutation = np.random.permutation(len(y_sample))
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for start in range(0, len(permutation), BATCH_SIZE):
        indices = permutation[start:start + BATCH_SIZE]

        batch_x = torch.from_numpy(x_sample[indices]).to(device)
        batch_x = batch_x.unsqueeze(-1)  # (batch, 70, 1)
        batch_y = torch.from_numpy(y_sample[indices]).long().to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        batch_count = len(batch_y)
        total_loss += loss.item() * batch_count
        total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        total_seen += batch_count

        del batch_x, batch_y, logits, loss

    return total_loss / total_seen, total_correct / total_seen


def evaluate(model, csv_file, feature_columns, label_to_id, criterion, device):
    """Evaluate in chunks and retain only aggregate metrics/confusion matrix."""

    model.eval()
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_loss = 0.0
    total_correct = 0
    total_rows = 0
    use_columns = feature_columns + [TARGET_COLUMN]

    with torch.no_grad():
        for chunk_number, chunk in enumerate(
            pd.read_csv(
                csv_file,
                usecols=use_columns,
                chunksize=CHUNK_SIZE,
            ),
            start=1,
        ):
            features = chunk[feature_columns].to_numpy(
                dtype=np.float32, copy=True
            )
            labels = (
                chunk[TARGET_COLUMN]
                .astype(str)
                .map(label_to_id)
                .to_numpy(dtype=np.int64)
            )

            if not np.isfinite(features).all() or (labels < 0).any():
                raise ValueError(
                    f"Invalid feature or unknown label in "
                    f"{csv_file.name}, chunk {chunk_number}."
                )

            for start in range(0, len(labels), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_x = torch.from_numpy(features[start:end]).to(device)
                batch_x = batch_x.unsqueeze(-1)
                batch_y = torch.from_numpy(labels[start:end].copy()).long().to(device)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                predictions = logits.argmax(dim=1)

                count = len(batch_y)
                total_loss += loss.item() * count
                total_correct += (predictions == batch_y).sum().item()
                total_rows += count

                matrix += confusion_matrix(
                    batch_y.cpu().numpy(),
                    predictions.cpu().numpy(),
                    labels=list(range(NUM_CLASSES)),
                )

                del batch_x, batch_y, logits, loss, predictions

            del chunk, features, labels

            if chunk_number % PROGRESS_EVERY_CHUNKS == 0:
                print(
                    f"  {csv_file.name}: processed "
                    f"{total_rows:,} rows",
                    flush=True,
                )

    return {
        "loss": total_loss / total_rows,
        "accuracy": total_correct / total_rows,
        "rows": total_rows,
        "confusion_matrix": matrix,
    }


def main():
    start_time = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    print("Device:", device, flush=True)
    print("CPU threads:", torch.get_num_threads(), flush=True)
    print("CSV chunk size:", CHUNK_SIZE, flush=True)
    print("Batch size:", BATCH_SIZE, flush=True)

    feature_columns = load_feature_columns()
    verify_inputs(feature_columns)

    label_to_id, class_names, training_counts, class_weights = (
        collect_training_labels(feature_columns)
    )

    x_sample, y_sample, sampled_counts = make_class_aware_sample(
        feature_columns, label_to_id
    )
    x_validation_sample, y_validation_sample = make_validation_sample(
        feature_columns, label_to_id
    )

    model = IntrusionBiLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_validation_loss = float("inf")
    history = []

    print("\nStarting BiLSTM training...", flush=True)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_accuracy = train_epoch(
            model, optimizer, criterion, x_sample, y_sample, device
        )

        validation_loss, validation_accuracy = evaluate_sample(
            model,
            x_validation_sample,
            y_validation_sample,
            criterion,
            device,
        )

        epoch_result = {
            "epoch": epoch,
            "training_loss": train_loss,
            "training_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(epoch_result)

        print(
            f"\nEpoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"train_accuracy={train_accuracy:.4f} | "
            f"val_loss={validation_loss:.4f} | "
            f"val_accuracy={validation_accuracy:.4f}",
            flush=True,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "label_to_id": label_to_id,
                    "feature_columns": feature_columns,
                    "model_config": {
                        "input_size": 1,
                        "sequence_length": NUM_FEATURES,
                        "hidden_size": HIDDEN_SIZE,
                        "num_layers": NUM_LAYERS,
                        "bidirectional": True,
                        "dropout": DROPOUT,
                        "number_of_classes": NUM_CLASSES,
                    },
                },
                MODEL_PATH,
            )
            print("Saved best validation checkpoint.", flush=True)

    # Release training sample before final test evaluation.
    del x_sample, y_sample
    del x_validation_sample, y_validation_sample
    del optimizer, criterion

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Recreate criterion for final test evaluation.
    criterion = nn.CrossEntropyLoss()

    print("\nEvaluating best checkpoint on full validation split...", flush=True)
    full_validation_result = evaluate(
        model,
        VALIDATION_FILE,
        feature_columns,
        label_to_id,
        criterion,
        device,
    )

    print("\nEvaluating best checkpoint on test split...", flush=True)
    test_result = evaluate(
        model,
        TEST_FILE,
        feature_columns,
        label_to_id,
        criterion,
        device,
    )

    matrix = test_result.pop("confusion_matrix")
    matrix_frame = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    )
    matrix_frame.to_csv(CONFUSION_MATRIX_PATH)

    # Derive per-class metrics directly from the accumulated confusion matrix.
    true_positive = np.diag(matrix).astype(np.float64)
    actual_count = matrix.sum(axis=1).astype(np.float64)
    predicted_count = matrix.sum(axis=0).astype(np.float64)

    precision = np.divide(
        true_positive, predicted_count,
        out=np.zeros_like(true_positive), where=predicted_count != 0
    )
    recall = np.divide(
        true_positive, actual_count,
        out=np.zeros_like(true_positive), where=actual_count != 0
    )
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(true_positive), where=(precision + recall) != 0
    )

    total = int(matrix.sum())
    macro_precision = float(precision.mean())
    macro_recall = float(recall.mean())
    macro_f1 = float(f1.mean())
    weighted_f1 = float(
        np.sum(f1 * actual_count) / actual_count.sum()
    )

    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("Class-wise test metrics\n")
        file.write("=" * 72 + "\n")
        for index, class_name in enumerate(class_names):
            file.write(
                f"{class_name}: precision={precision[index]:.6f}, "
                f"recall={recall[index]:.6f}, "
                f"f1={f1[index]:.6f}, "
                f"support={int(actual_count[index])}\n"
            )

    results = {
        "experiment_name": "Sprint 6 BiLSTM Intrusion Detection",
        "dataset": "CIC-IDS2017",
        "classification_type": "Multiclass classification",
        "number_of_classes": NUM_CLASSES,
        "number_of_input_features": NUM_FEATURES,
        "input_shape": "(batch_size, 70, 1)",
        "sequence_interpretation": (
            "Feature-position sequence, not a chronological network-traffic "
            "sequence. The 70 positions are ordered tabular features."
        ),
        "training_rows_in_full_split": int(sum(training_counts.values())),
        "training_sample_rows": int(sum(sampled_counts)),
        "maximum_samples_per_class": MAX_SAMPLES_PER_CLASS,
        "sampled_rows_per_class": {
            class_names[index]: int(sampled_counts[index])
            for index in range(NUM_CLASSES)
        },
        "class_names": class_names,
        "label_to_id": label_to_id,
        "training_history": history,
        "full_validation_results": {
            "loss": full_validation_result["loss"],
            "accuracy": full_validation_result["accuracy"],
            "rows": full_validation_result["rows"],
        },
        "test_results": {
            "loss": test_result["loss"],
            "accuracy": test_result["accuracy"],
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "rows": test_result["rows"],
        },
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "confusion_matrix_path": str(
            CONFUSION_MATRIX_PATH.relative_to(PROJECT_ROOT)
        ),
        "classwise_report_path": str(
            CLASSIFICATION_REPORT_PATH.relative_to(PROJECT_ROOT)
        ),
        "training_configuration": {
            "chunk_size": CHUNK_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "validation_checkpoint_selection": (
                "Class-aware validation sample, maximum 200 per class; "
                "full validation evaluated once after training."
            ),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "hidden_size": HIDDEN_SIZE,
            "lstm_layers": NUM_LAYERS,
            "bidirectional": True,
            "dropout": DROPOUT,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "class_imbalance_strategy": (
                "Class-aware reservoir sampling without class weights"
            ),
            "cpu_threads": CPU_THREADS,
        },
        "elapsed_seconds": time.time() - start_time,
        "limitations": [
            "The dataset does not provide the metadata needed to reconstruct "
            "reliable chronological flow sequences.",
            "The BiLSTM processes the ordered 70 tabular feature positions; "
            "these are not time steps.",
            "Feature order is inherited from the saved preprocessing schema.",
            "Training uses a bounded class-aware sample because of RAM limits.",
        ],
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results["test_results"], file, indent=4)

    print("\n========== BiLSTM TEST RESULTS ==========", flush=True)
    print(f"Test rows: {test_result['rows']:,}", flush=True)
    print(f"Accuracy: {test_result['accuracy']:.4f}", flush=True)
    print(f"Macro Precision: {macro_precision:.4f}", flush=True)
    print(f"Macro Recall: {macro_recall:.4f}", flush=True)
    print(f"Macro F1: {macro_f1:.4f}", flush=True)
    print(f"Weighted F1: {weighted_f1:.4f}", flush=True)
    print("\nSaved model:", MODEL_PATH, flush=True)
    print("Saved experiment config:", CONFIG_PATH, flush=True)
    print("Saved results:", RESULTS_PATH, flush=True)
    print("Saved confusion matrix:", CONFUSION_MATRIX_PATH, flush=True)
    print("Saved class-wise report:", CLASSIFICATION_REPORT_PATH, flush=True)


if __name__ == "__main__":
    main()
