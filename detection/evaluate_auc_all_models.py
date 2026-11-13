
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.special import softmax
from sklearn.metrics import roc_auc_score, average_precision_score


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

TEST_FILE = Path("data/processed/test_processed.csv")
FEATURE_FILE = Path("models/preprocessing/feature_columns.json")
OUTPUT_DIR = Path("reports/sprint8/auc")

TARGET_COLUMN = " Label"
CHUNK_SIZE = 10000
BATCH_SIZE = 8
CPU_THREADS = 2

DEVICE = torch.device("cpu")

torch.set_num_threads(CPU_THREADS)


# --------------------------------------------------
# NEURAL NETWORK ARCHITECTURES
# --------------------------------------------------

class IntrusionCNN(nn.Module):

    def __init__(self, num_classes=15):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 17, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class IntrusionBiLSTM(nn.Module):

    def __init__(self, num_classes=15):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


class CNNFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        return self.features(x)


class BiLSTMSequenceProcessor(nn.Module):

    def __init__(self):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x, _ = self.bilstm(x)
        x = x.mean(dim=1)
        return self.dropout(x)


class CNNBiLSTM(nn.Module):

    def __init__(self, num_classes=15):
        super().__init__()

        self.cnn = CNNFeatureExtractor()
        self.bilstm = BiLSTMSequenceProcessor()
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x = self.bilstm(x)
        return self.classifier(x)


# --------------------------------------------------
# LOAD CLASS AND FEATURE INFORMATION
# --------------------------------------------------

def load_metadata():

    with FEATURE_FILE.open("r", encoding="utf-8") as file:
        feature_columns = json.load(file)

    checkpoint = torch.load(
        "models/cnn/best_cnn_model_balanced.pt",
        map_location="cpu",
        weights_only=False
    )

    class_names = checkpoint["class_names"]

    print("Features:", len(feature_columns))
    print("Classes:", len(class_names))

    return feature_columns, class_names


# --------------------------------------------------
# LOAD NEURAL NETWORK
# --------------------------------------------------

def load_neural_model(model_name, class_names):

    if model_name == "CNN":
        path = "models/cnn/best_cnn_model_balanced.pt"
        model = IntrusionCNN(len(class_names))

    elif model_name == "BiLSTM":
        path = "models/bilstm/best_bilstm_model_unweighted.pt"
        model = IntrusionBiLSTM(len(class_names))

    else:
        path = "models/cnn_bilstm/best_cnn_bilstm.pt"
        model = CNNBiLSTM(len(class_names))

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    if checkpoint["class_names"] != class_names:
        raise ValueError(f"{model_name}: class ordering mismatch")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


# --------------------------------------------------
# LOAD BASELINE MODEL
# --------------------------------------------------

def load_baseline(model_name):

    paths = {
        "Logistic Regression":
            "models/baselines/logistic_regression_sgd.joblib",

        "Decision Tree":
            "models/baselines/decision_tree.joblib",

        "Random Forest":
            "models/baselines/random_forest.joblib"
    }

    return joblib.load(paths[model_name])


# --------------------------------------------------
# CREATE GROUND-TRUTH LABEL ARRAY
# --------------------------------------------------

def load_true_labels(class_names):

    label_to_id = {
        label: index
        for index, label in enumerate(class_names)
    }

    labels = []

    for chunk in pd.read_csv(
        TEST_FILE,
        usecols=[TARGET_COLUMN],
        chunksize=CHUNK_SIZE
    ):

        chunk_labels = chunk[TARGET_COLUMN].astype(str)

        unknown = set(chunk_labels.unique()) - set(label_to_id)

        if unknown:
            raise ValueError(f"Unknown test labels: {unknown}")

        labels.extend(
            chunk_labels.map(label_to_id).to_numpy()
        )

    return np.asarray(labels, dtype=np.int32)


# --------------------------------------------------
# GENERATE PROBABILITY SCORES
# --------------------------------------------------

def calculate_scores(
    model_name,
    feature_columns,
    class_names,
    y_true
):

    num_rows = len(y_true)
    num_classes = len(class_names)

    scores = np.empty(
        (num_rows, num_classes),
        dtype=np.float32
    )

    if model_name in [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ]:

        model = load_baseline(model_name)

        model_classes = list(model.classes_)

        if set(model_classes) != set(class_names):
            raise ValueError(
                f"{model_name}: baseline class ordering or labels mismatch"
            )

        column_order = [
            model_classes.index(label)
            for label in class_names
        ]

        offset = 0

        for chunk in pd.read_csv(
            TEST_FILE,
            chunksize=CHUNK_SIZE
        ):

            X = chunk[feature_columns].to_numpy(
                dtype=np.float32
            )

            probabilities = model.predict_proba(X)

            scores[offset:offset + len(X)] = (
                probabilities[:, column_order]
            )

            offset += len(X)

            del X, probabilities, chunk

    else:

        model = load_neural_model(
            model_name,
            class_names
        )

        offset = 0

        with torch.inference_mode():

            for chunk in pd.read_csv(
                TEST_FILE,
                chunksize=CHUNK_SIZE
            ):

                X = chunk[feature_columns].to_numpy(
                    dtype=np.float32
                )

                if not np.isfinite(X).all():
                    raise ValueError(
                        f"Invalid feature values found for {model_name}"
                    )

                for start in range(0, len(X), BATCH_SIZE):

                    batch = X[start:start + BATCH_SIZE]

                    tensor = torch.from_numpy(
                        batch.copy()
                    )

                    if model_name == "BiLSTM":
                        tensor = tensor.unsqueeze(-1)
                    else:
                        tensor = tensor.unsqueeze(1)

                    logits = model(tensor.to(DEVICE))

                    probabilities = torch.softmax(
                        logits, dim=1
                    ).cpu().numpy()

                    end = offset + len(batch)

                    scores[offset:end] = probabilities

                    offset = end

                del X, chunk

    if offset != num_rows:
        raise ValueError(
            f"{model_name}: processed {offset} rows, expected {num_rows}"
        )

    return scores


# --------------------------------------------------
# CALCULATE ROC-AUC AND PR-AUC
# --------------------------------------------------

def calculate_auc(model_name, scores, y_true, class_names):

    num_classes = len(class_names)

    per_class_results = []

    roc_values = []
    pr_values = []

    for class_id, class_name in enumerate(class_names):

        binary_true = (y_true == class_id).astype(np.uint8)

        positive_count = int(binary_true.sum())
        negative_count = len(binary_true) - positive_count

        if positive_count == 0 or negative_count == 0:

            roc_auc = np.nan
            pr_auc = np.nan

        else:

            roc_auc = roc_auc_score(
                binary_true,
                scores[:, class_id]
            )

            pr_auc = average_precision_score(
                binary_true,
                scores[:, class_id]
            )

            roc_values.append(roc_auc)
            pr_values.append(pr_auc)

        per_class_results.append({
            "Model": model_name,
            "Class": class_name,
            "Positive_Samples": positive_count,
            "Negative_Samples": negative_count,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc
        })

        print(
            f"{class_name}: "
            f"ROC-AUC={roc_auc:.5f}, "
            f"PR-AUC={pr_auc:.5f}"
            if not np.isnan(roc_auc)
            else f"{class_name}: AUC undefined"
        )

    macro_roc = (
        float(np.mean(roc_values))
        if roc_values else np.nan
    )

    macro_pr = (
        float(np.mean(pr_values))
        if pr_values else np.nan
    )

    summary = {
        "Model": model_name,
        "Macro_ROC_AUC": macro_roc,
        "Macro_PR_AUC": macro_pr
    }

    return summary, per_class_results


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_columns, class_names = load_metadata()

    y_true = load_true_labels(class_names)

    print("\nTest records:", len(y_true))

    model_names = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "CNN",
        "BiLSTM",
        "CNN-BiLSTM"
    ]

    all_summaries = []
    all_class_results = []

    for model_name in model_names:

        print("\n" + "=" * 55)
        print("Evaluating:", model_name)
        print("=" * 55)

        start_time = time.perf_counter()

        scores = calculate_scores(
            model_name,
            feature_columns,
            class_names,
            y_true
        )

        summary, class_results = calculate_auc(
            model_name,
            scores,
            y_true,
            class_names
        )

        summary["Evaluation_Seconds"] = (
            time.perf_counter() - start_time
        )

        all_summaries.append(summary)
        all_class_results.extend(class_results)

        del scores
        gc.collect()

        print("\nModel completed:", model_name)
        print("Macro ROC-AUC:", summary["Macro_ROC_AUC"])
        print("Macro PR-AUC:", summary["Macro_PR_AUC"])

    summary_df = pd.DataFrame(all_summaries)
    class_df = pd.DataFrame(all_class_results)

    summary_df.to_csv(
        OUTPUT_DIR / "auc_model_comparison.csv",
        index=False
    )

    class_df.to_csv(
        OUTPUT_DIR / "auc_per_class_results.csv",
        index=False
    )

    with (OUTPUT_DIR / "auc_model_comparison.json").open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            all_summaries,
            file,
            indent=4,
            allow_nan=True
        )

    print("\n" + "=" * 55)
    print("FINAL AUC COMPARISON")
    print("=" * 55)

    print(summary_df.to_string(index=False))

    print("\nSaved files:")
    print(OUTPUT_DIR / "auc_model_comparison.csv")
    print(OUTPUT_DIR / "auc_per_class_results.csv")
    print(OUTPUT_DIR / "auc_model_comparison.json")


if __name__ == "__main__":
    main()
