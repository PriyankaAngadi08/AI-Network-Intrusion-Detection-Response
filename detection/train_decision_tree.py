
from pathlib import Path
import json
import pandas as pd

import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "training_processed.csv"
VALIDATION_FILE = PROJECT_ROOT / "data" / "processed" / "validation_processed.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "baselines"
RESULTS_DIR = PROJECT_ROOT / "reports" / "sprint4"

RANDOM_STATE = 42
MAX_TRAINING_ROWS = 300_000
CHUNK_SIZE = 50_000
TARGET_COLUMN = " Label"


def load_feature_columns():
    feature_file = (
        PROJECT_ROOT / "models" / "preprocessing" / "feature_columns.json"
    )

    with open(feature_file, "r", encoding="utf-8") as file:
        return json.load(file)


def create_class_aware_sample(feature_columns):
    print("\nCreating class-aware training sample...")

    # First pass: count records belonging to each class.
    class_counts = pd.read_csv(
        TRAIN_FILE,
        usecols=[TARGET_COLUMN]
    )[TARGET_COLUMN].value_counts()

    # Allocate sample capacity proportionally, with a minimum for rare classes.
    # All available rows are retained when a class has fewer than 1,000 rows.
    allocations = {}
    remaining_budget = MAX_TRAINING_ROWS

    rare_classes = class_counts[class_counts <= 1000]

    for label, count in rare_classes.items():
        allocations[label] = int(count)
        remaining_budget -= int(count)

    common_classes = class_counts[class_counts > 1000]
    common_total = int(common_classes.sum())

    for label, count in common_classes.items():
        allocation = min(
            int(count),
            int(remaining_budget * count / common_total)
        )
        allocations[label] = allocation

    # Correct any rounding shortfall without exceeding available counts.
    current_total = sum(allocations.values())
    for label in class_counts.index:
        if current_total >= MAX_TRAINING_ROWS:
            break

        available = int(class_counts[label]) - allocations.get(label, 0)
        extra = min(available, MAX_TRAINING_ROWS - current_total)
        allocations[label] = allocations.get(label, 0) + extra
        current_total += extra

    print("\nRows allocated per class:")
    for label, count in allocations.items():
        print(f"{label}: {count}")

    # Second pass: randomly sample within each class.
    sampled_parts = []
    collected = {label: 0 for label in allocations}

    for chunk in pd.read_csv(
        TRAIN_FILE,
        chunksize=CHUNK_SIZE
    ):
        for label, target_count in allocations.items():
            needed = target_count - collected[label]

            if needed <= 0:
                continue

            class_rows = chunk[chunk[TARGET_COLUMN] == label]

            if class_rows.empty:
                continue

            take = min(needed, len(class_rows))

            selected = class_rows.sample(
                n=take,
                random_state=RANDOM_STATE + collected[label]
            )

            sampled_parts.append(selected)
            collected[label] += len(selected)

        if all(collected[label] >= allocations[label] for label in allocations):
            break

    if not sampled_parts:
        raise RuntimeError("No training rows were sampled.")

    sample = pd.concat(sampled_parts, ignore_index=True)

    # Keep feature order consistent with the saved preprocessing artifact.
    sample = sample[feature_columns + [TARGET_COLUMN]]

    sample = sample.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print(f"\nSample rows selected: {len(sample):,}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Number of classes represented: {sample[TARGET_COLUMN].nunique()}")

    print("\nFinal sample distribution:")
    print(sample[TARGET_COLUMN].value_counts().to_string())

    return sample


def train_decision_tree(sample):
    print("\nTraining Decision Tree...")

    X_train = sample.drop(columns=[TARGET_COLUMN])
    y_train = sample[TARGET_COLUMN].astype(str)

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    print("Decision Tree training completed.")
    print(f"Training rows: {len(X_train):,}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes learned: {len(model.classes_)}")
    print(f"Tree depth: {model.get_depth()}")
    print(f"Leaf nodes: {model.get_n_leaves()}")

    return model



def main():
    print("=== Decision Tree Baseline ===")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(f"Validation file not found: {VALIDATION_FILE}")

    # Step 1: Load the saved feature list.
    feature_columns = load_feature_columns()

    # Step 2: Create the training sample.
    sample_file = (
    PROJECT_ROOT
    / "reports"
    / "sprint4"
    / "decision_tree_training_sample_representative.csv"
    )

    sample = pd.read_csv(sample_file)
    print(f"Representative sample loaded: {len(sample):,} rows")

    # Step 3: Save the sample.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sample_file = RESULTS_DIR / "decision_tree_training_sample.csv"
    sample.to_csv(sample_file, index=False)

    print(f"\nSample saved to: {sample_file}")

    # Step 4: Train the Decision Tree.
    model = train_decision_tree(sample)

    # Step 5: Save the trained model.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_file = MODEL_DIR / "decision_tree.joblib"
    joblib.dump(model, model_file)

    print(f"\nModel saved to: {model_file}")
    print("Decision Tree training stage completed.")


if __name__ == "__main__":
    main()

