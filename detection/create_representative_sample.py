
from pathlib import Path
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    PROJECT_ROOT / "data" / "processed" / "training_processed.csv"
)
FEATURE_FILE = (
    PROJECT_ROOT / "models" / "preprocessing" / "feature_columns.json"
)
OUTPUT_FILE = (
    PROJECT_ROOT / "reports" / "sprint4"
    / "decision_tree_training_sample_representative.csv"
)

TARGET_COLUMN = " Label"
MAX_TRAINING_ROWS = 300_000
CHUNK_SIZE = 50_000
RANDOM_STATE = 42


def main():
    print("=== Create Representative Training Sample ===")

    with open(FEATURE_FILE, "r", encoding="utf-8") as file:
        feature_columns = json.load(file)

    # Count each class without loading all feature columns.
    counts = pd.read_csv(
        TRAIN_FILE,
        usecols=[TARGET_COLUMN]
    )[TARGET_COLUMN].value_counts()

    print(f"Classes found: {len(counts)}")
    print(f"Total training rows: {int(counts.sum()):,}")

    # Allocate the 300,000-row budget proportionally,
    # retaining every record for classes with <= 1,000 examples.
    allocations = {}
    rare_classes = counts[counts <= 1000]

    for label, count in rare_classes.items():
        allocations[label] = int(count)

    remaining = MAX_TRAINING_ROWS - sum(allocations.values())
    common_classes = counts[counts > 1000]
    common_total = int(common_classes.sum())

    for label, count in common_classes.items():
        allocations[label] = min(
            int(count),
            int(remaining * count / common_total)
        )

    # Distribute any rounding remainder.
    current_total = sum(allocations.values())

    for label in counts.index:
        if current_total >= MAX_TRAINING_ROWS:
            break

        available = int(counts[label]) - allocations.get(label, 0)
        extra = min(available, MAX_TRAINING_ROWS - current_total)
        allocations[label] = allocations.get(label, 0) + extra
        current_total += extra

    # Reservoir sampling separately for each class.
    rng = np.random.default_rng(RANDOM_STATE)
    reservoirs = {label: [] for label in allocations}
    seen = {label: 0 for label in allocations}

    print("\nScanning all training chunks...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(TRAIN_FILE, chunksize=CHUNK_SIZE),
        start=1
    ):
        for label, quota in allocations.items():
            rows = chunk[chunk[TARGET_COLUMN] == label]

            for row in rows.itertuples(index=False, name=None):
                seen[label] += 1

                if quota == 0:
                    continue

                if len(reservoirs[label]) < quota:
                    reservoirs[label].append(row)
                else:
                    position = int(rng.integers(0, seen[label]))

                    if position < quota:
                        reservoirs[label][position] = row

        print(f"Processed chunk {chunk_number}")

    # Rebuild the sampled dataset.
    all_rows = []

    for label, rows in reservoirs.items():
        all_rows.extend(rows)

    column_names = list(chunk.columns)
    sample = pd.DataFrame(all_rows, columns=column_names)

    # Preserve saved feature order and target position.
    sample = sample[feature_columns + [TARGET_COLUMN]]
    sample = sample.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_FILE, index=False)

    print("\nSample creation completed.")
    print(f"Sample rows: {len(sample):,}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Classes represented: {sample[TARGET_COLUMN].nunique()}")
    print("\nClass distribution:")
    print(sample[TARGET_COLUMN].value_counts().to_string())
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
