
from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_FILE = (
    PROJECT_ROOT
    / "reports"
    / "sprint4"
    / "decision_tree_training_sample_representative.csv"
)

MODEL_DIR = PROJECT_ROOT / "models" / "baselines"

TARGET_COLUMN = " Label"
RANDOM_STATE = 42


def main():
    print("=== Random Forest Baseline ===")

    print("\nLoading representative training sample...")
    data = pd.read_csv(SAMPLE_FILE)

    X_train = data.drop(columns=[TARGET_COLUMN])
    y_train = data[TARGET_COLUMN]

    print(f"Training rows: {len(data):,}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {y_train.nunique()}")

    model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_file = MODEL_DIR / "random_forest.joblib"
    joblib.dump(model, model_file)

    print("\nRandom Forest training completed.")
    print(f"Training rows: {len(X_train):,}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes learned: {len(model.classes_)}")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Model saved to: {model_file}")


if __name__ == "__main__":
    main()
