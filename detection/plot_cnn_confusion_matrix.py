import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FILE = (
    PROJECT_ROOT
    / "reports"
    / "sprint5"
    / "cnn_test_confusion_matrix.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "sprint5"
    / "cnn_test_confusion_matrix.png"
)

# Load the saved confusion matrix
confusion_df = pd.read_csv(CSV_FILE, index_col=0)

# Create the plot
plt.figure(figsize=(16, 13))

plt.imshow(
    confusion_df.values,
    interpolation="nearest",
    cmap="Blues",
)

plt.title("CNN Test Confusion Matrix", fontsize=16)
plt.colorbar(label="Number of Samples")

class_names = confusion_df.index.tolist()

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=90,
    fontsize=8,
)

plt.yticks(
    range(len(class_names)),
    class_names,
    fontsize=8,
)

plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("Actual Class", fontsize=12)

plt.tight_layout()

# Save the image
plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Confusion matrix image saved to: {OUTPUT_FILE}")