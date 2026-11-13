from pathlib import Path
import json
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HISTORY_FILE = (
    PROJECT_ROOT / "reports" / "sprint5"
    / "cnn_training_history.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "reports" / "sprint5"
    / "cnn_loss_curve.png"
)

with open(HISTORY_FILE, "r", encoding="utf-8") as file:
    history = json.load(file)

epochs = history["epoch"]
training_loss = history["training_loss"]
validation_loss = history["validation_loss"]

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    training_loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    validation_loss,
    marker="s",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training and Validation Loss")
plt.xticks(epochs)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()

print(f"Loss curve saved at: {OUTPUT_FILE}")