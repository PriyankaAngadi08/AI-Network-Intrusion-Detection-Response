import pandas as pd
from pathlib import Path
from collections import Counter

# Path to CIC-IDS2017 dataset
DATASET_PATH = Path(r"D:\Downloads\MachineLearningCSV\MachineLearningCVE")

# Get all CSV files
csv_files = list(DATASET_PATH.glob("*.csv"))

label_counts = Counter()
total_records = 0

print("=" * 70)
print("CIC-IDS2017 OVERALL LABEL DISTRIBUTION")
print("=" * 70)

for file_path in csv_files:

    print(f"\nProcessing: {file_path.name}")

    for chunk in pd.read_csv(file_path, chunksize=100000):

        # Last column is the label
        labels = chunk.iloc[:, -1].astype(str).str.strip()

        label_counts.update(labels)
        total_records += len(chunk)


print("\n" + "=" * 70)
print("OVERALL LABEL DISTRIBUTION")
print("=" * 70)

for label, count in label_counts.most_common():

    percentage = (count / total_records) * 100

    print(f"{label:35} {count:10} records   {percentage:8.2f}%")

print("\n" + "=" * 70)
print(f"TOTAL RECORDS: {total_records}")
print(f"TOTAL UNIQUE LABELS: {len(label_counts)}")
print("=" * 70)