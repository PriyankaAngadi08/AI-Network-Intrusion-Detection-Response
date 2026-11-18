import pandas as pd
import os

# Folder containing the CIC-IDS2017 CSV files
dataset_path = r"D:\Downloads\MachineLearningCSV\MachineLearningCVE"

# All 8 CIC-IDS2017 CSV files
files = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv"
]

total_rows = 0

print("=" * 70)
print("CIC-IDS2017 DATASET ANALYSIS")
print("=" * 70)

for file_name in files:

    file_path = os.path.join(dataset_path, file_name)

    print(f"\nProcessing: {file_name}")

    file_rows = 0
    label_counts = {}

    # Process the file in chunks to avoid loading the entire file into RAM
    for chunk in pd.read_csv(file_path, chunksize=100000):

        file_rows += len(chunk)

        # Strip whitespace only for temporary analysis of labels
        labels = chunk.iloc[:, -1].astype(str).str.strip()

        counts = labels.value_counts()

        for label, count in counts.items():
            label_counts[label] = label_counts.get(label, 0) + count

    total_rows += file_rows

    print(f"Number of records: {file_rows}")

    print("Labels:")
    for label, count in label_counts.items():
        print(f"    {label}: {count}")

print("\n" + "=" * 70)
print(f"TOTAL RECORDS ACROSS ALL FILES: {total_rows}")
print("=" * 70)