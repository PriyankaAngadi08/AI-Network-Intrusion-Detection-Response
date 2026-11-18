import pandas as pd
import glob
import os

print("=" * 60)
print("CIC-IDS2017 ATTACK CLASS DISTRIBUTION")
print("=" * 60)

dataset_folder = r"D:\Downloads\MachineLearningCSV\MachineLearningCVE"

files = glob.glob(os.path.join(dataset_folder, "*.csv"))

print("\nNumber of CSV files found:", len(files))

all_labels = []

for file in files:
    print("\nProcessing:", os.path.basename(file))

    df = pd.read_csv(file)

    labels = df[" Label"].value_counts()

    all_labels.append(labels)

overall_labels = pd.concat(all_labels, axis=1).fillna(0).sum(axis=1)

overall_labels = overall_labels.sort_values(ascending=False)

print("\n" + "=" * 60)
print("OVERALL ATTACK CLASS DISTRIBUTION")
print("=" * 60)

print(overall_labels)

print("\nPercentage distribution:")

percentage = (overall_labels / overall_labels.sum() * 100).round(2)

print(percentage)


print("\n" + "=" * 60)
print("CLASS IMBALANCE ANALYSIS")
print("=" * 60)

largest_class = overall_labels.max()
smallest_class = overall_labels.min()

largest_label = overall_labels.idxmax()
smallest_label = overall_labels.idxmin()

print("\nLargest class:")
print(largest_label, ":", int(largest_class))

print("\nSmallest class:")
print(smallest_label, ":", int(smallest_class))

print("\nImbalance ratio:")
print(round(largest_class / smallest_class, 2))