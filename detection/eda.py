import pandas as pd

file_path = r"D:\Downloads\MachineLearningCSV\MachineLearningCVE\Monday-WorkingHours.pcap_ISCX.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("CIC-IDS2017 EDA")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nTarget Labels:")
print(df[" Label"].value_counts())

print("\n" + "=" * 60)
print("NUMERICAL FEATURE ANALYSIS")
print("=" * 60)

numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

print("\nNumber of numerical features:")
print(len(numeric_columns))

print("\nNumerical feature names:")
print(numeric_columns.tolist())

print("\nNumerical feature statistics:")
print(df[numeric_columns].describe().T)

print("\n" + "=" * 60)
print("CATEGORICAL FEATURE ANALYSIS")
print("=" * 60)

categorical_columns = df.select_dtypes(include=["str"]).columns

print("\nNumber of categorical columns:")
print(len(categorical_columns))

print("\nCategorical column names:")
print(categorical_columns.tolist())

for column in categorical_columns:
    print("\n" + column)
    print(df[column].value_counts())


print("\n" + "=" * 60)
print("FEATURE DATA TYPE ANALYSIS")
print("=" * 60)

print("\nData types of all columns:")
print(df.dtypes)

print("\nData type summary:")
print(df.dtypes.value_counts())


print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing_values = df.isnull().sum()

print("\nTotal missing values:")
print(missing_values.sum())

print("\nColumns with missing values:")
print(missing_values[missing_values > 0])


print("\n" + "=" * 60)
print("MISSING VALUE PATTERN ANALYSIS")
print("=" * 60)

missing_rows = df[df.isnull().any(axis=1)]

print("\nNumber of rows containing missing values:")
print(len(missing_rows))

print("\nRows containing missing values:")
print(missing_rows.index.tolist())

print("\nLabels of rows containing missing values:")
print(missing_rows[" Label"].value_counts())

print("\nMissing-value rows:")
print(missing_rows[["Flow Bytes/s", " Label"]])

print("\n" + "=" * 60)
print("DUPLICATE RECORD ANALYSIS")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print("\nNumber of duplicate rows:")
print(duplicate_count)

print("\nPercentage of duplicate rows:")
print((duplicate_count / len(df)) * 100)


print("\n" + "=" * 60)
print("INFINITE VALUE ANALYSIS")
print("=" * 60)

infinite_values = df.select_dtypes(include=["int64", "float64"]).isin([float("inf"), float("-inf")])

print("\nTotal infinite values:")
print(infinite_values.sum().sum())

print("\nInfinite values by column:")
print(infinite_values.sum()[infinite_values.sum() > 0])


print("\n" + "=" * 60)
print("INVALID / INCONSISTENT VALUE ANALYSIS")
print("=" * 60)

# Check negative values
negative_values = (df.select_dtypes(include=["int64", "float64"]) < 0).sum()

print("\nColumns containing negative values:")
print(negative_values[negative_values > 0])

# Check destination port range
invalid_ports = ((df[" Destination Port"] < 0) | (df[" Destination Port"] > 65535)).sum()

print("\nInvalid Destination Port values:")
print(invalid_ports)

# Check negative packet counts
packet_columns = [
    " Total Fwd Packets",
    " Total Backward Packets"
]

print("\nNegative packet counts:")
for column in packet_columns:
    print(column, ":", (df[column] < 0).sum())


print("\n" + "=" * 60)
print("TARGET LABEL DISTRIBUTION")
print("=" * 60)

label_counts = df[" Label"].value_counts()

print("\nNumber of records for each label:")
print(label_counts)

print("\nPercentage of each label:")
print((label_counts / len(df) * 100).round(2))


print("\n" + "=" * 60)
print("FEATURE DISTRIBUTION ANALYSIS")
print("=" * 60)

numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

skewness = df[numeric_columns].skew()

print("\nFeature skewness:")
print(skewness.sort_values(ascending=False))



print("\n" + "=" * 60)
print("HIGH CORRELATION ANALYSIS")
print("=" * 60)

numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

correlation_matrix = df[numeric_columns].corr()

high_correlations = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):

        correlation = correlation_matrix.iloc[i, j]

        if abs(correlation) >= 0.90:
            high_correlations.append(
                (
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    correlation
                )
            )

print("\nHighly correlated feature pairs (|correlation| >= 0.90):")

for feature1, feature2, correlation in high_correlations:
    print(
        feature1,
        "<--->",
        feature2,
        ":",
        round(correlation, 4)
    )

print("\nTotal highly correlated pairs:")
print(len(high_correlations))



print("\n" + "=" * 60)
print("REDUNDANT FEATURE ANALYSIS")
print("=" * 60)

numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

correlation_matrix = df[numeric_columns].corr()

redundant_pairs = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):

        correlation = correlation_matrix.iloc[i, j]

        if abs(correlation) >= 0.9999:
            redundant_pairs.append(
                (
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    correlation
                )
            )

print("\nPotentially redundant feature pairs:")

for feature1, feature2, correlation in redundant_pairs:
    print(
        feature1,
        "<--->",
        feature2,
        ":",
        round(correlation, 4)
    )

print("\nTotal potentially redundant pairs:")
print(len(redundant_pairs))


print("\n" + "=" * 60)
print("DATA LEAKAGE ANALYSIS")
print("=" * 60)

print("\nTarget column:")
print(" Label")

print("\nChecking whether target label is included in numerical features:")

numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

if " Label" in numeric_columns:
    print("WARNING: Label is included in numerical features!")
else:
    print("OK: Label is not included in numerical features.")

print("\nChecking feature names for label-related information:")

label_related_features = [
    column for column in df.columns
    if "label" in column.lower()
]

print(label_related_features)

print("\nNumber of label-related columns:")
print(len(label_related_features))



print("\n" + "=" * 60)
print("IDENTIFIER FIELD ANALYSIS")
print("=" * 60)

print("\nAll dataset columns:")

for column in df.columns:
    print(column)

print("\nPotential identifier-related columns:")

identifier_keywords = [
    "flow id",
    "timestamp",
    "source ip",
    "destination ip",
    "id"
]

identifier_columns = []

for column in df.columns:
    column_lower = column.lower()

    for keyword in identifier_keywords:
        if keyword in column_lower:
            identifier_columns.append(column)
            break

print(identifier_columns)

print("\nNumber of potential identifier columns:")
print(len(identifier_columns))



print("\n" + "=" * 60)
print("IMPORTANT FEATURE DISTRIBUTIONS")
print("=" * 60)

import matplotlib.pyplot as plt
import numpy as np

features_to_plot = [
    " Flow Duration",
    " Total Fwd Packets",
    " Total Backward Packets",
    "Flow Bytes/s"
]

for feature in features_to_plot:

    values = df[feature].replace([np.inf, -np.inf], np.nan).dropna()

    plt.figure(figsize=(8, 5))

    plt.hist(
        values,
        bins=50
    )

    plt.title("Distribution of " + feature.strip())
    plt.xlabel(feature.strip())
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()