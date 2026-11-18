import pandas as pd

file_path = r"D:\Downloads\MachineLearningCSV\MachineLearningCVE\Monday-WorkingHours.pcap_ISCX.csv"

print("Reading dataset information...\n")

# Read only the header and first 5 rows
sample_df = pd.read_csv(file_path, nrows=5)

print("Number of columns:", len(sample_df.columns))

print("\nColumn names:")
for i, column in enumerate(sample_df.columns, start=1):
    print(f"{i}. {column}")

print("\nExact representation of the last column:")
print(repr(sample_df.columns[-1]))

print("\nFirst 5 values of the last column:")
print(sample_df.iloc[:, -1].tolist())
# Count all rows without loading the complete dataset
row_count = 0

for chunk in pd.read_csv(file_path, chunksize=100000):
    row_count += len(chunk)

print("\nTotal number of records:", row_count)

print("\nDataset structure:")
print(f"Rows    : {row_count}")
print(f"Columns : {len(sample_df.columns)}")
print(f"Features: {len(sample_df.columns) - 1}")
print("Target  : Label")