
from pathlib import Path

import pandas as pd


RAW_DIR = Path(
    r"D:\Downloads\MachineLearningCSV\MachineLearningCVE"
)

FEATURES = [
    " min_seg_size_forward",
    " Bwd Header Length",
]

CHUNK_SIZE = 100_000


def main():
    for feature in FEATURES:
        print(f"\n=== Negative values: {feature} ===")
        total_negative = 0

        for csv_file in sorted(RAW_DIR.glob("*.csv")):
            file_negative = 0
            examples = []

            for chunk in pd.read_csv(
                csv_file,
                usecols=[feature],
                chunksize=CHUNK_SIZE,
            ):
                values = pd.to_numeric(
                    chunk[feature],
                    errors="coerce",
                )

                negative_values = values[values < 0]
                file_negative += len(negative_values)

                if len(examples) < 5 and not negative_values.empty:
                    examples.extend(
                        negative_values.head(
                            5 - len(examples)
                        ).tolist()
                    )

            total_negative += file_negative

            if file_negative > 0:
                print(f"\nFile: {csv_file.name}")
                print("Negative records:", file_negative)
                print("Example values:", examples)

        print("\nTotal negative records:", total_negative)


if __name__ == "__main__":
    main()
