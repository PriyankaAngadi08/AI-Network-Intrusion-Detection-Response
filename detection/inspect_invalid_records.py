
from pathlib import Path
import pandas as pd

RAW_DIR = Path(
    r"D:\Downloads\MachineLearningCSV\MachineLearningCVE"
)

FEATURES = [
    " min_seg_size_forward",
    " Bwd Header Length"
]

CHUNK_SIZE = 100_000


def main():
    for csv_file in sorted(RAW_DIR.glob("*.csv")):
        print(f"\n{'=' * 70}")
        print("FILE:", csv_file.name)

        for chunk in pd.read_csv(
            csv_file,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):
            for feature in FEATURES:
                values = pd.to_numeric(
                    chunk[feature],
                    errors="coerce"
                )

                invalid_mask = values < 0

                if invalid_mask.any():
                    columns_to_show = [
                        feature,
                        " Label",
                        " Flow Duration",
                        " Total Fwd Packets",
                        " Total Backward Packets"
                    ]

                    available = [
                        col for col in columns_to_show
                        if col in chunk.columns
                    ]

                    print(
                        f"\nNegative values in {feature}:"
                    )

                    print(
                        chunk.loc[
                            invalid_mask, available
                        ].head(3).to_string(index=False)
                    )


if __name__ == "__main__":
    main()
