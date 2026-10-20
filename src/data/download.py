from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch UCI Heart Disease dataset, ID 45.
    dataset = fetch_ucirepo(id=45)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    print("Feature columns:")
    print(features.columns.tolist())

    print("\nTarget columns:")
    print(targets.columns.tolist())

    if targets.shape[1] != 1:
        raise ValueError("Expected exactly one target column.")

    # Combine features and target into a single CSV.
    df = features.copy()
    df["target"] = targets.iloc[:, 0]

    output_path = RAW_DIR / "heart_disease.csv"
    df.to_csv(output_path, index=False)

    print(f"\nDataset saved to: {output_path}")
    print(f"Dataset shape: {df.shape}")

    print("\nTarget values:")
    print(df["target"].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()