from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "heart_disease.csv"
PROCESSED_DIR = ROOT / "data" / "processed"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if "target" not in df.columns:
        raise ValueError("Dataset must contain a target column.")

    data = df.copy()

    # Convert common missing-value markers to NaN.
    data = data.replace(
        ["?", "NA", "N/A", ""],
        float("nan")
    )

    # Remove duplicate rows.
    data = data.drop_duplicates()

    # Remove records without a target.
    data["target"] = pd.to_numeric(
        data["target"],
        errors="coerce"
    )

    data = data.dropna(subset=["target"])

    # Verify this mapping against the downloaded UCI dataset variant.
    # For the Cleveland-style target, 0 means absence and values 1-4
    # indicate presence of disease.
    data["target"] = (
        data["target"] > 0
    ).astype(int)

    # Convert feature columns to numeric.
    for column in data.columns:
        if column != "target":
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    return data.reset_index(drop=True)


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            "Run python -m src.data.download first."
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_df = pd.read_csv(RAW_PATH)
    clean_df = clean_data(raw_df)

    output_path = (
        PROCESSED_DIR / "heart_disease_clean.csv"
    )

    clean_df.to_csv(output_path, index=False)

    print("Cleaned dataset saved:", output_path)
    print("Shape:", clean_df.shape)

    print("\nMissing values:")
    print(clean_df.isna().sum())

    print("\nTarget distribution:")
    print(clean_df["target"].value_counts())


if __name__ == "__main__":
    main()