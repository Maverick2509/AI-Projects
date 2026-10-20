from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed"
    / "heart_disease_clean.csv"
)

FIG_DIR = ROOT / "reports" / "figures"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Run the download and preparation scripts first."
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("\nColumn information:")
    print(df.info())

    print("\nSummary statistics:")
    print(df.describe())

    print("\nMissing values:")
    print(df.isna().sum())

    # 1. Histograms
    df.hist(
        figsize=(14, 10),
        bins=20
    )

    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "histograms.png",
        dpi=150
    )
    plt.close()

    # 2. Correlation heatmap
    numeric_df = df.select_dtypes(
        include="number"
    )

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        center=0
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "correlation_heatmap.png",
        dpi=150
    )
    plt.close()

    # 3. Target class balance
    plt.figure(figsize=(6, 4))

    sns.countplot(
        data=df,
        x="target"
    )

    plt.title("Target Class Distribution")
    plt.xlabel("Target: 0 = absent, 1 = present")
    plt.ylabel("Number of records")
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "class_balance.png",
        dpi=150
    )
    plt.close()

    print("\nEDA plots saved in:", FIG_DIR)


if __name__ == "__main__":
    main()