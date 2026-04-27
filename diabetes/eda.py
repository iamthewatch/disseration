# eda.py — Exploratory Data Analysis (EDA) for the Diabetes dataset

import os
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "diabetes.csv")

FEATURE_NAMES_EN = {
    'Pregnancies':               'Pregnancies',
    'Glucose':                   'Glucose',
    'BloodPressure':             'Blood Pressure',
    'SkinThickness':             'Skin Thickness',
    'Insulin':                   'Insulin',
    'BMI':                       'BMI',
    'DiabetesPedigreeFunction':  'Pedigree Function',
    'Age':                       'Age',
    'Outcome':                   'Outcome',
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    print("=" * 60)
    print("DATA LOADING")
    print("=" * 60)
    print(f"Dataset shape (rows, columns): {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDescriptive statistics:")
    print(df.describe())

    return df


def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df["Outcome"].value_counts()
    labels = ["No Diabetes (0)", "Diabetes (1)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(labels, counts.values, color=["steelblue", "tomato"], edgecolor="black")
    axes[0].set_title("Class Distribution", fontsize=14)
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

    axes[1].pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=["steelblue", "tomato"],
        startangle=90,
    )
    axes[1].set_title("Class Share", fontsize=14)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] {path}")


def plot_feature_histograms(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        label = FEATURE_NAMES_EN.get(col, col)
        axes[i].hist(df[col], bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        axes[i].set_title(label, fontsize=12)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Histograms", fontsize=16, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_histograms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    corr_en = corr.rename(index=FEATURE_NAMES_EN, columns=FEATURE_NAMES_EN)

    sns.heatmap(
        corr_en,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def plot_boxplots_by_outcome(df: pd.DataFrame) -> None:
    features = ["Glucose", "BMI", "Age", "Insulin"]
    feature_labels = [FEATURE_NAMES_EN.get(f, f) for f in features]

    fig, axes = plt.subplots(1, len(features), figsize=(16, 6))

    for ax, feature, label in zip(axes, features, feature_labels):
        data_groups = [
            df[df["Outcome"] == 0][feature].dropna(),
            df[df["Outcome"] == 1][feature].dropna(),
        ]
        bp = ax.boxplot(
            data_groups,
            patch_artist=True,
            labels=["No Diabetes", "Diabetes"],
            medianprops=dict(color="black", linewidth=2),
        )
        bp["boxes"][0].set_facecolor("steelblue")
        bp["boxes"][1].set_facecolor("tomato")
        ax.set_title(label, fontsize=12)
        ax.set_ylabel("Value")

    plt.suptitle("Boxplots by Key Features (grouped by Outcome)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "boxplots_by_outcome.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    plot_class_distribution(df)
    plot_feature_histograms(df)
    plot_correlation_heatmap(df)
    plot_boxplots_by_outcome(df)
    print("\nEDA complete. All plots saved to:", PLOTS_DIR)
