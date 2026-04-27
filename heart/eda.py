# eda.py — Exploratory Data Analysis (EDA)
# Dataset: heart.csv (Heart Disease Prediction)

import os
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

DATA_PATH = "data/heart.csv"
PLOTS_DIR = "heart/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURE_NAMES_EN = {
    'age':       'Age',
    'sex':       'Sex',
    'cp':        'Chest Pain Type',
    'trestbps':  'Resting Blood Pressure',
    'chol':      'Cholesterol',
    'fbs':       'Fasting Blood Sugar',
    'restecg':   'Resting ECG',
    'thalach':   'Max Heart Rate',
    'exang':     'Exercise Angina',
    'oldpeak':   'ST Depression',
    'slope':     'ST Slope',
    'ca':        'Major Vessels',
    'thal':      'Thalassemia',
    'target':    'Target',
}

# ── 1. Load dataset ───────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("DATASET SHAPE:")
print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\nDATA TYPES:")
print(df.dtypes)

print("\nMISSING VALUES:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "  No missing values found")

print("\nDESCRIPTIVE STATISTICS:")
print(df.describe())

# ── 2. Target class distribution ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
counts = df["target"].value_counts().sort_index()
bars = ax.bar(
    ["No Disease (0)", "Disease (1)"],
    counts.values,
    color=["#4C72B0", "#DD8452"],
    edgecolor="black",
)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            str(val), ha="center", va="bottom", fontweight="bold")

ax.set_title("Target Class Distribution", fontsize=14)
ax.set_ylabel("Number of Patients")
ax.set_xlabel("Class (Target)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=150)
plt.close()
print(f"\n[OK] Saved: {PLOTS_DIR}/class_distribution.png")

# ── 3. Feature histograms ─────────────────────────────────────────────────────
numeric_cols = df.select_dtypes(include="number").columns.tolist()

n_cols = 4
n_rows = -(-len(numeric_cols) // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    label = FEATURE_NAMES_EN.get(col, col)
    axes[i].hist(df[col], bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[i].set_title(label, fontsize=11)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Feature Histograms", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "histograms.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Saved: {PLOTS_DIR}/histograms.png")

# ── 4. Correlation heatmap ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = df.corr()
mask_upper = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
corr_en = corr_matrix.rename(index=FEATURE_NAMES_EN, columns=FEATURE_NAMES_EN)

sns.heatmap(
    corr_en,
    mask=mask_upper,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()
print(f"[OK] Saved: {PLOTS_DIR}/correlation_heatmap.png")

# ── 5. Boxplots by target class ───────────────────────────────────────────────
boxplot_features = ["age", "thalach", "chol", "trestbps"]
boxplot_labels   = [FEATURE_NAMES_EN.get(f, f) for f in boxplot_features]

fig, axes = plt.subplots(1, len(boxplot_features), figsize=(16, 6))

for ax, feature, label in zip(axes, boxplot_features, boxplot_labels):
    groups = [
        df.loc[df["target"] == 0, feature].dropna(),
        df.loc[df["target"] == 1, feature].dropna(),
    ]
    bp = ax.boxplot(groups, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    colors_bp = ["#4C72B0", "#DD8452"]
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels(["No Disease (0)", "Disease (1)"])
    ax.set_title(label, fontsize=12)
    ax.set_ylabel("Value")

plt.suptitle("Boxplots of Features by Target Class", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "boxplots.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Saved: {PLOTS_DIR}/boxplots.png")

print("\n[EDA complete] All plots saved to:", PLOTS_DIR)
