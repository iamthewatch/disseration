# EDA (Exploratory Data Analysis)
# Dataset: healthcare-dataset-stroke-data.csv

import os
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'healthcare-dataset-stroke-data.csv')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURE_NAMES_EN = {
    'age':               'Age',
    'avg_glucose_level': 'Avg Glucose Level',
    'bmi':               'BMI',
    'stroke':            'Stroke',
    'hypertension':      'Hypertension',
    'heart_disease':     'Heart Disease',
}

# ── 1. Load dataset ───────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATASET INFO")
print("=" * 60)
print(f"Size: {df.shape[0]} rows, {df.shape[1]} columns\n")

print("Data types:")
print(df.dtypes)
print()

print("Missing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0])
print(f"\nTotal missing values: {df.isnull().sum().sum()}")
print()

print("Descriptive statistics:")
print(df.describe())
print()

print("Target variable distribution (stroke):")
print(df['stroke'].value_counts())
print(f"Stroke rate: {df['stroke'].mean():.2%}")

# ── 2. Class distribution ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = df['stroke'].value_counts()
ax.bar(
    ['No Stroke (0)', 'Stroke (1)'],
    counts.values,
    color=['steelblue', 'tomato'],
    edgecolor='black'
)
for i, v in enumerate(counts.values):
    ax.text(i, v + 30, str(v), ha='center', fontweight='bold')
ax.set_title('Class Distribution: Stroke vs No Stroke')
ax.set_ylabel('Number of Patients')
ax.set_xlabel('Class')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'class_distribution.png'), dpi=150)
plt.close()
print("\n[Saved] class_distribution.png")

# ── 3. Numeric feature histograms ─────────────────────────────
numeric_cols = df.select_dtypes(include='number').columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ('id', 'stroke')]

fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5 * len(numeric_cols), 4))
for ax, col in zip(axes, numeric_cols):
    label = FEATURE_NAMES_EN.get(col, col)
    ax.hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_title(label)
    ax.set_xlabel(label)
    ax.set_ylabel('Frequency')
plt.suptitle('Numeric Feature Histograms', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'histograms.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] histograms.png")

# ── 4. Correlation heatmap ────────────────────────────────────
numeric_df = df[numeric_cols + ['stroke']].copy()
corr_matrix = numeric_df.corr()
corr_en = corr_matrix.rename(index=FEATURE_NAMES_EN, columns=FEATURE_NAMES_EN)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    corr_en,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    ax=ax,
    linewidths=0.5
)
ax.set_title('Numeric Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'), dpi=150)
plt.close()
print("[Saved] correlation_heatmap.png")

# ── 5. Boxplots by stroke group ───────────────────────────────
box_cols   = ['age', 'avg_glucose_level', 'bmi']
box_labels = [FEATURE_NAMES_EN.get(c, c) for c in box_cols]

fig, axes = plt.subplots(1, len(box_cols), figsize=(5 * len(box_cols), 5))
for ax, col, label in zip(axes, box_cols, box_labels):
    groups = [
        df.loc[df['stroke'] == 0, col].dropna(),
        df.loc[df['stroke'] == 1, col].dropna(),
    ]
    bp = ax.boxplot(groups, patch_artist=True, tick_labels=['No Stroke', 'Stroke'])
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('tomato')
    ax.set_title(label)
    ax.set_ylabel(label)
plt.suptitle('Feature Distribution by Stroke Group', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'boxplots.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] boxplots.png")

print("\nEDA complete. All plots saved to stroke/plots/")
