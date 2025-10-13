# ==========================================================
#  Create Balanced Dataset (1000 Cases)
#  Filter: Only Top 10 Statutes
# ==========================================================

import pandas as pd
import ast
from collections import Counter
from sklearn.utils import resample

# ---------------- Load Dataset ----------------
df = pd.read_csv("encoder.csv")

# Parse list-like strings
df["statutes"] = df["statutes"].apply(ast.literal_eval)
df["charges"]  = df["charges"].apply(ast.literal_eval)

# ---------------- Top 10 Statutes ----------------
top_statutes = [
    "IPC Sec 302",
    "IPC Sec 323",
    "IPC Sec 34",
    "IPC Sec 307",
    "CrPC Sec 313",
    "IPC Sec 149",
    "IPC Sec 148",
    "CrPC Sec 482",
    "IPC Sec 147",
    "IPC Sec 324"
]

# ---------------- Keep Only Top Statutes ----------------
def filter_statutes(lst):
    return [s for s in lst if s in top_statutes]

df["statutes"] = df["statutes"].apply(filter_statutes)
df = df[df["statutes"].apply(len) > 0].reset_index(drop=True)

print(f"✅ Cases after filtering to top statutes: {len(df)}")

# ---------------- One-Statute Expansion ----------------
# Create one row per statute to allow balancing
rows = []
for _, row in df.iterrows():
    for s in row["statutes"]:
        if s in top_statutes:
            new_row = row.copy()
            new_row["primary_statute"] = s
            rows.append(new_row)

df_expanded = pd.DataFrame(rows)
print(f"📦 Expanded to {len(df_expanded)} rows for balancing")

# ---------------- Balance Dataset ----------------
balanced_df = pd.DataFrame()

target_per_class = 100  # 100 cases per statute → 1000 total
for s in top_statutes:
    subset = df_expanded[df_expanded["primary_statute"] == s]
    if len(subset) > target_per_class:
        subset = resample(subset, n_samples=target_per_class, random_state=42)
    elif len(subset) < target_per_class:
        subset = resample(subset, n_samples=target_per_class, replace=True, random_state=42)
    balanced_df = pd.concat([balanced_df, subset])

balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✅ Final balanced dataset size: {len(balanced_df)}")

# ---------------- Statute Distribution Report ----------------
counts = balanced_df["primary_statute"].value_counts()
print("\n⚖️ Statute Distribution:")
for s, c in counts.items():
    print(f"   {s:<25} {c}")

# ---------------- Save Output ----------------
balanced_df.to_csv("dataset.csv", index=False)
print("\n💾 Saved balanced dataset as 'dataset.csv'")
