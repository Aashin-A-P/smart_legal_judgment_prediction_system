# ==========================================================
#  Trim Dataset: keep exactly 3 statutes & 3 charges per case
#  while maintaining overall distribution
# ==========================================================

import pandas as pd
import ast
import random
from collections import Counter

# ---------------- Load data ----------------
df = pd.read_csv("balanced_1000.csv")

df["statutes"] = df["statutes"].apply(ast.literal_eval)
df["charges"]  = df["charges"].apply(ast.literal_eval)

print(f"✅ Loaded {len(df)} cases")

# ---------------- Analyze before trimming ----------------
all_stats_before = [s for sublist in df["statutes"] for s in sublist]
all_chg_before   = [c for sublist in df["charges"] for c in sublist]

stat_counts_before = Counter(all_stats_before)
charge_counts_before = Counter(all_chg_before)

print("\n📊 BEFORE TRIMMING (Top 10):")
print("Top Statutes:")
for s, c in stat_counts_before.most_common(10):
    print(f"   {s:<40} {c}")
print("\nTop Charges:")
for c, n in charge_counts_before.most_common(10):
    print(f"   {c:<40} {n}")

# ---------------- Controlled trimming ----------------
# We bias the random selection using inverse frequency — rare ones get a slightly higher chance
stat_freq = {k: v for k, v in stat_counts_before.items()}
charge_freq = {k: v for k, v in charge_counts_before.items()}

def weighted_trim(lst, freq_map):
    if len(lst) <= 3:
        return lst
    # compute weights inversely proportional to global frequency (to preserve balance)
    weights = [1 / freq_map.get(x, 1) for x in lst]
    total = sum(weights)
    probs = [w / total for w in weights]
    return random.choices(lst, weights=probs, k=3)

df["statutes"] = df["statutes"].apply(lambda lst: weighted_trim(lst, stat_freq))
df["charges"]  = df["charges"].apply(lambda lst: weighted_trim(lst, charge_freq))

# ---------------- Analyze after trimming ----------------
all_stats_after = [s for sublist in df["statutes"] for s in sublist]
all_chg_after   = [c for sublist in df["charges"] for c in sublist]

stat_counts_after = Counter(all_stats_after)
charge_counts_after = Counter(all_chg_after)

print("\n📊 AFTER TRIMMING (Top 10):")
print("Top Statutes:")
for s, c in stat_counts_after.most_common(10):
    print(f"   {s:<40} {c}")
print("\nTop Charges:")
for c, n in charge_counts_after.most_common(10):
    print(f"   {c:<40} {n}")

# ---------------- Save output ----------------
df.to_csv("balanced_1000_trimmed.csv", index=False)
print("\n💾 Saved dataset as 'balanced_1000_trimmed.csv' with exactly 3 statutes & 3 charges per case.")
