import pandas as pd

# File paths (replace with your paths if needed)
files = [
    "cases_augmented.csv",
    "cases_augmented_2.csv",
    "cases_augmented_3.csv"
]

# Read and merge
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# Function to check valid entries
def is_valid(value):
    if pd.isna(value):
        return False
    s = str(value).strip()
    return not (s == "" or s == "[]" or s.lower() == "nan")

# Keep only rows where both statutes and charges are valid
valid_df = df[df['statutes'].apply(is_valid) & df['charges'].apply(is_valid)]

# Save result
valid_df.to_csv("cases_valid.csv", index=False)

print(f"✅ Merged file created: cases_valid.csv with {len(valid_df)} valid cases")
