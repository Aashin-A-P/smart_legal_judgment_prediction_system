import pandas as pd

# Step 1: Load all batch files
batch_files = [
    "batch_1_cases_with_facts.csv",
    "batch_2_cases_with_facts.csv",
    "batch_3_cases_with_facts.csv"
]
df = pd.concat([pd.read_csv(f) for f in batch_files], ignore_index=True)

# Step 2: Remove empty or [] facts
df = df[~df['facts'].isna()]
df = df[df['facts'].astype(str).str.strip() != '[]']
df = df[df['facts'].astype(str).str.strip() != '']

# Step 3: Load valid cases
valid_df = pd.read_csv("cases_valid_clean.csv")
valid_filenames = set(valid_df['filename'].astype(str).str.strip())

# Keep only filenames that exist in valid_df
df = df[df['filename'].astype(str).isin(valid_filenames)]

# Step 4: Merge in 'statutes' and 'charges' from valid_df
df = df.merge(valid_df[['filename', 'statutes', 'charges']], on='filename', how='left')

# Step 5: Keep only required columns
columns_to_keep = ['filename', 'label', 'statutes', 'charges', 'facts']
df = df[columns_to_keep]

# Step 6: Save as graph.csv
df.to_csv("graph.csv", index=False)

print(f"✅ Done! Saved 'graph.csv' with {len(df)} valid cases.")
