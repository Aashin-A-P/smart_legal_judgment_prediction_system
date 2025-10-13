import pandas as pd

# Load the original batch files (raw case text)
batch_files = ["batch_1.csv", "batch_2.csv", "batch_3.csv"]
batch_dfs = [pd.read_csv(f) for f in batch_files]
full_text_df = pd.concat(batch_dfs, ignore_index=True)

# Make sure the join key is consistent
# Assume: batch files use 'file' column, splits use 'filename'
full_text_df.rename(columns={"file": "filename"}, inplace=True)

# Load split datasets
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

# Merge text into splits using 'filename'
train_enriched = train_df.merge(full_text_df[["filename", "text"]], on="filename", how="left")
val_enriched = val_df.merge(full_text_df[["filename", "text"]], on="filename", how="left")
test_enriched = test_df.merge(full_text_df[["filename", "text"]], on="filename", how="left")

# Save enriched splits
train_enriched.to_csv("train_enriched.csv", index=False)
val_enriched.to_csv("validation_enriched.csv", index=False)
test_enriched.to_csv("test_enriched.csv", index=False)

print(f"✅ Train enriched: {len(train_enriched)} cases")
print(f"✅ Validation enriched: {len(val_enriched)} cases")
print(f"✅ Test enriched: {len(test_enriched)} cases")
