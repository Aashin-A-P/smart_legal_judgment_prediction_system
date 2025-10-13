import pandas as pd

# Load all batches
batch1 = pd.read_csv("batch_1.csv")
batch2 = pd.read_csv("batch_2.csv")
batch3 = pd.read_csv("batch_3.csv")

# Merge them together
merged_df = pd.concat([batch1, batch2, batch3], axis=0, ignore_index=True)

# Shuffle the dataset (optional, for randomness)
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save merged file
merged_df.to_csv("batch.csv", index=False)

print("Merged dataset shape:", merged_df.shape)
print(merged_df["label"].value_counts())
