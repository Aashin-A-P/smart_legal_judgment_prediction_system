import pandas as pd

# ===== Load the two input CSVs =====
batch_df = pd.read_csv("batch.csv")      # columns: filename, text, label
graph_df = pd.read_csv("graph.csv")      # columns: filename, label, statutes, charges, facts

# ===== Optional sanity checks =====
print("Batch shape:", batch_df.shape)
print("Graph shape:", graph_df.shape)

# ===== Merge on filename & label =====
merged_df = pd.merge(batch_df, graph_df, on=["filename", "label"], how="inner")

# ===== Reorder columns as requested =====
merged_df = merged_df[["filename", "label", "text", "statutes", "charges", "facts"]]

# ===== Save to encoder.csv =====
merged_df.to_csv("encoder.csv", index=False)

print(f"✅ encoder.csv created successfully with {len(merged_df)} rows.")
