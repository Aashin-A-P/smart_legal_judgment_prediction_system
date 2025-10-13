import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv("cases_valid_clean.csv")

# First split: train (70%) + temp (30%)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

# Second split: test (20%) + validation (10%)
# temp_df is 30%, so test_size = 2/3 → test = 20%, val = 10%
test_df, val_df = train_test_split(temp_df, test_size=1/3, random_state=42, shuffle=True)

# Save splits
train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"✅ Train: {len(train_df)} cases")
print(f"✅ Validation: {len(val_df)} cases")
print(f"✅ Test: {len(test_df)} cases")
