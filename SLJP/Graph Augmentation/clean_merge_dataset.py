import pandas as pd
import ast

# ========= Step 1: Load and Merge =========
files = [
    "cases_augmented.csv",
    "cases_augmented_2.csv",
    "cases_augmented_3.csv"
]

dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# ========= Step 2: Filter Valid Cases =========
def is_valid(value):
    if pd.isna(value):
        return False
    s = str(value).strip()
    return not (s == "" or s == "[]" or s.lower() == "nan")

valid_df = df[df['statutes'].apply(is_valid) & df['charges'].apply(is_valid)]

# ========= Step 3: Clean Stringified Lists =========
def clean_list_column(value):
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)  # Convert string → Python object
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip() != ""]
        else:
            return []
    except (ValueError, SyntaxError):
        return []

valid_df = valid_df.copy()  # avoid SettingWithCopyWarning
valid_df["statutes"] = valid_df["statutes"].apply(clean_list_column)
valid_df["charges"] = valid_df["charges"].apply(clean_list_column)

# ========= Step 4: Save Cleaned File =========
valid_df.to_csv("cases_valid_clean.csv", index=False)

print(f"✅ Cleaned file saved as 'cases_valid_clean.csv' with {len(valid_df)} valid cases")
