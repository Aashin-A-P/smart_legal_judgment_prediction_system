import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
import ast

# === Step 1: Load all splits ===
splits = ["train", "validation", "test"]

all_meta = []
all_embeddings = []

for split in splits:
    meta = pd.read_csv(f"{split}_enriched_meta.csv")
    emb = np.load(f"{split}_enriched_embeddings.npy")
    meta["split"] = split  # keep track of source split
    all_meta.append(meta)
    all_embeddings.append(emb)

df = pd.concat(all_meta, ignore_index=True)
embeddings = np.vstack(all_embeddings)  # shape [num_cases, 768]

print(f"✅ Loaded {len(df)} cases with embeddings shape {embeddings.shape}")

# === Step 2: Collect unique statutes and charges ===
all_statutes = sorted(set([s for sublist in df["statutes"].apply(ast.literal_eval) for s in sublist]))
all_charges = sorted(set([c for sublist in df["charges"].apply(ast.literal_eval) for c in sublist]))

statute2id = {s: i for i, s in enumerate(all_statutes)}
charge2id = {c: i for i, c in enumerate(all_charges)}

print(f"✅ {len(all_statutes)} unique statutes, {len(all_charges)} unique charges")

# === Step 3: Create HeteroData graph ===
data = HeteroData()

# Fact nodes
data["fact"].x = torch.tensor(embeddings, dtype=torch.float)

# Statute nodes (use one-hot for now)
data["statute"].x = torch.eye(len(all_statutes), dtype=torch.float)

# Charge nodes (use one-hot for now)
data["charge"].x = torch.eye(len(all_charges), dtype=torch.float)

# === Step 4: Add edges ===
fact2statute_edges = []
statute2charge_edges = []

for idx, row in df.iterrows():
    fact_id = idx
    statutes = ast.literal_eval(row["statutes"])
    charges = ast.literal_eval(row["charges"])

    # Fact → Statute
    for s in statutes:
        if s in statute2id:
            fact2statute_edges.append([fact_id, statute2id[s]])

    # Statute → Charge
    for s in statutes:
        for c in charges:
            if s in statute2id and c in charge2id:
                statute2charge_edges.append([statute2id[s], charge2id[c]])

# Convert to PyTorch tensors
if fact2statute_edges:
    data["fact", "mentions", "statute"].edge_index = torch.tensor(fact2statute_edges, dtype=torch.long).t().contiguous()
if statute2charge_edges:
    data["statute", "implies", "charge"].edge_index = torch.tensor(statute2charge_edges, dtype=torch.long).t().contiguous()

print("✅ Global graph created")
print(data)

