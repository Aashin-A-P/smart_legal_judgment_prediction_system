# ================================
# 1. Imports
# ================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv
from sklearn.metrics import accuracy_score, f1_score
import ast

# ================================
# 2. Load Splits and Embeddings
# ================================
splits = ["train", "validation", "test"]

all_meta = []
all_embeddings = []

for split in splits:
    meta = pd.read_csv(f"{split}_enriched_meta.csv")
    emb = np.load(f"{split}_enriched_embeddings.npy")
    meta["split"] = split
    all_meta.append(meta)
    all_embeddings.append(emb)

df = pd.concat(all_meta, ignore_index=True)
embeddings = np.vstack(all_embeddings)  # shape [num_cases, 768]

print(f"✅ Loaded {len(df)} cases with embeddings shape {embeddings.shape}")

# ================================
# 3. Collect Unique Statutes & Charges
# ================================
all_statutes = sorted(set([s for sublist in df["statutes"].apply(ast.literal_eval) for s in sublist]))
all_charges = sorted(set([c for sublist in df["charges"].apply(ast.literal_eval) for c in sublist]))

statute2id = {s: i for i, s in enumerate(all_statutes)}
charge2id = {c: i for i, c in enumerate(all_charges)}

print(f"✅ {len(all_statutes)} unique statutes, {len(all_charges)} unique charges")

# ================================
# 4. Construct HeteroData Graph
# ================================
data = HeteroData()

# Fact nodes
data["fact"].x = torch.tensor(embeddings, dtype=torch.float)

# Statute nodes (one-hot)
data["statute"].x = torch.eye(len(all_statutes), dtype=torch.float)

# Charge nodes (one-hot)
data["charge"].x = torch.eye(len(all_charges), dtype=torch.float)

# Add edges
fact2statute_edges = []
statute2charge_edges = []

for idx, row in df.iterrows():
    fact_id = idx
    statutes = ast.literal_eval(row["statutes"])
    charges = ast.literal_eval(row["charges"])

    for s in statutes:
        if s in statute2id:
            fact2statute_edges.append([fact_id, statute2id[s]])

    for s in statutes:
        for c in charges:
            if s in statute2id and c in charge2id:
                statute2charge_edges.append([statute2id[s], charge2id[c]])

# Make edges bidirectional
if fact2statute_edges:
    f2s = torch.tensor(fact2statute_edges, dtype=torch.long).t().contiguous()
    data["fact", "mentions", "statute"].edge_index = f2s
    data["statute", "rev_mentions", "fact"].edge_index = f2s.flip(0)

if statute2charge_edges:
    s2c = torch.tensor(statute2charge_edges, dtype=torch.long).t().contiguous()
    data["statute", "implies", "charge"].edge_index = s2c
    data["charge", "rev_implies", "statute"].edge_index = s2c.flip(0)

print("✅ Graph created with nodes and edges:")
print(data)

# ================================
# 5. Labels (Charges per Fact)
# ================================
y_true = torch.tensor(df["charges"].apply(
    lambda x: [1 if c in ast.literal_eval(x) else 0 for c in all_charges]
).tolist(), dtype=torch.float)

print("✅ Labels shape:", y_true.shape)

# ================================
# 6. Define GAT Model (Fixed)
# ================================
class GATPredictor(nn.Module):
    def __init__(self, hidden_dim=128, num_charges=10):
        super().__init__()
        self.gat = HeteroConv({
            ("fact", "mentions", "statute"): GATConv(
                (-1, -1), hidden_dim, heads=2, add_self_loops=False
            ),
            ("statute", "rev_mentions", "fact"): GATConv(
                (-1, -1), hidden_dim, heads=2, add_self_loops=False
            ),
            ("statute", "implies", "charge"): GATConv(
                (-1, -1), hidden_dim, heads=2, add_self_loops=False
            ),
            ("charge", "rev_implies", "statute"): GATConv(
                (-1, -1), hidden_dim, heads=2, add_self_loops=False
            ),
        }, aggr="mean")

        self.classifier = nn.Linear(hidden_dim * 2, num_charges)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.gat(x_dict, edge_index_dict)
        fact_emb = x_dict["fact"]   # ✅ fact nodes now updated
        out = self.classifier(fact_emb)
        return out

model = GATPredictor(num_charges=len(all_charges))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()  # multi-label loss

# ================================
# 7. Train Model
# ================================
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = criterion(out, y_true)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ================================
# 8. Evaluate
# ================================
model.eval()
with torch.no_grad():
    logits = model(data.x_dict, data.edge_index_dict)
    preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
    labels = y_true.cpu().numpy()

acc = accuracy_score(labels, preds)
f1_macro = f1_score(labels, preds, average="macro")
f1_micro = f1_score(labels, preds, average="micro")

print("\n===== Evaluation Results =====")
print("Accuracy:", acc)
print("Macro-F1:", f1_macro)
print("Micro-F1:", f1_micro)
