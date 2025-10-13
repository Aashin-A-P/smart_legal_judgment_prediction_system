# ==========================================================
#  LEGAL JUDGMENT PREDICTION (LJP)
#  InLegalBERT + Graph Attention Network + Cross Attention
#  ✅ CPU-ONLY VERSION
# ==========================================================

import pandas as pd
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score

# ----------------------------------------------------------
# Select device
# ----------------------------------------------------------
device = torch.device("cpu")
print(f"🧠 Using device: {device}")

# ==========================================================
# 1️⃣ Load and preprocess dataset
# ==========================================================
df = pd.read_csv("data.csv")
df = df.dropna(subset=["facts", "charges", "statutes"]).reset_index(drop=True)

def clean(text):
    import re
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["facts"] = df["facts"].apply(clean)
df["charges"] = df["charges"].apply(eval)
df["statutes"] = df["statutes"].apply(eval)

print("✅ Data Loaded:", df.shape)

# ==========================================================
# 2️⃣ Multi-label encoding for charges & statutes
# ==========================================================
mlb_charge = MultiLabelBinarizer()
mlb_statute = MultiLabelBinarizer()

Y_charge = torch.tensor(mlb_charge.fit_transform(df["charges"]), dtype=torch.float)
Y_statute = torch.tensor(mlb_statute.fit_transform(df["statutes"]), dtype=torch.float)

num_charge = Y_charge.shape[1]
num_statute = Y_statute.shape[1]
print(f"Charges: {num_charge} | Statutes: {num_statute}")

# ==========================================================
# 3️⃣ InLegalBERT encoding for case facts
# ==========================================================
tok = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
bert = AutoModel.from_pretrained("law-ai/InLegalBERT").to(device).eval()

@torch.no_grad()
def encode_texts(texts, batch_size=4):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with InLegalBERT"):
        batch = texts[i:i+batch_size]
        toks = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        out = bert(**toks).last_hidden_state.mean(1)
        embs.append(out.cpu())
    return torch.cat(embs)

fact_embs = encode_texts(df["facts"].tolist())  # [N, 768]
print("Facts encoded:", fact_embs.shape)

# ==========================================================
# 4️⃣ Graph construction
# ==========================================================
print("\n🔧 Constructing legal knowledge graph...")

sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
fact_sbert = sbert.encode(df["facts"].tolist(), convert_to_tensor=True, show_progress_bar=True)
sim_matrix = util.cos_sim(fact_sbert, fact_sbert)

edge_index = [[], []]
th = 0.8  # similarity threshold

# fact ↔ fact edges (semantic similarity)
for i in range(len(df)):
    sim_scores = sim_matrix[i]
    idx = torch.where(sim_scores > th)[0]
    for j in idx:
        if i != j:
            edge_index[0].append(i)
            edge_index[1].append(j)

# fact ↔ statute ↔ charge edges
fact_nodes = np.arange(len(df))
statute_nodes = np.arange(len(df), len(df) + num_statute)
charge_nodes = np.arange(len(df) + num_statute, len(df) + num_statute + num_charge)

for i, row in df.iterrows():
    for s in mlb_statute.transform([row["statutes"]])[0].nonzero()[0]:
        s_idx = statute_nodes[s]
        edge_index[0] += [i, s_idx]
        edge_index[1] += [s_idx, i]
    for c in mlb_charge.transform([row["charges"]])[0].nonzero()[0]:
        c_idx = charge_nodes[c]
        edge_index[0] += [i, c_idx]
        edge_index[1] += [c_idx, i]

edge_index = torch.tensor(edge_index, dtype=torch.long)

# node features: facts = InLegalBERT embeddings, others = random
statute_feat = torch.randn(num_statute, 768)
charge_feat = torch.randn(num_charge, 768)
x = torch.cat([fact_embs, statute_feat, charge_feat], dim=0)

graph_data = Data(x=x, edge_index=edge_index)
print(graph_data)

# ==========================================================
# 5️⃣ Graph Attention Network (GAT)
# ==========================================================
class GraphEncoder(nn.Module):
    def __init__(self, in_dim=768, hid=512, out=768, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_dim, hid, heads=heads, concat=True)
        self.gat2 = GATConv(hid*heads, out, heads=1, concat=False)
    def forward(self, data):
        x, edge = data.x, data.edge_index
        x = F.elu(self.gat1(x, edge))
        x = self.gat2(x, edge)
        return x

gat = GraphEncoder().to(device)
optimizer_gat = torch.optim.AdamW(gat.parameters(), lr=1e-4)

print("\n🧠 Training GAT encoder (CPU)...")
gat.train()
for epoch in range(2):  # keep epochs low for CPU speed
    optimizer_gat.zero_grad()
    out = gat(graph_data)
    loss = -out.var(0).mean()
    loss.backward()
    optimizer_gat.step()
    print(f"GAT Epoch {epoch+1}: loss={-loss.item():.4f}")

with torch.no_grad():
    graph_embeddings = gat(graph_data)[:len(df)]
print("Graph embeddings generated:", graph_embeddings.shape)

# ==========================================================
# 6️⃣ Cross-Attention + Multi-label Classifier
# ==========================================================
class CrossAttentionLJP(nn.Module):
    def __init__(self, hidden=768, n_stat=num_statute, n_charge=num_charge):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.scale = hidden ** 0.5
        self.fc_stat = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, n_stat)
        )
        self.fc_charge = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, n_charge)
        )
    def forward(self, f_emb, g_emb):
        Q = self.q_proj(f_emb)
        K = self.k_proj(g_emb)
        V = self.v_proj(g_emb)
        attn = torch.softmax((Q @ K.T) / self.scale, dim=-1)
        C = attn @ V
        fused = f_emb + C
        y_stat = self.fc_stat(fused)
        y_charge = self.fc_charge(fused)
        return y_stat, y_charge

model = CrossAttentionLJP().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

# ==========================================================
# 7️⃣ Train-test split
# ==========================================================
idx = np.arange(len(df))
np.random.shuffle(idx)
train_idx = idx[:int(0.8*len(idx))]
test_idx = idx[int(0.8*len(idx)):]

f_train, f_test = fact_embs[train_idx], fact_embs[test_idx]
g_train, g_test = graph_embeddings[train_idx], graph_embeddings[test_idx]
y_stat_train, y_stat_test = Y_statute[train_idx], Y_statute[test_idx]
y_chg_train, y_chg_test = Y_charge[train_idx], Y_charge[test_idx]

# ==========================================================
# 8️⃣ Training loop
# ==========================================================
print("\n🚀 Training Cross-Attention LJP model (CPU)...")
for epoch in range(3):  # fewer epochs for CPU
    model.train()
    opt.zero_grad()
    s_logits, c_logits = model(f_train, g_train)
    loss = criterion(s_logits, y_stat_train) + criterion(c_logits, y_chg_train)
    loss.backward()
    opt.step()
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# ==========================================================
# 9️⃣ Evaluation
# ==========================================================
print("\n📊 Evaluating model...")
model.eval()
with torch.no_grad():
    s_logits, c_logits = model(f_test, g_test)
    s_pred = (torch.sigmoid(s_logits) > 0.5).int().numpy()
    c_pred = (torch.sigmoid(c_logits) > 0.5).int().numpy()

stat_true = y_stat_test.numpy()
chg_true = y_chg_test.numpy()

print("\n=== Statute Prediction ===")
print("Macro F1:", f1_score(stat_true, s_pred, average="macro"))
print("Precision:", precision_score(stat_true, s_pred, average="macro"))
print("Recall:", recall_score(stat_true, s_pred, average="macro"))

print("\n=== Charge Prediction ===")
print("Macro F1:", f1_score(chg_true, c_pred, average="macro"))
print("Precision:", precision_score(chg_true, c_pred, average="macro"))
print("Recall:", recall_score(chg_true, c_pred, average="macro"))

torch.save(model.state_dict(), "LJP_CrossAttention_CPU.pt")
print("\n✅ Training completed and model saved as 'LJP_CrossAttention_CPU.pt'")
