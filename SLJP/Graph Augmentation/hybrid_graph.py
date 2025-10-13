import pandas as pd
import numpy as np
import json
import torch
import ast
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# ======================
# 1️⃣ Load graph.csv
# ======================
df = pd.read_csv("graph.csv")

# Convert stringified lists → Python lists
for col in ["facts", "statutes", "charges"]:
    df[col] = df[col].apply(ast.literal_eval)

# ======================
# 2️⃣ Collect unique nodes
# ======================
case_nodes = df["filename"].tolist()
fact_nodes = list({f for facts in df["facts"] for f in facts})
statute_nodes = list({s for statutes in df["statutes"] for s in statutes})
charge_nodes = list({c for charges in df["charges"] for c in charges})

print(f"Total nodes:\n  Cases     : {len(case_nodes)}"
      f"\n  Facts     : {len(fact_nodes)}"
      f"\n  Statutes  : {len(statute_nodes)}"
      f"\n  Charges   : {len(charge_nodes)}")

# ======================
# 3️⃣ Load models
# ======================
# light encoder for facts
mini_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# heavy encoder for statutes & charges
bert_tok = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
bert_model = AutoModel.from_pretrained("law-ai/InLegalBERT")

# ======================
# 4️⃣ Helper functions
# ======================

def embed_inlegalbert(texts, batch_size=8):
    """Encode with InLegalBERT"""
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="InLegalBERT"):
        batch = texts[i:i + batch_size]
        inputs = bert_tok(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=128)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_vecs.append(emb)
    return np.vstack(all_vecs)

def embed_minilm(texts, batch_size=128):
    """Encode with MiniLM"""
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="MiniLM"):
        batch = texts[i:i + batch_size]
        emb = mini_model.encode(batch, show_progress_bar=False,
                                convert_to_numpy=True)
        all_vecs.append(emb)
    return np.vstack(all_vecs)

def project_to_384(embeds):
    """Random linear projection from 768 → 384 dims"""
    W = np.random.randn(embeds.shape[1], 384) / np.sqrt(embeds.shape[1])
    return embeds @ W

# ======================
# 5️⃣ Generate embeddings
# ======================

# Facts → MiniLM (already 384 dims)
fact_embeddings = embed_minilm(fact_nodes)

# Statutes & Charges → InLegalBERT (768 dims → project 384)
statute_embeddings = project_to_384(embed_inlegalbert(statute_nodes))
charge_embeddings  = project_to_384(embed_inlegalbert(charge_nodes))

# Case nodes → random (placeholder, same 384 dims)
case_embeddings = np.random.randn(len(case_nodes), 384)

# ======================
# 6️⃣ Combine everything
# ======================
node_embeddings = np.vstack([
    case_embeddings,
    fact_embeddings,
    statute_embeddings,
    charge_embeddings
])

node_index = {
    "case": case_nodes,
    "fact": fact_nodes,
    "statute": statute_nodes,
    "charge": charge_nodes
}

# ======================
# 7️⃣ Save outputs
# ======================
np.save("node_embeddings_384.npy", node_embeddings)
with open("node_index.json", "w") as f:
    json.dump(node_index, f, indent=2)

print("✅ Saved:")
print(f"   node_embeddings_384.npy  → shape {node_embeddings.shape}")
print("   node_index.json")
