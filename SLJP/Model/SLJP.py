# ==========================================================
#  SMART LEGAL JUDGMENT PREDICTION SYSTEM (LJP)
#  InLegalBERT + GAT + Multi-Head Cross Attention
#  Multi-label Statute Prediction (CPU Optimized)
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
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

device = "cpu"
os.makedirs("ljp_final_outputs/plots", exist_ok=True)
os.makedirs("ljp_final_outputs/reports", exist_ok=True)

# ==========================================================
# 1️⃣ LOAD & CLEAN DATA
# ==========================================================
df = pd.read_csv("dataset.csv")
df = df.dropna(subset=["facts","statutes"]).reset_index(drop=True)
df["facts"] = df["facts"].astype(str).str.replace(r"\s+"," ",regex=True)
df["statutes"] = df["statutes"].apply(eval)
df["charges"] = df["charges"].apply(eval)
print("✅ Data Loaded:", df.shape)

# ==========================================================
# 2️⃣ LABEL ENCODING (Multi-label for Statutes)
# ==========================================================
mlb = MultiLabelBinarizer()
Y_statute = torch.tensor(mlb.fit_transform(df["statutes"]), dtype=torch.float)
num_statute = Y_statute.shape[1]
print(f"Total Statute Classes: {num_statute}")

# ==========================================================
# 3️⃣ ENCODE FACTS USING InLegalBERT
# ==========================================================
tok = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
bert = AutoModel.from_pretrained("law-ai/InLegalBERT").to(device)
for name, param in bert.named_parameters():
    param.requires_grad = any(k in name for k in ["encoder.layer.23","pooler"])  # fine-tune last layer

@torch.no_grad()
def encode_texts(texts, batch_size=4):
    embs = []
    for i in tqdm(range(0,len(texts),batch_size),desc="Encoding with InLegalBERT"):
        batch = texts[i:i+batch_size]
        toks = tok(batch,padding=True,truncation=True,max_length=512,return_tensors="pt").to(device)
        out = bert(**toks).last_hidden_state.mean(1)
        embs.append(out.cpu())
    return torch.cat(embs)

fact_embs = encode_texts(df["facts"].tolist())
print("Facts Encoded:", fact_embs.shape)

# ==========================================================
# 4️⃣ GRAPH CONSTRUCTION (fact, statute, charge)
# ==========================================================
print("\n🔧 Constructing Legal Knowledge Graph...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
fact_sbert = sbert.encode(df["facts"].tolist(), convert_to_tensor=True, show_progress_bar=True)
sim_matrix = util.cos_sim(fact_sbert, fact_sbert)

edge_index = [[],[]]
th = 0.8
for i in range(len(df)):
    idx = torch.where(sim_matrix[i] > th)[0]
    for j in idx:
        if i != j:
            edge_index[0].append(i)
            edge_index[1].append(j)

fact_nodes = np.arange(len(df))
statute_nodes = np.arange(len(df), len(df)+num_statute)
charge_set = sorted({c for sub in df["charges"] for c in sub})
charge_map = {c:i for i,c in enumerate(charge_set)}
num_charge = len(charge_map)
charge_nodes = np.arange(len(df)+num_statute, len(df)+num_statute+num_charge)

for i,row in df.iterrows():
    s_idx = np.where(mlb.transform([row["statutes"]])[0]==1)[0]
    for s in s_idx:
        edge_index[0]+=[i]; edge_index[1]+=[len(df)+s]
        edge_index[0]+=[len(df)+s]; edge_index[1]+=[i]
    for c in row["charges"]:
        if c in charge_map:
            ci = len(df)+num_statute+charge_map[c]
            edge_index[0]+=[i]; edge_index[1]+=[ci]
            edge_index[0]+=[ci]; edge_index[1]+=[i]

edge_index = torch.tensor(edge_index, dtype=torch.long)
statute_feat = torch.randn(num_statute,768)
charge_feat  = torch.randn(num_charge,768)
x = torch.cat([fact_embs, statute_feat, charge_feat],dim=0)
graph_data = Data(x=x, edge_index=edge_index)
print(graph_data)

# ==========================================================
# 5️⃣ GRAPH ATTENTION NETWORK
# ==========================================================
class GraphEncoder(nn.Module):
    def __init__(self,in_dim=768,hid=512,out=768,heads=4):
        super().__init__()
        self.gat1 = GATConv(in_dim,hid,heads=heads,concat=True)
        self.gat2 = GATConv(hid*heads,out,heads=1,concat=False)
    def forward(self,data):
        x,edge = data.x,data.edge_index
        x = F.elu(self.gat1(x,edge))
        x = self.gat2(x,edge)
        return x

gat = GraphEncoder().to(device)
opt_gat = torch.optim.AdamW(gat.parameters(),lr=1e-4)
for epoch in range(3):
    opt_gat.zero_grad()
    out = gat(graph_data)
    loss = -out.var(0).mean()
    loss.backward()
    opt_gat.step()
    print(f"GAT Epoch {epoch+1} | Loss: {loss.item():.4f}")
with torch.no_grad():
    graph_embs = gat(graph_data)[:len(df)]

# ==========================================================
# 6️⃣ MULTI-HEAD CROSS-ATTENTION + CLASSIFIER
# ==========================================================
class CrossAttentionLJP(nn.Module):
    def __init__(self,hidden=768,n_stat=num_statute,heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden,num_heads=heads,batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.fc = nn.Sequential(
            nn.Linear(hidden,hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2,n_stat)
        )
    def forward(self,f_emb,g_emb):
        f = f_emb.unsqueeze(1)
        g = g_emb.unsqueeze(1)
        attn_out,_ = self.attn(f,g,g)
        fused = self.norm(f+attn_out).squeeze(1)
        return self.fc(fused)

model = CrossAttentionLJP().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.8)
criterion = nn.BCEWithLogitsLoss()

# ==========================================================
# 7️⃣ SPLIT DATA (Train/Val/Test)
# ==========================================================
idx = np.arange(len(df))
np.random.shuffle(idx)
n = len(df)
train_idx = idx[:int(0.7*n)]
val_idx   = idx[int(0.7*n):int(0.85*n)]
test_idx  = idx[int(0.85*n):]

f_train,f_val,f_test = fact_embs[train_idx],fact_embs[val_idx],fact_embs[test_idx]
g_train,g_val,g_test = graph_embs[train_idx],graph_embs[val_idx],graph_embs[test_idx]
y_train,y_val,y_test = Y_statute[train_idx],Y_statute[val_idx],Y_statute[test_idx]

# ==========================================================
# 8️⃣ TRAINING LOOP
# ==========================================================
train_losses,val_losses = [],[]
epochs = 300
print("\n🚀 Training LJP Model...")
for epoch in range(epochs):
    model.train()
    opt.zero_grad()
    logits = model(f_train,g_train)
    loss = criterion(logits,y_train)
    loss.backward()
    opt.step()
    scheduler.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(f_val,g_val),y_val).item()
        val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

# ==========================================================
# 9️⃣ EVALUATION
# ==========================================================
model.eval()
with torch.no_grad():
    preds = torch.sigmoid(model(f_test,g_test))
pred_labels = (preds>0.5).int()

y_true = y_test.numpy()
y_pred = pred_labels.numpy()
report = classification_report(y_true,y_pred,target_names=mlb.classes_,zero_division=0,output_dict=True)
pd.DataFrame(report).transpose().to_csv("ljp_final_outputs/reports/classification_report.csv")
print("\n✅ Classification report saved!")

# ==========================================================
# 🔟 PLOTTING (10 Research-Grade Plots)
# ==========================================================
def saveplot(fig,name): plt.tight_layout(); fig.savefig(f"ljp_final_outputs/plots/{name}"); plt.close(fig)

# 1. Training & Validation Loss
fig = plt.figure(figsize=(6,4))
plt.plot(train_losses,label="Train"); plt.plot(val_losses,label="Val")
plt.title("Training & Validation Loss Curve"); plt.legend()
saveplot(fig,"1_loss_curves.png")

# 2. Macro Metrics
macro = report["macro avg"]
fig = plt.figure(figsize=(4,4))
sns.barplot(x=["Precision","Recall","F1"], y=[macro["precision"],macro["recall"],macro["f1-score"]])
plt.title("Macro-Average Metrics"); saveplot(fig,"2_macro_metrics.png")

# 3. Per-Class F1
f1s = [report[c]["f1-score"] for c in mlb.classes_]
fig = plt.figure(figsize=(10,5))
sns.barplot(x=mlb.classes_,y=f1s,palette="viridis")
plt.xticks(rotation=75); plt.title("Per-Class F1"); saveplot(fig,"3_per_class_f1.png")

# 4. Confusion Matrix (simplified)
cm = confusion_matrix(y_true.argmax(1),y_pred.argmax(1))
fig = plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,cmap="Blues"); plt.title("Confusion Matrix")
saveplot(fig,"4_confusion_matrix.png")

# 5. Prediction Distribution
fig = plt.figure(figsize=(6,4))
sns.countplot(x=pred_labels.sum(1)); plt.title("Predicted Statutes per Case")
saveplot(fig,"5_prediction_distribution.png")

# 6. Learning Rate Decay
fig = plt.figure(figsize=(6,4))
plt.plot([scheduler.get_last_lr()[0] for _ in range(epochs)])
plt.title("Learning Rate Schedule"); saveplot(fig,"6_lr_schedule.png")

# 7. Performance Heatmap
df_r = pd.DataFrame(report).iloc[:-3,:-1]
fig = plt.figure(figsize=(8,6))
sns.heatmap(df_r,annot=True,cmap="coolwarm"); plt.title("Performance Heatmap")
saveplot(fig,"7_performance_heatmap.png")

# 8. Training Loss Heatmap
fig = plt.figure(figsize=(6,4))
sns.heatmap(np.expand_dims(train_losses,axis=0),cmap="rocket",cbar=False)
plt.title("Training Loss Heatmap"); saveplot(fig,"8_loss_heatmap.png")

# 9. Statute Frequency
fig = plt.figure(figsize=(8,4))
sns.countplot(x=[s for sub in df.statutes for s in sub])
plt.xticks(rotation=75); plt.title("Statute Frequency in Dataset")
saveplot(fig,"9_statute_distribution.png")

# 10. Validation vs Test F1
val_f1 = f1_score(y_val, (torch.sigmoid(model(f_val,g_val))>0.5).int(), average="macro")
test_f1 = f1_score(y_test, pred_labels, average="macro")
fig = plt.figure(figsize=(4,4))
sns.barplot(x=["Validation","Test"],y=[val_f1,test_f1])
plt.title("F1 Comparison"); saveplot(fig,"10_val_test_f1.png")

print("\n📊 All plots saved in 'ljp_final_outputs/plots/'")
print("🎯 Enhanced LJP pipeline training completed successfully.")

from sklearn.metrics import accuracy_score, brier_score_loss

# ==========================================================
# 🔢  EXTRA METRICS & PLOTS (Advanced Evaluation)
# ==========================================================
print("\n📈 Computing extended metrics...")

# === 1️⃣ Compute Losses on all splits ===
with torch.no_grad():
    train_loss_final = criterion(model(f_train,g_train), y_train).item()
    val_loss_final = criterion(model(f_val,g_val), y_val).item()
    test_loss_final = criterion(model(f_test,g_test), y_test).item()

# === 2️⃣ Compute Accuracy ===
train_acc = accuracy_score(y_train.numpy(), (torch.sigmoid(model(f_train,g_train))>0.5).int().numpy())
val_acc   = accuracy_score(y_val.numpy(), (torch.sigmoid(model(f_val,g_val))>0.5).int().numpy())
test_acc  = accuracy_score(y_test.numpy(), y_pred)
print(f"Accuracy → Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")

# === 3️⃣ Compute Brier Score (for probabilistic calibration) ===
y_test_probs = preds.numpy().flatten()
y_test_true  = y_true.flatten()
brier = brier_score_loss(y_test_true, y_test_probs)
print(f"Brier Score: {brier:.6f}")

# ==========================================================
# 📉 PLOTS — NEW ADDITIONS
# ==========================================================

# === (A) Combined Line Plot: Train, Val, Test Loss ===
fig = plt.figure(figsize=(6,4))
plt.plot(train_losses, label="Train Loss", linewidth=2)
plt.plot(val_losses, label="Val Loss", linewidth=2)
plt.hlines(test_loss_final, 0, len(train_losses), colors="r", linestyles="--", label="Test Loss")
plt.xlabel("Epochs"); plt.ylabel("Loss")
plt.title("Train, Validation & Test Loss Curve")
plt.legend()
saveplot(fig, "11_train_val_test_loss_line.png")

# === (B) Radar Plot: Train vs Val vs Test (Precision, Recall, F1, Accuracy, Brier) ===
metrics = ["Precision", "Recall", "F1", "Accuracy", "Brier"]
train_metrics = [
    precision_score(y_train, (torch.sigmoid(model(f_train,g_train))>0.5).int(), average="macro"),
    recall_score(y_train, (torch.sigmoid(model(f_train,g_train))>0.5).int(), average="macro"),
    f1_score(y_train, (torch.sigmoid(model(f_train,g_train))>0.5).int(), average="macro"),
    train_acc,
    1 - brier  # invert Brier to represent higher-is-better
]
val_metrics = [
    precision_score(y_val, (torch.sigmoid(model(f_val,g_val))>0.5).int(), average="macro"),
    recall_score(y_val, (torch.sigmoid(model(f_val,g_val))>0.5).int(), average="macro"),
    f1_score(y_val, (torch.sigmoid(model(f_val,g_val))>0.5).int(), average="macro"),
    val_acc,
    1 - brier
]
test_metrics = [
    precision_score(y_test, y_pred, average="macro"),
    recall_score(y_test, y_pred, average="macro"),
    f1_score(y_test, y_pred, average="macro"),
    test_acc,
    1 - brier
]

# Radar setup
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
train_metrics += train_metrics[:1]
val_metrics += val_metrics[:1]
test_metrics += test_metrics[:1]
angles += angles[:1]

fig = plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)
ax.plot(angles, train_metrics, 'b-', linewidth=2, label='Train')
ax.fill(angles, train_metrics, 'b', alpha=0.1)
ax.plot(angles, val_metrics, 'g-', linewidth=2, label='Val')
ax.fill(angles, val_metrics, 'g', alpha=0.1)
ax.plot(angles, test_metrics, 'r-', linewidth=2, label='Test')
ax.fill(angles, test_metrics, 'r', alpha=0.1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=10)
ax.set_title("Radar Plot: Train vs Val vs Test Metrics", size=13)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
saveplot(fig, "12_radar_metrics.png")

# === (C) Accuracy Line Plot ===
fig = plt.figure(figsize=(6,4))
plt.plot([train_acc, val_acc, test_acc], marker='o', linewidth=2)
plt.xticks([0,1,2], ["Train","Validation","Test"])
plt.ylabel("Accuracy")
plt.title("Accuracy across Dataset Splits")
plt.ylim(0,1)
saveplot(fig, "13_accuracy_comparison.png")

# === (D) Brier Score Visualization ===
fig = plt.figure(figsize=(4,4))
sns.barplot(x=["Brier Score"], y=[brier], palette="coolwarm")
plt.title("Brier Score (Lower is Better)")
saveplot(fig, "14_brier_score.png")

print("✅ Added plots: Train/Val/Test Loss, Radar Metrics, Accuracy, Brier Score.")

# === (E) Detailed Brier Calibration Visualization ===
from sklearn.calibration import calibration_curve

# Compute calibration curve (probability vs. true outcome)
prob_true, prob_pred = calibration_curve(y_test_true, y_test_probs, n_bins=15)

fig = plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Calibration Curve')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
plt.title("Brier Calibration Curve")
plt.xlabel("Predicted Probability")
plt.ylabel("True Likelihood")
plt.legend()
saveplot(fig, "15_brier_calibration_curve.png")

# === (F) Probability Distribution Histogram ===
fig = plt.figure(figsize=(6,4))
plt.hist(y_test_probs, bins=30, color='royalblue', alpha=0.7)
plt.axvline(0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')
plt.title("Distribution of Predicted Probabilities")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.legend()
saveplot(fig, "16_probability_distribution.png")

print("✅ Added detailed Brier calibration and probability distribution plots.")
