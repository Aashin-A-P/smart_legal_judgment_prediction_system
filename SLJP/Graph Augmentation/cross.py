import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import numpy as np

# =====================================================
# CONFIG
# =====================================================
MODEL_NAME = "law-ai/InLegalBERT"
EPOCHS = 3
BATCH_SIZE = 4
LR = 2e-5
MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# CLEAN CSV
# =====================================================
with open("encoder.csv", "r", encoding="utf-8", errors="ignore") as f:
    content = f.read().replace("\x00", "")
with open("encoder_clean.csv", "w", encoding="utf-8") as f:
    f.write(content)
df = pd.read_csv("encoder_clean.csv")

# =====================================================
# SAFE UTILS
# =====================================================
def to_list_safe(val):
    """Safely converts string, list, or number into a list."""
    if pd.isna(val):
        return [0]
    try:
        parsed = eval(str(val))
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, (int, float)):
            return [parsed]
        elif isinstance(parsed, str) and "," in parsed:
            return [float(x.strip()) for x in parsed.split(",") if x.strip()]
        else:
            return list(parsed)
    except Exception:
        return [0]

# =====================================================
# DATASET
# =====================================================
class LegalDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Build vocabulary for textual labels/charges
        all_labels, all_charges = set(), set()
        for _, row in df.iterrows():
            val1, val2 = str(row.get("label", "")), str(row.get("charges", ""))
            if not any(c.isdigit() for c in val1):
                all_labels.update([x.strip(" []'\"") for x in val1.split(",") if x.strip()])
            if not any(c.isdigit() for c in val2):
                all_charges.update([x.strip(" []'\"") for x in val2.split(",") if x.strip()])

        # limit vocab to top-N to keep it trainable
        all_charges = list(all_charges)[:500]
        self.label_vocab = {lbl: i for i, lbl in enumerate(sorted(all_labels))} or {"dummy": 0}
        self.charge_vocab = {chg: i for i, chg in enumerate(sorted(all_charges))} or {"dummy": 0}
        print(f"🧾 Label vocab size: {len(self.label_vocab)} | Charge vocab size: {len(self.charge_vocab)}")

    def encode_labels(self, text, vocab):
        """Encodes labels into 0-1 tensor, clamped to valid range."""
        text = str(text)
        vec = torch.zeros(len(vocab), dtype=torch.float)

        if any(ch.isdigit() for ch in text):  # numeric list
            try:
                vals = torch.tensor(to_list_safe(text), dtype=torch.float)
                vals = torch.clamp(vals, 0, 1)
                vec[: min(len(vals), len(vec))] = vals[: min(len(vals), len(vec))]
            except Exception:
                pass
        else:  # textual list
            items = [x.strip(" []'\"") for x in text.split(",") if x.strip()]
            for item in items:
                if item in vocab:
                    vec[vocab[item]] = 1.0
        return vec

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text_inputs = self.tokenizer(
            str(row["text"]), padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        stat_inputs = self.tokenizer(
            str(row["statutes"]), padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        fact_inputs = self.tokenizer(
            str(row["facts"]), padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )

        statute_labels = self.encode_labels(row.get("label", ""), self.label_vocab)
        charge_labels = self.encode_labels(row.get("charges", ""), self.charge_vocab)

        return {
            "text": {k: v.squeeze(0) for k, v in text_inputs.items()},
            "statute": {k: v.squeeze(0) for k, v in stat_inputs.items()},
            "fact": {k: v.squeeze(0) for k, v in fact_inputs.items()},
            "statute_labels": statute_labels,
            "charge_labels": charge_labels
        }

# =====================================================
# MODEL
# =====================================================
class CrossAttentionLegalModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, hidden_dim=768, num_statutes=100, num_charges=50):
        super().__init__()
        self.encoder_text = AutoModel.from_pretrained(model_name)
        self.encoder_stat = AutoModel.from_pretrained(model_name)
        self.encoder_fact = AutoModel.from_pretrained(model_name)

        self.cross_attn_1 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=0.1)
        self.cross_attn_2 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=0.1)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.out_statute = nn.Linear(hidden_dim, num_statutes)
        self.out_charge = nn.Linear(hidden_dim, num_charges)

    def forward(self, text_inputs, statute_inputs, fact_inputs):
        text_out = self.encoder_text(**text_inputs).last_hidden_state.mean(dim=1)
        stat_out = self.encoder_stat(**statute_inputs).last_hidden_state.mean(dim=1)
        fact_out = self.encoder_fact(**fact_inputs).last_hidden_state.mean(dim=1)

        attn_output_1, _ = self.cross_attn_1(
            text_out.unsqueeze(0), stat_out.unsqueeze(0), stat_out.unsqueeze(0)
        )
        attn_output_2, _ = self.cross_attn_2(
            text_out.unsqueeze(0), fact_out.unsqueeze(0), fact_out.unsqueeze(0)
        )

        fused = torch.cat([attn_output_1.squeeze(0), attn_output_2.squeeze(0), text_out], dim=-1)
        fused = self.fusion(fused)
        pred_statute = torch.sigmoid(self.out_statute(fused))
        pred_charge = torch.sigmoid(self.out_charge(fused))
        return pred_statute, pred_charge

# =====================================================
# METRICS
# =====================================================
def compute_metrics(y_true_stat, y_pred_stat, y_true_chg, y_pred_chg, thr=0.3):
    y_pred_stat = (y_pred_stat > thr).int()
    y_pred_chg = (y_pred_chg > thr).int()
    return {
        "Statute_F1": f1_score(y_true_stat, y_pred_stat, average="macro", zero_division=0),
        "Charge_F1": f1_score(y_true_chg, y_pred_chg, average="macro", zero_division=0),
        "Statute_Prec": precision_score(y_true_stat, y_pred_stat, average="macro", zero_division=0),
        "Charge_Prec": precision_score(y_true_chg, y_pred_chg, average="macro", zero_division=0),
        "Statute_Rec": recall_score(y_true_stat, y_pred_stat, average="macro", zero_division=0),
        "Charge_Rec": recall_score(y_true_chg, y_pred_chg, average="macro", zero_division=0)
    }

# =====================================================
# MAIN
# =====================================================
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = LegalDataset(df, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    num_statutes = len(dataset.label_vocab)
    num_charges = len(dataset.charge_vocab)
    model = CrossAttentionLegalModel(MODEL_NAME, num_statutes=num_statutes, num_charges=num_charges).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        all_y_stat_true, all_y_stat_pred = [], []
        all_y_chg_true, all_y_chg_pred = [], []

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            text_inputs = {k: v.to(DEVICE) for k, v in batch["text"].items()}
            statute_inputs = {k: v.to(DEVICE) for k, v in batch["statute"].items()}
            fact_inputs = {k: v.to(DEVICE) for k, v in batch["fact"].items()}
            y_stat = batch["statute_labels"].to(DEVICE)
            y_chg = batch["charge_labels"].to(DEVICE)

            pred_stat, pred_chg = model(text_inputs, statute_inputs, fact_inputs)
            loss_stat = criterion(pred_stat, y_stat)
            loss_chg = criterion(pred_chg, y_chg)
            loss = loss_stat + loss_chg
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_y_stat_true.extend(y_stat.detach().cpu().numpy())
            all_y_stat_pred.extend(pred_stat.detach().cpu().numpy())
            all_y_chg_true.extend(y_chg.detach().cpu().numpy())
            all_y_chg_pred.extend(pred_chg.detach().cpu().numpy())

        metrics = compute_metrics(
            np.array(all_y_stat_true), np.array(all_y_stat_pred),
            np.array(all_y_chg_true), np.array(all_y_chg_pred)
        )
        print(f"\nEpoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
