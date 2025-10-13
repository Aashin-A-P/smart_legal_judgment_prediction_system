import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm

# Load InLegalBERT (force safetensors only)
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
model = AutoModel.from_pretrained("law-ai/InLegalBERT", use_safetensors=True)
model.eval()

def get_embedding(text):
    """Convert text into a 768-dim InLegalBERT embedding (mean pooled)."""
    if not isinstance(text, str) or text.strip() == "":
        return np.zeros(768)  # fallback for empty text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Process each split
for split in ["train_enriched.csv", "validation_enriched.csv", "test_enriched.csv"]:
    print(f"\n⚡ Processing {split} ...")
    df = pd.read_csv(split)

    embeddings = []
    for i, text in enumerate(tqdm(df["text"], desc=f"Embedding {split}")):
        emb = get_embedding(str(text))
        embeddings.append(emb)

    embeddings = np.array(embeddings)

    # Save embeddings separately
    np.save(split.replace(".csv", "_embeddings.npy"), embeddings)

    # Save CSV without embeddings (lightweight)
    df.to_csv(split.replace(".csv", "_meta.csv"), index=False)

    print(f"✅ Saved {split.replace('.csv', '_embeddings.npy')} (embeddings)")
    print(f"✅ Saved {split.replace('.csv', '_meta.csv')} (metadata)")
