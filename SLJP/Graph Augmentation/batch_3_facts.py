import os
import pandas as pd
import itertools
import json
import time
from tqdm import tqdm
from openai import OpenAI

# ========= CONFIG =========
MODELS = [
    "x-ai/grok-4-fast:free"
]
model_cycle = itertools.cycle(MODELS)

# ===== OpenRouter client =====

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"  # 🔑 replace with your actual key
)


# ========= PROMPT =========
def build_prompt(text):
    return f"""
You are a legal judgment assistant.
Your task is to carefully read the legal case text and extract the **major factual statements** only.

Strict Rules:
1. Extract between 3 and 7 key facts that are most relevant to the judgment.
2. Each fact must be concise, objective, and describe an event or action that occurred.
   Example: "The accused stabbed the victim with a knife" ✅
            "The accused is guilty of murder" ❌ (interpretive, not a fact).
3. Do not include legal conclusions, charges, or statutes — only events, actions, or circumstances.
4. Ensure each fact is a short sentence (max 20 words).
5. Always return a valid JSON object with exactly one key: "facts".
6. Do not include explanations, reasoning, or Markdown formatting.

Case Text:
{text[:120000]}
"""


# ========= CLEAN & PARSE =========
def safe_parse_json(raw_output):
    if not raw_output:
        return {}
    try:
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned)
    except Exception:
        return {}

# ========= API CALL =========
def call_model(model, text, retries=3):
    prompt = build_prompt(text)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[{model}] Error: {e} (attempt {attempt+1})")
            time.sleep(5)
    return None

# ========= PROCESS DATA =========
def process_dataset(input_path, output_path, limit=None):
    df = pd.read_csv(input_path)
    if limit:
        df = df.head(limit)

    # Initialize empty CSV with headers
    pd.DataFrame(columns=[
        "filename", "label", "model_used", "raw_response", "facts"
    ]).to_csv(output_path, index=False)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        model = next(model_cycle)
        raw_output = call_model(model, row["text"])

        parsed = safe_parse_json(raw_output)

        case_result = {
            "filename": row.get("filename", f"case_{i}"),
            "label": row.get("label", ""),
            "model_used": model,
            "raw_response": raw_output,
            "facts": parsed.get("facts", [])
        }

        # Append to CSV immediately
        pd.DataFrame([case_result]).to_csv(output_path, mode="a", header=False, index=False)

        print(f"💾 Saved case {i+1} → {output_path}")

        time.sleep(1)

    print(f"✅ All done. Results saved to {output_path}")

# ========= RUN =========
if __name__ == "__main__":
    process_dataset(
        input_path="batch_3.csv",           # input dataset (must have 'text' column)
        output_path="batch_3_cases_with_facts.csv", # output dataset
        limit=10000                       # adjust for testing
    )
