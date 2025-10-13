# ⚖️ JudgEx: Legal Reasoning Module
### _An Explainable Legal Reasoning Framework built on LLaMA-3-Instruct_

---

## 🧠 Overview
**JudgEx** is a generative reasoning module designed to produce **judge-style legal rationales** for predicted judgments.  
It forms the **explanation layer** of the larger _Smart Legal Judgment Prediction System (SLJP)_, ensuring that every model prediction is accompanied by a **transparent, human-interpretable explanation** grounded in statutory reasoning and factual evidence.

---

## 🎯 Key Objectives
- Provide **contextually coherent, legally grounded reasoning** for each predicted judgment.  
- Integrate factual embeddings, predicted statutes, and retrieved precedents into a unified **multi-attention prompt**.  
- Follow a structured **IRAC (Issue–Rule–Application–Conclusion)** format typical of judicial decisions.  
- Enable **explainable AI adoption** in legal NLP by combining **retrieval, attention, and generation**.

---

## 🏗️ Architecture Overview
JudgEx is powered by a fine-tuned **LLaMA-3-Instruct** model trained using two complementary paradigms:

1. **Continued Pretraining (CPT)**  
   - Adapts the base LLaMA-3 model to the _Indian legal domain_.  
   - Trained on millions of legal texts, court judgments, and statutes.  
   - Objective: Masked Language Modeling (MLM).  

2. **Supervised Fine-Tuning (SFT)**  
   - Teaches the model to map factual case descriptions → structured rationales.  
   - Uses the **PredEx dataset** containing factual summaries and expert-written judgments.  
   - Objective: Autoregressive cross-entropy loss for text generation.

---

## 🧩 Input Fusion and Context Retrieval
Before reasoning generation, the module fuses multiple input sources:

| Input Component | Description |
|-----------------|--------------|
| `f` | Factual case description |
| `Ŝ` | Predicted statutes from the LJP module |
| `Ĉh` | Predicted charges from the LJP module |
| `R_top-k` | Top-k similar precedents retrieved using FAISS semantic search |

These are concatenated into a composite prompt:

\[
P = [f, \hat{S}, \hat{Ch}, R_{top-k}]
\]

This ensures the generated reasoning remains **factually grounded and precedent-aware**.

---

## ⚙️ Generative Process
JudgEx employs **multi-level attention** across factual, statutory, and precedent segments:

\[
E = \text{LLaMA3}_{JudgEx}(P)
\]

### Steps:
1. Tokenize and embed the composite prompt.  
2. Compute cross-segment attention between factual and legal embeddings.  
3. Generate reasoning text conditioned on all inputs.  
4. Post-process the output into **structured IRAC format**.

---

## 📘 IRAC-Structured Output
JudgEx explanations follow a consistent four-part legal reasoning structure:

