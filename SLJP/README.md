# ⚖️ Smart Legal Judgment Prediction (SLJP) Module
### _Factual Understanding and Relational Reasoning for Judicial Decision Support_

---

## 🧠 Overview
The **Smart Legal Judgment Prediction (SLJP)** module is the **predictive backbone** of the Smart Legal Judgment Prediction System.  
It automatically identifies **relevant statutes**, **charges**, and **judgments** from factual case descriptions using a **hybrid multi-level attention and graph-reasoning architecture**.

This module leverages:
- **InLegalBERT** for deep semantic encoding of case facts  
- **Graph Attention Networks (GAT)** for relational reasoning among facts, statutes, and charges  
- **Cross-Attention Mechanisms** for aligning factual and statutory semantics  
- **Multi-Task Learning (MTL)** for joint prediction of statutes and charges  

---

## 🎯 Objectives
- Predict legal provisions and charges directly from factual narratives.  
- Capture **semantic**, **contextual**, and **relational** dependencies in legal text.  
- Achieve **higher interpretability** and **explainability** through attention weights.  
- Serve as the **input provider** for the reasoning module (JudgEx).  

---

## 🏗️ Architecture Overview

### 🧩 Pipeline


### 🧱 Components
1. **Factual Encoding (InLegalBERT)**  
   Encodes case facts into contextual embeddings capturing domain-specific legal semantics.
   \[
   h_f = \text{InLegalBERT}(f)
   \]

2. **Graph-Augmented Representation (GAT)**  
   Models dependencies among cases, statutes, and charges via graph attention:
   \[
   h_i' = \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j \right)
   \]
   where \( \alpha_{ij} \) denotes the legal relevance between nodes.

3. **Cross-Attention Integration**  
   Aligns factual embeddings with statutory embeddings:
   \[
   \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   \]

4. **Multi-Task Learning Layer (MTL)**  
   Jointly predicts statutes (\(\hat{Y}_S\)) and charges (\(\hat{Y}_{Ch}\)):
   \[
   \mathcal{L} = \lambda_1 \mathcal{L}_S + \lambda_2 \mathcal{L}_{Ch}
   \]
   addressing class imbalance using weighted binary cross-entropy.

---

## 📘 Input Representation
Each case file is represented as a tuple:
\[
C = \{f, S, Ch, Y\}
\]

| Symbol | Meaning |
|:-------|:---------|
| `f` | Factual case description |
| `S` | Statutory references |
| `Ch` | Charges invoked |
| `Y` | Judgment label |

During **training**, all four components are used.  
During **inference**, only `f` (facts) is given → the model predicts `Ŝ`, `Ĉh`, and `Ŷ`.

---

## ⚙️ Implementation Details

### 💡 Algorithm (Simplified)
```python
# Step 1: Encode facts
h_f = InLegalBERT(f)

# Step 2: Build graph of facts–statutes–charges
G = build_legal_graph(facts, statutes, charges)

# Step 3: Relational propagation
h_f_prime = GAT(G, h_f)

# Step 4: Cross-attention with statutes
aligned = cross_attention(Q=h_f_prime, K=h_statutes, V=h_statutes)

# Step 5: Joint prediction
Y_S = sigmoid(W_S @ aligned)
Y_Ch = sigmoid(W_Ch @ aligned)

