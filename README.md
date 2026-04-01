# Smart Legal Judgment Prediction System using Multi-Attention and Graph-Augmented Reasoning
An AI powered Legal Judgment Prediction System that predicts the outcome of a case in the Indian Court of Law based on multi-attention layers and graph-augmented reasoning.

This project has been done under Dr. M.R. Sumalatha as part of the Final Year Project (Project I) of 7th Semester. The project deals with predicting statutes and charges provided the factual descriptions and also providing explanation for the prediction. 

The project involves two modules: \
    1. Smart Legal Judgment Prediction Module (SLJP) \
    2. Legal Reasoning Module (Judgment + Explainability Module [JudgEx Module]) 

## 📘 Abstract
Indian law is rooted in the Constitution of India and administered through a hierarchical judiciary—from the Supreme Court to subordinate Civil and Criminal Courts. However, this framework remains complex and often inaccessible to the public.

To address this, the **Smart Legal Judgment Prediction System (SLJP)** integrates **multi-level attention** and **graph-augmented reasoning** to predict statutes and charges from factual case descriptions. The architecture employs:
- **InLegalBERT** for factual encoding  
- **Graph Attention Network (GAT)** for relational learning  
- **Cross-Attention Mechanism** to align facts with statutory semantics  
- **JudgEx (LLaMA-3)** for contextual, judge-style rationales  

Experiments on the **Indian Legal Corpus** show significant improvements in macro-F1, precision, and interpretability over **LegalBERT** and **CaseLaw-GNN**. By producing **role-aware, transparent judgments**, SLJP enhances consistency and accountability across modern court ecosystems.

---

## 🔍 Keywords
`Legal Judgment Prediction` · `Graph-Augmented Reasoning` · `Multi-Level Attention` · `InLegalBERT` · `LLaMA-3`

---

## ⚖️ 1. Introduction
Indian courts face mounting backlogs and interpretative inconsistencies due to complex statutes and limited judicial resources.  
While prior LJP models (LegalBERT, CaseLaw-GNN, CAIL) achieved accuracy, they **lacked relational understanding and interpretability**.

SLJP overcomes this by:
- Capturing **semantic dependencies** via multi-level attention  
- Modeling **relational links** through GAT  
- Providing **human-readable reasoning** using JudgEx  

---

## 📚 2. Related Work
### 🏛 Domain-Specific Legal Transformers
- **LegalBERT**, **Lawformer**, and **InLegalBERT** improved legal text understanding.  
- InLegalBERT, trained on Indian judgments, outperformed multilingual BERTs for factual encoding.

### 🔗 Graph-Based Legal Reasoning
- GNN-based models (CaseLaw-GNN, Zhao et al., Xu et al.) modeled case–statute–charge relations.  
- However, interpretability remained limited.

### 🎯 Attention and Alignment
- Cross- and multi-attention mechanisms bridge facts with statutory semantics.  
- Dual-residual and multi-task models improved contextual understanding.

### 🧩 Retrieval-Augmented Explainability
- RAG-based systems and LLaMA-series models enabled **fact-grounded explanations**.  
- **JudgEx** extends this through fine-tuned **LLaMA-3-Instruct**, offering judge-style reasoning.

---

## 🗂️ 3. Dataset
### 📄 Custom Indian Legal Corpus
- ~1,500 judicial documents from Supreme and High Courts.  
- Fields: `{filename, facts, statutes, charges, judgment}`  
- Cleaned and tokenized via **InLegalBERT**.  
- Split: 70% train / 10% val / 20% test.

### 🧠 PredEx Dataset
- Contains factual summaries + expert-authored rationales.  
- Used to train JudgEx for explainability using BLEU, METEOR, and BERTScore.

---

## 🧮 4. Methodology
### 🧱 Architecture Overview
Two synergistic modules:
1. **Legal Judgment Prediction Module**  
   → Predicts statutes and charges using InLegalBERT + GAT + Cross-Attention.  
2. **Legal Reasoning Module (JudgEx)**  
   → Generates human-interpretable, context-aware rationales.

### ⚙️ Core Components
- **Input Representation:**  
  Each case \(C = \{f, S, Ch, Y\}\) with facts, statutes, charges, and judgment.  
- **InLegalBERT Encoder:**  
  Produces contextual embeddings \(h_f\).  
- **Graph Attention Network:**  
  Learns relational dependencies across cases/statutes.  
- **Cross-Attention Layer:**  
  Aligns facts ↔ statutes for semantic clarity.  
- **Multi-Task Learning:**  
  Jointly optimizes statute and charge predictions with weighted BCE loss.

### 🤖 JudgEx: Legal Reasoning Module
- Built on **LLaMA-3-Instruct**  
- Trained via:
  - **CPT (Continued Pretraining):** learns Indian legal language  
  - **SFT (Supervised Fine-Tuning):** maps facts → reasons  
- Integrates **precedent retrieval (FAISS)** for context grounding.  
- Generates structured **IRAC-style explanations (Issue–Rule–Application–Conclusion).**

---

## 📊 5. Results and Discussion
### ⚙️ Experimental Setup
- **Libraries:** PyTorch, Transformers, PyTorch Geometric  
- **Hardware:** NVIDIA T4 / P100 (16 GB VRAM)  
- **Optimizer:** AdamW (lr = 2e-5) with early stopping  
- **Metrics:** Macro-F1, Precision, Recall, Accuracy; BLEU, METEOR, ROUGE for reasoning  

### 📈 Baseline Comparison
| Model | Precision | Recall | Macro-F1 |
|:------|:----------:|:------:|:--------:|
| XGBoost | 0.6875 | 0.6154 | 0.6247 |
| LegalBERT | 0.7023 | 0.6710 | 0.6825 |
| CaseLaw-GNN | 0.7134 | 0.6932 | 0.7015 |
| InLegalBERT | 0.7452 | 0.7184 | 0.7247 |
| **Ours (Final)** | **0.8421** | **0.8034** | **0.8212** |

### 🔬 Ablation Study
| Variant | Precision | Recall | Macro-F1 |
|:--------|:----------:|:------:|:--------:|
| InLegalBERT | 0.7452 | 0.7184 | 0.7247 |
| + GAT | 0.7558 | 0.7391 | 0.7456 |
| + Cross-Attention | 0.7832 | 0.7586 | 0.7695 |
| **+ MTL (Final)** | **0.8421** | **0.8034** | **0.8212** |

### 📊 Visual Insights
- **ROC AUC:** 0.992 (near-perfect discrimination)  
- **Stable loss curves** across training/validation/test  
- **High F1** on IPC 302 (Murder), IPC 420 (Cheating)  

### 🧾 Reasoning (JudgEx) Evaluation
| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR |
|:------|:--------:|:--------:|:--------:|:----:|:------:|
| LLaMA-2 | 0.327 | 0.187 | 0.214 | 0.061 | 0.176 |
| LLaMA-2-SFT | 0.499 | 0.444 | 0.451 | 0.252 | 0.376 |
| **JudgEx (Ours)** | **0.512** | **0.446** | **0.452** | **0.268** | **0.391** |

---

## 🧩 6. Discussion
- SLJP integrates **semantic (InLegalBERT)**, **relational (GAT)**, and **interpretive (JudgEx)** reasoning.  
- Outperforms baselines across accuracy, precision, and interpretability.  
- Follows ethical AI principles by **justifying every prediction**.  
- Limitations: computational cost, rare statute underrepresentation.  
- Future: Retrieval-Augmented Generation (RAG), dynamic precedent reasoning, RLHF fine-tuning.

---

## ✅ 7. Conclusion & Future Work
The Smart Legal Judgment Prediction System unifies factual inference and explainable reasoning for the Indian judiciary.  
It achieves:
- **Macro-F1 = 0.82**  
- **AUC = 0.992**  
- **ROUGE-L = 0.452**, **BLEU = 0.268**, **METEOR = 0.391**

Future directions:
- Multi-lingual legal corpora  
- Retrieval-augmented precedent graphs  
- RLHF for factual–reason alignment  
- Integration into **judicial analytics & legal education platforms**

---

## 🤝 Credits
This research was conducted as part of the **Final-Year Project, Department of Information Technology, Madras Institute of Technology, Anna University**.

**Supervisor:** Dr. M. R. Sumalatha  
**Project:** _Smart Legal Judgment Prediction System using Multi-Level Attention and Graph-Augmented Reasoning_

---

## ⚖️ Disclosure
The authors declare **no competing interests** related to this work.

---

### 🧾 Citation
If you use this repository, please cite:
> Aashin et al. (2025). *Smart Legal Judgment Prediction System using Multi-Level Attention and Graph-Augmented Reasoning.*  
> Department of Information Technology, MIT – Anna University.

---

### 📬 Contact
**Author:** Aashin  
📧 [mailto:apaashin@gmail.com](apaashin@gmail.com)  
🎓 Final-Year Student @ MIT – Anna University  
