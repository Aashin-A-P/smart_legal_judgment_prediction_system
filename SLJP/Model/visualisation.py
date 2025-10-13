# ==========================================================
# 🔍 TESTING AND FINAL EVALUATION VISUALS
# ==========================================================
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, f1_score, precision_score, recall_score
)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12})

# ========== COMPUTE FINAL METRICS ==========
test_precision = precision_score(y_test, pred_labels, average="macro", zero_division=0)
test_recall = recall_score(y_test, pred_labels, average="macro", zero_division=0)
test_f1 = f1_score(y_test, pred_labels, average="macro")
print(f"\n🎯 Test Metrics | Precision: {test_precision:.3f} | Recall: {test_recall:.3f} | F1: {test_f1:.3f}")

# ========== 1️⃣ TEST METRIC BARPLOT ==========
fig = plt.figure(figsize=(6,4))
sns.barplot(x=["Precision","Recall","F1-Score"], 
            y=[test_precision,test_recall,test_f1],
            palette=["#007acc","#66cc66","#ff9933"])
plt.title("Test Performance Metrics")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("ljp_final_outputs/plots/test_metrics_barplot.png", dpi=300)
plt.show()

# ========== 2️⃣ CONFUSION MATRIX ==========
cm = confusion_matrix(y_test.argmax(1), pred_labels.argmax(1))
fig = plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("ljp_final_outputs/plots/test_confusion_matrix.png", dpi=300)
plt.show()

# ========== 3️⃣ ROC CURVE ==========
y_true_flat = y_test.ravel()
y_score_flat = preds.ravel()
fpr, tpr, _ = roc_curve(y_true_flat, y_score_flat)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color="#ff6600", lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Macro-Average ROC Curve (Test Set)")
plt.legend()
plt.tight_layout()
plt.savefig("ljp_final_outputs/plots/test_roc_curve.png", dpi=300)
plt.show()

# ========== 4️⃣ TOP-10 STATUTES: TRUE VS PREDICTED ==========
true_counts = pd.Series(y_test.sum(0), index=mlb.classes_)
pred_counts = pd.Series(pred_labels.sum(0), index=mlb.classes_)

df_compare = pd.DataFrame({
    "Statute": mlb.classes_,
    "True_Freq": true_counts,
    "Pred_Freq": pred_counts
}).sort_values("True_Freq", ascending=False).head(10)

fig = plt.figure(figsize=(10,6))
x = np.arange(len(df_compare))
plt.bar(x-0.2, df_compare["True_Freq"], 0.4, label="True", color="#007acc")
plt.bar(x+0.2, df_compare["Pred_Freq"], 0.4, label="Predicted", color="#ff9933")
plt.xticks(x, df_compare["Statute"], rotation=45, ha="right")
plt.ylabel("Frequency")
plt.title("Top 10 Statutes: True vs Predicted Counts")
plt.legend()
plt.tight_layout()
plt.savefig("ljp_final_outputs/plots/test_true_vs_predicted.png", dpi=300)
plt.show()

# ========== 5️⃣ PERFORMANCE COMPARISON ==========
val_f1 = f1_score(y_val, (torch.sigmoid(model(f_val,g_val))>0.5).int(), average="macro")
fig = plt.figure(figsize=(5,4))
sns.barplot(x=["Validation","Test"], y=[val_f1, test_f1], palette=["#5a9bd4","#ed7d31"])
plt.ylim(0,1)
plt.title("Validation vs Test Macro F1 Comparison")
plt.tight_layout()
plt.savefig("ljp_final_outputs/plots/final_val_test_comparison.png", dpi=300)
plt.show()

print("\n✅ All test evaluation plots saved in ljp_final_outputs/plots/")
