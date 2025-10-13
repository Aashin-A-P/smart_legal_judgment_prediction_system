# ==========================================================
#  Full Benchmark Suite – Individual Folders per Model
#  10 Plots + CSV Reports per Model + Summary Comparison
#  Models: NaiveBayes | RandomForest | XGBoost | MLP | SVM
# ==========================================================

import pandas as pd
import ast
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve, accuracy_score
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ---------------- Load dataset ----------------
df = pd.read_csv("dataset.csv")
df["statutes"] = df["statutes"].apply(ast.literal_eval)
df["target"] = df["statutes"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
df = df.dropna(subset=["facts", "target"])
print(f"✅ Loaded {len(df)} samples")

# ---------------- Encode + Vectorize ----------------
le = LabelEncoder()
y = le.fit_transform(df["target"])
class_names = list(le.classes_)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
X = tfidf.fit_transform(df["facts"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ---------------- Models ----------------
models = {
    "NaiveBayes": MultinomialNB(),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, eval_metric="mlogloss", use_label_encoder=False, random_state=42
    ),
    "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42),
    "SVM_RBF": SVC(kernel="rbf", C=2, gamma="scale", probability=True, decision_function_shape="ovo")
}

os.makedirs("model_outputs", exist_ok=True)

# ---------------- Plot utilities ----------------
def plot_confusion_matrix(y_true, y_pred, name, plot_dir):
    plt.figure(figsize=(9,7))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix – {name}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{name}_1_confusion_matrix.png")
    plt.close()

def plot_f1_bar(report, name, plot_dir):
    f1_scores = [report[str(i)]["f1-score"] if str(i) in report else 0 for i in range(len(class_names))]
    plt.figure(figsize=(9,6))
    sns.barplot(x=class_names, y=f1_scores, palette="viridis")
    plt.xticks(rotation=70)
    plt.title(f"Per-Class F1 Scores – {name}")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{name}_2_f1_bar.png")
    plt.close()

def plot_macro_scores(report, name, plot_dir):
    plt.figure(figsize=(5,4))
    vals = [report["macro avg"]["precision"], report["macro avg"]["recall"], report["macro avg"]["f1-score"]]
    sns.barplot(x=["Precision","Recall","F1"], y=vals, palette="crest")
    plt.title(f"Macro-Average Metrics – {name}")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{name}_3_macro_scores.png")
    plt.close()

def plot_feature_importance(model, name, tfidf, plot_dir):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        idx = np.argsort(imp)[-20:]
        plt.figure(figsize=(8,6))
        plt.barh(np.array(tfidf.get_feature_names_out())[idx], imp[idx])
        plt.title(f"Top 20 Feature Importances – {name}")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{name}_4_feature_importance.png")
        plt.close()

def plot_roc_curve(model, X_test, y_test, name, plot_dir):
    try:
        y_prob = model.predict_proba(X_test)
        y_bin = label_binarize(y_test, classes=np.arange(len(class_names)))
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, color='darkorange')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Micro-ROC Curve – {name}")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{name}_5_roc_curve.png")
        plt.close()
    except Exception:
        pass

def plot_precision_recall(model, X_test, y_test, name, plot_dir):
    try:
        y_prob = model.predict_proba(X_test)
        y_bin = label_binarize(y_test, classes=np.arange(len(class_names)))
        prec, rec, _ = precision_recall_curve(y_bin.ravel(), y_prob.ravel())
        plt.figure(figsize=(6,5))
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve – {name}")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{name}_6_precision_recall.png")
        plt.close()
    except Exception:
        pass

def plot_class_distribution(y_pred, name, plot_dir):
    plt.figure(figsize=(8,4))
    sns.countplot(x=le.inverse_transform(y_pred), palette="flare")
    plt.xticks(rotation=70)
    plt.title(f"Predicted Class Distribution – {name}")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{name}_7_class_distribution.png")
    plt.close()

def plot_heatmap_scores(report, name, plot_dir):
    df_r = pd.DataFrame(report).iloc[:-3, :-1]
    plt.figure(figsize=(8,6))
    sns.heatmap(df_r, annot=True, cmap="coolwarm")
    plt.title(f"Performance Heatmap – {name}")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{name}_8_heatmap.png")
    plt.close()

# ---------------- Main training loop ----------------
summary_rows = []

for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    model_dir = f"model_outputs/{name}"
    plot_dir = f"{model_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy ({name}): {acc:.3f}")

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(f"{model_dir}/{name}_classification_report.csv")

    summary_rows.append({
        "Model": name,
        "Accuracy": acc,
        "Macro Precision": report_dict["macro avg"]["precision"],
        "Macro Recall": report_dict["macro avg"]["recall"],
        "Macro F1": report_dict["macro avg"]["f1-score"]
    })

    # Generate plots into that model's folder
    plot_confusion_matrix(y_test, y_pred, name, plot_dir)
    plot_f1_bar(report_dict, name, plot_dir)
    plot_macro_scores(report_dict, name, plot_dir)
    plot_feature_importance(model, name, tfidf, plot_dir)
    plot_roc_curve(model, X_test, y_test, name, plot_dir)
    plot_precision_recall(model, X_test, y_test, name, plot_dir)
    plot_class_distribution(y_pred, name, plot_dir)
    plot_heatmap_scores(report_dict, name, plot_dir)

    print(f"✅ Saved report & plots for {name} in '{model_dir}'")

# ---------------- Save global summary ----------------
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("model_outputs/model_summary.csv", index=False)
print("\n🎯 All models completed! Each model has its own folder with plots & CSVs.")
