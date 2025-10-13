# ==========================================================
#  Multi-class Statute Prediction using SVM (RBF Kernel)
#  TF-IDF Features + Evaluation Plots
# ==========================================================

import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Load dataset ----------------
df = pd.read_csv("encoder_top10_balanced_1000.csv")

# Convert lists from string form
df["statutes"] = df["statutes"].apply(eval)

# ---------------- Prepare Target ----------------
target_col = "statutes"

# Take first statute in each list (primary statute)
df["target"] = df[target_col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)

# Drop any missing entries
df = df.dropna(subset=["facts", "target"])
print(f"✅ Loaded {len(df)} samples for statute prediction")

# ---------------- TF-IDF Feature Extraction ----------------
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X = tfidf.fit_transform(df["facts"])
y = df["target"]

# ---------------- Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- Train SVM Model ----------------
svm_model = SVC(kernel="rbf", C=2, gamma="scale", decision_function_shape='ovo')
svm_model.fit(X_train, y_train)

# ---------------- Predictions ----------------
y_pred = svm_model.predict(X_test)

# ---------------- Classification Report ----------------
print("\n📊 Classification Report (Statute Prediction):")
print(classification_report(y_test, y_pred))

# ---------------- Plot 1: Confusion Matrix ----------------
plt.figure(figsize=(10,8))
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=svm_model.classes_)
disp.plot(cmap="Blues", xticks_rotation=90, colorbar=False)
plt.title("Confusion Matrix – SVM (RBF) for Statute Prediction")
plt.tight_layout()
plt.show()

# ---------------- Plot 2: F1 Score per Statute ----------------
report = classification_report(y_test, y_pred, output_dict=True)
f1_scores = {label: metrics["f1-score"] for label, metrics in report.items() if label in svm_model.classes_}

plt.figure(figsize=(10,6))
sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()), palette="viridis")
plt.xticks(rotation=75)
plt.title("F1-score per Statute – SVM (RBF) Model")
plt.ylabel("F1-score")
plt.xlabel("Statute Label")
plt.tight_layout()
plt.show()
