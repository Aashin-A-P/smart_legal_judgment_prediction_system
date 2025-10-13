import pandas as pd
import matplotlib.pyplot as plt

# ======================
# 1. Filtered Data
# ======================
data = {
    "Models": [
        "LLaMa-2",
        "LLaMa-2 SFT",
        "LLaMa-2 CPT",
        "JudgEx"  # renamed
    ],
    "ROUGE-1": [32.11, 49.72, 33.55, 50.76],
    "ROUGE-2": [18.86, 43.21, 15.49, 43.38],
    "ROUGE-L": [21.09, 43.99, 22.87, 43.79],
    "BLEU":    [5.99, 25.31, 8.98, 25.55],
    "METEOR":  [17.60, 36.30, 23.26, 36.43]
}

df = pd.DataFrame(data)

# ======================
# 2. Show Table
# ======================
print("\n===== Lexical Evaluation Comparison =====\n")
print(df.to_string(index=False))

# ======================
# 3. Plot Bar Graph
# ======================
metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "METEOR"]

ax = df.set_index("Models")[metrics].plot(
    kind="bar",
    figsize=(10, 6),
    rot=0,
    colormap="viridis"
)

plt.title("Lexical Evaluation: LLaMa-2 vs. JudgEx")
plt.ylabel("Score (%)")
plt.xlabel("Models")
plt.legend(title="Metrics")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
