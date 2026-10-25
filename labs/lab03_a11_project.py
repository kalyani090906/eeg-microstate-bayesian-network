"""
Lab 3 A11: repeat experiments A2-A10 and the k-means of A11 on the PROJECT data
(features_final_16.csv, 28 subjects x 16 microstate features), using your own functions.

Run:  eeg_env\\Scripts\\python.exe labs\\lab03_a11_project.py
Outputs go to outputs/labs/ (plots are saved, not shown, so the script never stops halfway).
"""

from lab_common import FEATURE_LIST, LAB_OUT, SEED, TABLES, load_feature_names, load_table
from lab03_functions import (dot_product, euclidean_norm, kmeans, mean, minkowski_distance,
                             self_statistics, standard_deviation, variance)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# ---------------- settings ----------------
SUBJECT_A, SUBJECT_B = "h01", "s01"     # two feature vectors for A5-A7 (one control, one patient)
HIST_FEATURE = "state0_occurrence"      # feature for the A10 histogram
K = 2                                   # clusters for A11 (two groups: control / patient?)
# ------------------------------------------

df = load_table(TABLES["with s14"])
features = load_feature_names(FEATURE_LIST)
X = df[features]

# ---------- A1 [THINKING]: data types ----------
print("A1. Data types")
print("  subject : nominal (an identifier, not used as a feature)")
print("  label   : nominal, 2 classes (0 = healthy, 1 = schizophrenia)")
print("  all 16 features: ratio (durations in seconds, occurrences per second,")
print("  transition probabilities 0-1; all have a true zero)")

# ---------- A2-A3: encoding and dimensionality ----------
categorical = [c for c in features if X[c].dtype == object]
print(f"\nA2-A3. Categorical feature columns: {categorical if categorical else 'none'}")
print(f"  So no encoding is needed: dimensionality stays {X.shape[1]} features x {X.shape[0]} subjects.")

# ---------- A4-A6: Minkowski distance, p = 1..10, own vs scipy ----------
va = X[df["subject"] == SUBJECT_A].values[0]
vb = X[df["subject"] == SUBJECT_B].values[0]
z = (X - X.mean()) / X.std(ddof=0)                       # standardised copy (every feature same scale)
za = z[df["subject"] == SUBJECT_A].values[0]
zb = z[df["subject"] == SUBJECT_B].values[0]

print(f"\nA4-A6. Minkowski distance between {SUBJECT_A} and {SUBJECT_B}")
print("   p   own (raw)   scipy (raw)   own (standardised)")
rows = []
for p in range(1, 11):
    own_raw, _ = minkowski_distance(va, vb, p)
    own_std, _ = minkowski_distance(za, zb, p)
    rows.append({"p": p, "own_raw": own_raw, "scipy_raw": minkowski(va, vb, p),
                 "own_standardised": own_std})
    print(f"  {p:2d}   {own_raw:9.4f}   {minkowski(va, vb, p):11.4f}   {own_std:12.4f}")
dist = pd.DataFrame(rows)
print(f"  Largest own-vs-scipy difference: {np.abs(dist.own_raw - dist.scipy_raw).max():.2e}")
plt.figure(figsize=(7, 4))
plt.plot(dist.p, dist.own_raw, marker="o", label="raw features")
plt.plot(dist.p, dist.own_standardised, marker="s", label="standardised features")
plt.xlabel("p"), plt.ylabel("Minkowski distance"), plt.legend()
plt.title(f"Minkowski distance {SUBJECT_A} vs {SUBJECT_B}, p = 1..10")
plt.tight_layout(), plt.savefig(LAB_OUT / "lab03_a5_minkowski.png", dpi=160), plt.close()

# ---------- A7: dot product and Euclidean norm, own vs numpy ----------
n = len(va)
print(f"\nA7. Dot product {SUBJECT_A}.{SUBJECT_B}: own {dot_product(va, vb, n):.6f}, numpy {np.dot(va, vb):.6f}")
print(f"    Norm of {SUBJECT_A}: own {euclidean_norm(va, n):.6f}, numpy {np.linalg.norm(va):.6f}")

# ---------- A8-A9: mean / variance / std, own vs numpy ----------
own_stats = self_statistics(X)
own_stats["numpy_mean"] = np.mean(X.values, axis=0)
own_stats["numpy_std"] = np.std(X.values, axis=0)       # ddof=0, same definition as ours
print("\nA8-A9. Own vs numpy (first 4 features):")
print(own_stats.head(4).round(5).to_string())
print(f"  Largest difference: mean {np.abs(own_stats['mean'] - own_stats['numpy_mean']).max():.2e}, "
      f"std {np.abs(own_stats['std'] - own_stats['numpy_std']).max():.2e}")
own_stats.round(6).to_csv(LAB_OUT / "lab03_a8_a9_statistics.csv")

# ---------- A10: histogram of one feature ----------
values = X[HIST_FEATURE]
counts, edges = np.histogram(values, bins=6)
print(f"\nA10. {HIST_FEATURE}: mean {mean(values):.4f}, variance {variance(values):.6f}")
print(f"  bucket counts {counts.tolist()}")
plt.figure(figsize=(7, 4))
plt.hist([values[df.label == 0], values[df.label == 1]], bins=edges, stacked=True,
         label=["healthy", "schizophrenia"], edgecolor="black")
plt.xlabel(HIST_FEATURE), plt.ylabel("Number of subjects"), plt.legend()
plt.title(f"Histogram of {HIST_FEATURE} (28 subjects)")
plt.tight_layout(), plt.savefig(LAB_OUT / "lab03_a10_histogram.png", dpi=160), plt.close()

# ---------- A11: your k-means on the standardised features ----------
# (k-means uses distances, so features are standardised with your own mean/std first)
standardised = [[(row[j] - mean(X.iloc[:, j])) / standard_deviation(X.iloc[:, j])
                 for j in range(n)] for row in X.values.tolist()]
centroids, clusters = kmeans(standardised, K)
sk = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(np.array(standardised))

table = pd.crosstab(pd.Series(clusters, name="own cluster"),
                    df["label"].map({0: "healthy", 1: "schizophrenia"}))
print(f"\nA11. Own k-means, k = {K}, on 16 standardised features:")
print(table.to_string())
print(f"  Agreement with diagnosis (adjusted Rand index): {adjusted_rand_score(df.label, clusters):.3f}"
      "  (0 = chance, 1 = perfect)")
print(f"  Agreement own vs sklearn KMeans: {adjusted_rand_score(clusters, sk.labels_):.3f}")
table.to_csv(LAB_OUT / "lab03_a11_kmeans_vs_labels.csv")
print(f"\nSaved plots and tables in {LAB_OUT}")
