"""
Lab 4 A1 + A3 on the PROJECT data:
  A1 - repeat the Lab 3 experiments with the AI-tool functions (lab04_functions.py)
  A3 - performance comparison of the two k-means versions (own Lab 3 vs AI Lab 4):
       run time (average of 10 runs), within-cluster error (SSE), agreement with each other
       and with the diagnosis.

Run:  eeg_env\\Scripts\\python.exe labs\\lab04_a1_a3_project.py
"""

from lab_common import FEATURE_LIST, LAB_OUT, SEED, TABLES, load_feature_names, load_table
import lab03_functions as own
import lab04_functions as ai

import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# ---------------- settings ----------------
SUBJECT_A, SUBJECT_B = "h01", "s01"
K = 2
N_RUNS = 10
# ------------------------------------------

df = load_table(TABLES["with s14"])
features = load_feature_names(FEATURE_LIST)
X = df[features]
va = X[df["subject"] == SUBJECT_A].values[0].tolist()
vb = X[df["subject"] == SUBJECT_B].values[0].tolist()

# ---------- A1: Lab 3 experiments with the AI functions ----------
print("A1. AI-tool functions on the project data")
print(f"  Minkowski p=1..3 ({SUBJECT_A} vs {SUBJECT_B}): AI / scipy")
for p in [1, 2, 3]:
    print(f"    p={p}: {ai.minkowski_dis(va, vb, p):.6f} / {minkowski(va, vb, p):.6f}")
print(f"  Dot product: AI {ai.dot_prod(va, vb):.6f}, numpy {np.dot(va, vb):.6f}")
print(f"  Norm:        AI {ai.vec_norm(va):.6f}, numpy {np.linalg.norm(va):.6f}")
means, variances, stds = ai.matrix_stats(X.values.tolist())
print(f"  Means equal numpy:          {np.allclose(means, X.values.mean(axis=0))}")
print(f"  Std equal numpy (ddof=0):   {np.allclose(stds, X.values.std(axis=0, ddof=0))}")
print(f"  Std equal pandas (ddof=1):  {np.allclose(stds, X.std().values)}  "
      "<- expected False: pandas divides by n-1")

# ---------- A3: own k-means vs AI k-means ----------
Z = ((X - X.mean()) / X.std(ddof=0)).values.tolist()          # standardised features


def sse(data, centroids, labels):
    # within-cluster sum of squared distances (lower = tighter clusters)
    return sum(own.point_distance(p, centroids[c]) ** 2 for p, c in zip(data, labels))


def timed(fn):
    # run fn N_RUNS times; return the last result and the average time
    times = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - start)
    return out, sum(times) / N_RUNS


(own_c, own_l), own_t = timed(lambda: own.kmeans(Z, K))
(ai_c, ai_l), ai_t = timed(lambda: ai.kmeans(Z, K, seed=SEED))
sk, sk_t = timed(lambda: KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(np.array(Z)))

rows = []
for name, centroids, labels, t in [("Own k-means (Lab 3)", own_c, own_l, own_t),
                                   ("AI k-means (Lab 4)", ai_c, ai_l, ai_t),
                                   ("sklearn KMeans", sk.cluster_centers_.tolist(),
                                    sk.labels_.tolist(), sk_t)]:
    rows.append({"version": name, "avg_time_s": t, "sse": sse(Z, centroids, labels),
                 "ARI_vs_own": adjusted_rand_score(own_l, labels),
                 "ARI_vs_diagnosis": adjusted_rand_score(df["label"], labels),
                 "cluster_sizes": sorted(pd.Series(labels).value_counts().tolist())})
table = pd.DataFrame(rows)
print(f"\nA3. k-means comparison (k = {K}, 16 standardised features, time = mean of {N_RUNS} runs)")
print(table.round(4).to_string(index=False))
print("Different start points can give different (equally valid) clusters; SSE says which is tighter.")
table.round(5).to_csv(LAB_OUT / "lab04_a3_kmeans_comparison.csv", index=False)
print(f"\nSaved: {LAB_OUT / 'lab04_a3_kmeans_comparison.csv'}")
