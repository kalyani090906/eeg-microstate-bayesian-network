"""
Project item 7: clustering on the extracted features, compared with the diagnosis.

The clustering never sees the labels. For each method (k-means, Ward hierarchical, Gaussian
mixture) and each k = 2, 3, 4, on the 16 standardised features:
  - contingency table (cluster x diagnosis)
  - adjusted Rand index (ARI; 0 = chance, 1 = clusters = diagnosis)
  - normalised mutual information (NMI; 0 = unrelated, 1 = identical)
  - purity (share of subjects in their cluster's majority diagnosis)
  - permutation test: shuffle the labels 1,000 times; p = how often shuffled labels match the
    clusters at least as well (ARI) as the real labels
  - silhouette score (how well separated the clusters are, ignoring labels; -1..1)
Run with and without s14. Also a PCA picture of the k = 2 k-means clusters.

Run:  eeg_env\\Scripts\\python.exe scripts\\item7_clustering.py
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
RESULTS_DIR = OUTPUT_DIR / "items"
TABLE_16 = OUTPUT_DIR / "features_final_16.csv"
FEATURE_LIST = OUTPUT_DIR / "features_final_16_list.txt"
K_VALUES = [2, 3, 4]
N_PERMUTATIONS = 1000
SEED = 42
# --------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METHODS = {
    "k-means": lambda k: KMeans(n_clusters=k, n_init=20, random_state=SEED),
    "Ward hierarchical": lambda k: AgglomerativeClustering(n_clusters=k, linkage="ward"),
    # 'diag' covariance: 28 subjects are too few to estimate a full 16x16 covariance per cluster
    "Gaussian mixture": lambda k: GaussianMixture(n_components=k, covariance_type="diag",
                                                  n_init=10, random_state=SEED),
}


def purity(labels, clusters):
    # each cluster votes for its majority diagnosis; share of subjects that agree
    table = pd.crosstab(clusters, labels)
    return table.max(axis=1).sum() / len(labels)


def permutation_p(labels, clusters, rng):
    # how often do shuffled labels agree with the clusters at least as well as the real ones?
    observed = adjusted_rand_score(labels, clusters)
    shuffled = [adjusted_rand_score(rng.permutation(labels), clusters) for _ in range(N_PERMUTATIONS)]
    return (1 + sum(s >= observed for s in shuffled)) / (1 + N_PERMUTATIONS)


if __name__ == "__main__":
    features = [l.strip() for l in FEATURE_LIST.read_text().splitlines() if l.strip()]
    df_all = pd.read_csv(TABLE_16)
    rng = np.random.default_rng(SEED)

    rows, tables = [], []
    for table_name, df in [("with s14", df_all), ("without s14", df_all[df_all.subject != "s14"])]:
        # Step 1: standardise the features (no labels involved)
        X = StandardScaler().fit_transform(df[features])
        labels = df["label"].values
        diagnosis = df["label"].map({0: "healthy", 1: "schizophrenia"}).values
        print(f"\n=== {table_name}: {len(df)} subjects ===")
        print(f"  {'method':18s} k   ARI     NMI    purity  silhouette  permutation p")

        for method, make in METHODS.items():
            for k in K_VALUES:
                # Step 2: cluster without labels
                clusters = make(k).fit_predict(X)
                # Step 3: compare the clusters with the diagnosis
                row = {"table": table_name, "method": method, "k": k,
                       "ARI": adjusted_rand_score(labels, clusters),
                       "NMI": normalized_mutual_info_score(labels, clusters),
                       "purity": purity(labels, clusters),
                       "silhouette": silhouette_score(X, clusters),
                       "permutation_p": permutation_p(labels, clusters, rng),
                       "cluster_sizes": sorted(np.bincount(clusters).tolist(), reverse=True)}
                rows.append(row)
                print(f"  {method:18s} {k}  {row['ARI']:6.3f}  {row['NMI']:6.3f}  {row['purity']:6.3f}"
                      f"   {row['silhouette']:6.3f}      {row['permutation_p']:.3f}")
                # Step 4: keep the contingency table
                ct = pd.crosstab(pd.Series(clusters, name="cluster"), pd.Series(diagnosis, name="diagnosis"))
                ct.insert(0, "k", k), ct.insert(0, "method", method), ct.insert(0, "table", table_name)
                tables.append(ct.reset_index())

        # Step 5: picture of k-means k = 2 in the first two principal components
        if table_name == "with s14":
            clusters = METHODS["k-means"](2).fit_predict(X)
            pcs = PCA(n_components=2, random_state=SEED).fit(X)
            Z = pcs.transform(X)
            plt.figure(figsize=(7, 5))
            for c, colour in [(0, "#7f8c8d"), (1, "#e67e22")]:
                for lab, marker in [(0, "o"), (1, "^")]:
                    m = (clusters == c) & (labels == lab)
                    plt.scatter(Z[m, 0], Z[m, 1], c=colour, marker=marker, edgecolors="k", s=60)
            for i, s in enumerate(df["subject"]):
                plt.annotate(s, Z[i], fontsize=7, xytext=(3, 3), textcoords="offset points")
            plt.scatter([], [], c="#7f8c8d", label="cluster 0"), plt.scatter([], [], c="#e67e22", label="cluster 1")
            plt.scatter([], [], c="white", edgecolors="k", marker="o", label="healthy")
            plt.scatter([], [], c="white", edgecolors="k", marker="^", label="schizophrenia")
            ev = pcs.explained_variance_ratio_
            plt.xlabel(f"PC1 ({ev[0]:.0%} of variance)"), plt.ylabel(f"PC2 ({ev[1]:.0%})")
            plt.title("k-means (k = 2) on 16 standardised features: colour = cluster, shape = diagnosis")
            plt.legend(fontsize=8), plt.tight_layout()
            plt.savefig(RESULTS_DIR / "item7_kmeans_pca.png", dpi=160), plt.close()

    res = pd.DataFrame(rows)
    res.round(4).to_csv(RESULTS_DIR / "item7_clustering.csv", index=False)
    pd.concat(tables).fillna(0).to_csv(RESULTS_DIR / "item7_contingency_tables.csv", index=False)
    print(f"\nSaved in {RESULTS_DIR}")
    n_tests = len(res[res.table == "with s14"])
    print(f"Note: {n_tests} tests per table. With that many, one p < 0.05 can appear by chance;")
    print(f"a Bonferroni-safe threshold is 0.05 / {n_tests} = {0.05 / n_tests:.4f}.")
