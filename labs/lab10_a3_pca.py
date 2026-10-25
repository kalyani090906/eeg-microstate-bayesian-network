"""
Lab 10 A3 (also part of project item 4): PCA keeping 99% of the variance, then the classifiers
on the PCA features, compared with the same classifiers on the original 16 features.

PCA is fitted INSIDE each leave-one-subject-out fold (training subjects only).
A cumulative-variance plot on all subjects is also made; PCA uses no labels, so that plot is
description only.

Run:  eeg_env\\Scripts\\python.exe labs\\lab10_a3_pca.py
"""

from lab_common import FEATURE_LIST, LAB_OUT, SEED, TABLES, load_feature_names, load_table, run_models, save

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

VARIANCE = 0.99

# Step 1: the classifiers to compare
CLASSIFIERS = {
    "Logistic regression": lambda: LogisticRegression(max_iter=5000, random_state=SEED),
    "Random forest": lambda: RandomForestClassifier(n_estimators=300, random_state=SEED),
    "AdaBoost": lambda: AdaBoostClassifier(n_estimators=100, random_state=SEED),
    "Naive Bayes": lambda: GaussianNB(),
    "kNN (k=5)": lambda: KNeighborsClassifier(n_neighbors=5),
}


def with_pca(make_clf):
    # scale -> PCA keeping 99% variance -> classifier, all refitted inside each fold
    return lambda: Pipeline([("scale", StandardScaler()),
                             ("pca", PCA(n_components=VARIANCE, random_state=SEED)),
                             ("clf", make_clf())])


def without_pca(make_clf):
    return lambda: Pipeline([("scale", StandardScaler()), ("clf", make_clf())])


if __name__ == "__main__":
    features = load_feature_names(FEATURE_LIST)

    # Step 2: how many components are needed for 99%? (all subjects, description only)
    df = load_table(TABLES["with s14"])
    X = StandardScaler().fit_transform(df[features].values)
    cumulative = np.cumsum(PCA(random_state=SEED).fit(X).explained_variance_ratio_)
    n_99 = int(np.searchsorted(cumulative, VARIANCE) + 1)
    print(f"Components for {VARIANCE:.0%} variance (all 28 subjects): {n_99} of {len(features)}")
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(cumulative) + 1), cumulative, marker="o")
    plt.axhline(VARIANCE, color="grey", linestyle="--")
    plt.axvline(n_99, color="grey", linestyle=":")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title(f"PCA on the 16 standardised features: {n_99} components reach {VARIANCE:.0%}")
    plt.tight_layout()
    plt.savefig(LAB_OUT / "lab10_a3_pca_variance.png", dpi=160)
    plt.close()

    # Step 3: how many components does each training fold keep? (should be close to n_99)
    counts = []
    for subject in df["subject"]:
        train = df[df["subject"] != subject][features].values
        pca = PCA(n_components=VARIANCE, random_state=SEED).fit(StandardScaler().fit_transform(train))
        counts.append(pca.n_components_)
    print(f"Components kept per training fold: min {min(counts)}, max {max(counts)}")

    # Step 4: same classifiers with and without PCA (pipelines already scale, so scale=False)
    models = {}
    for name, make_clf in CLASSIFIERS.items():
        models[f"{name} | original 16"] = without_pca(make_clf)
        models[f"{name} | PCA 99%"] = with_pca(make_clf)
    results = run_models(models, scale=False)
    save(results, "lab10_a3_pca.csv")
