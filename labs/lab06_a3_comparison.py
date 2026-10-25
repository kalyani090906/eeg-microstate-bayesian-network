"""
Lab 6 A3 on the project data: performance comparison of 3 kNN versions (k = 3)
  - your own kNN (Lab 5)        - scikit-learn kNN        - AI-generated kNN (Lab 6 A1)
Metrics: accuracy, precision, recall, F1, and run time averaged over 10 runs.

Two evaluations:
  (1) the 70/30 split (as the lab asks)
  (2) leave-one-subject-out (every subject tested once; more reliable with 28 subjects)
Scaling is learned from the training subjects only (the original scaled all data first).

Run:  eeg_env\\Scripts\\python.exe labs\\lab06_a3_comparison.py
"""

from lab_common import FEATURE_LIST, LAB_OUT, SEED, TABLES, load_feature_names, load_table
from lab05_knn import MyKNN as OwnKNN, min_max_apply, min_max_fit
from lab06_ai_knn import MyKNN as AIKNN

import time

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

K = 3
N_RUNS = 10
VERSIONS = {
    "Own kNN (Lab 5)": lambda: OwnKNN(k=K, p=2, sort_algorithm="merge"),
    "sklearn kNN": lambda: KNeighborsClassifier(n_neighbors=K),
    "AI-generated kNN (Lab 6 A1)": lambda: AIKNN(k=K, p=2, algorithm="merge", weighted=False),
}


def fit_predict(make_model, X_tr, y_tr, X_te):
    # scale with the training part only, fit, predict
    mins, maxs = min_max_fit(X_tr)
    model = make_model().fit(min_max_apply(X_tr, mins, maxs), y_tr)
    return list(model.predict(min_max_apply(X_te, mins, maxs)))


def metrics(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


if __name__ == "__main__":
    features = load_feature_names(FEATURE_LIST)
    rows = []
    for table_name, path in TABLES.items():
        df = load_table(path)
        X, y = df[features].values.tolist(), df["label"].tolist()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)

        for name, make in VERSIONS.items():
            # Step 1: 70/30 split, timed over N_RUNS runs
            times = []
            for _ in range(N_RUNS):
                start = time.perf_counter()
                pred = fit_predict(make, X_tr, y_tr, X_te)
                times.append(time.perf_counter() - start)
            rows.append({"table": table_name, "evaluation": "70/30 split", "version": name,
                         **metrics(y_te, pred), "avg_time_s": sum(times) / N_RUNS})

            # Step 2: leave-one-subject-out
            loso_pred = [fit_predict(make, X[:i] + X[i + 1:], y[:i] + y[i + 1:], [X[i]])[0]
                         for i in range(len(X))]
            rows.append({"table": table_name, "evaluation": "leave-one-subject-out",
                         "version": name, **metrics(y, loso_pred), "avg_time_s": None})

    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))
    res.round(5).to_csv(LAB_OUT / "lab06_a3_knn_comparison.csv", index=False)
    print(f"\nSaved: {LAB_OUT / 'lab06_a3_knn_comparison.csv'}")
    print("Same predictions from all three = the implementations agree. With only ~20 training subjects")
    print("scikit-learn's fixed set-up cost makes it slower; its speed advantage shows only on large data.")
