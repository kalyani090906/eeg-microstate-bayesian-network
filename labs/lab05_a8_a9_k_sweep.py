"""
Lab 5 A8-A9 on the project data: accuracy for k = 1..15 for
sklearn kNN, your own kNN, and your own weighted kNN.

Two views:
  (1) the 70/30 split the lab asks for, with TRAIN and TEST accuracy (for over/under-fitting)
  (2) leave-one-subject-out accuracy (all 28 subjects tested once; far less noisy)
Scaling is always learned from the training subjects only.

Run:  eeg_env\\Scripts\\python.exe labs\\lab05_a8_a9_k_sweep.py
"""

from lab_common import FEATURE_LIST, LAB_OUT, SEED, TABLES, load_feature_names, load_table
from lab05_knn import MyKNN, MyWeightedKNN, min_max_apply, min_max_fit

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

K_VALUES = range(1, 16)
MODELS = {
    "sklearn kNN": lambda k: KNeighborsClassifier(n_neighbors=k),
    "own kNN": lambda k: MyKNN(k=k, p=2, sort_algorithm="merge"),
    "own weighted kNN": lambda k: MyWeightedKNN(k=k, p=2, sort_algorithm="merge"),
}


def scaled(X_train, X_test):
    # min-max learned on the training part, applied to both
    mins, maxs = min_max_fit(X_train)
    return min_max_apply(X_train, mins, maxs), min_max_apply(X_test, mins, maxs)


def loso_accuracy(make_model, X, y):
    # leave one subject out: train on the rest, predict the one left out
    correct = 0
    for i in range(len(X)):
        X_tr, X_te = scaled(X[:i] + X[i + 1:], [X[i]])
        model = make_model().fit(X_tr, y[:i] + y[i + 1:])
        correct += int(list(model.predict(X_te))[0] == y[i])
    return correct / len(X)


if __name__ == "__main__":
    features = load_feature_names(FEATURE_LIST)
    rows = []
    for table_name, path in TABLES.items():
        df = load_table(path)
        X, y = df[features].values.tolist(), df["label"].tolist()

        # Step 1: the lab's 70/30 split (scaled with the training part only)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
        X_tr, X_te = scaled(X_tr, X_te)

        for k in K_VALUES:
            for name, make in MODELS.items():
                model = make(k).fit(X_tr, y_tr)
                rows.append({
                    "table": table_name, "k": k, "model": name,
                    "train_acc": model.score(X_tr, y_tr),               # step 2: train accuracy
                    "test_acc": model.score(X_te, y_te),                # step 3: test accuracy
                    "loso_acc": loso_accuracy(lambda: make(k), X, y),   # step 4: leave-one-out
                })
    res = pd.DataFrame(rows)
    res.round(3).to_csv(LAB_OUT / "lab05_a8_a9_k_sweep.csv", index=False)

    # Step 5: print and plot the main table
    main = res[res.table == "with s14"]
    print(main.pivot(index="k", columns="model", values=["test_acc", "loso_acc"]).round(3).to_string())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (name, part) in zip(axes, main.groupby("model")):
        ax.plot(part.k, part.train_acc, marker="o", label="train (70%)")
        ax.plot(part.k, part.test_acc, marker="s", label="test (30%, 9 subjects)")
        ax.plot(part.k, part.loso_acc, marker="^", label="leave-one-subject-out")
        ax.axhline(0.5, color="grey", linestyle=":")
        ax.set_title(name), ax.set_xlabel("k"), ax.legend(fontsize=8)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle("kNN accuracy vs k (28 subjects, 16 features, min-max scaled on training data)")
    fig.tight_layout()
    fig.savefig(LAB_OUT / "lab05_a8_a9_k_sweep.png", dpi=160)
    print(f"\nSaved: {LAB_OUT / 'lab05_a8_a9_k_sweep.csv'} and .png")
    print("k = 1 always gives train accuracy 1.0 (each point is its own neighbour): overfitting.")
