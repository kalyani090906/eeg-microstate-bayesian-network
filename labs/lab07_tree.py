"""
Lab 7 A1-A5 - your own entropy, Gini, binning, root-node selection and decision tree
(from ML-Lab-Assignments/Lab-07/A1-5.py), run on the project data.

Changes from the original, and why:
  - The tree could be built but not USED: there was no predict function, so it could not be
    scored. Each node now stores its bin edges, and predict() walks the tree.
  - Binning is split into bin_edges() (learn the cut points) and assign_bins() (apply them),
    so a new subject can be put into the same bins. bin_feature() keeps its old signature and
    default parameters (A4).
  - The own tree is scored with leave-one-subject-out next to scikit-learn's tree.
  - The data-loading code moved under  if __name__ == "__main__"  so other scripts can import this.

Run:  eeg_env\\Scripts\\python.exe labs\\lab07_tree.py
"""

import numpy as np
import pandas as pd


# A1: entropy of a set of labels
def calculate_entropy(labels):
    probabilities = pd.Series(labels).value_counts() / len(labels)
    entropy = 0
    for p in probabilities:
        entropy += -p * np.log2(p)
    return entropy


# A2: Gini index of a set of labels
def calculate_gini(labels):
    probabilities = pd.Series(labels).value_counts() / len(labels)
    gini = 1
    for p in probabilities:
        gini -= p ** 2
    return gini


# A4: binning (equal width or equal frequency; defaults used when no parameters are passed)
def bin_edges(values, bin_type="equal_width", n_bins=4):
    # Inner cut points that split the values into n_bins bins
    if bin_type == "equal_width":
        lo, hi = min(values), max(values)
        if hi == lo:
            return []
        width = (hi - lo) / n_bins
        return [lo + width * i for i in range(1, n_bins)]
    if bin_type == "equal_frequency":
        cuts = np.quantile(values, [i / n_bins for i in range(1, n_bins)])
        return sorted(set(float(c) for c in cuts))
    raise ValueError("bin_type must be 'equal_width' or 'equal_frequency'")


def assign_bins(values, edges):
    # Bin number of each value: how many cut points it is at or above
    return [sum(v >= e for e in edges) for v in values]


def bin_feature(values, bin_type="equal_width", n_bins=4):
    return assign_bins(values, bin_edges(values, bin_type, n_bins))


# A3: pick the feature with the highest information gain
def information_gain(values, y, bin_type="equal_width", n_bins=4):
    bins = bin_feature(values, bin_type, n_bins)
    weighted_entropy = 0
    for b in set(bins):
        labels_in_bin = [y[i] for i in range(len(bins)) if bins[i] == b]
        weighted_entropy += len(labels_in_bin) / len(y) * calculate_entropy(labels_in_bin)
    return calculate_entropy(y) - weighted_entropy


def select_root_feature(X, y, bin_type="equal_width", n_bins=4):
    best_feature, best_gain = None, -1
    for feature in X.columns:
        gain = information_gain(X[feature].tolist(), y, bin_type, n_bins)
        if gain > best_gain:
            best_feature, best_gain = feature, gain
    return best_feature, best_gain


# A5: build the tree
def majority_class(labels):
    return pd.Series(labels).value_counts().idxmax()


def build_tree(X, y, bin_type="equal_width", n_bins=4, depth=0, max_depth=3):
    y = list(y)
    # Stop: pure node, too deep, or too few subjects -> leaf with the majority class
    if len(set(y)) == 1 or depth >= max_depth or len(y) < 2:
        return {"leaf": True, "prediction": majority_class(y)}
    best_feature, best_gain = select_root_feature(X, y, bin_type, n_bins)
    if best_gain <= 0:
        return {"leaf": True, "prediction": majority_class(y)}

    # Split the subjects by the bins of the best feature and grow one branch per bin
    edges = bin_edges(X[best_feature].tolist(), bin_type, n_bins)
    bins = assign_bins(X[best_feature].tolist(), edges)
    branches = {}
    for b in sorted(set(bins)):
        rows = [i for i in range(len(bins)) if bins[i] == b]
        branches[b] = build_tree(X.iloc[rows], [y[i] for i in rows], bin_type, n_bins,
                                 depth + 1, max_depth)
    return {"leaf": False, "feature": best_feature, "edges": edges, "branches": branches,
            "majority": majority_class(y)}


def predict_one(tree, row):
    # Walk down the tree; a bin never seen in training falls back to the node's majority class
    while not tree["leaf"]:
        b = assign_bins([row[tree["feature"]]], tree["edges"])[0]
        if b not in tree["branches"]:
            return tree["majority"]
        tree = tree["branches"][b]
    return tree["prediction"]


def predict(tree, X):
    return [predict_one(tree, X.iloc[i]) for i in range(len(X))]


def print_tree(tree, indent=""):
    if tree["leaf"]:
        print(f"{indent}-> predict {'schizophrenia' if tree['prediction'] == 1 else 'healthy'}")
        return
    for b, child in tree["branches"].items():
        print(f"{indent}{tree['feature']} in bin {b}:")
        print_tree(child, indent + "    ")


if __name__ == "__main__":
    from lab_common import FEATURE_LIST, LAB_OUT, SEED, TABLES, load_feature_names, load_table
    from sklearn.tree import DecisionTreeClassifier

    df = load_table(TABLES["with s14"])
    features = load_feature_names(FEATURE_LIST)
    X, y = df[features], df["label"].tolist()

    # A1-A2
    print(f"A1. Entropy of the labels: {calculate_entropy(y):.4f}  (1.0 = perfectly balanced classes)")
    print(f"A2. Gini index of the labels: {calculate_gini(y):.4f}  (0.5 = perfectly balanced)")

    # A3-A4: root node with both binning types
    for bin_type in ["equal_width", "equal_frequency"]:
        f, g = select_root_feature(X, y, bin_type=bin_type)
        print(f"A3-A4. Root feature ({bin_type}, 4 bins): {f}  (information gain {g:.4f})")

    # A5: the tree on all subjects
    tree = build_tree(X, y, max_depth=3)
    print("\nA5. Own decision tree (all 28 subjects, depth 3):")
    print_tree(tree)

    # Score: own tree vs sklearn tree, leave-one-subject-out, both tables
    rows = []
    for table_name, path in TABLES.items():
        d = load_table(path)
        Xd, yd = d[features], d["label"].tolist()
        own_pred, sk_pred = [], []
        for i in range(len(d)):
            train = [j for j in range(len(d)) if j != i]
            t = build_tree(Xd.iloc[train], [yd[j] for j in train], max_depth=3)
            own_pred.append(predict(t, Xd.iloc[[i]])[0])
            sk = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=SEED)
            sk_pred.append(sk.fit(Xd.iloc[train], [yd[j] for j in train]).predict(Xd.iloc[[i]])[0])
        for name, pred in [("own tree", own_pred), ("sklearn tree", sk_pred)]:
            rows.append({"table": table_name, "model": name,
                         "loso_accuracy": float(np.mean(np.array(pred) == np.array(yd)))})
    res = pd.DataFrame(rows)
    print("\nLeave-one-subject-out accuracy (depth 3):")
    print(res.round(3).to_string(index=False))
    print("(lab07_a8 can differ by a subject or two: a 50/50 leaf counts as 'healthy' here but as")
    print(" 'patient' in evaluate(), which thresholds the probability at >= 0.5.)")
    res.round(4).to_csv(LAB_OUT / "lab07_a5_tree_loso.csv", index=False)
    print(f"\nSaved: {LAB_OUT / 'lab07_a5_tree_loso.csv'}")
