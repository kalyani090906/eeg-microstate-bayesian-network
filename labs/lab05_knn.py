"""
Lab 5 A1-A2 - your own kNN and weighted kNN (from ML-Lab-Assignments/Lab-05/A1_A2.py),
as a module that the other Lab 5 and Lab 6 scripts import.

Changes from the original, and why:
  - The data-loading code was at module level, so every "from A1_A2 import MyKNN" re-ran it.
    It is now only in the demo under  if __name__ == "__main__".
  - NEW min_max_fit / min_max_apply: kNN uses distances, and our features have very different
    scales (durations ~0.06 s, occurrences ~3.5 per s), so unscaled distances are dominated by a
    few features. The scaling is learned from the TRAINING subjects only (no leakage).
  - fit() returns self (so it can be chained, like scikit-learn).
  - A1_2.py (older interactive copy with input()) is not carried over; this module already has
    3 sorting algorithms as a config parameter.
Sorting is stable (equal distances keep training order), which is the tie-break for equal
distances (A1 e); majority_vote breaks vote ties by the nearest neighbour (A1 f).
"""

import pandas as pd


# a. Encoding
def encode_if_categorical(df, column):
    if df[column].dtype == object:
        mapping = {value: i for i, value in enumerate(df[column].unique())}
        return df[column].map(mapping)
    return df[column]


# b. Data imputation
def impute_column(column, method="mean"):
    if method == "mean":
        fill_value = column.mean()
    elif method == "median":
        fill_value = column.median()
    elif method == "mode":
        fill_value = column.mode()[0]
    else:
        raise ValueError("method must be 'mean', 'median', or 'mode'")
    return column.copy().fillna(fill_value)


# NEW: min-max scaling learned on the training rows only
def min_max_fit(X_train):
    n = len(X_train[0])
    mins = [min(row[j] for row in X_train) for j in range(n)]
    maxs = [max(row[j] for row in X_train) for j in range(n)]
    return mins, maxs


def min_max_apply(X, mins, maxs):
    return [[(v - mins[j]) / (maxs[j] - mins[j]) if maxs[j] != mins[j] else 0.0
             for j, v in enumerate(row)] for row in X]


# c. Distance calculation
def minkowski_distance(v1, v2, p):
    distance = 0
    for i in range(len(v1)):
        distance += abs(v1[i] - v2[i]) ** p
    return distance ** (1 / p)


# d. Sorting algorithms (bubble, merge, quick)
def bubble_sort(items):
    items = items.copy()
    n = len(items)
    for i in range(n):
        for j in range(0, n - i - 1):
            if items[j][0] > items[j + 1][0]:
                items[j], items[j + 1] = items[j + 1], items[j]
    return items


def merge_sort(items):
    if len(items) <= 1:
        return items
    mid = len(items) // 2
    left, right = merge_sort(items[:mid]), merge_sort(items[mid:])
    merged, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i][0] <= right[j][0]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def quick_sort(items):
    if len(items) <= 1:
        return items
    pivot = items[len(items) // 2]
    left = [x for x in items if x[0] < pivot[0]]
    middle = [x for x in items if x[0] == pivot[0]]
    right = [x for x in items if x[0] > pivot[0]]
    return quick_sort(left) + middle + quick_sort(right)


SORT_FUNCTIONS = {"bubble": bubble_sort, "merge": merge_sort, "quick": quick_sort}


def sort_by_distance(distance_label_pairs, algorithm="merge"):
    if algorithm not in SORT_FUNCTIONS:
        raise ValueError(f"algorithm must be one of {list(SORT_FUNCTIONS.keys())}")
    return SORT_FUNCTIONS[algorithm](distance_label_pairs)


# e. Identify k nearest neighbours
def get_k_neighbors(sorted_distance_label_pairs, k):
    return sorted_distance_label_pairs[:k]


# f. Class evaluation and assignment (tie -> class of the nearest neighbour among the tied)
def majority_vote(k_neighbors):
    votes = {}
    for distance, label in k_neighbors:
        votes[label] = votes.get(label, 0) + 1
    max_votes = max(votes.values())
    tied_classes = [label for label, count in votes.items() if count == max_votes]
    if len(tied_classes) == 1:
        return tied_classes[0]
    for distance, label in k_neighbors:
        if label in tied_classes:
            return label


class MyKNN:
    def __init__(self, k=3, p=2, sort_algorithm="merge"):
        self.k = k
        self.p = p
        self.sort_algorithm = sort_algorithm
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = list(X_train)
        self.y_train = list(y_train)
        return self

    def _neighbors(self, x):
        pairs = [(minkowski_distance(x, self.X_train[i], self.p), self.y_train[i])
                 for i in range(len(self.X_train))]
        return get_k_neighbors(sort_by_distance(pairs, self.sort_algorithm), self.k)

    def _predict_one(self, x):
        return majority_vote(self._neighbors(x))

    def predict(self, X_test):
        return [self._predict_one(x) for x in X_test]

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        return sum(1 for pred, true in zip(predictions, y_test) if pred == true) / len(y_test)


# A2: weighted kNN (closer neighbours count more: weight = 1 / distance)
def weighted_vote(k_neighbors):
    weighted_votes = {}
    for distance, label in k_neighbors:
        weighted_votes[label] = weighted_votes.get(label, 0) + 1 / (distance + 1e-6)
    max_weight = max(weighted_votes.values())
    tied_classes = [label for label, w in weighted_votes.items() if w == max_weight]
    if len(tied_classes) == 1:
        return tied_classes[0]
    for distance, label in k_neighbors:
        if label in tied_classes:
            return label


class MyWeightedKNN(MyKNN):
    def _predict_one(self, x):
        return weighted_vote(self._neighbors(x))


if __name__ == "__main__":
    # Demo: load the project table and apply encoding + imputation (A1 a, b)
    from lab_common import FEATURE_LIST, TABLES, load_feature_names, load_table
    df = load_table(TABLES["with s14"])
    feature_cols = load_feature_names(FEATURE_LIST)
    for col in feature_cols:
        df[col] = impute_column(encode_if_categorical(df, col), method="mean")
    print("Number of samples :", len(df))
    print("Number of features:", len(feature_cols))
    print("Missing values    :", int(df[feature_cols].isna().sum().sum()))
    print("Classes           :", sorted(int(c) for c in df["label"].unique()))
