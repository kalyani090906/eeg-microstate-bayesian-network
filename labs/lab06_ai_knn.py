"""
Lab 6 A1 - the AI-generated kNN (from ML-Lab-Assignments/Lab-06/A1.py), run on the project data.
GenAI tool used: <write the tool you used in Lab 6, e.g. ChatGPT>

Changes from the original, and why:
  - The dataset path now points to outputs/features_final_16.csv (the old eeg_features.csv was
    the superseded table).
  - The original scaled ALL subjects before the train/test split, so the test subjects shaped the
    scaling (leakage). NEW min_max_scale_from(): min/max come from the training subjects only.
Functions 1-7 below are unchanged.
"""

import csv
import random


# ---------------------------------------------------------
# 1. Distance function
# ---------------------------------------------------------
def minkowski_distance(v1, v2, p):
    """
    Minkowski distance of order p between two equal-length vectors.
    p = 1 -> Manhattan distance
    p = 2 -> Euclidean distance
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of equal length")

    total = 0
    for a, b in zip(v1, v2):
        total += abs(a - b) ** p

    return total ** (1 / p)


# ---------------------------------------------------------
# 2. Sorting (ascending by distance)
# ---------------------------------------------------------
def _bubble_sort(pairs):
    pairs = pairs[:]  # copy, don't mutate original
    n = len(pairs)
    for i in range(n - 1):
        swapped = False
        for j in range(n - 1 - i):
            if pairs[j][0] > pairs[j + 1][0]:
                pairs[j], pairs[j + 1] = pairs[j + 1], pairs[j]
                swapped = True
        if not swapped:
            break
    return pairs


def _merge_sort(pairs):
    if len(pairs) <= 1:
        return pairs[:]

    mid = len(pairs) // 2
    left = _merge_sort(pairs[:mid])
    right = _merge_sort(pairs[mid:])

    merged = []
    i = j = 0
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


def _quick_sort(pairs):
    if len(pairs) <= 1:
        return pairs[:]

    pivot = pairs[len(pairs) // 2][0]
    less = [p for p in pairs if p[0] < pivot]
    equal = [p for p in pairs if p[0] == pivot]
    greater = [p for p in pairs if p[0] > pivot]

    return _quick_sort(less) + equal + _quick_sort(greater)


def sort_distance(distance_label_pairs, algorithm="quick"):
    """
    Sorts a list of (distance, label) pairs in ascending order of distance.
    algorithm: 'bubble', 'merge', or 'quick'
    """
    algorithm = algorithm.lower()
    if algorithm == "bubble":
        return _bubble_sort(distance_label_pairs)
    elif algorithm == "merge":
        return _merge_sort(distance_label_pairs)
    elif algorithm == "quick":
        return _quick_sort(distance_label_pairs)
    else:
        raise ValueError("algorithm must be 'bubble', 'merge', or 'quick'")


# ---------------------------------------------------------
# 3. Get k nearest neighbors
# ---------------------------------------------------------
def get_k_neighbors(sorted_pairs, k):
    """
    Returns the first k (distance, label) pairs from an already
    ascending-sorted list.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    return sorted_pairs[:k]


# ---------------------------------------------------------
# 4. Majority vote (with nearest-neighbor tie-break)
# ---------------------------------------------------------
def majority_vote(k_neighbors):
    """
    k_neighbors: list of (distance, label) pairs, ordered nearest first.
    Returns the majority label. If there is a tie in counts, the label
    belonging to the nearest neighbor among the tied classes wins.
    """
    counts = {}
    for dist, label in k_neighbors:
        counts[label] = counts.get(label, 0) + 1

    max_count = max(counts.values())
    tied_labels = [label for label, c in counts.items() if c == max_count]

    if len(tied_labels) == 1:
        return tied_labels[0]

    # Tie-break: pick the label whose closest occurrence appears first
    # in k_neighbors (i.e. smallest distance).
    for dist, label in k_neighbors:
        if label in tied_labels:
            return label


# ---------------------------------------------------------
# 5. Weighted vote (weight = 1 / distance)
# ---------------------------------------------------------
def weighted_vote(k_neighbors):
    """
    k_neighbors: list of (distance, label) pairs.
    Each neighbor votes with weight 1/distance (distance 0 -> infinite
    weight, i.e. exact match wins outright).
    Returns the label with the highest total weight.
    """
    weights = {}
    for dist, label in k_neighbors:
        if dist == 0:
            return label  # exact match, decisive
        w = 1 / dist
        weights[label] = weights.get(label, 0) + w

    best_label = None
    best_weight = -1
    for label, w in weights.items():
        if w > best_weight:
            best_weight = w
            best_label = label

    return best_label


# ---------------------------------------------------------
# 6. MyKNN class
# ---------------------------------------------------------
class MyKNN:
    def __init__(self, k=3, p=2, algorithm="quick", weighted=False):
        """
        k         : number of neighbors
        p         : order for Minkowski distance
        algorithm : sorting algorithm ('bubble', 'merge', 'quick')
        weighted  : if True, use weighted_vote instead of majority_vote
        """
        self.k = k
        self.p = p
        self.algorithm = algorithm
        self.weighted = weighted
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        """
        X: list of feature vectors (list of lists)
        y: list of labels
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self.X_train = X
        self.y_train = y
        return self

    def _predict_one(self, x):
        distance_label_pairs = []
        for xi, yi in zip(self.X_train, self.y_train):
            d = minkowski_distance(x, xi, self.p)
            distance_label_pairs.append((d, yi))

        sorted_pairs = sort_distance(distance_label_pairs, self.algorithm)
        k_neighbors = get_k_neighbors(sorted_pairs, self.k)

        if self.weighted:
            return weighted_vote(k_neighbors)
        else:
            return majority_vote(k_neighbors)

    def predict(self, X):
        """
        X: list of feature vectors to predict labels for.
        Returns a list of predicted labels.
        """
        return [self._predict_one(x) for x in X]

    def score(self, X, y):
        """
        Returns accuracy (fraction of correct predictions) on X, y.
        """
        predictions = self.predict(X)
        correct = 0
        for pred, actual in zip(predictions, y):
            if pred == actual:
                correct += 1
        return correct / len(y)


# ---------------------------------------------------------
# 7. Dataset loading and preparation
# ---------------------------------------------------------
def load_eeg_dataset(filename):
    """
    Loads the EEG feature CSV using only Python's csv module.

    The dataset contains:
        subject -> identifier, excluded from X
        label   -> target, kept as y
        24 remaining columns -> numerical EEG features
    """
    with open(filename, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError("Dataset is empty")

    feature_names = [
        name for name in reader.fieldnames
        if name not in ["subject", "label"]
    ]

    X = []
    y = []
    subjects = []

    for row in rows:
        subjects.append(row["subject"])
        y.append(int(row["label"]))
        X.append([float(row[name]) for name in feature_names])

    return subjects, X, y, feature_names


def min_max_scale(X):
    """
    Min-Max normalizes every feature to [0, 1].

    This is useful for KNN because KNN uses distances and
    features with larger numerical scales could otherwise dominate.
    """
    if not X:
        return []

    n_features = len(X[0])
    mins = [min(row[j] for row in X) for j in range(n_features)]
    maxs = [max(row[j] for row in X) for j in range(n_features)]

    scaled = []
    for row in X:
        new_row = []
        for j, value in enumerate(row):
            if maxs[j] == mins[j]:
                new_row.append(0.0)
            else:
                new_row.append(
                    (value - mins[j]) / (maxs[j] - mins[j])
                )
        scaled.append(new_row)

    return scaled


def stratified_train_test_split(subjects, X, y, test_size=0.30, seed=42):
    """
    Simple stratified 70/30 split using only Python.

    The split is done separately for each class so that both
    healthy and schizophrenia subjects are represented in train/test.
    """
    random.seed(seed)

    class_indices = {}
    for i, label in enumerate(y):
        class_indices.setdefault(label, []).append(i)

    train_indices = []
    test_indices = []

    for label, indices in class_indices.items():
        indices = indices[:]
        random.shuffle(indices)

        n_test = round(len(indices) * test_size)

        # Ensure at least one test sample for each class.
        n_test = max(1, n_test)

        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])

    random.shuffle(train_indices)
    random.shuffle(test_indices)

    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    subjects_train = [subjects[i] for i in train_indices]
    subjects_test = [subjects[i] for i in test_indices]

    return (
        subjects_train, X_train, y_train,
        subjects_test, X_test, y_test
    )



def min_max_scale_from(X, reference):
    """
    NEW: Min-Max scale X using the min/max of `reference` (the training subjects),
    so the test subjects never influence the scaling.
    """
    n_features = len(reference[0])
    mins = [min(row[j] for row in reference) for j in range(n_features)]
    maxs = [max(row[j] for row in reference) for j in range(n_features)]
    return [[(v - mins[j]) / (maxs[j] - mins[j]) if maxs[j] != mins[j] else 0.0
             for j, v in enumerate(row)] for row in X]


# ---------------------------------------------------------
# 8. Run on the project dataset
# ---------------------------------------------------------
if __name__ == "__main__":
    from lab_common import SEED, TABLES

    subjects, X, y, feature_names = load_eeg_dataset(TABLES["with s14"])
    print("Dataset shape:")
    print("  Samples :", len(X))
    print("  Features:", len(feature_names))
    print("  Classes :", sorted(set(y)))
    print()

    # Split first, then scale with the training subjects only
    (subjects_train, X_train, y_train,
     subjects_test, X_test, y_test) = stratified_train_test_split(
        subjects, X, y, test_size=0.30, seed=SEED)
    X_train, X_test = min_max_scale_from(X_train, X_train), min_max_scale_from(X_test, X_train)
    print("Train/Test split:")
    print("  Training samples:", len(X_train))
    print("  Test samples    :", len(X_test), subjects_test)
    print()

    for weighted in (False, True):
        model = MyKNN(k=3, p=2, algorithm="merge", weighted=weighted).fit(X_train, y_train)
        print("Weighted custom KNN" if weighted else "Custom KNN")
        print("  Predictions:", model.predict(X_test))
        print("  Actual     :", y_test)
        print("  Accuracy   :", round(model.score(X_test, y_test), 4))
        print()
