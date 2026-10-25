"""
Lab 3 - your own functions (from ML-Lab-Assignments/Lab-03), collected in one module so the
project-data script (lab03_a11_project.py) and the Lab 4 tests can import them.

Changes from the original, and why:
  - k-means (A11): an empty cluster used to crash (mean of an empty list); it now keeps its old
    centroid. A max_iters limit stops endless loops. K is a parameter instead of input().
  - k-means returns the cluster number of every point, so clusters can be compared with labels.
Nothing here reads data; no other logic was changed.
"""

import math

import pandas as pd


# ---------------- A2: encoding ----------------
def label_encode(column):
    # Give every distinct value a number 0, 1, 2, ... in order of first appearance
    mapping = {}
    for i, value in enumerate(column.unique()):
        mapping[value] = i
    return column.map(mapping)


def one_hot_encode(column):
    # One 0/1 column per distinct value
    encoded_df = pd.DataFrame()
    for value in column.unique():
        encoded_df[value] = (column == value).astype(int)
    return encoded_df


# ---------------- A4: Minkowski distance ----------------
def minkowski_distance(v1, v2, p):
    # p = 1 Manhattan, p = 2 Euclidean, otherwise general Minkowski
    distance = 0
    for i in range(len(v1)):
        distance += abs(v1[i] - v2[i]) ** p
    distance = distance ** (1 / p)
    if p == 1:
        dis_type = "Manhattan"
    elif p == 2:
        dis_type = "Euclidean"
    else:
        dis_type = "Minkowski"
    return distance, dis_type


# ---------------- A7: dot product and norm ----------------
def dot_product(vec1, vec2, length):
    vec1, vec2 = list(vec1), list(vec2)
    total = 0
    for i in range(length):
        total += vec1[i] * vec2[i]
    return total


def euclidean_norm(vec, length):
    vec = list(vec)
    total = 0
    for i in range(length):
        total += vec[i] * vec[i]
    return math.sqrt(total)


# ---------------- A8: mean, variance, standard deviation ----------------
def mean(values):
    values = list(values)
    total = 0
    for i in range(len(values)):
        total = total + values[i]
    return total / len(values)


def variance(values):
    # Population variance (divide by n), same as numpy.var / numpy.std default
    values = list(values)
    avg = mean(values)
    var = 0
    for i in range(len(values)):
        var = var + ((values[i] - avg) ** 2)
    return var / len(values)


def standard_deviation(values):
    return math.sqrt(variance(values))


def self_statistics(df):
    # Mean, variance and standard deviation of every column, as a table
    rows = []
    for col in df.columns:
        rows.append({"feature": col, "mean": mean(df[col]),
                     "variance": variance(df[col]), "std": standard_deviation(df[col])})
    return pd.DataFrame(rows).set_index("feature")


# ---------------- A11: k-means ----------------
def point_distance(a, b):
    # Euclidean distance between two points
    total = 0
    for i in range(len(a)):
        total = total + ((a[i] - b[i]) ** 2)
    return math.sqrt(total)


def nearest_centroid(point, centroids):
    closest = point_distance(point, centroids[0])
    cluster = 0
    for i in range(1, len(centroids)):
        dist = point_distance(point, centroids[i])
        if dist < closest:
            closest = dist
            cluster = i
    return cluster


def cluster_mean(cluster):
    centroid = []
    for i in range(len(cluster[0])):
        total = 0
        for j in range(len(cluster)):
            total = total + cluster[j][i]
        centroid.append(float(total / len(cluster)))
    return centroid


def kmeans(data, k, max_iters=100):
    """
    data: list of points (list of lists). Returns (centroids, assignments) where
    assignments[i] is the cluster number of data[i].
    Algorithm 8.1 (Tan, Steinbach & Kumar): start with the first k points as centroids,
    assign each point to its nearest centroid, recompute centroids, stop when they do not change.
    """
    centroids = [list(map(float, data[i])) for i in range(k)]
    for _ in range(max_iters):
        # Step 1: assign every point to its nearest centroid
        assignments = [nearest_centroid(point, centroids) for point in data]
        # Step 2: recompute each centroid; an empty cluster keeps its old centroid
        new_centroids = []
        for c in range(k):
            members = [data[i] for i in range(len(data)) if assignments[i] == c]
            new_centroids.append(cluster_mean(members) if members else centroids[c])
        # Step 3: stop when the centroids no longer change
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return centroids, assignments
