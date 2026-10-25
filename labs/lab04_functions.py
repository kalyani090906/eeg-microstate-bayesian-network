"""
Lab 4 - the AI-tool versions of the Lab 3 functions (from ML-Lab-Assignments/Lab-04).
GenAI tool used for these functions: <write the tool you used in Lab 4, e.g. ChatGPT>

Changes from the original, and why:
  - A8-A9: the original compared our std (divides by n) with pandas .std() (divides by n-1),
    so "Standard Deviation same: False" was a definition mismatch, not a bug in the code.
    The comparison script now uses numpy std with ddof=0 (same definition).
  - k-means is unchanged apart from returning the labels of the LAST centroids
    (the original returned labels from one iteration earlier).
Nothing here reads data.
"""

import random

import numpy as np
import pandas as pd


# ---------------- A2: encoding ----------------
def label_encode(df, col):
    df = df.copy()
    df[col], _ = pd.factorize(df[col])
    return df


def one_hot_encode(df, col):
    df = df.copy()
    dummies = pd.get_dummies(df[col], prefix=col)
    return df.drop(columns=[col]).join(dummies)


# ---------------- A4: Minkowski distance ----------------
def minkowski_dis(vec1, vec2, p):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length")
    if p < 1:
        raise ValueError("p must be >= 1")
    total = sum(abs(a - b) ** p for a, b in zip(vec1, vec2))
    return total ** (1 / p)


# ---------------- A7: dot product and norm ----------------
def dot_prod(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must be of the same length")
    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def vec_norm(a):
    total = 0
    for i in range(len(a)):
        total += a[i] ** 2
    return total ** 0.5


# ---------------- A8: statistics ----------------
def mean(data):
    return sum(data) / len(data)


def variance(data):
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)


def std_dev(data):
    return variance(data) ** 0.5


def matrix_stats(data_2d):
    # Mean, variance and std of every column of a list-of-rows matrix
    n_rows, n_cols = len(data_2d), len(data_2d[0])
    means, variances, std_devs = [], [], []
    for col in range(n_cols):
        column_values = [data_2d[row][col] for row in range(n_rows)]
        means.append(mean(column_values))
        variances.append(variance(column_values))
        std_devs.append(std_dev(column_values))
    return means, variances, std_devs


# ---------------- A11: k-means ----------------
def initialize_centroids(data, k, seed=None):
    if seed is not None:
        random.seed(seed)
    return [list(c) for c in random.sample(data, k)]


def euclidean_distance(a, b):
    total = 0
    for i in range(len(a)):
        total += (a[i] - b[i]) ** 2
    return total ** 0.5


def assign_clusters(data, centroids):
    labels = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        labels.append(distances.index(min(distances)))
    return labels


def update_centroids(data, labels, k):
    n_features = len(data[0])
    sums = [[0] * n_features for _ in range(k)]
    counts = [0] * k
    for point, label in zip(data, labels):
        counts[label] += 1
        for j in range(n_features):
            sums[label][j] += point[j]
    return [None if counts[i] == 0 else [sums[i][j] / counts[i] for j in range(n_features)]
            for i in range(k)]


def kmeans(data, k, max_iters=100, tol=0.0001, seed=None):
    centroids = initialize_centroids(data, k, seed)
    for _ in range(max_iters):
        labels = assign_clusters(data, centroids)                 # assign to nearest centroid
        new_centroids = update_centroids(data, labels, k)          # recompute centroids
        for i in range(k):                                         # empty cluster keeps old one
            if new_centroids[i] is None:
                new_centroids[i] = centroids[i]
        shift = sum(euclidean_distance(centroids[i], new_centroids[i]) for i in range(k))
        centroids = new_centroids
        if shift < tol:                                            # stop when centroids settle
            break
    return centroids, assign_clusters(data, centroids)
