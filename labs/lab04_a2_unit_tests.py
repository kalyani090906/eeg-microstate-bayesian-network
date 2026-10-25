"""
Lab 4 A2: unit tests for the Lab 3 (own) and Lab 4 (AI-tool) functions.
GenAI tool used to generate these test cases: Claude (Anthropic).

Run:  eeg_env\\Scripts\\python.exe labs\\lab04_a2_unit_tests.py
"""

import math
import unittest

import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski

import lab03_functions as own
import lab04_functions as ai

A = [3, 7, 2, 9, 5]
B = [8, 1, 6, 4, 3]
DATA = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]          # mean 5, population std 2
TWO_BLOBS = [[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]]


class TestEncoding(unittest.TestCase):
    def test_own_label_encode(self):
        out = own.label_encode(pd.Series(["a", "b", "a", "c"]))
        self.assertEqual(out.tolist(), [0, 1, 0, 2])

    def test_own_one_hot(self):
        out = own.one_hot_encode(pd.Series(["a", "b", "a"]))
        self.assertEqual(out.shape, (3, 2))
        self.assertEqual(out.sum(axis=1).tolist(), [1, 1, 1])     # exactly one 1 per row

    def test_ai_label_encode(self):
        out = ai.label_encode(pd.DataFrame({"x": ["a", "b", "a"]}), "x")
        self.assertEqual(out["x"].tolist(), [0, 1, 0])

    def test_ai_one_hot(self):
        out = ai.one_hot_encode(pd.DataFrame({"x": ["a", "b", "a"]}), "x")
        self.assertEqual(sorted(out.columns), ["x_a", "x_b"])


class TestDistance(unittest.TestCase):
    def test_own_matches_scipy(self):
        for p in range(1, 11):
            self.assertAlmostEqual(own.minkowski_distance(A, B, p)[0], minkowski(A, B, p))

    def test_own_names(self):
        self.assertEqual(own.minkowski_distance(A, B, 1)[1], "Manhattan")
        self.assertEqual(own.minkowski_distance(A, B, 2)[1], "Euclidean")

    def test_ai_matches_scipy(self):
        for p in range(1, 11):
            self.assertAlmostEqual(ai.minkowski_dis(A, B, p), minkowski(A, B, p))

    def test_known_values(self):
        self.assertAlmostEqual(ai.minkowski_dis([0, 0], [3, 4], 2), 5.0)
        self.assertAlmostEqual(own.minkowski_distance([0, 0], [3, 4], 1)[0], 7.0)

    def test_ai_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            ai.minkowski_dis([1, 2], [1], 2)
        with self.assertRaises(ValueError):
            ai.minkowski_dis([1, 2], [1, 2], 0.5)


class TestVectors(unittest.TestCase):
    def test_dot(self):
        self.assertEqual(own.dot_product(A, B, len(A)), np.dot(A, B))
        self.assertEqual(ai.dot_prod(A, B), np.dot(A, B))

    def test_norm(self):
        self.assertAlmostEqual(own.euclidean_norm(A, len(A)), np.linalg.norm(A))
        self.assertAlmostEqual(ai.vec_norm(A), np.linalg.norm(A))


class TestStatistics(unittest.TestCase):
    def test_own(self):
        self.assertAlmostEqual(own.mean(DATA), 5.0)
        self.assertAlmostEqual(own.variance(DATA), 4.0)
        self.assertAlmostEqual(own.standard_deviation(DATA), 2.0)

    def test_ai(self):
        self.assertAlmostEqual(ai.mean(DATA), 5.0)
        self.assertAlmostEqual(ai.std_dev(DATA), np.std(DATA))

    def test_ai_matrix_stats(self):
        matrix = [[1, 10], [3, 30], [5, 50]]
        means, variances, stds = ai.matrix_stats(matrix)
        self.assertTrue(np.allclose(means, [3, 30]))
        self.assertTrue(np.allclose(stds, np.std(matrix, axis=0)))


class TestKMeans(unittest.TestCase):
    def _two_groups(self, labels):
        # the first three points share one cluster, the last three the other
        return len(set(labels[:3])) == 1 and len(set(labels[3:])) == 1 and labels[0] != labels[3]

    def test_own_finds_two_blobs(self):
        _, labels = own.kmeans(TWO_BLOBS, 2)
        self.assertTrue(self._two_groups(labels))

    def test_ai_finds_two_blobs(self):
        _, labels = ai.kmeans(TWO_BLOBS, 2, seed=42)
        self.assertTrue(self._two_groups(labels))

    def test_own_empty_cluster_does_not_crash(self):
        # identical points: one cluster stays empty; must not raise
        centroids, labels = own.kmeans([[1, 1]] * 4, 2)
        self.assertEqual(len(centroids), 2)

    def test_centroid_is_mean(self):
        centroids, labels = own.kmeans(TWO_BLOBS, 2)
        for c in range(2):
            members = [p for p, l in zip(TWO_BLOBS, labels) if l == c]
            self.assertTrue(np.allclose(centroids[c], np.mean(members, axis=0)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
