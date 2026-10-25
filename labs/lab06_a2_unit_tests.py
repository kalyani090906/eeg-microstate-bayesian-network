"""
A2 - Unit Tests for KNN Implementations

Tests both:
1. Previous Lab KNN implementation -> lab05_knn.py
2. AI-generated KNN implementation  -> lab06_ai_knn.py

AI tool used:
ChatGPT - used to generate the unit test cases.
"""

import unittest


# =========================================================
# Import Previous Lab Implementation
# =========================================================

from lab05_knn import (
    minkowski_distance as PreviousMinkowski,
    sort_by_distance as PreviousSort,
    get_k_neighbors as PreviousGetK,
    majority_vote as PreviousMajorityVote,
    weighted_vote as PreviousWeightedVote,
    MyKNN as PreviousKNN,
    MyWeightedKNN as PreviousWeightedKNN,
)


# =========================================================
# Import AI-generated Implementation
# =========================================================

from lab06_ai_knn import (
    minkowski_distance as AIMinkowski,
    sort_distance as AISort,
    get_k_neighbors as AIGetK,
    majority_vote as AIMajorityVote,
    weighted_vote as AIWeightedVote,
    MyKNN as AIKNN,
)


# =========================================================
# 1. Previous Lab - Minkowski Distance
# =========================================================

class TestPreviousMinkowski(unittest.TestCase):

    def test_euclidean_distance(self):
        result = PreviousMinkowski(
            [0, 0],
            [3, 4],
            2
        )

        self.assertAlmostEqual(result, 5.0)

    def test_manhattan_distance(self):
        result = PreviousMinkowski(
            [1, 1],
            [4, 5],
            1
        )

        self.assertAlmostEqual(result, 7.0)


# =========================================================
# 2. AI Version - Minkowski Distance
# =========================================================

class TestAIMinkowski(unittest.TestCase):

    def test_euclidean_distance(self):
        result = AIMinkowski(
            [0, 0],
            [3, 4],
            2
        )

        self.assertAlmostEqual(result, 5.0)

    def test_manhattan_distance(self):
        result = AIMinkowski(
            [1, 1],
            [4, 5],
            1
        )

        self.assertAlmostEqual(result, 7.0)

    def test_mismatched_vectors_raise_error(self):
        with self.assertRaises(ValueError):
            AIMinkowski(
                [1, 2],
                [1, 2, 3],
                2
            )


# =========================================================
# 3. Previous Lab - Sorting
# =========================================================

class TestPreviousSorting(unittest.TestCase):

    def setUp(self):
        self.pairs = [
            (3.0, "B"),
            (1.0, "A"),
            (2.0, "C")
        ]

        self.expected = [
            (1.0, "A"),
            (2.0, "C"),
            (3.0, "B")
        ]

    def test_merge_sort(self):
        result = PreviousSort(
            self.pairs,
            "merge"
        )

        self.assertEqual(
            result,
            self.expected
        )

    def test_quick_sort(self):
        result = PreviousSort(
            self.pairs,
            "quick"
        )

        self.assertEqual(
            result,
            self.expected
        )

    def test_bubble_sort(self):
        result = PreviousSort(
            self.pairs,
            "bubble"
        )

        self.assertEqual(
            result,
            self.expected
        )


# =========================================================
# 4. AI Version - Sorting
# =========================================================

class TestAISorting(unittest.TestCase):

    def setUp(self):
        self.pairs = [
            (3.0, "B"),
            (1.0, "A"),
            (2.0, "C")
        ]

        self.expected = [
            (1.0, "A"),
            (2.0, "C"),
            (3.0, "B")
        ]

    def test_merge_sort(self):
        result = AISort(
            self.pairs,
            "merge"
        )

        self.assertEqual(
            result,
            self.expected
        )

    def test_quick_sort(self):
        result = AISort(
            self.pairs,
            "quick"
        )

        self.assertEqual(
            result,
            self.expected
        )

    def test_bubble_sort(self):
        result = AISort(
            self.pairs,
            "bubble"
        )

        self.assertEqual(
            result,
            self.expected
        )


# =========================================================
# 5. Previous Lab - K Nearest Neighbors
# =========================================================

class TestPreviousKNeighbors(unittest.TestCase):

    def test_returns_first_k(self):

        pairs = [
            (1.0, "A"),
            (2.0, "B"),
            (3.0, "C")
        ]

        result = PreviousGetK(
            pairs,
            2
        )

        expected = [
            (1.0, "A"),
            (2.0, "B")
        ]

        self.assertEqual(
            result,
            expected
        )


# =========================================================
# 6. AI Version - K Nearest Neighbors
# =========================================================

class TestAIKNeighbors(unittest.TestCase):

    def test_returns_first_k(self):

        pairs = [
            (1.0, "A"),
            (2.0, "B"),
            (3.0, "C")
        ]

        result = AIGetK(
            pairs,
            2
        )

        expected = [
            (1.0, "A"),
            (2.0, "B")
        ]

        self.assertEqual(
            result,
            expected
        )


# =========================================================
# 7. Previous Lab - Majority Vote
# =========================================================

class TestPreviousMajorityVote(unittest.TestCase):

    def test_majority_wins(self):

        neighbors = [
            (1.0, "A"),
            (2.0, "A"),
            (3.0, "B")
        ]

        result = PreviousMajorityVote(
            neighbors
        )

        self.assertEqual(
            result,
            "A"
        )

    def test_tie_uses_nearest_neighbor(self):

        neighbors = [
            (1.0, "B"),
            (2.0, "A")
        ]

        result = PreviousMajorityVote(
            neighbors
        )

        self.assertEqual(
            result,
            "B"
        )


# =========================================================
# 8. AI Version - Majority Vote
# =========================================================

class TestAIMajorityVote(unittest.TestCase):

    def test_majority_wins(self):

        neighbors = [
            (1.0, "A"),
            (2.0, "A"),
            (3.0, "B")
        ]

        result = AIMajorityVote(
            neighbors
        )

        self.assertEqual(
            result,
            "A"
        )

    def test_tie_uses_nearest_neighbor(self):

        neighbors = [
            (1.0, "B"),
            (2.0, "A")
        ]

        result = AIMajorityVote(
            neighbors
        )

        self.assertEqual(
            result,
            "B"
        )


# =========================================================
# 9. Previous Lab - Weighted Vote
# =========================================================

class TestPreviousWeightedVote(unittest.TestCase):

    def test_closer_neighbor_has_more_weight(self):

        neighbors = [
            (1.0, "B"),
            (4.0, "A"),
            (4.0, "A")
        ]

        result = PreviousWeightedVote(
            neighbors
        )

        self.assertEqual(
            result,
            "B"
        )


# =========================================================
# 10. AI Version - Weighted Vote
# =========================================================

class TestAIWeightedVote(unittest.TestCase):

    def test_closer_neighbor_has_more_weight(self):

        neighbors = [
            (1.0, "B"),
            (4.0, "A"),
            (4.0, "A")
        ]

        result = AIWeightedVote(
            neighbors
        )

        self.assertEqual(
            result,
            "B"
        )

    def test_zero_distance_wins(self):

        neighbors = [
            (0.0, "A"),
            (1.0, "B"),
            (2.0, "B")
        ]

        result = AIWeightedVote(
            neighbors
        )

        self.assertEqual(
            result,
            "A"
        )


# =========================================================
# 11. Previous Lab - MyKNN
# =========================================================

class TestPreviousKNN(unittest.TestCase):

    def setUp(self):

        self.X_train = [
            [1, 1],
            [2, 2],
            [8, 8],
            [9, 9]
        ]

        self.y_train = [
            "A",
            "A",
            "B",
            "B"
        ]

    def test_prediction(self):

        model = PreviousKNN(
            k=1,
            p=2,
            sort_algorithm="quick"
        )

        model.fit(
            self.X_train,
            self.y_train
        )

        predictions = model.predict([
            [1.5, 1.5],
            [8.5, 8.5]
        ])

        self.assertEqual(
            predictions,
            ["A", "B"]
        )


# =========================================================
# 12. AI Version - MyKNN
# =========================================================

class TestAIKNN(unittest.TestCase):

    def setUp(self):

        self.X_train = [
            [1, 1],
            [2, 2],
            [8, 8],
            [9, 9]
        ]

        self.y_train = [
            "A",
            "A",
            "B",
            "B"
        ]

    def test_prediction(self):

        model = AIKNN(
            k=1,
            p=2,
            algorithm="quick"
        )

        model.fit(
            self.X_train,
            self.y_train
        )

        predictions = model.predict([
            [1.5, 1.5],
            [8.5, 8.5]
        ])

        self.assertEqual(
            predictions,
            ["A", "B"]
        )


# =========================================================
# Run Tests
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("UNIT TESTING")
    print("Previous Lab KNN + AI-generated KNN")
    print("=" * 60)
    print()

    suite = unittest.defaultTestLoader.loadTestsFromModule(
        __import__(__name__)
    )

    runner = unittest.TextTestRunner(
        verbosity=2
    )

    result = runner.run(suite)

    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    total = result.testsRun
    failed = len(result.failures)
    errors = len(result.errors)
    passed = total - failed - errors

    print("Total tests :", total)
    print("Passed      :", passed)
    print("Failed      :", failed)
    print("Errors      :", errors)

    if result.wasSuccessful():
        print()
        print("RESULT: ALL TESTS PASSED")
    else:
        print()
        print("RESULT: SOME TESTS FAILED")

    print("=" * 60)