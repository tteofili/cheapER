import unittest

import numpy as np

from cheaper.metrics import recall_at_k


class RecallAtKTest(unittest.TestCase):
    def test_uses_actual_relevant_count_and_deduplicates(self):
        results = np.array([[1, 2, 2], [3, 4, 5]])
        ground_truth = np.array([[1, 1, -1], [4, 4, -1]])

        score = recall_at_k(results, ground_truth, 2, ignore_values={-1})

        self.assertEqual(score, 1.0)

    def test_filters_nan_padding(self):
        results = np.array([[1.0, np.nan], [3.0, 4.0]])
        ground_truth = np.array([[np.nan, 1.0], [np.nan, 5.0]])

        score = recall_at_k(results, ground_truth, 2)

        self.assertEqual(score, 0.5)

    def test_rejects_non_positive_k(self):
        with self.assertRaises(ValueError):
            recall_at_k(np.array([[1, 2]]), np.array([[1, 2]]), 0)

    def test_rejects_mismatched_rows(self):
        with self.assertRaises(ValueError):
            recall_at_k(np.array([[1, 2], [3, 4]]), np.array([[1, 2]]), 2)

    def test_rejects_not_enough_prediction_columns(self):
        with self.assertRaises(ValueError):
            recall_at_k(np.array([[1, 2]]), np.array([[1, 2, 3]]), 3)

    def test_rejects_not_enough_ground_truth_columns(self):
        with self.assertRaises(ValueError):
            recall_at_k(np.array([[1, 2, 3]]), np.array([[1, 2]]), 3)

    def test_rejects_non_2d_inputs(self):
        with self.assertRaises(ValueError):
            recall_at_k(np.array([1, 2]), np.array([[1, 2]]), 1)


if __name__ == "__main__":
    unittest.main()
