import unittest

import numpy as np

from foodmind_ml.pipeline import (
    NumpyLogisticRegression,
    build_cf_scores,
    normalize_dish_name,
    rating_to_label,
)


class PipelineTests(unittest.TestCase):
    def test_normalize_dish_name_removes_noise(self):
        self.assertEqual(normalize_dish_name("  Chef's SPECIAL Chicken-Rice!! "), "chef s special chicken rice")

    def test_rating_to_label_keeps_neutral_unknown(self):
        self.assertEqual(rating_to_label(5), 1)
        self.assertEqual(rating_to_label(1), 0)
        self.assertIsNone(rating_to_label(3))
        self.assertIsNone(rating_to_label(0))

    def test_build_cf_scores_sets_availability_flags(self):
        matrix = np.array(
            [
                [5.0, 4.0, 0.0],
                [5.0, 0.0, 1.0],
                [0.0, 4.0, 1.0],
            ]
        )

        user_cf, user_ok, item_cf, item_ok = build_cf_scores(matrix, [(1, 1), (0, 2)])

        self.assertTrue(user_ok[0])
        self.assertTrue(item_ok[0])
        self.assertGreater(user_cf[0], 0)
        self.assertGreater(item_cf[0], 0)
        self.assertTrue(user_ok[1])
        self.assertTrue(item_ok[1])

    def test_numpy_logistic_regression_learns_simple_boundary(self):
        x = np.array([[0.0], [0.1], [0.9], [1.0]])
        y = np.array([0, 0, 1, 1])

        model = NumpyLogisticRegression(learning_rate=0.5, epochs=400)
        model.fit(x, y)
        pred = model.predict_proba(x)

        self.assertLess(pred[0], 0.5)
        self.assertGreater(pred[-1], 0.5)


if __name__ == "__main__":
    unittest.main()
