import unittest

import numpy as np

from foodmind_ml.collaborative import ItemCF, UserCF


class CollaborativeFilteringTests(unittest.TestCase):
    def test_user_cf_scores_candidate_from_similar_users(self):
        matrix = np.array(
            [
                [5.0, 4.0, 0.0],
                [5.0, 0.0, 1.0],
                [0.0, 4.0, 1.0],
            ]
        )

        model = UserCF(k=25).fit(matrix)
        score, available = model.score(user_idx=1, item_idx=1)

        self.assertTrue(available)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_item_cf_scores_candidate_from_similar_items(self):
        matrix = np.array(
            [
                [5.0, 4.0, 0.0],
                [5.0, 0.0, 1.0],
                [0.0, 4.0, 1.0],
            ]
        )

        model = ItemCF(k=25).fit(matrix)
        score, available = model.score(user_idx=0, item_idx=2)

        self.assertTrue(available)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_cf_returns_unavailable_when_no_positive_neighbour_signal(self):
        matrix = np.array(
            [
                [5.0, 0.0],
                [0.0, 4.0],
            ]
        )

        user_score, user_available = UserCF().fit(matrix).score(user_idx=0, item_idx=1)
        item_score, item_available = ItemCF().fit(matrix).score(user_idx=0, item_idx=1)

        self.assertEqual(user_score, 0.0)
        self.assertFalse(user_available)
        self.assertEqual(item_score, 0.0)
        self.assertFalse(item_available)


if __name__ == "__main__":
    unittest.main()
