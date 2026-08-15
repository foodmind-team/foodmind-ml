import unittest

import numpy as np
import pandas as pd

from foodmind_ml.pipeline import (
    NumpyLogisticRegression,
    add_user_preference_scores,
    build_candidate_feature_frame,
    build_cf_scores,
    build_user_preference_profiles,
    filter_restaurant_candidates,
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

    def test_filter_restaurant_candidates_applies_budget_distance_and_mapping(self):
        menu = pd.DataFrame(
            [
                {
                    "restaurant_menu_id": "RM1",
                    "restaurant_id": "R1",
                    "restaurant_name": "Near Cafe",
                    "menu_item_name": "Spicy Noodles",
                    "price_sgd": "8.00",
                    "category_group": "main_dish",
                    "availability": "true",
                    "latitude": 1.3000,
                    "longitude": 103.7700,
                },
                {
                    "restaurant_menu_id": "RM2",
                    "restaurant_id": "R1",
                    "restaurant_name": "Near Cafe",
                    "menu_item_name": "Iced Tea",
                    "price_sgd": "3.00",
                    "category_group": "drink",
                    "availability": "true",
                    "latitude": 1.3000,
                    "longitude": 103.7700,
                },
                {
                    "restaurant_menu_id": "RM3",
                    "restaurant_id": "R2",
                    "restaurant_name": "Far Cafe",
                    "menu_item_name": "Chicken Rice",
                    "price_sgd": "8.00",
                    "category_group": "main_dish",
                    "availability": "true",
                    "latitude": 1.3600,
                    "longitude": 103.8500,
                },
            ]
        )
        mapping = pd.DataFrame(
            [
                {
                    "restaurant_id": "R1",
                    "menu_item_name": "Spicy Noodles",
                    "dish_id": "D001",
                    "match_confidence": "0.91",
                    "review_status": "auto_matched",
                },
                {
                    "restaurant_id": "R2",
                    "menu_item_name": "Chicken Rice",
                    "dish_id": "D002",
                    "match_confidence": "1.00",
                    "review_status": "auto_matched",
                },
            ]
        )

        candidates = filter_restaurant_candidates(
            menu,
            mapping,
            user_latitude=1.3001,
            user_longitude=103.7701,
            budget_sgd=10.0,
            radius_km=2.0,
        )

        self.assertEqual(candidates["restaurant_menu_id"].tolist(), ["RM1"])
        self.assertEqual(candidates.iloc[0]["dish_id"], "D001")
        self.assertLess(candidates.iloc[0]["distance_km"], 0.1)
        self.assertAlmostEqual(candidates.iloc[0]["price_budget_ratio"], 0.8)

    def test_user_preference_score_rewards_matching_profile(self):
        interactions = pd.DataFrame(
            [
                {"user_id": "U1", "dish_id": "D_spicy", "rating": 5},
                {"user_id": "U1", "dish_id": "D_plain", "rating": 1},
            ]
        )
        dish_features = pd.DataFrame(
            [
                {"dish_id": "D_spicy", "is_spicy": 1, "is_sweet": 0, "is_main_dish": 1, "cuisine": "thai"},
                {"dish_id": "D_plain", "is_spicy": 0, "is_sweet": 0, "is_main_dish": 1, "cuisine": "american"},
            ]
        ).set_index("dish_id")
        profiles = build_user_preference_profiles(interactions, dish_features)
        candidates = pd.DataFrame(
            [
                {"user_id": "U1", "dish_id": "D_spicy"},
                {"user_id": "U1", "dish_id": "D_plain"},
            ]
        )

        scored = add_user_preference_scores(candidates, profiles, dish_features)

        spicy_score = scored.loc[scored["dish_id"] == "D_spicy", "user_preference_score"].iloc[0]
        plain_score = scored.loc[scored["dish_id"] == "D_plain", "user_preference_score"].iloc[0]
        self.assertGreater(spicy_score, plain_score)
        self.assertGreaterEqual(spicy_score, 0.5)

    def test_build_candidate_feature_frame_keeps_cf_user_and_context_features(self):
        dish_features = pd.DataFrame(
            [
                {
                    "dish_id": "D1",
                    "is_spicy": 1,
                    "is_sweet": 0,
                    "is_main_dish": 1,
                    "dish_avg_rating": 4.5,
                    "dish_rating_count": 10,
                    "cuisine": "thai",
                }
            ]
        ).set_index("dish_id")
        profiles = pd.DataFrame(
            [
                {
                    "user_id": "U1",
                    "user_spicy_preference": 1.0,
                    "user_sweet_preference": 0.0,
                    "user_main_dish_preference": 1.0,
                }
            ]
        ).set_index("user_id")
        candidates = pd.DataFrame(
            [
                {
                    "restaurant_menu_id": "RM1",
                    "dish_id": "D1",
                    "is_main_dish": "true",
                    "price_budget_ratio": 0.6,
                    "budget_fit_score": 0.4,
                    "distance_km": 0.8,
                    "distance_fit_score": 0.6,
                    "match_confidence": 0.9,
                }
            ]
        )
        matrix = np.array([[5.0], [4.0]])

        x, rows = build_candidate_feature_frame(
            user_id="U1",
            candidates=candidates,
            matrix=matrix,
            user_index={"U1": 0, "U2": 1},
            dish_index={"D1": 0},
            dish_features=dish_features,
            user_avg={"U1": 5.0},
            user_profiles=profiles,
        )

        self.assertEqual(x.shape[0], 1)
        self.assertAlmostEqual(rows.iloc[0]["user_preference_score"], 1.0)
        self.assertAlmostEqual(rows.iloc[0]["price_budget_ratio"], 0.6)
        self.assertAlmostEqual(rows.iloc[0]["distance_km"], 0.8)
        self.assertTrue(rows.iloc[0]["user_cf_available"])

    def test_build_candidate_feature_frame_fills_missing_context_defaults(self):
        dish_features = pd.DataFrame(
            [
                {
                    "dish_id": "D1",
                    "is_spicy": 0,
                    "is_sweet": 0,
                    "is_main_dish": 1,
                    "dish_avg_rating": 4.0,
                    "dish_rating_count": 3,
                    "cuisine": "",
                }
            ]
        ).set_index("dish_id")

        x, rows = build_candidate_feature_frame(
            user_id="U1",
            candidates=pd.DataFrame([{"dish_id": "D1"}]),
            matrix=np.array([[5.0]]),
            user_index={"U1": 0},
            dish_index={"D1": 0},
            dish_features=dish_features,
            user_avg={"U1": 5.0},
            user_profiles=pd.DataFrame(),
        )

        self.assertEqual(x.shape[1], 19)
        self.assertAlmostEqual(rows.iloc[0]["price_budget_ratio"], 1.0)
        self.assertAlmostEqual(rows.iloc[0]["match_confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
