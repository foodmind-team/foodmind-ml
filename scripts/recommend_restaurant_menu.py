"""Recommend real restaurant menu items for one FoodMind user context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from foodmind_ml.pipeline import (
    FEATURE_NAMES,
    NumpyLogisticRegression,
    _labelled,
    _rating_matrix,
    build_candidate_feature_frame,
    build_user_preference_profiles,
    filter_restaurant_candidates,
)


def load_model(path: Path) -> NumpyLogisticRegression:
    payload = np.load(path, allow_pickle=True)
    saved_features = [str(value) for value in payload["feature_names"].tolist()]
    if saved_features != FEATURE_NAMES:
        raise ValueError(
            "model feature schema is out of date; run "
            "`python scripts/train_hybrid_model.py` before restaurant recommendation"
        )
    model = NumpyLogisticRegression()
    model.weights_ = payload["weights"]
    model.mean_ = payload["mean"]
    model.std_ = payload["std"]
    return model


def reason_codes(row: pd.Series) -> str:
    codes = []
    if bool(row.get("user_cf_available")) and float(row.get("user_cf_score", 0)) >= 0.7:
        codes.append("similar_users_liked")
    if bool(row.get("item_cf_available")) and float(row.get("item_cf_score", 0)) >= 0.7:
        codes.append("similar_to_user_history")
    if float(row.get("user_preference_score", 0)) >= 0.7:
        codes.append("matches_user_profile")
    if float(row.get("budget_fit_score", 0)) >= 0.2:
        codes.append("comfortably_within_budget")
    if float(row.get("distance_fit_score", 0)) >= 0.5:
        codes.append("nearby")
    if float(row.get("match_confidence", 0)) < 0.8:
        codes.append("dish_mapping_needs_review")
    return "|".join(codes)


def build_indices(
    train: pd.DataFrame, candidate_dish_ids: list[str], user_id: str, max_users: int, max_dishes: int
) -> tuple[dict[str, int], dict[str, int], np.ndarray]:
    users = train["user_id"].value_counts().head(max_users).index.tolist()
    if user_id not in users:
        users.append(user_id)

    dishes = train["dish_id"].value_counts().head(max_dishes).index.tolist()
    for dish_id in candidate_dish_ids:
        if dish_id and dish_id not in dishes:
            dishes.append(dish_id)

    user_index = {user: idx for idx, user in enumerate(users)}
    dish_index = {dish: idx for idx, dish in enumerate(dishes)}
    matrix = _rating_matrix(train, user_index, dish_index)
    return user_index, dish_index, matrix


def recommend(args: argparse.Namespace) -> pd.DataFrame:
    processed_dir = Path(args.processed_dir)
    model = load_model(Path(args.artifacts_dir) / "hybrid_lr_model.npz")

    train = _labelled(pd.read_csv(processed_dir / "train_interactions.csv"))
    dish_features = pd.read_csv(processed_dir / "dish_features.csv").set_index("dish_id")
    dishes = pd.read_csv(processed_dir / "dishes.csv", usecols=["dish_id", "dish_name"]).set_index("dish_id")
    restaurant_menu = pd.read_csv(processed_dir / "restaurant_menu.csv")
    menu_mapping = pd.read_csv(processed_dir / "menu_dish_mapping.csv")

    candidates = filter_restaurant_candidates(
        restaurant_menu,
        menu_mapping,
        user_latitude=args.lat,
        user_longitude=args.lon,
        budget_sgd=args.budget,
        radius_km=args.radius_km,
        require_mapped=True,
    )
    if candidates.empty:
        return pd.DataFrame()

    user_index, dish_index, matrix = build_indices(
        train,
        candidates["dish_id"].dropna().astype(str).unique().tolist(),
        args.user_id,
        args.max_users,
        args.max_dishes,
    )
    user_avg = train.groupby("user_id")["rating"].mean().to_dict()
    user_profiles = build_user_preference_profiles(train, dish_features)
    x, scored_rows = build_candidate_feature_frame(
        args.user_id,
        candidates,
        matrix,
        user_index,
        dish_index,
        dish_features,
        user_avg,
        user_profiles,
    )
    if scored_rows.empty:
        return pd.DataFrame()

    scored_rows = scored_rows.copy()
    scored_rows["acceptance_probability"] = model.predict_proba(x)
    scored_rows["dish_name"] = (
        scored_rows["dish_id"].map(dishes["dish_name"]).fillna(scored_rows["canonical_dish_name"])
    )
    scored_rows["reason_codes"] = scored_rows.apply(reason_codes, axis=1)
    keep = [
        "restaurant_menu_id",
        "restaurant_id",
        "restaurant_name",
        "dish_id",
        "dish_name",
        "menu_item_name",
        "price_sgd",
        "distance_km",
        "acceptance_probability",
        "user_cf_score",
        "item_cf_score",
        "user_preference_score",
        "price_budget_ratio",
        "match_confidence",
        "source_reliability",
        "reason_codes",
    ]
    return scored_rows.sort_values("acceptance_probability", ascending=False)[keep].head(args.top_n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--budget", required=True, type=float)
    parser.add_argument("--lat", required=True, type=float)
    parser.add_argument("--lon", required=True, type=float)
    parser.add_argument("--radius-km", default=3.0, type=float)
    parser.add_argument("--top-n", default=5, type=int)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--artifacts-dir", default="artifacts/candidate")
    parser.add_argument("--output", default="reports/metrics/restaurant_recommendations.csv")
    parser.add_argument("--max-users", default=1200, type=int)
    parser.add_argument("--max-dishes", default=1200, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = recommend(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output, index=False)
    print(json.dumps({"rows": int(len(rows)), "output": str(output)}, indent=2))
    if not rows.empty:
        print(rows.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
