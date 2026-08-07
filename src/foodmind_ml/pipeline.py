from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from foodmind_ml.collaborative import ItemCF, UserCF, cosine_similarity, weighted_rating_score


CUISINE_TAGS = {
    "african",
    "american",
    "asian",
    "australian",
    "british",
    "cajun",
    "canadian",
    "caribbean",
    "chinese",
    "cuban",
    "european",
    "french",
    "german",
    "greek",
    "hawaiian",
    "indian",
    "indonesian",
    "irish",
    "italian",
    "japanese",
    "korean",
    "malaysian",
    "mediterranean",
    "mexican",
    "middle-eastern",
    "moroccan",
    "south-american",
    "spanish",
    "thai",
    "vietnamese",
}


def normalize_dish_name(value: object) -> str:
    text = "" if value is None else str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def rating_to_label(rating: object) -> int | None:
    value = float(rating)
    if value >= 4:
        return 1
    if 0 < value <= 2:
        return 0
    return None


def _tags(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = value.split(",")
    if not isinstance(parsed, list):
        return []
    return [normalize_dish_name(tag).replace(" ", "-") for tag in parsed]


def infer_cuisine(value: object) -> str:
    tags = _tags(value)
    for tag in tags:
        if tag in CUISINE_TAGS:
            return tag
    return ""


def _has_any(text: str, words: tuple[str, ...]) -> int:
    return int(any(word in text for word in words))


def _dish_feature_flags(name: object, tags: object) -> dict[str, int]:
    text = f"{normalize_dish_name(name)} {' '.join(_tags(tags)).replace('-', ' ')}"
    return {
        "is_spicy": _has_any(text, ("spicy", "chili", "chilli", "curry", "hot")),
        "is_sweet": _has_any(text, ("sweet", "dessert", "cake", "cookie", "sugar")),
        "is_main_dish": _has_any(text, ("main dish", "main ingredient", "dinner", "lunch", "rice", "noodle", "pasta", "meat")),
    }


def prepare_foodcom(
    raw_dir: Path,
    processed_dir: Path,
    min_user_interactions: int = 5,
    min_dish_interactions: int = 5,
) -> dict[str, int]:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    recipes = pd.read_csv(
        raw_dir / "RAW_recipes.csv",
        usecols=["id", "name", "ingredients", "tags", "nutrition"],
    )
    recipes = recipes.rename(columns={"id": "original_recipe_id"})
    recipes["canonical_dish_name"] = recipes["name"].map(normalize_dish_name)
    recipes = recipes[recipes["canonical_dish_name"] != ""].copy()
    names = sorted(recipes["canonical_dish_name"].unique())
    dish_ids = {name: f"D{idx + 1:06d}" for idx, name in enumerate(names)}
    recipes["dish_id"] = recipes["canonical_dish_name"].map(dish_ids)
    recipes["cuisine"] = recipes["tags"].map(infer_cuisine)

    recipe_map = recipes[["original_recipe_id", "dish_id", "canonical_dish_name"]].copy()
    recipe_map.to_csv(processed_dir / "recipe_dish_map.csv", index=False)

    dishes = (
        recipes.sort_values(["dish_id", "original_recipe_id"])
        .groupby("dish_id", as_index=False)
        .first()[["dish_id", "canonical_dish_name", "original_recipe_id", "ingredients", "tags", "cuisine", "nutrition"]]
        .rename(columns={"canonical_dish_name": "dish_name"})
    )
    dishes.to_csv(processed_dir / "dishes.csv", index=False)

    interactions = pd.read_csv(
        raw_dir / "RAW_interactions.csv",
        usecols=["user_id", "recipe_id", "date", "rating"],
    )
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    interactions = interactions[interactions["rating"].notna() & (interactions["rating"] > 0)].copy()
    interactions = interactions.merge(
        recipe_map,
        left_on="recipe_id",
        right_on="original_recipe_id",
        how="inner",
    )
    interactions["interaction_time"] = pd.to_datetime(interactions["date"], errors="coerce")
    interactions = interactions[interactions["interaction_time"].notna()].copy()
    interactions = interactions.sort_values(["user_id", "dish_id", "interaction_time"])
    interactions = interactions.groupby(["user_id", "dish_id"], as_index=False).tail(1)

    previous = -1
    while previous != len(interactions):
        previous = len(interactions)
        user_counts = interactions["user_id"].value_counts()
        dish_counts = interactions["dish_id"].value_counts()
        interactions = interactions[
            interactions["user_id"].isin(user_counts[user_counts >= min_user_interactions].index)
            & interactions["dish_id"].isin(dish_counts[dish_counts >= min_dish_interactions].index)
        ].copy()

    interactions["user_id"] = "U" + interactions["user_id"].astype(str)
    interactions["label"] = interactions["rating"].map(rating_to_label)
    interactions["action"] = "rated"
    interactions["source"] = "foodcom"
    interactions = interactions[
        ["user_id", "dish_id", "rating", "label", "action", "interaction_time", "source"]
    ].sort_values(["user_id", "interaction_time", "dish_id"])
    interactions.to_csv(processed_dir / "user_dish_interactions.csv", index=False)

    train, validation, test = _time_split_by_user(interactions)
    train.to_csv(processed_dir / "train_interactions.csv", index=False)
    validation.to_csv(processed_dir / "validation_interactions.csv", index=False)
    test.to_csv(processed_dir / "test_interactions.csv", index=False)

    stats = interactions.groupby("dish_id")["rating"].agg(["mean", "count"]).reset_index()
    stats = stats.rename(columns={"mean": "dish_avg_rating", "count": "dish_rating_count"})
    feature_rows = []
    for row in dishes.to_dict("records"):
        feature_rows.append({"dish_id": row["dish_id"], "cuisine": row["cuisine"], **_dish_feature_flags(row["dish_name"], row["tags"])})
    dish_features = pd.DataFrame(feature_rows).merge(stats, on="dish_id", how="left")
    dish_features["dish_avg_rating"] = dish_features["dish_avg_rating"].fillna(0)
    dish_features["dish_rating_count"] = dish_features["dish_rating_count"].fillna(0).astype(int)
    dish_features.to_csv(processed_dir / "dish_features.csv", index=False)

    summary = {
        "recipes": int(len(recipes)),
        "dishes": int(len(dishes)),
        "interactions": int(len(interactions)),
        "train": int(len(train)),
        "validation": int(len(validation)),
        "test": int(len(test)),
    }
    (processed_dir / "foodcom_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _time_split_by_user(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranked = interactions.sort_values(["user_id", "interaction_time"]).copy()
    ranked["_rank"] = ranked.groupby("user_id").cumcount()
    ranked["_n"] = ranked.groupby("user_id")["dish_id"].transform("size")
    test = ranked[ranked["_rank"] == ranked["_n"] - 1].copy()
    validation = ranked[ranked["_rank"] == ranked["_n"] - 2].copy()
    train = ranked[ranked["_rank"] < ranked["_n"] - 2].copy()
    drop = ["_rank", "_n"]
    return train.drop(columns=drop), validation.drop(columns=drop), test.drop(columns=drop)


def build_cf_scores(matrix: np.ndarray, pairs: list[tuple[int, int]], k: int = 25) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    user_cf = UserCF(k=k).fit(matrix)
    item_cf = ItemCF(k=k).fit(matrix)
    user_scores, user_ok = user_cf.score_many(pairs)
    item_scores, item_ok = item_cf.score_many(pairs)
    return user_scores, user_ok, item_scores, item_ok


def _cosine(matrix: np.ndarray) -> np.ndarray:
    return cosine_similarity(matrix)


def _weighted_rating(ratings: np.ndarray, weights: np.ndarray, k: int) -> tuple[float, bool]:
    return weighted_rating_score(ratings, weights, k)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    d_phi = math.radians(float(lat2) - float(lat1))
    d_lambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "available"}


def _first_non_empty(left: pd.Series, right: pd.Series) -> pd.Series:
    left_text = left.fillna("").astype(str)
    return left.where(left_text.str.strip() != "", right)


def _numeric_series(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in frame.columns:
        values = frame[column]
    else:
        values = pd.Series(default, index=frame.index)
    return pd.to_numeric(values, errors="coerce").fillna(default)


def filter_restaurant_candidates(
    restaurant_menu: pd.DataFrame,
    menu_dish_mapping: pd.DataFrame,
    user_latitude: float,
    user_longitude: float,
    budget_sgd: float,
    radius_km: float,
    require_mapped: bool = True,
) -> pd.DataFrame:
    """Apply FoodMind hard constraints before model ranking."""
    rows = restaurant_menu.copy()
    mapping_cols = ["restaurant_id", "menu_item_name", "dish_id", "match_confidence", "review_status"]
    available_mapping_cols = [col for col in mapping_cols if col in menu_dish_mapping.columns]
    if {"restaurant_id", "menu_item_name"}.issubset(menu_dish_mapping.columns):
        rows = rows.merge(
            menu_dish_mapping[available_mapping_cols].drop_duplicates(["restaurant_id", "menu_item_name"]),
            on=["restaurant_id", "menu_item_name"],
            how="left",
            suffixes=("", "_mapped"),
        )
        if "dish_id_mapped" in rows.columns:
            rows["dish_id"] = _first_non_empty(rows.get("dish_id", pd.Series("", index=rows.index)), rows["dish_id_mapped"])
            rows = rows.drop(columns=["dish_id_mapped"])

    rows["price_sgd"] = pd.to_numeric(rows.get("price_sgd"), errors="coerce")
    rows["latitude"] = pd.to_numeric(rows.get("latitude"), errors="coerce")
    rows["longitude"] = pd.to_numeric(rows.get("longitude"), errors="coerce")
    rows["match_confidence"] = pd.to_numeric(rows.get("match_confidence", 1.0), errors="coerce").fillna(0.0)
    rows["availability"] = rows.get("availability", "true")

    rows = rows[
        (rows["category_group"].astype(str) == "main_dish")
        & rows["availability"].map(_truthy)
        & rows["price_sgd"].notna()
        & rows["latitude"].notna()
        & rows["longitude"].notna()
        & (rows["price_sgd"] <= float(budget_sgd))
    ].copy()
    if require_mapped:
        rows = rows[rows.get("dish_id", "").fillna("").astype(str).str.strip() != ""].copy()
    if rows.empty:
        return rows

    rows["distance_km"] = rows.apply(
        lambda row: haversine_km(user_latitude, user_longitude, row["latitude"], row["longitude"]),
        axis=1,
    )
    rows = rows[rows["distance_km"] <= float(radius_km)].copy()
    if rows.empty:
        return rows

    rows["price_budget_ratio"] = rows["price_sgd"] / float(budget_sgd) if budget_sgd else 1.0
    rows["budget_fit_score"] = (1.0 - rows["price_budget_ratio"]).clip(lower=0.0, upper=1.0)
    rows["distance_fit_score"] = (1.0 - rows["distance_km"] / float(radius_km)).clip(lower=0.0, upper=1.0) if radius_km else 0.0
    return rows.sort_values(["distance_km", "price_sgd", "restaurant_menu_id"], kind="stable").reset_index(drop=True)


def build_user_preference_profiles(interactions: pd.DataFrame, dish_features: pd.DataFrame) -> pd.DataFrame:
    feature_cols = ["is_spicy", "is_sweet", "is_main_dish"]
    joined = interactions.copy()
    if "label" not in joined.columns:
        joined["label"] = joined["rating"].map(rating_to_label)
    joined = joined[joined["label"] == 1].join(dish_features[feature_cols], on="dish_id", how="inner")
    if joined.empty:
        return pd.DataFrame(columns=["user_id", "user_spicy_preference", "user_sweet_preference", "user_main_dish_preference"]).set_index("user_id")
    profiles = joined.groupby("user_id")[feature_cols].mean()
    return profiles.rename(
        columns={
            "is_spicy": "user_spicy_preference",
            "is_sweet": "user_sweet_preference",
            "is_main_dish": "user_main_dish_preference",
        }
    )


def add_user_preference_scores(
    rows: pd.DataFrame,
    user_profiles: pd.DataFrame,
    dish_features: pd.DataFrame,
) -> pd.DataFrame:
    scored = rows.copy()
    needed = ["is_spicy", "is_sweet", "is_main_dish"]
    if not set(needed).issubset(scored.columns):
        scored = scored.join(dish_features[needed], on="dish_id", rsuffix="_feature")
    if user_profiles is not None and not user_profiles.empty:
        scored = scored.join(user_profiles, on="user_id")
    defaults = {
        "user_spicy_preference": 0.5,
        "user_sweet_preference": 0.5,
        "user_main_dish_preference": 0.5,
        "is_spicy": 0.0,
        "is_sweet": 0.0,
        "is_main_dish": 1.0,
    }
    for col, default in defaults.items():
        scored[col] = _numeric_series(scored, col, default)
    scored["spice_preference_match"] = 1.0 - (scored["user_spicy_preference"] - scored["is_spicy"]).abs()
    scored["sweet_preference_match"] = 1.0 - (scored["user_sweet_preference"] - scored["is_sweet"]).abs()
    scored["main_dish_preference_match"] = 1.0 - (scored["user_main_dish_preference"] - scored["is_main_dish"]).abs()
    scored["user_preference_score"] = scored[
        ["spice_preference_match", "sweet_preference_match", "main_dish_preference_match"]
    ].mean(axis=1)
    return scored


class NumpyLogisticRegression:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 300, l2: float = 0.001):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyLogisticRegression":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        xs = (x - self.mean_) / self.std_
        xb = np.c_[np.ones(xs.shape[0]), xs]
        weights = np.zeros(xb.shape[1], dtype=float)
        for _ in range(self.epochs):
            pred = _sigmoid(xb @ weights)
            grad = (xb.T @ (pred - y)) / len(y)
            grad[1:] += self.l2 * weights[1:]
            weights -= self.learning_rate * grad
        self.weights_ = weights
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None or self.weights_ is None:
            raise ValueError("model is not fitted")
        xs = (np.asarray(x, dtype=float) - self.mean_) / self.std_
        xb = np.c_[np.ones(xs.shape[0]), xs]
        return _sigmoid(xb @ self.weights_)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35, 35)))


FEATURE_NAMES = [
    "user_cf_score",
    "user_cf_available",
    "item_cf_score",
    "item_cf_available",
    "user_preference_score",
    "spice_preference_match",
    "sweet_preference_match",
    "main_dish_preference_match",
    "dish_avg_rating",
    "dish_rating_count_log",
    "user_avg_rating",
    "is_spicy",
    "is_sweet",
    "is_main_dish",
    "price_budget_ratio",
    "budget_fit_score",
    "distance_km",
    "distance_fit_score",
    "match_confidence",
]


def train_hybrid_model(
    processed_dir: Path,
    artifacts_dir: Path,
    reports_dir: Path,
    max_users: int = 1200,
    max_dishes: int = 1200,
) -> dict[str, object]:
    processed_dir = Path(processed_dir)
    artifacts_dir = Path(artifacts_dir)
    reports_dir = Path(reports_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train = _labelled(pd.read_csv(processed_dir / "train_interactions.csv"))
    validation = _labelled(pd.read_csv(processed_dir / "validation_interactions.csv"))
    test = _labelled(pd.read_csv(processed_dir / "test_interactions.csv"))
    dish_features = pd.read_csv(processed_dir / "dish_features.csv").set_index("dish_id")
    dishes = pd.read_csv(processed_dir / "dishes.csv", usecols=["dish_id", "dish_name"]).set_index("dish_id")

    users = train["user_id"].value_counts().head(max_users).index.tolist()
    dish_ids = train["dish_id"].value_counts().head(max_dishes).index.tolist()
    user_index = {user: idx for idx, user in enumerate(users)}
    dish_index = {dish: idx for idx, dish in enumerate(dish_ids)}
    matrix = _rating_matrix(train, user_index, dish_index)
    user_avg = train.groupby("user_id")["rating"].mean().to_dict()
    user_profiles = build_user_preference_profiles(train, dish_features)

    x_train, y_train, _ = _features(train, matrix, user_index, dish_index, dish_features, user_avg, user_profiles)
    x_val, y_val, val_rows = _features(validation, matrix, user_index, dish_index, dish_features, user_avg, user_profiles)
    x_test, y_test, _ = _features(test, matrix, user_index, dish_index, dish_features, user_avg, user_profiles)
    if len(y_train) == 0:
        raise ValueError("no labelled training rows after filtering")

    majority = int(np.mean(y_train) >= 0.5)
    model = NumpyLogisticRegression()
    model.fit(x_train, y_train)

    metrics = {
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "selected_users": int(len(users)),
        "selected_dishes": int(len(dish_ids)),
        "majority_class": majority,
        "majority_validation": _metrics(y_val, np.full(len(y_val), majority, dtype=float)),
        "hybrid_lr_validation": _metrics(y_val, model.predict_proba(x_val)) if len(y_val) else {},
        "hybrid_lr_test": _metrics(y_test, model.predict_proba(x_test)) if len(y_test) else {},
        "feature_names": FEATURE_NAMES,
    }
    (reports_dir / "hybrid_lr_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez(
        artifacts_dir / "hybrid_lr_model.npz",
        weights=model.weights_,
        mean=model.mean_,
        std=model.std_,
        feature_names=np.array(FEATURE_NAMES),
    )
    _write_sample_recommendations(
        reports_dir / "sample_recommendations.csv",
        model,
        val_rows,
        matrix,
        user_index,
        dish_index,
        dish_features,
        dishes,
        user_avg,
        train,
    )
    return metrics


def _labelled(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["label"] = frame["label"].where(frame["label"].notna(), frame["rating"].map(rating_to_label))
    return frame[frame["label"].notna()].copy()


def _rating_matrix(frame: pd.DataFrame, user_index: dict[str, int], dish_index: dict[str, int]) -> np.ndarray:
    matrix = np.zeros((len(user_index), len(dish_index)), dtype=float)
    for row in frame.itertuples(index=False):
        user_idx = user_index.get(row.user_id)
        dish_idx = dish_index.get(row.dish_id)
        if user_idx is not None and dish_idx is not None:
            matrix[user_idx, dish_idx] = float(row.rating)
    return matrix


def _features(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    user_index: dict[str, int],
    dish_index: dict[str, int],
    dish_features: pd.DataFrame,
    user_avg: dict[str, float],
    user_profiles: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows = frame[frame["user_id"].isin(user_index) & frame["dish_id"].isin(dish_index)].copy()
    pairs = [(user_index[row.user_id], dish_index[row.dish_id]) for row in rows.itertuples(index=False)]
    if not pairs:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0,)), rows
    user_cf, user_ok, item_cf, item_ok = build_cf_scores(matrix, pairs)
    rows["user_cf_score"] = user_cf
    rows["user_cf_available"] = user_ok.astype(int)
    rows["item_cf_score"] = item_cf
    rows["item_cf_available"] = item_ok.astype(int)
    rows["user_avg_rating"] = rows["user_id"].map(user_avg).fillna(0) / 5.0
    overlapping_feature_cols = [col for col in dish_features.columns if col in rows.columns]
    rows = rows.drop(columns=overlapping_feature_cols)
    joined = rows.join(dish_features, on="dish_id")
    joined = add_user_preference_scores(joined, user_profiles if user_profiles is not None else pd.DataFrame(), dish_features)
    joined["dish_avg_rating"] = joined["dish_avg_rating"].fillna(0) / 5.0
    joined["dish_rating_count_log"] = joined["dish_rating_count"].fillna(0).map(lambda value: math.log1p(value))
    for col in ["is_spicy", "is_sweet", "is_main_dish"]:
        joined[col] = joined[col].fillna(0)
    context_defaults = {
        "price_budget_ratio": 1.0,
        "budget_fit_score": 0.0,
        "distance_km": 0.0,
        "distance_fit_score": 0.0,
        "match_confidence": 1.0,
    }
    for col, default in context_defaults.items():
        joined[col] = _numeric_series(joined, col, default)
    return joined[FEATURE_NAMES].to_numpy(dtype=float), joined["label"].to_numpy(dtype=int), joined


def build_candidate_feature_frame(
    user_id: str,
    candidates: pd.DataFrame,
    matrix: np.ndarray,
    user_index: dict[str, int],
    dish_index: dict[str, int],
    dish_features: pd.DataFrame,
    user_avg: dict[str, float],
    user_profiles: pd.DataFrame | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    candidate_rows = candidates.copy()
    candidate_rows["user_id"] = user_id
    candidate_rows["rating"] = 0
    candidate_rows["label"] = 0
    x, _, rows = _features(candidate_rows, matrix, user_index, dish_index, dish_features, user_avg, user_profiles)
    return x, rows


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    if len(y_true) == 0:
        return {}
    pred = (probabilities >= 0.5).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": float((tp + tn) / len(y_true)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "roc_auc": float(_roc_auc(y_true, probabilities)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    wins = sum(float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg)) for p in pos)
    return wins / (len(pos) * len(neg))


def _write_sample_recommendations(
    path: Path,
    model: NumpyLogisticRegression,
    validation_rows: pd.DataFrame,
    matrix: np.ndarray,
    user_index: dict[str, int],
    dish_index: dict[str, int],
    dish_features: pd.DataFrame,
    dishes: pd.DataFrame,
    user_avg: dict[str, float],
    train: pd.DataFrame,
) -> None:
    users = validation_rows["user_id"].drop_duplicates().head(20).tolist()
    already = train.groupby("user_id")["dish_id"].apply(set).to_dict()
    out = []
    reverse_dishes = list(dish_index.keys())
    for user in users:
        candidates = [dish for dish in reverse_dishes if dish not in already.get(user, set())]
        candidate_rows = pd.DataFrame({"user_id": user, "dish_id": candidates, "rating": 0, "label": 0})
        user_profiles = build_user_preference_profiles(train, dish_features)
        x, _, rows = _features(candidate_rows, matrix, user_index, dish_index, dish_features, user_avg, user_profiles)
        if len(rows) == 0:
            continue
        rows = rows.copy()
        rows["acceptance_probability"] = model.predict_proba(x)
        rows = rows.sort_values("acceptance_probability", ascending=False).head(3)
        for row in rows.itertuples(index=False):
            out.append(
                {
                    "user_id": row.user_id,
                    "dish_id": row.dish_id,
                    "dish_name": dishes.loc[row.dish_id, "dish_name"] if row.dish_id in dishes.index else "",
                    "acceptance_probability": round(float(row.acceptance_probability), 4),
                    "user_cf_score": round(float(row.user_cf_score), 4),
                    "item_cf_score": round(float(row.item_cf_score), 4),
                }
            )
    pd.DataFrame(out).to_csv(path, index=False)


def prepare_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/foodcom")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-dish-interactions", type=int, default=5)
    args = parser.parse_args()
    summary = prepare_foodcom(Path(args.raw_dir), Path(args.processed_dir), args.min_user_interactions, args.min_dish_interactions)
    print(json.dumps(summary, indent=2))


def train_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--artifacts-dir", default="artifacts/candidate")
    parser.add_argument("--reports-dir", default="reports/metrics")
    parser.add_argument("--max-users", type=int, default=1200)
    parser.add_argument("--max-dishes", type=int, default=1200)
    args = parser.parse_args()
    metrics = train_hybrid_model(Path(args.processed_dir), Path(args.artifacts_dir), Path(args.reports_dir), args.max_users, args.max_dishes)
    print(json.dumps(metrics, indent=2))
