"""Deterministic, privacy-preserving UserCF and ItemCF serving index builder."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from foodmind_ml.collaborative import ItemCF, UserCF


INDEX_SCHEMA_VERSION = "foodmind-collaborative-index-v1"


def build_index(
    snapshot: Path,
    *,
    min_neighbor_support: int = 3,
    min_item_support: int = 2,
    top_k: int = 25,
) -> dict[str, Any]:
    """Build a bounded index from a Backend training snapshot.

    The input contains HMAC pseudonyms only.  Only rows with the Backend's
    positive ``collaborativeStrength`` participate; a rejection is not an
    implicit negative signal and therefore never enters the matrix.
    """
    if min_neighbor_support < 1 or min_item_support < 1 or top_k < 1:
        raise ValueError("collaborative index thresholds must be positive")
    raw = snapshot.read_bytes()
    interactions: dict[str, dict[str, float]] = defaultdict(dict)
    for line in raw.decode("utf-8").splitlines():
        row = json.loads(line)
        user_key, meal_key = row.get("userKey"), row.get("mealKey")
        strength = row.get("collaborativeStrength", 0.0)
        if not isinstance(user_key, str) or not isinstance(meal_key, str):
            continue
        if not isinstance(strength, (int, float)) or isinstance(strength, bool):
            continue
        if 0.0 < float(strength) <= 1.0:
            interactions[user_key][meal_key] = max(interactions[user_key].get(meal_key, 0.0), float(strength))

    user_keys = sorted(interactions)
    meal_keys = sorted({meal for meals in interactions.values() for meal in meals})
    matrix = np.zeros((len(user_keys), len(meal_keys)), dtype=float)
    user_index = {key: i for i, key in enumerate(user_keys)}
    meal_index = {key: i for i, key in enumerate(meal_keys)}
    for user_key, meals in interactions.items():
        for meal_key, strength in meals.items():
            matrix[user_index[user_key], meal_index[meal_key]] = strength * 5.0

    user_cf = UserCF(k=top_k).fit(matrix) if matrix.size else None
    item_cf = ItemCF(k=top_k).fit(matrix) if matrix.size else None
    user_scores: dict[str, dict[str, dict[str, float | int]]] = {}
    item_scores: dict[str, dict[str, dict[str, float | int]]] = {}
    for user_key, user_i in user_index.items():
        user_entries: dict[str, dict[str, float | int]] = {}
        item_entries: dict[str, dict[str, float | int]] = {}
        for meal_key, meal_i in meal_index.items():
            if matrix[user_i, meal_i] > 0:
                continue
            assert user_cf is not None and item_cf is not None
            score, available = user_cf.score(user_i, meal_i)
            neighbor_support = int(np.count_nonzero(
                (matrix[:, meal_i] > 0) & (user_cf.user_similarity_[user_i, :] > 0)
            ))
            if available and neighbor_support >= min_neighbor_support:
                user_entries[meal_key] = {"score": round(float(score), 8), "support": neighbor_support}
            item_score, item_available = item_cf.score(user_i, meal_i)
            item_support = int(np.count_nonzero(
                (matrix[user_i, :] > 0) & (item_cf.item_similarity_[meal_i, :] > 0)
            ))
            if item_available and item_support >= min_item_support:
                item_entries[meal_key] = {"score": round(float(item_score), 8), "support": item_support}
        if user_entries:
            user_scores[user_key] = user_entries
        if item_entries:
            item_scores[user_key] = item_entries

    return {
        "schemaVersion": INDEX_SCHEMA_VERSION,
        "sourceSnapshotSha256": hashlib.sha256(raw).hexdigest(),
        "positiveOnly": True,
        "thresholds": {
            "minNeighborSupport": min_neighbor_support,
            "minItemSupport": min_item_support,
            "topK": top_k,
        },
        "userCount": len(user_keys),
        "mealCount": len(meal_keys),
        "userCf": user_scores,
        "itemCf": item_scores,
    }


def write_index(snapshot: Path, output: Path, **kwargs: Any) -> dict[str, Any]:
    index = build_index(snapshot, **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return index
