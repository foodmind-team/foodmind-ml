from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    return (matrix @ matrix.T) / np.outer(norms, norms)


def weighted_rating_score(ratings: np.ndarray, weights: np.ndarray, k: int) -> tuple[float, bool]:
    keep = np.flatnonzero(weights > 0)
    if keep.size == 0:
        return 0.0, False
    if keep.size > k:
        keep = keep[np.argsort(weights[keep])[-k:]]
    denom = float(np.sum(np.abs(weights[keep])))
    if denom == 0:
        return 0.0, False
    return float(np.dot(ratings[keep], weights[keep]) / denom / 5.0), True


@dataclass
class UserCF:
    k: int = 25

    def fit(self, user_item_matrix: np.ndarray) -> "UserCF":
        self.matrix_ = np.asarray(user_item_matrix, dtype=float)
        self.user_similarity_ = cosine_similarity(self.matrix_)
        np.fill_diagonal(self.user_similarity_, 0.0)
        return self

    def score(self, user_idx: int, item_idx: int) -> tuple[float, bool]:
        self._check_fitted()
        raters = np.flatnonzero(self.matrix_[:, item_idx] > 0)
        weights = self.user_similarity_[user_idx, raters]
        return weighted_rating_score(self.matrix_[raters, item_idx], weights, self.k)

    def score_many(self, pairs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
        scores, available = [], []
        for user_idx, item_idx in pairs:
            score, ok = self.score(user_idx, item_idx)
            scores.append(score)
            available.append(ok)
        return np.array(scores, dtype=float), np.array(available, dtype=bool)

    def _check_fitted(self) -> None:
        if not hasattr(self, "matrix_") or not hasattr(self, "user_similarity_"):
            raise ValueError("UserCF is not fitted")


@dataclass
class ItemCF:
    k: int = 25

    def fit(self, user_item_matrix: np.ndarray) -> "ItemCF":
        self.matrix_ = np.asarray(user_item_matrix, dtype=float)
        self.item_similarity_ = cosine_similarity(self.matrix_.T)
        np.fill_diagonal(self.item_similarity_, 0.0)
        return self

    def score(self, user_idx: int, item_idx: int) -> tuple[float, bool]:
        self._check_fitted()
        rated_items = np.flatnonzero(self.matrix_[user_idx, :] > 0)
        weights = self.item_similarity_[item_idx, rated_items]
        return weighted_rating_score(self.matrix_[user_idx, rated_items], weights, self.k)

    def score_many(self, pairs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
        scores, available = [], []
        for user_idx, item_idx in pairs:
            score, ok = self.score(user_idx, item_idx)
            scores.append(score)
            available.append(ok)
        return np.array(scores, dtype=float), np.array(available, dtype=bool)

    def _check_fitted(self) -> None:
        if not hasattr(self, "matrix_") or not hasattr(self, "item_similarity_"):
            raise ValueError("ItemCF is not fitted")
