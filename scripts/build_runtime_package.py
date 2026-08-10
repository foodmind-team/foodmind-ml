"""Build the local Recommendation Inference package from the trained ML artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


RUNTIME_FEATURES = (
    "preference_match",
    "want_to_try",
    "group_preference_rate",
    "group_available",
    "context_match",
    "cleanliness_observed",
)


def build(source: Path, output: Path) -> None:
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source, allow_pickle=False) as model:
        names = tuple(str(name) for name in model["feature_names"].tolist())
        weights = np.asarray(model["weights"], dtype=float)
        mean = np.asarray(model["mean"], dtype=float)
        std = np.asarray(model["std"], dtype=float)

    if weights.shape != (len(names) + 1,) or mean.shape != (len(names),) or std.shape != mean.shape:
        raise ValueError("source ML artifact has incompatible tensor shapes")
    source_index = {name: index for index, name in enumerate(names)}
    # These mappings preserve learned signals that are supplied by the runtime
    # contract; unavailable source-model features are intentionally neutral.
    mappings = {
        "preference_match": "user_preference_score",
        "want_to_try": None,
        "group_preference_rate": "dish_avg_rating",
        "group_available": "dish_rating_count_log",
        "context_match": "distance_fit_score",
        "cleanliness_observed": None,
    }
    runtime_weights = [float(weights[0])]
    runtime_mean = []
    runtime_std = []
    for name in RUNTIME_FEATURES:
        source_name = mappings[name]
        if source_name is None:
            runtime_weights.append(0.0)
            runtime_mean.append(0.0)
            runtime_std.append(1.0)
            continue
        index = source_index[source_name]
        runtime_weights.append(float(weights[index + 1]))
        runtime_mean.append(float(mean[index]))
        runtime_std.append(float(std[index]) if std[index] > 0 else 1.0)

    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "hybrid_lr_model.npz"
    np.savez(
        artifact,
        weights=np.asarray(runtime_weights, dtype=float),
        mean=np.asarray(runtime_mean, dtype=float),
        std=np.asarray(runtime_std, dtype=float),
        feature_names=np.asarray(RUNTIME_FEATURES),
    )
    manifest = {
        "packageVersion": "recommendation-package-v1",
        "modelVersion": "hybrid-ranking-v1",
        "featureSchemaVersion": "recommendation-features-v2",
        "inferenceContractVersion": "recommendation-inference-v1",
        "modelKeyVersion": "hmac-sha256-v1",
        "approvedFor": ["local"],
        "artifact": artifact.name,
        "artifactSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "featureNames": list(RUNTIME_FEATURES),
        "createdAt": datetime.now(UTC).isoformat(),
        "sourceArtifact": str(source),
        "sourceArtifactSha256": source_digest,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("artifacts/candidate/hybrid_lr_model.npz"))
    parser.add_argument("--output", type=Path, default=Path(".tmp/runtime/model-package"))
    options = parser.parse_args()
    build(options.source, options.output)
