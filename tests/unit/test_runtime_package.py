import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_runtime_package import RUNTIME_FEATURES, build


class RuntimePackageTest(unittest.TestCase):
    def test_builds_inference_compatible_local_package(self):
        names = [
            "user_preference_score",
            "dish_avg_rating",
            "dish_rating_count_log",
            "distance_fit_score",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            np.savez(
                source,
                weights=np.arange(len(names) + 1, dtype=float),
                mean=np.ones(len(names)),
                std=np.ones(len(names)),
                feature_names=np.array(names),
            )
            output = root / "package"

            build(source, output)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            with np.load(output / "hybrid_lr_model.npz", allow_pickle=False) as artifact:
                self.assertEqual(tuple(artifact["feature_names"].tolist()), RUNTIME_FEATURES)
                self.assertEqual(artifact["weights"].shape, (len(RUNTIME_FEATURES) + 1,))
            self.assertEqual(manifest["featureNames"], list(RUNTIME_FEATURES))
            self.assertEqual(manifest["approvedFor"], ["local"])
            self.assertIsNone(manifest["collaborativeIndex"])
            self.assertEqual(len(manifest["sourceArtifactSha256"]), 64)

    def test_copies_only_a_versioned_collaborative_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            names = ["user_cf_score", "item_cf_score"]
            np.savez(
                source,
                weights=np.arange(len(names) + 1, dtype=float),
                mean=np.ones(len(names)),
                std=np.ones(len(names)),
                feature_names=np.array(names),
            )
            index = root / "index.json"
            index.write_text(
                json.dumps(
                    {
                        "schemaVersion": "foodmind-collaborative-index-v1",
                        "sourceSnapshotSha256": "a" * 64,
                        "positiveOnly": True,
                        "userCf": {},
                        "itemCf": {},
                    }
                ),
                encoding="utf-8",
            )
            output = root / "package"
            build(source, output, index)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["collaborativeIndex"]["artifact"], "collaborative_index.json")
            self.assertTrue(manifest["collaborativeIndex"]["positiveOnly"])
            self.assertTrue((output / "collaborative_index.json").is_file())


if __name__ == "__main__":
    unittest.main()
