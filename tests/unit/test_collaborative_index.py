import json
import tempfile
import unittest
from pathlib import Path

from foodmind_ml.collaborative.index import INDEX_SCHEMA_VERSION, build_index, write_index


class CollaborativeIndexTest(unittest.TestCase):
    def test_builds_deterministic_positive_only_user_and_item_signals(self):
        rows = []
        for user, meals in {
            "user-0": ("meal-a", "meal-b"),
            "user-1": ("meal-a", "meal-b", "meal-c"),
            "user-2": ("meal-a", "meal-b", "meal-c"),
            "user-3": ("meal-a", "meal-b", "meal-c"),
        }.items():
            rows.extend({"userKey": user, "mealKey": meal, "collaborativeStrength": 1.0} for meal in meals)
        # A rejected event is deliberately ignored, not treated as a negative rating.
        rows.append({"userKey": "user-0", "mealKey": "meal-c", "collaborativeStrength": 0.0})
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot.ndjson"
            snapshot.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            first = build_index(snapshot)
            output = Path(directory) / "index.json"
            second = write_index(snapshot, output)

        self.assertEqual(first, second)
        self.assertEqual(first["schemaVersion"], INDEX_SCHEMA_VERSION)
        self.assertTrue(first["positiveOnly"])
        self.assertEqual(first["userCf"]["user-0"]["meal-c"]["support"], 3)
        self.assertEqual(first["itemCf"]["user-0"]["meal-c"]["support"], 2)
        self.assertGreater(first["userCf"]["user-0"]["meal-c"]["score"], 0)


if __name__ == "__main__":
    unittest.main()
