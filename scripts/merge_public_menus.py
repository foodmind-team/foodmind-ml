"""Merge public web menu extracts into the collected restaurant menu table."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "data/interim/restaurant_menu_collected.csv"
PUBLIC_PATH = ROOT / "data/interim/public_web_menus_raw.csv"
TARGETS_PATH = ROOT / "data/external/menu_search_targets_curated.csv"

MAIN_FIELDS = [
    "restaurant_id",
    "restaurant_name",
    "menu_item_name",
    "raw_menu_item_name",
    "price_sgd",
    "price_note",
    "category",
    "source",
    "source_url",
    "source_file",
    "collected_at",
    "address",
    "latitude",
    "longitude",
    "restaurant_rating",
    "branch_match_status",
    "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def main() -> int:
    main_rows = [row for row in read_csv(MAIN_PATH) if row.get("source") != "public_web_reference"]
    public_rows = read_csv(PUBLIC_PATH)
    added = []
    public_ids = set()
    seen = set()
    for row in public_rows:
        name = (row.get("raw_menu_item_name") or "").strip()
        if not name:
            continue
        rid = (row.get("restaurant_id") or "").strip()
        key = (rid, name, row.get("price_sgd", ""), row.get("raw_category", ""), row.get("source_url", ""))
        if key in seen:
            continue
        seen.add(key)
        public_ids.add(rid)
        note_parts = []
        if row.get("source_type"):
            note_parts.append(f"source_type={row['source_type']}")
        if row.get("notes"):
            note_parts.append(row["notes"])
        is_exact_foodpanda_outlet = (
            rid == "T026"
            and "foodpanda.sg/restaurant/ch6o/syed-cafe-nuh" in (row.get("source_url") or "")
        )
        added.append(
            {
                "restaurant_id": rid,
                "restaurant_name": (row.get("restaurant_name") or "").strip(),
                "menu_item_name": name,
                "raw_menu_item_name": name,
                "price_sgd": (row.get("price_sgd") or "").strip(),
                "price_note": (row.get("price_note") or "").strip(),
                "category": (row.get("raw_category") or "").strip(),
                "source": "public_web_reference",
                "source_url": (row.get("source_url") or "").strip(),
                "source_file": (row.get("source_file") or "").strip(),
                "collected_at": (row.get("collected_at") or "").strip(),
                "address": "",
                "latitude": "",
                "longitude": "",
                "restaurant_rating": "",
                "branch_match_status": "target_branch_match" if is_exact_foodpanda_outlet else "public_web_reference_needs_branch_review",
                "notes": "; ".join(note_parts),
            }
        )
    write_csv(MAIN_PATH, main_rows + added, MAIN_FIELDS)

    with TARGETS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        target_fields = reader.fieldnames or []
        targets = list(reader)
    for target in targets:
        target_id = target.get("target_id", "")
        if target_id in public_ids and target.get("menu_status") == "to_find":
            target["menu_status"] = "collected_public_web_reference"
            urls = sorted(
                {
                    row.get("source_url", "")
                    for row in public_rows
                    if row.get("restaurant_id") == target_id and row.get("raw_menu_item_name")
                }
            )
            target["menu_source_url"] = " | ".join(url for url in urls if url)
            target["notes"] = (
                "Public web menu rows collected; branch applicability should be reviewed "
                "before treating as exact outlet data"
            )
    write_csv(TARGETS_PATH, targets, target_fields)

    print(f"kept_non_public_rows: {len(main_rows)}")
    print(f"added_public_rows: {len(added)}")
    print(f"restaurant_menu_collected_rows: {len(main_rows) + len(added)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
