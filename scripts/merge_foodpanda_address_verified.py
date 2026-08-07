"""Merge address-verified Foodpanda browser snapshots into collected menus.

Rows whose actual Foodpanda address matches the target branch become exact
branch rows. Same-brand rows that jumped to another branch are retained as
brand reference menus, with branch status preserved.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/interim/foodpanda_menus_address_verified.csv"
VALIDATION_PATH = ROOT / "data/interim/foodpanda_address_verified_branch_validation.csv"
MAIN_PATH = ROOT / "data/interim/restaurant_menu_collected.csv"

OUTPUT_FIELDS = [
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

MANAGED_SOURCES = {
    "foodpanda_address_verified_exact",
    "foodpanda_address_verified_brand_reference",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in OUTPUT_FIELDS} for row in rows])


def main() -> int:
    raw_rows = [row for row in read_csv(RAW_PATH) if row.get("raw_menu_item_name")]
    validation = {row["target_id"]: row for row in read_csv(VALIDATION_PATH)}
    main_rows = [row for row in read_csv(MAIN_PATH) if row.get("source") not in MANAGED_SOURCES]

    converted = []
    seen = set()
    for row in raw_rows:
        rid = row.get("restaurant_id", "")
        status = validation.get(rid, {}).get("branch_match_status", "brand_menu_wrong_or_uncertain_branch")
        exact = status == "target_branch_match"
        source = "foodpanda_address_verified_exact" if exact else "foodpanda_address_verified_brand_reference"
        name = row.get("raw_menu_item_name", "")
        key = (rid, name, row.get("raw_price", ""), row.get("raw_category", ""), source)
        if key in seen:
            continue
        seen.add(key)
        converted.append(
            {
                "restaurant_id": rid,
                "restaurant_name": row.get("restaurant_name", ""),
                "menu_item_name": name,
                "raw_menu_item_name": name,
                "price_sgd": row.get("price_sgd", ""),
                "price_note": row.get("price_note", ""),
                "category": row.get("raw_category", ""),
                "source": source,
                "source_url": row.get("source_url", ""),
                "source_file": "",
                "collected_at": row.get("collected_at", ""),
                "address": row.get("address", ""),
                "latitude": row.get("latitude", ""),
                "longitude": row.get("longitude", ""),
                "restaurant_rating": row.get("restaurant_rating", ""),
                "branch_match_status": status,
                "notes": (
                    "Foodpanda browser snapshot after manual human verification; target branch address matched"
                    if exact
                    else "Foodpanda browser snapshot after manual human verification; same brand but actual branch differs from target"
                ),
            }
        )

    write_csv(MAIN_PATH, main_rows + converted)
    print(f"kept_other_rows: {len(main_rows)}")
    print(f"merged_foodpanda_rows: {len(converted)}")
    print(f"exact_rows: {sum(1 for row in converted if row['source'] == 'foodpanda_address_verified_exact')}")
    print(f"brand_reference_rows: {sum(1 for row in converted if row['source'] == 'foodpanda_address_verified_brand_reference')}")
    print(f"restaurant_menu_collected_rows: {len(main_rows) + len(converted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
