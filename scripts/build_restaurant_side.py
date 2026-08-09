"""Build restaurant-side datasets for FoodMind.

This script turns the collected restaurant/menu snapshots into model-ready CSVs:

- data/processed/restaurants.csv
- data/processed/restaurant_menu.csv
- data/processed/menu_dish_mapping.csv
- data/processed/restaurant_data_quality_report.csv

It keeps uncertain menu sources visible instead of silently treating them as
exact branch data.
"""

from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
TARGETS_PATH = ROOT / "data/external/menu_search_targets_curated.csv"
MENU_PATH = ROOT / "data/interim/restaurant_menu_collected.csv"
DISHES_PATH = ROOT / "data/processed/dishes.csv"
NEA_PATH = ROOT / "data/external/nea_menu_search_candidates.csv"
OUT_DIR = ROOT / "data/processed"


RESTAURANTS_OUT = OUT_DIR / "restaurants.csv"
RESTAURANT_MENU_OUT = OUT_DIR / "restaurant_menu.csv"
MENU_DISH_MAPPING_OUT = OUT_DIR / "menu_dish_mapping.csv"
QUALITY_REPORT_OUT = OUT_DIR / "restaurant_data_quality_report.csv"
COLLECTION_BACKLOG_OUT = OUT_DIR / "restaurant_collection_backlog.csv"


KNOWN_LOCATION_COORDS = {
    "1 Create Way Singapore 138602": ("1.30399654877953", "103.774032940413", "onemap_known_result"),
    "1 Lower Kent Ridge Road Singapore 119082": ("1.29494705784662", "103.784879314406", "onemap_known_result"),
    "3155 Commonwealth Avenue West Singapore 129588": ("1.31509768025918", "103.764313438645", "onemap_known_result"),
    "321 Clementi Avenue 3 Singapore 129905": ("1.31158680791682", "103.76490400713", "onemap_known_result"),
    "2 Science Park Drive Singapore 118222": ("1.29085322001822", "103.784631727374", "onemap_known_result"),
    "University Sports Centre NUS": ("1.300793", "103.775817", "manual_landmark_estimate"),
    "Engineering Block E4 NUS": ("1.299166", "103.770409", "manual_landmark_estimate"),
    "The Ridge NUS": ("1.296775", "103.772149", "manual_landmark_estimate"),
    "CDE NUS": ("1.300300", "103.771875", "manual_landmark_estimate"),
    "Ventus NUS": ("1.295725", "103.771494", "manual_landmark_estimate"),
}


ALIASES = {
    "chicken rice": "hainanese chicken rice",
    "hainan chicken rice": "hainanese chicken rice",
    "hainanese chicken": "hainanese chicken rice",
    "laksa": "laksa",
    "nasi lemak": "nasi lemak",
    "beef rendang": "beef rendang",
    "mapo tofu": "mapo tofu",
    "ma po tofu": "mapo tofu",
    "fish and chips": "fish chips",
    "fish & chips": "fish chips",
    "mac and cheese": "macaroni and cheese",
    "mac & cheese": "macaroni and cheese",
}


NON_MAIN_CATEGORY_WORDS = {
    "drink",
    "drinks",
    "beverage",
    "beverages",
    "coffee",
    "tea",
    "lemonade",
    "soda",
    "dessert",
    "desserts",
    "cake",
    "cakes",
    "ice cream",
    "gelato",
    "side",
    "sides",
    "fries",
    "add on",
    "addon",
    "snack",
    "snacks",
    "sauce",
}


MAIN_CATEGORY_HINTS = {
    "main",
    "mains",
    "pasta",
    "burger",
    "burgers",
    "sandwich",
    "sandwiches",
    "rice",
    "noodle",
    "noodles",
    "soup",
    "paofan",
    "ramen",
    "spaghetti",
    "steak",
    "chop",
    "cutlet",
    "chicken",
    "beef",
    "seafood",
    "fish",
    "set",
    "combo",
    "pasta ala carte",
}


CUISINE_RULES = [
    ("korean", ("korean", "hae bok", "ha jun", "jinjja", "vons")),
    ("japanese", ("udon", "sushi", "ichiban", "ramen", "saizeriya")),
    ("chinese", ("crystal jade", "soup restaurant", "xin wang", "mei heong", "kopitiam")),
    ("singaporean", ("nasi lemak", "crave", "laksa", "saap saap", "syed", "masta")),
    ("western", ("burger", "subway", "royals", "sloppy", "delifrance", "toasties", "jewel", "starbucks")),
    ("thai", ("thai", "charcoal thai", "saap saap")),
    ("indian", ("dabba", "syed")),
    ("dessert", ("burnt cones", "polar", "mei heong", "dessert")),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fieldnames} for row in rows])


def clean_text(value: Any) -> str:
    value = "" if value is None else str(value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(value).lower()).strip()


def slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_text(value).lower())


def price_float(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    try:
        price = float(text)
    except ValueError:
        match = re.search(r"(\d+(?:\.\d{1,2})?)", text.replace(",", ""))
        if not match:
            return ""
        price = float(match.group(1))
    if price <= 0:
        return ""
    return f"{price:.2f}"


def infer_cuisine(name: str, category: str = "") -> str:
    haystack = norm(f"{name} {category}")
    matches = []
    for cuisine, words in CUISINE_RULES:
        if any(word in haystack for word in words):
            matches.append(cuisine)
    return "|".join(dict.fromkeys(matches))


def classify_menu_item(category: str, name: str) -> tuple[str, str, str, str, str]:
    c = norm(category)
    n = norm(name)
    joined = f"{c} {n}"
    tokens = set(joined.split())
    is_drink = (
        bool(tokens & {"drink", "drinks", "beverage", "beverages", "coffee", "tea", "latte", "soda", "lemonade", "juice", "water", "kopi", "teh", "milo"})
        or any(phrase in joined for phrase in ("iced lemon tea", "soya milk", "grass jelly"))
    )
    savory_tokens = {"crab", "fish", "chicken", "beef", "salmon", "prawn", "seafood", "meat", "burger", "rice", "noodle", "spaghetti"}
    is_dessert = (
        bool(tokens & {"dessert", "desserts", "cake", "cakes", "brownie", "pudding", "gelato", "tart", "tarts"})
        or any(phrase in joined for phrase in ("ice cream", "snow ice", "ice kacang"))
    ) and not bool(tokens & savory_tokens)
    is_add_on = any(word in c for word in ("add on", "addon", "extra", "topping")) or n in {"sauce", "topping"}
    is_side = any(word in joined for word in ("side", "fries", "garlic bread", "tater tots", "wings", "samosa", "springroll"))
    is_main = (
        not is_drink
        and not is_dessert
        and not is_add_on
        and any(word in joined for word in MAIN_CATEGORY_HINTS)
    )
    if is_drink:
        group = "drink"
    elif is_dessert:
        group = "dessert"
    elif is_add_on:
        group = "add_on"
    elif is_side:
        group = "side"
    elif is_main:
        group = "main_dish"
    else:
        group = "needs_review"
    return str(is_main).lower(), str(is_drink).lower(), str(is_dessert).lower(), str(is_add_on or is_side).lower(), group


def canonical_menu_name(name: str) -> str:
    cleaned = norm(name)
    remove_words = [
        "signature",
        "classic",
        "chef s",
        "chef",
        "best seller",
        "popular",
        "set",
        "combo",
        "large",
        "small",
        "regular",
        "meal",
    ]
    for word in remove_words:
        cleaned = re.sub(rf"\b{re.escape(word)}\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return ALIASES.get(cleaned, cleaned)


def source_reliability(source: str, branch_status: str) -> str:
    if branch_status == "target_branch_match":
        return "high"
    if source == "manual_menu_image":
        return "high"
    if "needs_review" in branch_status:
        return "medium"
    if source == "public_web_reference":
        return "medium_brand_reference"
    return "medium"


def onemap_geocode(query: str) -> tuple[str, str, str]:
    query = clean_text(query)
    if not query:
        return "", "", "missing_address"
    if query in KNOWN_LOCATION_COORDS:
        lat, lon, status = KNOWN_LOCATION_COORDS[query]
        return lat, lon, status
    try:
        params = urlencode({"searchVal": query, "returnGeom": "Y", "getAddrDetails": "Y", "pageNum": 1})
        request = Request(
            f"https://www.onemap.gov.sg/api/common/elastic/search?{params}",
            headers={"User-Agent": "FoodMind academic dataset preparation"},
        )
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
        results = payload.get("results") or []
        if not results:
            return "", "", "not_found"
        first = results[0]
        return first.get("LATITUDE", ""), first.get("LONGITUDE", ""), "onemap_search_result"
    except Exception as exc:
        return "", "", f"geocode_failed:{type(exc).__name__}"


def build_dish_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in read_csv(DISHES_PATH):
        name = norm(row.get("dish_name"))
        dish_id = row.get("dish_id", "")
        if name and dish_id and name not in lookup:
            lookup[name] = dish_id
    return lookup


def nea_exact_matches(targets: list[dict[str, str]], nea_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    matches = {}
    for target in targets:
        target_id = target["target_id"]
        name_key = slug(target.get("display_name"))
        address_key = slug(target.get("address_hint"))
        best = None
        best_score = 0
        for nea in nea_rows:
            premises = slug(nea.get("premises_address"))
            licensee = slug(nea.get("licensee_name"))
            score = 0
            if address_key and address_key in premises:
                score += 3
            if name_key and (name_key in licensee or licensee in name_key):
                score += 2
            if target.get("location_hint") and slug(target.get("location_hint")) in premises:
                score += 1
            if score > best_score:
                best = nea
                best_score = score
        if best and best_score >= 3:
            matches[target_id] = {**best, "nea_match_score": str(best_score)}
    return matches


def build_restaurants(targets: list[dict[str, str]], menu_rows: list[dict[str, str]], nea_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    menu_by_restaurant = defaultdict(list)
    for row in menu_rows:
        menu_by_restaurant[row.get("restaurant_id", "")].append(row)
    nea_matches = nea_exact_matches(targets, nea_rows)
    restaurants = []
    for target in targets:
        rid = target["target_id"]
        rows = menu_by_restaurant.get(rid, [])
        exact_rows = [row for row in rows if row.get("branch_match_status") in {"target_branch_match", "manual_user_provided_menu"}]
        address = clean_text(target.get("address_hint") or "")
        if rows:
            row_address_counts = Counter(clean_text(row.get("address")) for row in rows if clean_text(row.get("address")))
            if row_address_counts and not address:
                address = row_address_counts.most_common(1)[0][0]
        lat, lon, geocode_status = onemap_geocode(address or target.get("location_hint", ""))
        nea = nea_matches.get(rid, {})
        restaurants.append(
            {
                "restaurant_id": rid,
                "restaurant_name": target.get("display_name", ""),
                "area": target.get("area", ""),
                "location_hint": target.get("location_hint", ""),
                "address": address,
                "latitude": lat,
                "longitude": lon,
                "geocode_status": geocode_status,
                "restaurant_rating": "",
                "cuisine": infer_cuisine(target.get("display_name", ""), target.get("source_note", "")),
                "menu_status": target.get("menu_status", ""),
                "menu_rows_total": len(rows),
                "menu_rows_exact_or_manual": len(exact_rows),
                "has_exact_branch_menu": str(bool(exact_rows)).lower(),
                "source_url": target.get("menu_source_url", ""),
                "nea_licensee_name": nea.get("licensee_name", ""),
                "nea_licence_number": nea.get("licence_number", ""),
                "nea_grade": nea.get("grade", ""),
                "nea_demerit_points": nea.get("demerit_points", ""),
                "nea_match_score": nea.get("nea_match_score", ""),
                "notes": target.get("notes", ""),
            }
        )
    return restaurants


def build_restaurant_menu(menu_rows: list[dict[str, str]], restaurants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    restaurant_lookup = {row["restaurant_id"]: row for row in restaurants}
    output = []
    seen = set()
    for index, row in enumerate(menu_rows, 1):
        rid = row.get("restaurant_id", "")
        name = clean_text(row.get("menu_item_name") or row.get("raw_menu_item_name"))
        if not rid or not name:
            continue
        price = price_float(row.get("price_sgd"))
        category = clean_text(row.get("category"))
        is_main, is_drink, is_dessert, is_add_on, group = classify_menu_item(category, name)
        canonical = canonical_menu_name(name)
        branch_status = row.get("branch_match_status", "")
        source = row.get("source", "")
        key = (rid, name, price, category, source)
        if key in seen:
            continue
        seen.add(key)
        restaurant = restaurant_lookup.get(rid, {})
        output.append(
            {
                "restaurant_menu_id": f"RM{index:06d}",
                "restaurant_id": rid,
                "restaurant_name": row.get("restaurant_name") or restaurant.get("restaurant_name", ""),
                "dish_id": "",
                "canonical_dish_name": canonical,
                "menu_item_name": name,
                "raw_menu_item_name": row.get("raw_menu_item_name", ""),
                "price_sgd": price,
                "price_note": row.get("price_note", ""),
                "category": category,
                "category_group": group,
                "is_main_dish": is_main,
                "is_drink": is_drink,
                "is_dessert": is_dessert,
                "is_add_on_or_side": is_add_on,
                "availability": "true",
                "latitude": row.get("latitude") or restaurant.get("latitude", ""),
                "longitude": row.get("longitude") or restaurant.get("longitude", ""),
                "address": row.get("address") or restaurant.get("address", ""),
                "restaurant_rating": row.get("restaurant_rating") or restaurant.get("restaurant_rating", ""),
                "branch_match_status": branch_status,
                "source": source,
                "source_reliability": source_reliability(source, branch_status),
                "source_url": row.get("source_url", ""),
                "source_file": row.get("source_file", ""),
                "collected_at": row.get("collected_at", ""),
                "needs_manual_review": str(branch_status != "target_branch_match" and source != "manual_menu_image").lower(),
                "notes": row.get("notes", ""),
            }
        )
    return output


def build_menu_dish_mapping(restaurant_menu: list[dict[str, Any]], dish_lookup: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for row in restaurant_menu:
        key = (row["restaurant_id"], row["menu_item_name"])
        if key in seen:
            continue
        seen.add(key)
        canonical = row["canonical_dish_name"]
        dish_id = ""
        method = "unmatched"
        confidence = "0.00"
        review_status = "needs_review"
        if canonical in dish_lookup:
            dish_id = dish_lookup[canonical]
            method = "exact_canonical_name"
            confidence = "1.00"
            review_status = "auto_matched"
        elif canonical in ALIASES and ALIASES[canonical] in dish_lookup:
            dish_id = dish_lookup[ALIASES[canonical]]
            method = "alias_dictionary"
            confidence = "0.95"
            review_status = "auto_matched"
        rows.append(
            {
                "restaurant_id": row["restaurant_id"],
                "restaurant_name": row["restaurant_name"],
                "menu_item_name": row["menu_item_name"],
                "canonical_dish_name": canonical,
                "dish_id": dish_id,
                "match_method": method,
                "match_confidence": confidence,
                "review_status": review_status,
                "category_group": row["category_group"],
                "source_reliability": row["source_reliability"],
            }
        )
    return rows


def build_quality_report(restaurants: list[dict[str, Any]], restaurant_menu: list[dict[str, Any]], mapping: list[dict[str, Any]]) -> list[dict[str, Any]]:
    menu_by_status = Counter(row["branch_match_status"] for row in restaurant_menu)
    menu_by_source = Counter(row["source"] for row in restaurant_menu)
    exact_restaurants = sum(1 for row in restaurants if row["has_exact_branch_menu"] == "true")
    matched_mappings = sum(1 for row in mapping if row["dish_id"])
    return [
        {"metric": "restaurants_total", "value": len(restaurants), "notes": "Targets in curated restaurant scope"},
        {"metric": "restaurants_with_any_menu", "value": sum(1 for row in restaurants if int(row["menu_rows_total"]) > 0), "notes": ""},
        {"metric": "restaurants_with_exact_or_manual_menu", "value": exact_restaurants, "notes": ""},
        {"metric": "menu_rows_total", "value": len(restaurant_menu), "notes": ""},
        {"metric": "menu_rows_main_dish", "value": sum(1 for row in restaurant_menu if row["category_group"] == "main_dish"), "notes": ""},
        {"metric": "menu_rows_target_branch_match", "value": menu_by_status.get("target_branch_match", 0), "notes": ""},
        {"metric": "menu_rows_manual", "value": menu_by_status.get("manual_user_provided_menu", 0), "notes": ""},
        {"metric": "menu_rows_public_reference", "value": menu_by_source.get("public_web_reference", 0), "notes": "Brand/reference menus need branch review"},
        {"metric": "menu_dish_mapping_rows", "value": len(mapping), "notes": ""},
        {"metric": "menu_dish_auto_matched", "value": matched_mappings, "notes": "Exact/alias matches only; fuzzy/LLM review not applied"},
        {"metric": "generated_at", "value": date.today().isoformat(), "notes": ""},
    ]


def build_collection_backlog(restaurants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in restaurants:
        total = int(row["menu_rows_total"])
        exact = int(row["menu_rows_exact_or_manual"])
        status = row["menu_status"]
        if exact > 0:
            priority = "done"
            missing_piece = "none"
            suggested_action = "Use in exact branch restaurant recommendation"
        elif total > 0:
            priority = "medium"
            missing_piece = "exact_branch_validation"
            suggested_action = "Verify outlet-specific PDF/photo/menu page before using as exact branch data"
        elif status == "foodpanda_brand_reference_only":
            priority = "high"
            missing_piece = "foodpanda_exact_branch_failed"
            suggested_action = "Collect in-store menu photo/PDF or manually set Foodpanda address and verify returned outlet address"
        else:
            priority = "high"
            missing_piece = "menu_rows"
            suggested_action = "Find official menu, mall page with prices, public PDF, or user-provided menu photo"
        rows.append(
            {
                "restaurant_id": row["restaurant_id"],
                "restaurant_name": row["restaurant_name"],
                "area": row["area"],
                "location_hint": row["location_hint"],
                "menu_status": status,
                "menu_rows_total": total,
                "menu_rows_exact_or_manual": exact,
                "priority": priority,
                "missing_piece": missing_piece,
                "suggested_action": suggested_action,
                "notes": row["notes"],
            }
        )
    priority_order = {"high": 0, "medium": 1, "done": 2}
    return sorted(rows, key=lambda item: (priority_order.get(item["priority"], 9), item["area"], item["restaurant_name"]))


def main() -> int:
    targets = read_csv(TARGETS_PATH)
    menu_rows = read_csv(MENU_PATH)
    nea_rows = read_csv(NEA_PATH)
    dish_lookup = build_dish_lookup()

    restaurants = build_restaurants(targets, menu_rows, nea_rows)
    restaurant_menu = build_restaurant_menu(menu_rows, restaurants)
    mapping = build_menu_dish_mapping(restaurant_menu, dish_lookup)
    report = build_quality_report(restaurants, restaurant_menu, mapping)
    backlog = build_collection_backlog(restaurants)

    write_csv(
        RESTAURANTS_OUT,
        restaurants,
        [
            "restaurant_id",
            "restaurant_name",
            "area",
            "location_hint",
            "address",
            "latitude",
            "longitude",
            "geocode_status",
            "restaurant_rating",
            "cuisine",
            "menu_status",
            "menu_rows_total",
            "menu_rows_exact_or_manual",
            "has_exact_branch_menu",
            "source_url",
            "nea_licensee_name",
            "nea_licence_number",
            "nea_grade",
            "nea_demerit_points",
            "nea_match_score",
            "notes",
        ],
    )
    write_csv(
        RESTAURANT_MENU_OUT,
        restaurant_menu,
        [
            "restaurant_menu_id",
            "restaurant_id",
            "restaurant_name",
            "dish_id",
            "canonical_dish_name",
            "menu_item_name",
            "raw_menu_item_name",
            "price_sgd",
            "price_note",
            "category",
            "category_group",
            "is_main_dish",
            "is_drink",
            "is_dessert",
            "is_add_on_or_side",
            "availability",
            "latitude",
            "longitude",
            "address",
            "restaurant_rating",
            "branch_match_status",
            "source",
            "source_reliability",
            "source_url",
            "source_file",
            "collected_at",
            "needs_manual_review",
            "notes",
        ],
    )
    write_csv(
        MENU_DISH_MAPPING_OUT,
        mapping,
        [
            "restaurant_id",
            "restaurant_name",
            "menu_item_name",
            "canonical_dish_name",
            "dish_id",
            "match_method",
            "match_confidence",
            "review_status",
            "category_group",
            "source_reliability",
        ],
    )
    write_csv(QUALITY_REPORT_OUT, report, ["metric", "value", "notes"])
    write_csv(
        COLLECTION_BACKLOG_OUT,
        backlog,
        [
            "restaurant_id",
            "restaurant_name",
            "area",
            "location_hint",
            "menu_status",
            "menu_rows_total",
            "menu_rows_exact_or_manual",
            "priority",
            "missing_piece",
            "suggested_action",
            "notes",
        ],
    )

    print(f"restaurants: {len(restaurants)} -> {RESTAURANTS_OUT}")
    print(f"restaurant_menu: {len(restaurant_menu)} -> {RESTAURANT_MENU_OUT}")
    print(f"menu_dish_mapping: {len(mapping)} -> {MENU_DISH_MAPPING_OUT}")
    print(f"quality_report: {len(report)} -> {QUALITY_REPORT_OUT}")
    print(f"collection_backlog: {len(backlog)} -> {COLLECTION_BACKLOG_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
