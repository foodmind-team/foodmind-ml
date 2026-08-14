"""Export static Foodpanda restaurant menu snapshots.

This parser only reads product cards already present in publicly accessible
restaurant HTML. It does not call private APIs, log in, bypass location gates,
or place orders.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DEFAULT_URLS_FILE = "data/external/foodpanda_menu_urls.csv"
DEFAULT_OUTPUT = "data/interim/foodpanda_menus_raw.csv"
FIELDNAMES = [
    "restaurant_id",
    "restaurant_name",
    "raw_menu_item_name",
    "raw_price",
    "raw_category",
    "source_url",
    "source_file",
    "collected_at",
    "notes",
    "price_sgd",
    "price_note",
    "description",
    "address",
    "postal_code",
    "latitude",
    "longitude",
    "restaurant_rating",
    "rating_count",
    "cuisine",
]

SECTION_RE = re.compile(r'<div class="[^"]*\bdish-category-section\b[^"]*"[^>]*data-testid="menu-category-section"')
TITLE_RE = re.compile(r"<h2[^>]*>(.*?)</h2>", re.DOTALL)
PRODUCT_RE = re.compile(
    r'<li class="[^"]*\bproduct-tile\b[^"]*"[^>]*data-testid="menu-product".*?</li>',
    re.DOTALL,
)
NAME_RE = re.compile(
    r'data-testid="(?:menu-product-name|menu-popular-tile-name)"[^>]*>(.*?)</',
    re.DOTALL,
)
PRICE_RE = re.compile(
    r'data-testid="(?:menu-product-price|menu-popular-tile-price)"[^>]*>(.*?)</',
    re.DOTALL,
)
DESCRIPTION_RE = re.compile(
    r'data-testid="menu-product-description"[^>]*>(.*?)</',
    re.DOTALL,
)
JSON_LD_RE = re.compile(
    r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>',
    re.DOTALL,
)


def clean_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def fetch(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0 Safari/537.36 FoodMind academic static snapshot"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-SG,en;q=0.9",
        },
    )
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_price(value: str) -> tuple[str, str]:
    text = clean_text(value)
    note = "fixed"
    if text.lower().startswith("from"):
        note = "starting_price"
    match = re.search(r"(\d+(?:\.\d{1,2})?)", text.replace(",", ""))
    if not match:
        return "", "missing_or_unparsed"
    return match.group(1), note


def parse_restaurant_json_ld(page_html: str) -> dict[str, Any]:
    for match in JSON_LD_RE.finditer(page_html):
        try:
            data = json.loads(clean_text(match.group(1)))
        except json.JSONDecodeError:
            continue
        if data.get("@type") != "Restaurant":
            continue
        address = data.get("address") or {}
        geo = data.get("geo") or {}
        rating = data.get("aggregateRating") or {}
        return {
            "restaurant_name": data.get("name", ""),
            "address": address.get("streetAddress", ""),
            "postal_code": address.get("postalCode", ""),
            "latitude": geo.get("latitude", ""),
            "longitude": geo.get("longitude", ""),
            "restaurant_rating": rating.get("ratingValue", ""),
            "rating_count": rating.get("ratingCount", ""),
            "cuisine": "|".join(data.get("servesCuisine") or []),
        }
    return {}


def parse_sections(page_html: str) -> list[tuple[str, str]]:
    starts = [match.start() for match in SECTION_RE.finditer(page_html)]
    sections: list[tuple[str, str]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(page_html)
        block = page_html[start:end]
        title_match = TITLE_RE.search(block)
        title = clean_text(title_match.group(1)) if title_match else ""
        sections.append((title, block))
    return sections


def parse_products(page_html: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    sections = parse_sections(page_html)

    if not sections:
        sections = [("", page_html)]

    for category, section_html in sections:
        for product_match in PRODUCT_RE.finditer(section_html):
            product_html = product_match.group(0)
            name_match = NAME_RE.search(product_html)
            price_match = PRICE_RE.search(product_html)
            if not name_match or not price_match:
                continue
            name = clean_text(name_match.group(1))
            raw_price = clean_text(price_match.group(1))
            price_sgd, price_note = parse_price(raw_price)
            description_match = DESCRIPTION_RE.search(product_html)
            description = clean_text(description_match.group(1)) if description_match else ""

            key = (category, name, price_sgd)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "raw_category": category,
                    "raw_menu_item_name": name,
                    "raw_price": raw_price,
                    "price_sgd": price_sgd,
                    "price_note": price_note,
                    "description": description,
                }
            )
    return rows


def read_url_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def restaurant_id_from_url(url: str, fallback_index: int) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    if len(parts) >= 2 and parts[0] == "restaurant":
        return f"FP_{parts[1].upper()}"
    return f"FP_{fallback_index:03d}"


def export_rows(input_rows: list[dict[str, str]], delay_seconds: float) -> list[dict[str, Any]]:
    collected_at = date.today().isoformat()
    output_rows: list[dict[str, Any]] = []

    for index, input_row in enumerate(input_rows, 1):
        url = input_row.get("source_url") or input_row.get("url") or ""
        if not url:
            continue
        restaurant_id = input_row.get("restaurant_id") or restaurant_id_from_url(url, index)
        restaurant_name = input_row.get("restaurant_name") or input_row.get("name") or ""

        try:
            page_html = fetch(url)
            restaurant = parse_restaurant_json_ld(page_html)
            products = parse_products(page_html)
            restaurant_name = restaurant_name or restaurant.get("restaurant_name") or ""
        except Exception as exc:
            output_rows.append(
                {
                    "restaurant_id": restaurant_id,
                    "restaurant_name": restaurant_name,
                    "raw_menu_item_name": "",
                    "raw_price": "",
                    "raw_category": "",
                    "source_url": url,
                    "source_file": "",
                    "collected_at": collected_at,
                    "notes": f"needs_manual_review: fetch failed: {exc}",
                    "price_sgd": "",
                    "price_note": "",
                    "description": "",
                    "address": "",
                    "postal_code": "",
                    "latitude": "",
                    "longitude": "",
                    "restaurant_rating": "",
                    "rating_count": "",
                    "cuisine": "",
                }
            )
            if index < len(input_rows) and delay_seconds > 0:
                time.sleep(delay_seconds)
            continue

        if not products:
            output_rows.append(
                {
                    "restaurant_id": restaurant_id,
                    "restaurant_name": restaurant_name,
                    "raw_menu_item_name": "",
                    "raw_price": "",
                    "raw_category": "",
                    "source_url": url,
                    "source_file": "",
                    "collected_at": collected_at,
                    "notes": "needs_manual_review: no public product cards parsed",
                    "price_sgd": "",
                    "price_note": "",
                    "description": "",
                    "address": restaurant.get("address", ""),
                    "postal_code": restaurant.get("postal_code", ""),
                    "latitude": restaurant.get("latitude", ""),
                    "longitude": restaurant.get("longitude", ""),
                    "restaurant_rating": restaurant.get("restaurant_rating", ""),
                    "rating_count": restaurant.get("rating_count", ""),
                    "cuisine": restaurant.get("cuisine", ""),
                }
            )
            if index < len(input_rows) and delay_seconds > 0:
                time.sleep(delay_seconds)
            continue

        for product in products:
            output_rows.append(
                {
                    "restaurant_id": restaurant_id,
                    "restaurant_name": restaurant_name,
                    "raw_menu_item_name": product["raw_menu_item_name"],
                    "raw_price": product["raw_price"],
                    "raw_category": product["raw_category"],
                    "source_url": url,
                    "source_file": "",
                    "collected_at": collected_at,
                    "notes": "Foodpanda static public page snapshot",
                    "price_sgd": product["price_sgd"],
                    "price_note": product["price_note"],
                    "description": product["description"],
                    "address": restaurant.get("address", ""),
                    "postal_code": restaurant.get("postal_code", ""),
                    "latitude": restaurant.get("latitude", ""),
                    "longitude": restaurant.get("longitude", ""),
                    "restaurant_rating": restaurant.get("restaurant_rating", ""),
                    "rating_count": restaurant.get("rating_count", ""),
                    "cuisine": restaurant.get("cuisine", ""),
                }
            )

        if index < len(input_rows) and delay_seconds > 0:
            time.sleep(delay_seconds)

    return output_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def merge_success_rows(existing_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in existing_rows + new_rows:
        if not row.get("raw_menu_item_name"):
            continue
        key = (
            row.get("restaurant_id", ""),
            row.get("raw_menu_item_name", ""),
            row.get("raw_price", ""),
            row.get("raw_category", ""),
        )
        merged[key] = {field: row.get(field, "") for field in FIELDNAMES}
    return list(merged.values())


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls", nargs="*", help="Foodpanda restaurant URLs to export.")
    parser.add_argument("--urls-file", default=DEFAULT_URLS_FILE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--delay-seconds", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=0, help="Maximum input rows to request this run.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many input rows before requesting.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_rows = read_url_rows(Path(args.urls_file))
    input_rows.extend({"source_url": url} for url in args.urls)
    if args.offset:
        input_rows = input_rows[args.offset :]
    if args.limit:
        input_rows = input_rows[: args.limit]
    if not input_rows:
        print(f"No URLs provided. Add rows to {args.urls_file} or pass URLs as arguments.")
        return 2

    rows = export_rows(input_rows, args.delay_seconds)
    menu_row_count = sum(1 for row in rows if row["raw_menu_item_name"])
    review_row_count = sum(1 for row in rows if not row["raw_menu_item_name"])
    output_path = Path(args.output)
    if menu_row_count:
        rows = merge_success_rows(read_existing_rows(output_path), rows)
        write_csv(output_path, rows)
        written_path = output_path
    else:
        written_path = output_path.with_name(output_path.stem + "_failed.csv")
        write_csv(written_path, rows)

    print(f"restaurants_or_inputs: {len(input_rows)}")
    print(f"menu_rows: {menu_row_count}")
    print(f"review_rows: {review_row_count}")
    print(f"output: {written_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
