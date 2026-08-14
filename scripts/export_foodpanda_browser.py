"""Export Foodpanda menus from an already-open Chrome browser.

Start Chrome with remote debugging, complete any human check manually, then run:

  python scripts/export_foodpanda_browser.py --delay-seconds 20

The script reads only the visible public restaurant pages through the browser
DOM and writes a static menu snapshot CSV.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

DEFAULT_URLS_FILE = "data/external/foodpanda_menu_urls.csv"
DEFAULT_OUTPUT = "data/interim/foodpanda_menus_raw.csv"
DEFAULT_CDP = "http://127.0.0.1:9222"

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


def read_url_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def clean_text(value: Any) -> str:
    value = "" if value is None else str(value)
    return re.sub(r"\s+", " ", value).strip()


def parse_price(raw_price: str) -> tuple[str, str]:
    text = clean_text(raw_price)
    note = "starting_price" if text.lower().startswith("from") else "fixed"
    matches = re.findall(r"(\d+(?:\.\d{1,2})?)", text.replace(",", ""))
    if not matches:
        return "", "missing_or_unparsed"
    return matches[0], note


def merge_rows(existing_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def auto_scroll(page: Any) -> None:
    previous_height = 0
    for _ in range(8):
        height = page.evaluate("document.body.scrollHeight")
        if height == previous_height:
            break
        previous_height = height
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1200)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(500)


def set_delivery_address(page: Any, address: str) -> bool:
    address = clean_text(address)
    if not address:
        return True

    try:
        page.goto("https://www.foodpanda.sg/", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)
        page.get_by_test_id("location-search-button").click(timeout=10000)
    except Exception:
        pass

    try:
        page.wait_for_timeout(1000)
        inputs = page.locator("input")
        target = None
        for index in range(inputs.count() - 1, -1, -1):
            candidate = inputs.nth(index)
            placeholder = candidate.get_attribute("placeholder") or ""
            aria = candidate.get_attribute("aria-label") or ""
            if "address" in placeholder.lower() or "address" in aria.lower() or index == inputs.count() - 1:
                target = candidate
                break
        if target is None:
            return False
        target.click(timeout=5000)
        target.press("Control+A")
        target.fill(address)
        page.wait_for_timeout(5000)
        target.press("ArrowDown")
        page.wait_for_timeout(500)
        target.press("Enter")
        page.wait_for_timeout(6000)
        body = page.locator("body").inner_text(timeout=10000).lower()
        return "generic location" not in body and "select your address" not in body[:800]
    except Exception:
        return False


def extract_products(page: Any) -> list[dict[str, str]]:
    return page.evaluate(
        """
        () => {
          const text = (el) => el ? el.textContent.replace(/\\s+/g, ' ').trim() : '';
          const rows = [];
          const sections = [...document.querySelectorAll('[data-testid="menu-category-section"]')];
          const containers = sections.length ? sections : [document.body];
          for (const section of containers) {
            const titleEl = section.querySelector('[data-testid="menu-category-section-title"] h2, h2');
            const category = text(titleEl);
            const products = [...section.querySelectorAll('[data-testid="menu-product"]')];
            for (const product of products) {
              const nameSelector = [
                '[data-testid="menu-product-name"]',
                '[data-testid="menu-popular-tile-name"]',
              ].join(', ');
              const priceSelector = [
                '[data-testid="menu-product-price"]',
                '[data-testid="menu-popular-tile-price"]',
              ].join(', ');
              const name = text(product.querySelector(nameSelector));
              const price = text(product.querySelector(priceSelector));
              const description = text(product.querySelector('[data-testid="menu-product-description"]'));
              if (name && price) rows.push({category, name, price, description});
            }
          }
          const seen = new Set();
          return rows.filter((row) => {
            const key = `${row.category}\\t${row.name}\\t${row.price}`;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
          });
        }
        """
    )


def extract_restaurant_info(page: Any, fallback_name: str) -> dict[str, Any]:
    return page.evaluate(
        """
        (fallbackName) => {
          const text = (el) => el ? el.textContent.replace(/\\s+/g, ' ').trim() : '';
          let data = {};
          for (const script of document.querySelectorAll('script[type="application/ld+json"]')) {
            try {
              const parsed = JSON.parse(script.textContent);
              if (parsed['@type'] === 'Restaurant') {
                data = parsed;
                break;
              }
            } catch (_) {}
          }
          const address = data.address || {};
          const geo = data.geo || {};
          const rating = data.aggregateRating || {};
          const cuisine = Array.isArray(data.servesCuisine) ? data.servesCuisine.join('|') : '';
          return {
            restaurant_name: data.name || fallbackName || text(document.querySelector('h1')),
            address: address.streetAddress || '',
            postal_code: address.postalCode || '',
            latitude: geo.latitude || '',
            longitude: geo.longitude || '',
            restaurant_rating: rating.ratingValue || '',
            rating_count: rating.ratingCount || '',
            cuisine,
          };
        }
        """,
        fallback_name,
    )


def scrape_row(page: Any, input_row: dict[str, str], collected_at: str) -> list[dict[str, Any]]:
    url = input_row.get("source_url") or input_row.get("url") or ""
    restaurant_id = input_row.get("restaurant_id", "")
    fallback_name = input_row.get("restaurant_name") or input_row.get("name") or ""
    delivery_address = input_row.get("delivery_address", "")
    address_set = set_delivery_address(page, delivery_address)
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(6000)
    auto_scroll(page)

    body_text = page.locator("body").inner_text(timeout=10000)
    if "confirm you are a human" in body_text.lower() or "请确认您是人类" in body_text:
        return [
            {
                "restaurant_id": restaurant_id,
                "restaurant_name": fallback_name,
                "raw_menu_item_name": "",
                "raw_price": "",
                "raw_category": "",
                "source_url": url,
                "source_file": "",
                "collected_at": collected_at,
                "notes": "needs_manual_review: browser reached human verification page",
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
        ]

    restaurant = extract_restaurant_info(page, fallback_name)
    products = extract_products(page)
    if not products:
        return [
            {
                "restaurant_id": restaurant_id,
                "restaurant_name": restaurant.get("restaurant_name", fallback_name),
                "raw_menu_item_name": "",
                "raw_price": "",
                "raw_category": "",
                "source_url": url,
                "source_file": "",
                "collected_at": collected_at,
                "notes": "needs_manual_review: no menu products found in browser DOM",
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
        ]

    rows = []
    for product in products:
        price_sgd, price_note = parse_price(product["price"])
        rows.append(
            {
                "restaurant_id": restaurant_id,
                "restaurant_name": restaurant.get("restaurant_name", fallback_name),
                "raw_menu_item_name": product["name"],
                "raw_price": product["price"],
                "raw_category": product["category"],
                "source_url": url,
                "source_file": "",
                "collected_at": collected_at,
                "notes": (
                    "Foodpanda static browser snapshot"
                    if address_set
                    else "Foodpanda static browser snapshot; delivery address not confirmed"
                ),
                "price_sgd": price_sgd,
                "price_note": price_note,
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
    return rows


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urls-file", default=DEFAULT_URLS_FILE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cdp-url", default=DEFAULT_CDP)
    parser.add_argument("--delay-seconds", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_rows = read_url_rows(Path(args.urls_file))
    if args.offset:
        input_rows = input_rows[args.offset :]
    if args.limit:
        input_rows = input_rows[: args.limit]
    if not input_rows:
        print("No input URLs.")
        return 2

    collected_at = date.today().isoformat()
    all_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(args.cdp_url)
        context = browser.contexts[0]
        page = context.pages[0] if context.pages else context.new_page()
        for index, input_row in enumerate(input_rows, 1):
            try:
                rows = scrape_row(page, input_row, collected_at)
            except Exception as exc:
                rows = [
                    {
                        "restaurant_id": input_row.get("restaurant_id", ""),
                        "restaurant_name": input_row.get("restaurant_name", ""),
                        "raw_menu_item_name": "",
                        "raw_price": "",
                        "raw_category": "",
                        "source_url": input_row.get("source_url", ""),
                        "source_file": "",
                        "collected_at": collected_at,
                        "notes": f"needs_manual_review: browser scrape failed: {exc}",
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
                ]
            menu_count = sum(1 for row in rows if row["raw_menu_item_name"])
            restaurant_id = input_row.get("restaurant_id")
            restaurant_name = input_row.get("restaurant_name")
            print(f"[{index}/{len(input_rows)}] {restaurant_id} {restaurant_name}: {menu_count}")
            if menu_count:
                all_rows.extend(rows)
            else:
                failed_rows.extend(rows)
            if index < len(input_rows) and args.delay_seconds > 0:
                time.sleep(args.delay_seconds)

    output_path = Path(args.output)
    if all_rows:
        merged = merge_rows(read_existing_rows(output_path), all_rows)
        write_csv(output_path, merged)
    if failed_rows:
        write_csv(output_path.with_name(output_path.stem + "_browser_failed.csv"), failed_rows)

    print(f"menu_rows_this_run: {sum(1 for row in all_rows if row['raw_menu_item_name'])}")
    print(f"failed_inputs_this_run: {len(failed_rows)}")
    print(f"output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
