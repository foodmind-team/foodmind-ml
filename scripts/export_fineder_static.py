"""Export FINeDER restaurant and set-menu data to CSV.

FINeDER renders its public restaurant catalog into the Next.js
``__NEXT_DATA__`` payload on the home page. This script reads that payload
directly, so it does not need Playwright or a browser.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


BASE_URL = "https://fineder.sg"
NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)


def fetch_homepage(url: str = BASE_URL) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "FoodMind academic dataset collection; "
                "contact via project owner; low-frequency request"
            )
        },
    )
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def extract_restaurants(page_html: str) -> list[dict[str, Any]]:
    match = NEXT_DATA_RE.search(page_html)
    if not match:
        raise ValueError("Could not find Next.js __NEXT_DATA__ payload")

    payload = json.loads(html.unescape(match.group(1)))
    restaurants = payload["props"]["pageProps"]["restaurants"]
    if not isinstance(restaurants, list):
        raise ValueError("Unexpected FINeDER payload: restaurants is not a list")
    return restaurants


def join_list(values: Any) -> str:
    if not values:
        return ""
    return "|".join(str(value) for value in values)


def day_list(values: Any) -> str:
    if not values:
        return ""
    day_names = {
        0: "Sunday",
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
        7: "Public Holiday",
    }
    return "|".join(day_names.get(int(value), str(value)) for value in values)


def normalise_price(raw_price: Any) -> tuple[Any, str]:
    if raw_price in (None, ""):
        return "", "missing_on_source"
    try:
        price = float(raw_price)
    except (TypeError, ValueError):
        return "", "invalid_on_source"
    if price <= 0:
        return "", "missing_on_source"
    if price.is_integer():
        return int(price), "fixed"
    return price, "fixed"


def restaurant_rows(restaurants: list[dict[str, Any]], collected_at: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for restaurant in restaurants:
        location = restaurant.get("location") or {}
        rows.append(
            {
                "fineder_restaurant_id": restaurant.get("id", ""),
                "restaurant_name": restaurant.get("name", ""),
                "cuisine": join_list(restaurant.get("cuisine")),
                "michelin_stars": restaurant.get("michelinStars", ""),
                "google_rating": restaurant.get("googleRating", ""),
                "address": restaurant.get("address", ""),
                "latitude": location.get("lat", ""),
                "longitude": location.get("lon", ""),
                "contact_number": restaurant.get("contactNumber", ""),
                "website": restaurant.get("website", ""),
                "reservation_provider": restaurant.get("reservationProvider", ""),
                "reservation_url": restaurant.get("reservationUrl", ""),
                "google_maps_link": restaurant.get("googleMapsLink", ""),
                "dress_code": restaurant.get("dressCode", ""),
                "halal": restaurant.get("halal", ""),
                "has_alcohol_pairing": restaurant.get("hasAlcoholPairing", ""),
                "has_vegetarian_sets": restaurant.get("hasVegetarianSets", ""),
                "private_room_capacity": restaurant.get("privateRoomCapacity", ""),
                "asias50best": restaurant.get("asias50Best", ""),
                "worlds50best": restaurant.get("worlds50Best", ""),
                "lunch_days": day_list(restaurant.get("lunchDays")),
                "dinner_days": day_list(restaurant.get("dinnerDays")),
                "source_url": BASE_URL,
                "collected_at": collected_at,
            }
        )
    return rows


def menu_rows(restaurants: list[dict[str, Any]], collected_at: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for restaurant in restaurants:
        base = {
            "fineder_restaurant_id": restaurant.get("id", ""),
            "restaurant_name": restaurant.get("name", ""),
            "cuisine": join_list(restaurant.get("cuisine")),
            "michelin_stars": restaurant.get("michelinStars", ""),
            "google_rating": restaurant.get("googleRating", ""),
            "address": restaurant.get("address", ""),
            "website": restaurant.get("website", ""),
        }
        for meal_period, sets_key, days_key in (
            ("lunch", "lunchSets", "lunchDays"),
            ("dinner", "dinnerSets", "dinnerDays"),
        ):
            for item in restaurant.get(sets_key) or []:
                price_sgd, price_note = normalise_price(item.get("price"))
                rows.append(
                    {
                        **base,
                        "menu_level": "set_menu",
                        "meal_period": meal_period,
                        "available_days": day_list(restaurant.get(days_key)),
                        "menu_name": item.get("name", ""),
                        "raw_price_sgd": item.get("price", ""),
                        "price_sgd": price_sgd,
                        "price_note": price_note,
                        "menu_link": item.get("link", ""),
                        "expires_at": item.get("expiresAt", ""),
                        "source_url": BASE_URL,
                        "collected_at": collected_at,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/raw/fineder_export_package",
        help="Directory where CSV files will be written.",
    )
    parser.add_argument(
        "--source-url",
        default=BASE_URL,
        help="FINeDER page containing the Next.js data payload.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    collected_at = date.today().isoformat()
    restaurants = extract_restaurants(fetch_homepage(args.source_url))

    output_dir = Path(args.output_dir)
    restaurants_path = output_dir / "fineder_restaurants.csv"
    menus_path = output_dir / "fineder_menus.csv"

    restaurants_csv = restaurant_rows(restaurants, collected_at)
    menus_csv = menu_rows(restaurants, collected_at)
    write_csv(restaurants_path, restaurants_csv)
    write_csv(menus_path, menus_csv)

    print(f"restaurants: {len(restaurants_csv)} -> {restaurants_path}")
    print(f"menus: {len(menus_csv)} -> {menus_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
