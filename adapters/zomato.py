#!/usr/bin/env python3
"""
adapters/zomato.py

TrueRating dataset adapter for the Kaggle "Zomato Bangalore Restaurants"
dataset (zomato.csv: url, address, name, online_order, book_table, rate,
votes, phone, location, rest_type, dish_liked, cuisines,
approx_cost(for two people), reviews_list, menu_item, listed_in(type),
listed_in(city)).

Format quirks this adapter handles:
  - Reviews live inside a single `reviews_list` column per row, as a
    stringified Python list of ("Rated X.X", "RATED\\n <text>") tuples.
  - The same physical restaurant appears on MULTIPLE rows (once per
    listed_in(type) category such as Delivery / Dine-out / Buffet), and
    those duplicate rows do NOT always carry an identical reviews_list --
    some carry only a subset. So restaurant identity is (name, address),
    and every row's reviews are unioned + de-duplicated by exact review
    text before being handed off.
  - There is no per-review date. ReviewRecord.date is left as None.
  - `rate` ("4.1/5", or "NEW"/"-"/blank for unrated places) is a real,
    independent crowd rating, preserved as RestaurantRecord.reference_rating
    specifically so evaluate.py can validate TrueRating's own computed
    score against it later.

Unlike the Yelp adapter, this one does NOT need to pre-sample for memory
reasons -- the whole file is a few hundred MB and reviews are already
scoped per row, so it hands ingest.py the FULL de-duplicated candidate
pool and lets the shared filter-and-sample step do the real selection.
"""

import ast
import csv
import re
import sys
from collections import defaultdict
from typing import List

from adapters.base import BaseDatasetAdapter, RestaurantRecord, ReviewRecord

RATE_PATTERN = re.compile(r"([\d.]+)\s*/\s*5")
RATED_PATTERN = re.compile(r"Rated\s+([\d.]+)", re.IGNORECASE)

csv.field_size_limit(sys.maxsize)


class ZomatoAdapter(BaseDatasetAdapter):
    name = "zomato"
    description = "Kaggle Zomato Bangalore Restaurants dataset (zomato.csv)"

    def add_cli_arguments(self, parser) -> None:
        parser.add_argument("--input", default="zomato.csv", help="Path to zomato.csv")

    def load(self, args) -> List[RestaurantRecord]:
        groups, header_idx = self._load_and_group_rows(args.input)
        records = self._build_restaurant_records(groups, header_idx)
        return records

    # -- internal helpers -----------------------------------------------
    @staticmethod
    def _parse_rate(raw: str):
        if not raw:
            return None
        match = RATE_PATTERN.search(raw)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _parse_cost(raw: str):
        if not raw:
            return None
        cleaned = raw.replace(",", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    @staticmethod
    def _parse_votes(raw: str) -> int:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _parse_reviews_list(cls, raw: str):
        if not raw or raw.strip() in ("", "[]"):
            return []
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []

        reviews = []
        for entry in parsed:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            rating_str, text_str = entry[0], entry[1]

            stars = None
            if isinstance(rating_str, str):
                m = RATED_PATTERN.search(rating_str)
                if m:
                    try:
                        stars = float(m.group(1))
                    except ValueError:
                        stars = None

            text = text_str if isinstance(text_str, str) else ""
            text = re.sub(r"^\s*RATED\s*", "", text, flags=re.IGNORECASE).strip()

            if text:
                reviews.append((stars, text))

        return reviews

    def _load_and_group_rows(self, input_file: str):
        groups = defaultdict(list)
        total_rows = 0

        try:
            with open(input_file, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader)
                idx = {name: i for i, name in enumerate(header)}

                required = [
                    "name", "address", "reviews_list", "rate", "votes",
                    "cuisines", "rest_type", "location",
                    "approx_cost(for two people)",
                ]
                missing = [col for col in required if col not in idx]
                if missing:
                    raise ValueError(f"Input CSV is missing expected columns: {missing}")

                for row in reader:
                    total_rows += 1
                    if total_rows % 10_000 == 0:
                        print(f"[ZOMATO] Scanned {total_rows:,} rows, "
                              f"{len(groups):,} distinct restaurants so far...")

                    key = (row[idx["name"]].strip(), row[idx["address"]].strip())
                    if not key[0]:
                        continue
                    groups[key].append(row)
        except FileNotFoundError:
            print(f"[ZOMATO][ERROR] File not found: {input_file}", file=sys.stderr)
            raise
        except OSError as exc:
            print(f"[ZOMATO][ERROR] Could not read {input_file}: {exc}", file=sys.stderr)
            raise

        print(f"[ZOMATO] Finished scanning {total_rows:,} rows -> "
              f"{len(groups):,} distinct restaurants (by name + address).")
        return groups, idx

    def _build_restaurant_records(self, groups: dict, header_idx: dict) -> List[RestaurantRecord]:
        records = []

        for (name, address), rows in groups.items():
            categories = None
            city = None
            approx_cost = None
            votes_total = 0
            reference_rating = None
            seen_review_texts = set()
            deduped_reviews = []

            for row in rows:
                if categories is None and row[header_idx["cuisines"]].strip():
                    categories = row[header_idx["cuisines"]].strip()
                    rest_type = row[header_idx["rest_type"]].strip()
                    if rest_type:
                        categories = f"{categories} | {rest_type}"

                if city is None and row[header_idx["location"]].strip():
                    city = row[header_idx["location"]].strip()

                if approx_cost is None:
                    approx_cost = self._parse_cost(row[header_idx["approx_cost(for two people)"]])

                if reference_rating is None:
                    reference_rating = self._parse_rate(row[header_idx["rate"]])

                votes_total = max(votes_total, self._parse_votes(row[header_idx["votes"]]))

                for stars, text in self._parse_reviews_list(row[header_idx["reviews_list"]]):
                    if text not in seen_review_texts:
                        seen_review_texts.add(text)
                        deduped_reviews.append(ReviewRecord(text=text, stars=stars, date=None))

            records.append(
                RestaurantRecord(
                    name=name,
                    categories=categories,
                    city=city,
                    reference_rating=reference_rating,
                    votes=votes_total,
                    approx_cost=approx_cost,
                    reviews=deduped_reviews,
                )
            )

        return records
