#!/usr/bin/env python3
"""
ingest.py

Unified Phase 1 entry point for the TrueRating platform.

TrueRating is not tied to any one dataset: this script loads whichever
dataset you point it at into the platform's canonical `restaurants` /
`reviews` SQLite schema by delegating to a dataset-specific adapter
(adapters/yelp.py, adapters/zomato.py, ...). Every dataset-agnostic
mechanic -- schema creation, minimum-review filtering, random sampling,
stable ID assignment, batched writes, indexing -- lives here exactly
once, so adding a new dataset means writing ONE new adapter file, not
another end-to-end script.

Everything downstream (extract.py, credibility.py, score.py, rag.py,
ab_test.py, app.py, evaluate.py) only ever reads this schema and doesn't
care which adapter produced it.

Usage:
    python ingest.py yelp \
        --business-file yelp_academic_dataset_business.json \
        --review-file yelp_academic_dataset_review.json \
        --db truerating.db

    python ingest.py zomato \
        --input zomato.csv \
        --db truerating_zomato.db \
        --sample-size 500 \
        --min-reviews 10

    python ingest.py --list          # show every registered adapter
"""

import argparse
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import List

from adapters import ADAPTERS
from adapters.base import RestaurantRecord

DEFAULT_SAMPLE_SIZE = 500
DEFAULT_MIN_REVIEWS = 10
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Schema — shared by every dataset. Yelp-sourced restaurants simply leave
# reference_rating / votes / approx_cost as NULL; nothing downstream reads
# those columns, so this is fully backward compatible.
# ---------------------------------------------------------------------------
def init_database(db_path: str) -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("PRAGMA synchronous = OFF;")
        cur.execute("PRAGMA temp_store = MEMORY;")

        cur.execute("DROP TABLE IF EXISTS restaurants;")
        cur.execute("DROP TABLE IF EXISTS reviews;")

        cur.execute("""
            CREATE TABLE restaurants (
                id               TEXT PRIMARY KEY,
                name             TEXT,
                categories       TEXT,
                city             TEXT,
                reference_rating REAL,
                votes            INTEGER,
                approx_cost      REAL
            );
        """)

        cur.execute("""
            CREATE TABLE reviews (
                id            TEXT PRIMARY KEY,
                restaurant_id TEXT,
                text          TEXT,
                stars         REAL,
                date          TEXT
            );
        """)

        conn.commit()
        print("[DB] Schema initialized (restaurants, reviews).")
        return conn
    except sqlite3.Error as exc:
        print(f"[DB][ERROR] Failed to initialize database: {exc}", file=sys.stderr)
        raise


# ---------------------------------------------------------------------------
# Dataset-agnostic filtering, sampling, and persistence
# ---------------------------------------------------------------------------
def filter_and_sample(
    records: List[RestaurantRecord], min_reviews: int, sample_size: int
) -> List[RestaurantRecord]:
    candidates = [r for r in records if len(r.reviews) >= min_reviews]
    print(f"[FILTER] {len(candidates):,} of {len(records):,} restaurants have "
          f">= {min_reviews} de-duplicated reviews.")

    have_reference = sum(1 for r in candidates if r.reference_rating is not None)
    print(f"[FILTER] {have_reference:,} of those have a usable reference_rating "
          f"for later accuracy evaluation (evaluate.py --stage scoring).")

    if len(candidates) <= sample_size:
        print(f"[SAMPLE] Only {len(candidates)} candidates available; "
              f"requested {sample_size}. Using all candidates.")
        return candidates

    random.seed(RANDOM_SEED)
    sampled = random.sample(candidates, sample_size)
    print(f"[SAMPLE] Randomly sampled {len(sampled)} restaurants (seed={RANDOM_SEED}).")
    return sampled


def save_restaurants_and_reviews(
    conn: sqlite3.Connection, sampled: List[RestaurantRecord], id_prefix: str
) -> None:
    restaurant_rows = []
    review_rows = []

    for i, record in enumerate(sampled):
        restaurant_id = f"{id_prefix}_{i:05d}"
        restaurant_rows.append((
            restaurant_id,
            record.name,
            record.categories,
            record.city,
            record.reference_rating,
            record.votes,
            record.approx_cost,
        ))
        for j, review in enumerate(record.reviews):
            review_id = f"{id_prefix}_{i:05d}_rev{j:04d}"
            review_rows.append((review_id, restaurant_id, review.text, review.stars, review.date))

    try:
        cur = conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO restaurants "
            "(id, name, categories, city, reference_rating, votes, approx_cost) "
            "VALUES (?, ?, ?, ?, ?, ?, ?);",
            restaurant_rows,
        )
        conn.commit()
        print(f"Extracted {len(restaurant_rows)} restaurants.")

        BATCH_SIZE = 10_000
        for start in range(0, len(review_rows), BATCH_SIZE):
            batch = review_rows[start:start + BATCH_SIZE]
            cur.executemany(
                "INSERT OR REPLACE INTO reviews (id, restaurant_id, text, stars, date) "
                "VALUES (?, ?, ?, ?, ?);",
                batch,
            )
            conn.commit()
            print(f"[DB] Inserted {min(start + BATCH_SIZE, len(review_rows)):,} / "
                  f"{len(review_rows):,} reviews...")

    except sqlite3.Error as exc:
        print(f"[DB][ERROR] Failed to insert restaurants/reviews: {exc}", file=sys.stderr)
        raise


def build_indexes(conn: sqlite3.Connection) -> None:
    try:
        cur = conn.cursor()
        cur.execute("CREATE INDEX IF NOT EXISTS idx_reviews_restaurant_id "
                    "ON reviews(restaurant_id);")
        conn.commit()
        print("[DB] Built index: idx_reviews_restaurant_id.")
    except sqlite3.Error as exc:
        print(f"[DB][ERROR] Failed to build indexes: {exc}", file=sys.stderr)
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="TrueRating ingestion: load any supported dataset into the platform's schema."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List every registered dataset adapter and exit."
    )

    subparsers = parser.add_subparsers(dest="source", metavar="<source>")
    for adapter_name, adapter in ADAPTERS.items():
        sub = subparsers.add_parser(adapter_name, help=adapter.description)
        adapter.add_cli_arguments(sub)
        sub.add_argument("--db", default=f"truerating_{adapter_name}.db",
                          help="Output SQLite database path")
        sub.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                          help="How many restaurants to sample (default: 500)")
        sub.add_argument("--min-reviews", type=int, default=DEFAULT_MIN_REVIEWS,
                          help="Minimum de-duplicated reviews required to keep a restaurant")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.list or not args.source:
        print("Registered dataset adapters:")
        for adapter_name, adapter in ADAPTERS.items():
            print(f"  {adapter_name:10s} {adapter.description}")
        if not args.source:
            print("\nRun `python ingest.py <source> --help` for that adapter's options.")
        sys.exit(0)

    adapter = ADAPTERS[args.source]
    start_time = time.time()

    conn = None
    try:
        print(f"[INGEST] Loading '{args.source}' via {adapter.__class__.__name__}...")
        records = adapter.load(args)
        if not records:
            print("[FATAL] Adapter returned no restaurants. Check your input file(s).",
                  file=sys.stderr)
            sys.exit(1)

        sampled = filter_and_sample(records, args.min_reviews, args.sample_size)
        if not sampled:
            print("[FATAL] No restaurants survived filtering. Try lowering --min-reviews.",
                  file=sys.stderr)
            sys.exit(1)

        conn = init_database(args.db)
        id_prefix = args.source[:3]
        save_restaurants_and_reviews(conn, sampled, id_prefix)
        build_indexes(conn)

        elapsed = time.time() - start_time
        total_reviews = sum(len(r.reviews) for r in sampled)
        print(f"Database successfully generated at '{args.db}' "
              f"({len(sampled)} restaurants, {total_reviews:,} reviews) in {elapsed:.1f}s.")

    except Exception as exc:
        print(f"[FATAL] Ingestion failed for source '{args.source}': {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
