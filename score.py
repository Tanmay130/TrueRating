#!/usr/bin/env python3
"""
score.py

Scoring stage of the TrueRating platform (formerly phase4_scoring.py).
Dataset-agnostic: works against any database built by ingest.py, regardless
of which adapter (Yelp, Zomato, ...) produced it.

Aggregates extract.py (aspect_scores) and credibility.py (review_weights)
output into a single credibility-weighted "TrueRating" per restaurant:

  - For each aspect (taste, hygiene, service, value, delivery):
        aspect_score = sum(aspect_value * weight) / sum(weight)
    computed only over reviews where that aspect was actually mentioned
    (non-null) and excluding anything flagged as spam.
  - true_overall_rating = mean of the (non-null) aspect scores.

This is a full recompute every run (not incremental) since the underlying
weights/scores can change upstream; `restaurant_scores` is dropped and
rebuilt each time.

Usage:
    python score.py --db truerating.db
    python score.py --db truerating_zomato.db
"""

import argparse
import sqlite3
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ASPECTS = ["taste", "hygiene", "service", "value", "delivery"]
DEFAULT_DB_FILE = "truerating.db"
TOP_N = 10


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_db_connection(db_path: str) -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn
    except sqlite3.Error as exc:
        print(f"[DB][ERROR] Could not connect to {db_path}: {exc}", file=sys.stderr)
        raise


def ensure_scores_table(conn: sqlite3.Connection) -> None:
    """(Re)create restaurant_scores. This is a full recompute each run."""
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS restaurant_scores;")
        cur.execute(
            """
            CREATE TABLE restaurant_scores (
                restaurant_id       TEXT PRIMARY KEY,
                taste               REAL,
                hygiene             REAL,
                service             REAL,
                "value"             REAL,
                delivery            REAL,
                true_overall_rating REAL,
                review_count        INTEGER
            );
            """
        )
        conn.commit()
        print("[DB] Table 'restaurant_scores' (re)created.")
    except sqlite3.Error as exc:
        print(f"[DB][ERROR] Failed to create restaurant_scores table: {exc}", file=sys.stderr)
        raise


def load_credible_reviews(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Join reviews -> aspect_scores -> review_weights, excluding spam.
    Returns one row per (non-spam) scored review with its aspect values
    and credibility weight.
    """
    query = """
        SELECT
            r.restaurant_id AS restaurant_id,
            a.taste         AS taste,
            a.hygiene       AS hygiene,
            a.service       AS service,
            a."value"       AS value,
            a.delivery      AS delivery,
            w.weight        AS weight
        FROM reviews r
        JOIN aspect_scores a ON r.id = a.review_id
        JOIN review_weights w ON r.id = w.review_id
        WHERE w.is_spam = 0;
    """
    try:
        df = pd.read_sql_query(query, conn)
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        print(f"[DB][ERROR] Failed to load joined review data: {exc}", file=sys.stderr)
        raise

    df["weight"] = df["weight"].fillna(0.0)
    print(f"[DATA] Loaded {len(df):,} credible (non-spam) scored reviews "
          f"across {df['restaurant_id'].nunique():,} restaurants.")
    return df


def load_restaurant_lookup(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load restaurant id/name/city for display purposes only."""
    try:
        return pd.read_sql_query(
            'SELECT id AS restaurant_id, name, city FROM restaurants;', conn
        )
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        print(f"[DB][ERROR] Failed to load restaurants table: {exc}", file=sys.stderr)
        raise


# ---------------------------------------------------------------------------
# Scoring math
# ---------------------------------------------------------------------------
def compute_restaurant_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized weighted-average aggregation per restaurant_id.

    For each aspect: only rows where that aspect is non-null contribute to
    both the numerator (value * weight) and the denominator (weight sum).
    """
    if df.empty:
        return pd.DataFrame(
            columns=["restaurant_id"] + ASPECTS + ["true_overall_rating", "review_count"]
        )

    work = df.copy()

    for aspect in ASPECTS:
        has_value = work[aspect].notna()
        work[f"{aspect}_wsum"] = work[aspect] * work["weight"]
        work[f"{aspect}_wgt"] = np.where(has_value, work["weight"], 0.0)

    agg_cols = {f"{a}_wsum": "sum" for a in ASPECTS}
    agg_cols.update({f"{a}_wgt": "sum" for a in ASPECTS})

    grouped = work.groupby("restaurant_id").agg(agg_cols)
    review_counts = work.groupby("restaurant_id").size().rename("review_count")

    for aspect in ASPECTS:
        wgt_sum = grouped[f"{aspect}_wgt"]
        # Replace a zero denominator with NaN *before* dividing so we never
        # actually perform a 0/0 division (which pandas/numpy can raise as
        # a real ZeroDivisionError for scalar-like float blocks).
        safe_denom = wgt_sum.where(wgt_sum > 0, other=np.nan)
        grouped[aspect] = grouped[f"{aspect}_wsum"] / safe_denom

    grouped["true_overall_rating"] = grouped[ASPECTS].mean(axis=1, skipna=True)
    grouped = grouped.join(review_counts)

    result = grouped[ASPECTS + ["true_overall_rating", "review_count"]].reset_index()
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_scores(conn: sqlite3.Connection, scores: pd.DataFrame) -> int:
    if scores.empty:
        print("[DB][WARN] No restaurant scores to save.")
        return 0

    rows = [
        (
            row.restaurant_id,
            None if pd.isna(row.taste) else float(row.taste),
            None if pd.isna(row.hygiene) else float(row.hygiene),
            None if pd.isna(row.service) else float(row.service),
            None if pd.isna(row.value) else float(row.value),
            None if pd.isna(row.delivery) else float(row.delivery),
            None if pd.isna(row.true_overall_rating) else float(row.true_overall_rating),
            int(row.review_count),
        )
        for row in scores.itertuples(index=False)
    ]

    try:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT OR REPLACE INTO restaurant_scores
                (restaurant_id, taste, hygiene, service, "value", delivery,
                 true_overall_rating, review_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
        conn.commit()
        print(f"[DB] Saved scores for {len(rows):,} restaurants.")
        return len(rows)
    except sqlite3.Error as exc:
        print(f"[DB][ERROR] Failed to bulk insert restaurant_scores: {exc}", file=sys.stderr)
        raise


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_top_n(scores: pd.DataFrame, restaurants: pd.DataFrame, n: int = TOP_N) -> None:
    if scores.empty:
        print("[INFO] No scores to display.")
        return

    display = scores.merge(restaurants, on="restaurant_id", how="left")
    top = display.sort_values("true_overall_rating", ascending=False).head(n)

    print(f"\nTop {n} Restaurants by TrueOverallRating:")
    header = f"{'Rank':<5}{'Name':<32}{'City':<18}{'Overall':<9}{'Reviews':<8}"
    print(header)
    print("-" * len(header))

    for rank, row in enumerate(top.itertuples(index=False), start=1):
        name = (row.name or "Unknown")[:30]
        city = (row.city or "?")[:16]
        overall = f"{row.true_overall_rating:.3f}" if pd.notna(row.true_overall_rating) else "N/A"
        print(f"{rank:<5}{name:<32}{city:<18}{overall:<9}{row.review_count:<8}")


def print_coverage_diagnostic(scores: pd.DataFrame, restaurants: pd.DataFrame) -> None:
    """Flag restaurants with zero credible reviews (excluded from scoring)."""
    scored_ids = set(scores["restaurant_id"]) if not scores.empty else set()
    all_ids = set(restaurants["restaurant_id"])
    missing = all_ids - scored_ids
    if missing:
        print(
            f"[INFO] {len(missing)} of {len(all_ids)} restaurants have no "
            f"credible (non-spam) scored reviews yet and were skipped."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="TrueRating: credibility-weighted restaurant scoring."
    )
    parser.add_argument("--db", default=DEFAULT_DB_FILE, help="Path to the target database")
    parser.add_argument("--top-n", type=int, default=TOP_N, help="How many top restaurants to print")
    return parser.parse_args()


def main():
    args = parse_args()

    conn = None
    try:
        conn = get_db_connection(args.db)
        ensure_scores_table(conn)

        reviews_df = load_credible_reviews(conn)
        restaurants_df = load_restaurant_lookup(conn)

        scores_df = compute_restaurant_scores(reviews_df)
        save_scores(conn, scores_df)

        print_coverage_diagnostic(scores_df, restaurants_df)
        print_top_n(scores_df, restaurants_df, n=args.top_n)

        print("\nScoring complete.")

    except Exception as exc:
        print(f"[FATAL] Scoring pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
