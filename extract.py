#!/usr/bin/env python3
"""
extract.py

Aspect extraction stage of the TrueRating platform (formerly
phase2_extraction.py). Dataset-agnostic: works against any database built
by ingest.py, regardless of which adapter (Yelp, Zomato, ...) produced it.

Reads reviews from the target database, sends them to Groq (via LangChain)
in batches of 10, and extracts per-aspect sentiment scores (taste, hygiene,
service, value, delivery) plus an "informative" flag. Results are persisted
incrementally into an `aspect_scores` table so the script is safely
resumable — reviews already scored are never re-sent to the API.

Provider: Groq (llama-3.1-8b-instant), chosen over llama-3.3-70b-versatile
after hitting its 100,000 Tokens Per Day free-tier cap at batch 27. The 8b
model has a much higher daily token allowance; batch size was also dropped
from 15 to 10 to stay safely under the per-minute token limit.

Usage:
    python extract.py --db truerating.db
    python extract.py --db truerating_zomato.db

Requires (add to requirements.txt if not already present):
    langchain
    langchain-groq
    python-dotenv
    pydantic
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field, RootModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BATCH_SIZE = 10          # reduced from 15 to stay safely under Groq's TPM limit
MAX_RETRIES = 2          # additional retries after the first attempt (3 total tries)
RETRY_DELAY_SECONDS = 3
BATCH_SLEEP_SECONDS = 2.5  # Groq free tier allows 30 RPM; 2.5s keeps us well under it

# llama-3.3-70b-versatile has a 100,000 Tokens Per Day (TPD) free-tier cap,
# which we hit at batch 27. llama-3.1-8b-instant has a much higher daily
# token allowance, so it's used here to get through the full dataset.
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_DB_FILE = "truerating.db"

SYSTEM_PROMPT = """You are a strict data parser, not a creative writer.

You will be given a numbered list of restaurant reviews. For EACH review,
extract sentiment scores for the following aspects, each on a scale from
-1.0 (very negative) to 1.0 (very positive):

  - taste
  - hygiene
  - service
  - value
  - delivery

Also determine `informative`: true if the review gives specific, useful
detail (e.g. names a dish, describes wait times, mentions cleanliness,
explains a price complaint); false if it is generic noise (e.g. "good food",
"would recommend", "5 stars").

CRITICAL RULE: DO NOT guess or infer scores. If an aspect (like hygiene or
delivery) is not explicitly mentioned in the text, its value MUST be null.
Do not fill in a neutral 0.0 as a substitute for "not mentioned" — null and
0.0 (neutral sentiment) are different things and must not be confused.

Return exactly one result per input review, preserving its original
review_id.

{format_instructions}
"""

HUMAN_PROMPT = "Reviews to evaluate:\n\n{reviews_block}"


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------
class AspectScore(BaseModel):
    review_id: str
    taste: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    hygiene: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    service: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    value: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    delivery: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    informative: bool


class BatchExtraction(RootModel[List[AspectScore]]):
    """
    Wraps a raw JSON array of AspectScore objects. Groq/Llama models tend to
    return a bare `[...]` list rather than a `{"results": [...]}` object, so
    a RootModel is used here instead of a BaseModel with a named field.
    """

    pass


# ---------------------------------------------------------------------------
# Environment / API key
# ---------------------------------------------------------------------------
def load_api_key() -> str:
    """Load GROQ_API_KEY from the local .env file."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        print(
            "[FATAL] GROQ_API_KEY is missing or empty. Set it in your .env file.",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


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


def ensure_aspect_table(conn: sqlite3.Connection) -> None:
    """Create the aspect_scores table if it doesn't already exist (idempotent)."""
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS aspect_scores (
                review_id   TEXT PRIMARY KEY,
                taste       REAL,
                hygiene     REAL,
                service     REAL,
                "value"     REAL,
                delivery    REAL,
                informative INTEGER
            );
            """
        )
        conn.commit()
        print("[DB] Table 'aspect_scores' ready (created if missing).")
    except sqlite3.Error as exc:
        print(f"[DB][ERROR] Failed to create aspect_scores table: {exc}", file=sys.stderr)
        raise


def fetch_pending_reviews(conn: sqlite3.Connection, limit: int) -> List[Tuple[str, str]]:
    """
    Return (id, text) for up to `limit` reviews that do not yet have a row
    in aspect_scores.

    Ordering is round-robin BY RESTAURANT rather than plain insertion order:
    each restaurant's 1st not-yet-scored review is queued before anyone's
    2nd, each 2nd before anyone's 3rd, and so on. Plain insertion order
    would process one restaurant fully before starting the next (reviews
    are written restaurant-by-restaurant in ingest.py), so a partial run
    with, say, --limit 1340 would only ever cover a couple dozen
    restaurants out of the full set. Round-robin ordering means even a
    small partial run touches as many restaurants as possible, which
    matters for score.py/evaluate.py's per-restaurant aggregation and the
    reference_rating correlation check.
    """
    try:
        cur = conn.cursor()
        cur.execute(
            """
            WITH pending AS (
                SELECT
                    r.id,
                    r.text,
                    r.restaurant_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY r.restaurant_id ORDER BY r.id
                    ) AS rn
                FROM reviews r
                LEFT JOIN aspect_scores a ON r.id = a.review_id
                WHERE a.review_id IS NULL
            )
            SELECT id, text
            FROM pending
            ORDER BY rn, restaurant_id
            LIMIT ?;
            """,
            (limit,),
        )
        rows = cur.fetchall()
        print(f"[DB] Found {len(rows):,} pending reviews (not yet in aspect_scores), "
              f"ordered round-robin by restaurant for broad coverage.")
        return rows
    except sqlite3.Error as exc:
        print(f"[DB][ERROR] Failed to fetch pending reviews: {exc}", file=sys.stderr)
        raise


def save_batch_results(
    conn: sqlite3.Connection,
    batch_result: BatchExtraction,
    expected_ids: set,
    batch_num: int,
) -> int:
    """
    Bulk-insert a batch's extracted scores. Returns rows written.

    NOTE: attribute access directly off the RootModel's items (e.g.
    `item.review_id`) was silently producing None/blank fields in
    production, which then got written straight into SQLite as empty rows.
    `.model_dump()` + `.get()` is the bulletproof extraction path below.
    """
    data_to_insert = []
    for item in batch_result.root:
        data = item.model_dump()
        data_to_insert.append((
            data.get('review_id'),
            data.get('taste'),
            data.get('hygiene'),
            data.get('service'),
            data.get('value'),
            data.get('delivery'),
            data.get('informative')
        ))

    returned_ids = {row[0] for row in data_to_insert}
    missing = expected_ids - returned_ids
    unexpected = returned_ids - expected_ids
    if missing:
        print(
            f"[BATCH {batch_num}][WARN] Model omitted {len(missing)} review_id(s): "
            f"{sorted(missing)}"
        )
    if unexpected:
        print(
            f"[BATCH {batch_num}][WARN] Model returned {len(unexpected)} unknown "
            f"review_id(s) not in this batch; discarding them: {sorted(unexpected)}"
        )

    # Only keep rows with a review_id we actually asked about — this also
    # guards against any None/blank review_id ever reaching the database.
    rows = [row for row in data_to_insert if row[0] in expected_ids]

    if not rows:
        return 0

    try:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT OR REPLACE INTO aspect_scores
                (review_id, taste, hygiene, service, "value", delivery, informative)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
        conn.commit()
        return len(rows)
    except sqlite3.Error as exc:
        print(f"[BATCH {batch_num}][ERROR] Failed to write results to DB: {exc}", file=sys.stderr)
        raise


# ---------------------------------------------------------------------------
# LangChain pipeline
# ---------------------------------------------------------------------------
def build_chain(model_name: str, temperature: float, api_key: str):
    parser = PydanticOutputParser(pydantic_object=BatchExtraction)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        groq_api_key=api_key,
    )

    chain = prompt | llm | parser
    return chain


def format_batch(batch: List[Tuple[str, str]]) -> str:
    """Format a batch of (id, text) rows into a numbered string block."""
    lines = [
        f"Review {i}: [{review_id}] - {text}"
        for i, (review_id, text) in enumerate(batch, start=1)
    ]
    return "\n".join(lines)


def chunk_list(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def invoke_with_retry(chain, reviews_block: str, batch_num: int) -> Optional[BatchExtraction]:
    """
    Invoke the chain, retrying up to MAX_RETRIES times on parsing errors or
    rate-limit / transient API exceptions. Returns None if all attempts fail.
    """
    attempt = 0
    while True:
        try:
            result = chain.invoke({"reviews_block": reviews_block})
            return result
        except Exception as exc:
            attempt += 1
            print(
                f"[BATCH {batch_num}][WARN] Attempt {attempt} failed "
                f"({type(exc).__name__}): {exc}"
            )
            if attempt > MAX_RETRIES:
                print(
                    f"[BATCH {batch_num}][ERROR] Exceeded max retries "
                    f"({MAX_RETRIES}). Skipping this batch."
                )
                return None
            print(f"[BATCH {batch_num}] Retrying in {RETRY_DELAY_SECONDS}s...")
            time.sleep(RETRY_DELAY_SECONDS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="TrueRating: aspect-based sentiment extraction via Groq."
    )
    parser.add_argument("--db", default=DEFAULT_DB_FILE, help="Path to the target database")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Groq model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Reviews per API call"
    )
    parser.add_argument(
        "--limit", type=int, default=5000,
        help="Max pending reviews to process this run (default: 3000). "
             "Reviews are queued round-robin by restaurant, so a smaller "
             "--limit still spreads coverage across as many restaurants as "
             "possible instead of finishing one restaurant at a time.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    api_key = load_api_key()

    conn = None
    try:
        conn = get_db_connection(args.db)
        ensure_aspect_table(conn)

        pending = fetch_pending_reviews(conn, args.limit)
        if not pending:
            print("[INFO] No pending reviews found. Nothing to do.")
            return

        chain = build_chain(args.model, args.temperature, api_key)

        total_batches = (len(pending) + args.batch_size - 1) // args.batch_size
        total_saved = 0
        total_skipped_batches = 0

        for batch_num, batch in enumerate(chunk_list(pending, args.batch_size), start=1):
            expected_ids = {review_id for review_id, _ in batch}
            reviews_block = format_batch(batch)

            print(f"[BATCH {batch_num}/{total_batches}] Sending {len(batch)} reviews to Groq...")
            result = invoke_with_retry(chain, reviews_block, batch_num)

            if result is None:
                total_skipped_batches += 1
                print(f"[BATCH {batch_num}/{total_batches}] Skipped after repeated failures.")
                continue

            saved = save_batch_results(conn, result, expected_ids, batch_num)
            total_saved += saved
            print(
                f"[BATCH {batch_num}/{total_batches}] Saved {saved}/{len(batch)} "
                f"aspect scores. Running total: {total_saved:,}."
            )

            # Rate-limit throttle: Groq allows 30 RPM. Sleeping 2.5s after each
            # successful batch keeps us safely under that limit.
            time.sleep(BATCH_SLEEP_SECONDS)

        elapsed = time.time() - start_time
        print(
            f"Extraction complete. {total_saved:,} reviews scored, "
            f"{total_skipped_batches} batch(es) skipped, in {elapsed:.1f}s."
        )

    except Exception as exc:
        print(f"[FATAL] Extraction pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
