#!/usr/bin/env python3
"""
credibility.py

Credibility weighting stage of the TrueRating platform (formerly
phase3_credibility.py). Dataset-agnostic: works against any database built
by ingest.py, regardless of which adapter (Yelp, Zomato, ...) produced it.

For every review that has already passed through extract.py (aspect
extraction), this script:
  1. Generates a local sentence embedding (all-MiniLM-L6-v2, CPU-only).
  2. Flags near-duplicate ("copy-paste") spam within the same restaurant
     using cosine similarity of those embeddings.
  3. Computes a credibility weight in [0.1, 1.0] from review length, aspect
     coverage, and the `informative` flag from extract.py.
  4. Persists embeddings + metadata to a local ChromaDB collection, and
     weights/spam flags to a `review_weights` table in the target database.

The script is idempotent / resumable: reviews already present in
`review_weights` are never re-processed or re-embedded.

Usage:
    python credibility.py --db truerating.db --chroma-path ./chroma_db
    python credibility.py --db truerating_zomato.db --chroma-path ./chroma_db_zomato

Requires (already present in requirements.txt):
    sentence-transformers
    chromadb
    torch
    numpy
"""

import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime
from math import ceil
from typing import Dict, List, Set

import numpy as np
import sqlite3

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BATCH_SIZE = 250
SIMILARITY_THRESHOLD = 0.90   # cosine similarity above this => near-duplicate
ENCODE_SUB_BATCH_SIZE = 32    # internal batch size for SentenceTransformer.encode

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DB_FILE = "truerating.db"
DEFAULT_CHROMA_PATH = "./chroma_db"
DEFAULT_COLLECTION_NAME = "truerating_reviews"

MIN_WEIGHT = 0.1
MAX_WEIGHT = 1.0
LENGTH_NORM_WORDS = 40   # word count that saturates the length score at 1.0
NUM_ASPECTS = 5          # taste, hygiene, service, value, delivery
INFORMATIVE_BONUS = 0.2


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_db_connection(db_path: str) -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn
    except sqlite3.Error as exc:
        print(f"[DB][ERROR] Could not connect to {db_path}: {exc}", file=sys.stderr)
        raise


def ensure_weights_table(conn: sqlite3.Connection) -> None:
    """Create the review_weights table if it doesn't already exist (idempotent)."""
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS review_weights (
                review_id TEXT PRIMARY KEY,
                weight    REAL,
                is_spam   BOOLEAN
            );
            """
        )
        conn.commit()
        print("[DB] Table 'review_weights' ready (created if missing).")
    except sqlite3.Error as exc:
        print(f"[DB][ERROR] Failed to create review_weights table: {exc}", file=sys.stderr)
        raise


def fetch_pending_reviews(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """
    Return one row per review that:
      - has a matching row in aspect_scores (extract.py already ran on it), and
      - does NOT already have a row in review_weights (not yet processed).
    """
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                r.id            AS review_id,
                r.restaurant_id AS restaurant_id,
                r.text          AS text,
                r.date          AS date,
                a.taste         AS taste,
                a.hygiene       AS hygiene,
                a.service       AS service,
                a."value"       AS value_score,
                a.delivery      AS delivery,
                a.informative   AS informative
            FROM reviews r
            JOIN aspect_scores a ON r.id = a.review_id
            LEFT JOIN review_weights w ON r.id = w.review_id
            WHERE w.review_id IS NULL;
            """
        )
        rows = cur.fetchall()
        print(f"[DB] Found {len(rows):,} reviews pending credibility weighting.")
        return rows
    except sqlite3.Error as exc:
        print(f"[DB][ERROR] Failed to fetch pending reviews: {exc}", file=sys.stderr)
        raise


def save_weights(conn: sqlite3.Connection, rows: List[tuple]) -> None:
    """Bulk insert (review_id, weight, is_spam) into review_weights."""
    if not rows:
        return
    try:
        cur = conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO review_weights (review_id, weight, is_spam) "
            "VALUES (?, ?, ?);",
            rows,
        )
        conn.commit()
    except sqlite3.Error as exc:
        print(f"[DB][ERROR] Failed to write review_weights batch: {exc}", file=sys.stderr)
        raise


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
def init_chroma_collection(chroma_path: str, collection_name: str):
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_or_create_collection(name=collection_name)
        print(f"[CHROMA] Connected to persistent store at '{chroma_path}', "
              f"collection '{collection_name}'.")
        return collection
    except Exception as exc:
        print(f"[CHROMA][ERROR] Failed to initialize collection: {exc}", file=sys.stderr)
        raise


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
def load_embedding_model() -> SentenceTransformer:
    try:
        print(f"[MODEL] Loading '{MODEL_NAME}' on CPU...")
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        print("[MODEL] Ready.")
        return model
    except Exception as exc:
        print(f"[MODEL][ERROR] Failed to load embedding model: {exc}", file=sys.stderr)
        raise


# ---------------------------------------------------------------------------
# Spam detection (cosine similarity within restaurant groups)
# ---------------------------------------------------------------------------
def parse_date(date_str) -> datetime:
    """Best-effort parse of the review date; falls back to datetime.min."""
    if not date_str:
        return datetime.min
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    return datetime.min


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity for a small (n x d) block of embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized = embeddings / norms
    return normalized @ normalized.T


def detect_spam(batch: List[sqlite3.Row], embeddings: np.ndarray) -> Set[int]:
    """
    Group batch indices by restaurant_id, compute cosine similarity within
    each group, and mark the newer review of any pair exceeding
    SIMILARITY_THRESHOLD as spam. Returns a set of spam batch-indices.
    """
    spam_indices: Set[int] = set()
    groups: Dict[str, List[int]] = defaultdict(list)

    for idx, row in enumerate(batch):
        groups[row["restaurant_id"]].append(idx)

    for restaurant_id, indices in groups.items():
        if len(indices) < 2:
            continue

        group_embeddings = embeddings[indices]
        sim_matrix = cosine_similarity_matrix(group_embeddings)
        n = len(indices)

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] > SIMILARITY_THRESHOLD:
                    idx_i, idx_j = indices[i], indices[j]
                    date_i = parse_date(batch[idx_i]["date"])
                    date_j = parse_date(batch[idx_j]["date"])
                    newer_idx = idx_j if date_j >= date_i else idx_i
                    spam_indices.add(newer_idx)

    return spam_indices


# ---------------------------------------------------------------------------
# Credibility weight math
# ---------------------------------------------------------------------------
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_aspect_score(row: sqlite3.Row) -> float:
    aspects = [row["taste"], row["hygiene"], row["service"], row["value_score"], row["delivery"]]
    non_null_count = sum(1 for a in aspects if a is not None)
    return non_null_count / NUM_ASPECTS


def compute_credibility_weight(row: sqlite3.Row) -> float:
    """
    weight = clamp( (length_score + aspect_score) / 2 + informative_bonus, 0.1, 1.0 )

      length_score   = min(1.0, word_count / 40)
      aspect_score   = (# non-null aspects) / 5
      informative_bonus = 0.2 if informative else 0.0
    """
    text = row["text"] or ""
    word_count = len(text.split())
    length_score = min(1.0, word_count / LENGTH_NORM_WORDS)

    aspect_score = compute_aspect_score(row)
    informative_bonus = INFORMATIVE_BONUS if row["informative"] else 0.0

    raw_weight = (length_score + aspect_score) / 2.0 + informative_bonus
    return clamp(raw_weight, MIN_WEIGHT, MAX_WEIGHT)


# ---------------------------------------------------------------------------
# Batch utilities
# ---------------------------------------------------------------------------
def chunk_list(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="TrueRating: embedding-based spam filtering + credibility weighting."
    )
    parser.add_argument("--db", default=DEFAULT_DB_FILE, help="Path to the target database")
    parser.add_argument(
        "--chroma-path", default=DEFAULT_CHROMA_PATH, help="ChromaDB persistent store path"
    )
    parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION_NAME, help="ChromaDB collection name"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Reviews per processing batch"
    )
    return parser.parse_args()


def process_batch(
    conn: sqlite3.Connection,
    collection,
    model: SentenceTransformer,
    batch: List[sqlite3.Row],
    batch_num: int,
    total_batches: int,
) -> int:
    """Embed, spam-check, weight, and persist a single batch. Returns spam count."""
    texts = [row["text"] for row in batch]

    embeddings = model.encode(
        texts,
        batch_size=ENCODE_SUB_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    spam_indices = detect_spam(batch, embeddings)

    weight_rows = []
    chroma_ids = []
    chroma_embeddings = []
    chroma_metadatas = []
    chroma_documents = []

    for idx, row in enumerate(batch):
        is_spam = idx in spam_indices
        weight = 0.0 if is_spam else compute_credibility_weight(row)

        weight_rows.append((row["review_id"], weight, int(is_spam)))

        chroma_ids.append(row["review_id"])
        chroma_embeddings.append(embeddings[idx].tolist())
        chroma_metadatas.append(
            {
                "restaurant_id": row["restaurant_id"],
                "weight": weight,
                "is_spam": is_spam,
            }
        )
        chroma_documents.append(row["text"])

    save_weights(conn, weight_rows)

    try:
        collection.upsert(
            ids=chroma_ids,
            embeddings=chroma_embeddings,
            metadatas=chroma_metadatas,
            documents=chroma_documents,
        )
    except Exception as exc:
        print(f"[BATCH {batch_num}][ERROR] ChromaDB upsert failed: {exc}", file=sys.stderr)
        raise

    print(
        f"Processed batch {batch_num}/{total_batches}. "
        f"Found {len(spam_indices)} spam reviews."
    )
    return len(spam_indices)


def main():
    args = parse_args()
    start_time = time.time()

    conn = None
    try:
        conn = get_db_connection(args.db)
        ensure_weights_table(conn)

        pending = fetch_pending_reviews(conn)
        if not pending:
            print("[INFO] No pending reviews found. Nothing to do.")
            return

        collection = init_chroma_collection(args.chroma_path, args.collection)
        model = load_embedding_model()

        total_batches = ceil(len(pending) / args.batch_size)
        total_saved = 0
        total_spam = 0

        for batch_num, batch in enumerate(chunk_list(pending, args.batch_size), start=1):
            try:
                spam_count = process_batch(conn, collection, model, batch, batch_num, total_batches)
            except Exception as exc:
                print(
                    f"[BATCH {batch_num}/{total_batches}][ERROR] Skipping batch due to: {exc}",
                    file=sys.stderr,
                )
                continue

            total_saved += len(batch)
            total_spam += spam_count

        elapsed = time.time() - start_time
        print(
            f"Credibility weighting complete. {total_saved:,} reviews scored, "
            f"{total_spam:,} marked spam, in {elapsed:.1f}s."
        )

    except Exception as exc:
        print(f"[FATAL] Credibility pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
