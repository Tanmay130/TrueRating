#!/usr/bin/env python3
"""
evaluate.py

Evaluation harness for the TrueRating platform (formerly evaluate_model.py).
Computes metrics across every stage to check AUTHENTICITY (does it reflect
reality?), ACCURACY (are its numbers defensible?), and PERFORMANCE (is it
fast/reliable?) -- on whichever database you point it at (Yelp's
truerating.db, Zomato's truerating_zomato.db, or any future dataset built
by ingest.py the same way).

Run it incrementally as you complete each stage -- earlier stages need no
API key at all:

    data         ingest.py output only (restaurants/reviews). No API key needed.
    extraction   + extract.py output (aspect_scores). No API key needed.
                 Includes a genuine accuracy check: correlates the LLM's
                 extracted sentiment against the review's OWN star rating
                 (ground truth the reviewer themselves gave).
    credibility  + credibility.py output (review_weights). No API key needed.
    scoring      + score.py output (restaurant_scores). No API key needed.
                 THE key authenticity check: correlates TrueRating's
                 computed true_overall_rating against the dataset's real,
                 independent crowd rating (Zomato's reference_rating, when
                 present), plus MAE/RMSE and top-K ranking overlap.
    rag          + rag.py infra (ChromaDB + Groq). NEEDS GROQ_API_KEY.
                 Retrieval relevance, latency, and (optional) LLM-judged
                 answer quality on a small batch of test queries.
    all          Runs every stage the database currently supports.

Usage:
    python evaluate.py --db truerating_zomato.db --stage all
    python evaluate.py --db truerating_zomato.db --stage scoring
    python evaluate.py --db truerating_zomato.db --stage rag --judge-sample 5
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ASPECTS = ["taste", "hygiene", "service", "value", "delivery"]

DEFAULT_TEST_QUERIES = [
    "What are the best places for biryani?",
    "Which restaurants have slow or poor service?",
    "Where can I get good food on a budget?",
    "Any good places for North Indian food with a nice ambience?",
    "Which restaurants do people say have great desserts?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_conn(db_path: str) -> sqlite3.Connection:
    if not Path(db_path).exists():
        print(f"[FATAL] Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    return conn


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (name,)
    ).fetchone()
    return row is not None


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table});")]
    return column in cols


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """
    Spearman rank correlation without a scipy dependency: it's just the
    Pearson correlation of the two series' ranks.
    """
    return float(a.rank().corr(b.rank(), method="pearson"))


def _round_all(d: dict, digits: int = 4) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and not np.isnan(v):
            out[k] = round(v, digits)
        elif isinstance(v, dict):
            out[k] = _round_all(v, digits)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Stage 1: Data layer (ingest.py) — no API key needed
# ---------------------------------------------------------------------------
def evaluate_data_layer(conn: sqlite3.Connection) -> dict:
    if not table_exists(conn, "restaurants") or not table_exists(conn, "reviews"):
        return {"error": "restaurants/reviews tables not found — run ingest.py first."}

    restaurants = pd.read_sql_query("SELECT * FROM restaurants;", conn)
    reviews = pd.read_sql_query("SELECT * FROM reviews;", conn)

    metrics = {
        "restaurant_count": len(restaurants),
        "review_count": len(reviews),
    }

    per_restaurant = reviews.groupby("restaurant_id").size()
    metrics["reviews_per_restaurant"] = {
        "min": int(per_restaurant.min()) if len(per_restaurant) else 0,
        "max": int(per_restaurant.max()) if len(per_restaurant) else 0,
        "mean": float(per_restaurant.mean()) if len(per_restaurant) else 0.0,
        "median": float(per_restaurant.median()) if len(per_restaurant) else 0.0,
    }

    for col in ("categories", "city"):
        if col in restaurants.columns:
            metrics[f"{col}_null_rate"] = float(restaurants[col].isna().mean())

    if "reference_rating" in restaurants.columns:
        metrics["reference_rating_coverage"] = float(
            1 - restaurants["reference_rating"].isna().mean()
        )

    text = reviews["text"].fillna("")
    word_counts = text.str.split().apply(len)
    metrics["review_word_count"] = {
        "mean": float(word_counts.mean()),
        "median": float(word_counts.median()),
        "p10": float(word_counts.quantile(0.10)),
        "p90": float(word_counts.quantile(0.90)),
    }

    metrics["duplicate_review_text_rate"] = (
        float(1 - reviews["text"].nunique() / len(reviews)) if len(reviews) else None
    )
    metrics["empty_or_whitespace_review_rate"] = float(
        (text.str.strip() == "").mean()
    )

    return _round_all(metrics)


# ---------------------------------------------------------------------------
# Stage 2: Aspect extraction (extract.py) — no API key needed
# ---------------------------------------------------------------------------
def evaluate_extraction(conn: sqlite3.Connection) -> dict:
    if not table_exists(conn, "aspect_scores"):
        return {"error": "aspect_scores table not found — run extract.py first."}

    aspect_scores = pd.read_sql_query("SELECT * FROM aspect_scores;", conn)
    total_reviews = conn.execute("SELECT COUNT(*) FROM reviews;").fetchone()[0]

    metrics = {
        "reviews_scored": len(aspect_scores),
        "total_reviews_in_db": total_reviews,
        "extraction_coverage": (
            len(aspect_scores) / total_reviews if total_reviews else None
        ),
    }

    for aspect in ASPECTS:
        if aspect in aspect_scores.columns:
            metrics[f"{aspect}_mentioned_rate"] = float(aspect_scores[aspect].notna().mean())
            metrics[f"{aspect}_mean_when_mentioned"] = float(aspect_scores[aspect].mean())
            metrics[f"{aspect}_std_when_mentioned"] = float(aspect_scores[aspect].std())

    if "informative" in aspect_scores.columns:
        metrics["informative_rate"] = float(aspect_scores["informative"].astype(float).mean())

    # --- Authenticity check: does extracted sentiment track the reviewer's
    # own star rating? (only meaningful if reviews.stars is populated) ---
    joined = pd.read_sql_query(
        """
        SELECT a.*, r.stars
        FROM aspect_scores a
        JOIN reviews r ON a.review_id = r.id
        WHERE r.stars IS NOT NULL;
        """,
        conn,
    )
    if not joined.empty:
        present_aspects = [a for a in ASPECTS if a in joined.columns]
        joined["avg_extracted_sentiment"] = joined[present_aspects].mean(axis=1, skipna=True)
        valid = joined.dropna(subset=["avg_extracted_sentiment", "stars"])
        if len(valid) >= 10:
            metrics["sentiment_vs_reviewer_stars_pearson_r"] = float(
                valid["avg_extracted_sentiment"].corr(valid["stars"], method="pearson")
            )
            metrics["sentiment_vs_reviewer_stars_spearman_r"] = _spearman(
                valid["avg_extracted_sentiment"], valid["stars"]
            )
            metrics["sentiment_vs_reviewer_stars_n"] = int(len(valid))
        else:
            metrics["sentiment_vs_reviewer_stars_note"] = (
                f"Only {len(valid)} reviews have both a star rating and an "
                f"extracted score — need >= 10 for a meaningful correlation."
            )
    else:
        metrics["sentiment_vs_reviewer_stars_note"] = (
            "No reviews with a non-null star rating found — this dataset "
            "may not carry per-review stars (e.g. raw Yelp import before "
            "ingest.py stored them), so this check is skipped."
        )

    return _round_all(metrics)


# ---------------------------------------------------------------------------
# Stage 3: Credibility & spam detection (credibility.py) — no API key needed
# ---------------------------------------------------------------------------
def evaluate_credibility(conn: sqlite3.Connection) -> dict:
    if not table_exists(conn, "review_weights"):
        return {"error": "review_weights table not found — run credibility.py first."}

    df = pd.read_sql_query(
        """
        SELECT w.review_id, w.weight, w.is_spam, r.text
        FROM review_weights w
        JOIN reviews r ON w.review_id = r.id;
        """,
        conn,
    )

    metrics = {
        "reviews_weighted": len(df),
        "spam_rate": float(df["is_spam"].astype(float).mean()) if len(df) else None,
    }

    non_spam = df[df["is_spam"] == 0]
    if not non_spam.empty:
        metrics["weight_mean"] = float(non_spam["weight"].mean())
        metrics["weight_std"] = float(non_spam["weight"].std())
        metrics["weight_min"] = float(non_spam["weight"].min())
        metrics["weight_max"] = float(non_spam["weight"].max())

        bins = pd.cut(non_spam["weight"], bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0])
        metrics["weight_histogram"] = {
            str(interval): int(count) for interval, count in bins.value_counts().sort_index().items()
        }

        word_counts = non_spam["text"].fillna("").str.split().apply(len)
        metrics["weight_vs_review_length_correlation"] = float(
            non_spam["weight"].corr(word_counts)
        )

    return _round_all(metrics)


# ---------------------------------------------------------------------------
# Stage 4: Scoring accuracy (score.py) — no API key needed
# ---------------------------------------------------------------------------
def evaluate_scoring(conn: sqlite3.Connection) -> dict:
    if not table_exists(conn, "restaurant_scores"):
        return {"error": "restaurant_scores table not found — run score.py first."}

    scores = pd.read_sql_query("SELECT * FROM restaurant_scores;", conn)
    metrics = {"restaurants_scored": len(scores)}

    if not column_exists(conn, "restaurants", "reference_rating"):
        metrics["reference_comparison_note"] = (
            "restaurants.reference_rating not present in this database — "
            "this dataset has no independent crowd rating to validate "
            "against (that's normal for the original Yelp pipeline)."
        )
        return _round_all(metrics)

    restaurants = pd.read_sql_query(
        "SELECT id AS restaurant_id, reference_rating FROM restaurants;", conn
    )
    merged = scores.merge(restaurants, on="restaurant_id", how="inner").dropna(
        subset=["true_overall_rating", "reference_rating"]
    )
    metrics["comparable_restaurants"] = len(merged)

    if len(merged) < 10:
        metrics["reference_comparison_note"] = (
            f"Only {len(merged)} restaurants have both a computed score and "
            f"a reference_rating — need >= 10 for meaningful stats."
        )
        return _round_all(metrics)

    # Rescale our [-1, 1] score onto the dataset's native [0, 5] scale so
    # MAE/RMSE are interpretable in the same units as reference_rating.
    rescaled_ours = (merged["true_overall_rating"] + 1) / 2 * 5
    errors = rescaled_ours - merged["reference_rating"]

    metrics["pearson_r"] = float(
        merged["true_overall_rating"].corr(merged["reference_rating"], method="pearson")
    )
    metrics["spearman_r"] = _spearman(
        merged["true_overall_rating"], merged["reference_rating"]
    )
    metrics["mae_rescaled_0_to_5"] = float(errors.abs().mean())
    metrics["rmse_rescaled_0_to_5"] = float(np.sqrt((errors ** 2).mean()))

    k = min(20, len(merged))
    top_ours = set(merged.nlargest(k, "true_overall_rating")["restaurant_id"])
    top_theirs = set(merged.nlargest(k, "reference_rating")["restaurant_id"])
    overlap = len(top_ours & top_theirs)
    metrics[f"top_{k}_overlap_count"] = overlap
    metrics[f"top_{k}_overlap_pct"] = float(overlap / k)

    return _round_all(metrics)


# ---------------------------------------------------------------------------
# Stage 5: RAG / system performance (rag.py) — NEEDS GROQ_API_KEY
# ---------------------------------------------------------------------------
def evaluate_rag(
    conn: sqlite3.Connection,
    chroma_path: str,
    collection_name: str,
    test_queries: list = None,
    judge_sample: int = 0,
) -> dict:
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        return {"error": f"Missing dependency for rag stage: {exc}"}

    test_queries = test_queries or DEFAULT_TEST_QUERIES
    metrics = {"queries_tested": len(test_queries)}

    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_or_create_collection(name=collection_name)
        metrics["chroma_vector_count"] = collection.count()
    except Exception as exc:
        return {"error": f"Could not open ChromaDB at {chroma_path}: {exc}"}

    try:
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
    except Exception as exc:
        return {"error": f"Could not load embedding model: {exc}"}

    embed_times, retrieve_times, hit_counts = [], [], []
    for query in test_queries:
        t0 = time.time()
        embedding = model.encode(query, convert_to_numpy=True).tolist()
        t1 = time.time()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=5,
            where={"$and": [{"weight": {"$gt": 0.5}}, {"is_spam": False}]},
        )
        t2 = time.time()
        embed_times.append(t1 - t0)
        retrieve_times.append(t2 - t1)
        hit_counts.append(len(results.get("ids", [[]])[0]))

    metrics["avg_embed_seconds"] = float(np.mean(embed_times))
    metrics["avg_retrieve_seconds"] = float(np.mean(retrieve_times))
    metrics["avg_hits_per_query"] = float(np.mean(hit_counts))
    metrics["zero_result_query_rate"] = float(np.mean([h == 0 for h in hit_counts]))

    if judge_sample > 0:
        judge_metrics = _run_judge_sample(
            conn, chroma_path, collection_name, test_queries[:judge_sample]
        )
        metrics["llm_judge"] = judge_metrics

    return _round_all(metrics)


def _run_judge_sample(conn, chroma_path, collection_name, queries: list) -> dict:
    """
    Optional: generate a real RAG answer for each query and have an LLM
    referee score it 1-10, the same way ab_test.py does. This is the only
    part of the whole evaluation suite that spends Groq API calls.
    """
    import os

    from dotenv import load_dotenv
    from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq
    from pydantic import BaseModel, Field
    import chromadb
    from sentence_transformers import SentenceTransformer

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"error": "GROQ_API_KEY not set — skipping LLM-judged quality check."}

    class Verdict(BaseModel):
        score: int = Field(ge=1, le=10)
        explanation: str

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name=collection_name)
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful restaurant assistant. Use the credible "
                   "reviews to answer the user's question. If the reviews don't "
                   "cover it, say you don't know."),
        ("human", "Credible reviews:\n{context}\n\nQuestion: {question}"),
    ])
    rag_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, groq_api_key=api_key)
    rag_chain = rag_prompt | rag_llm | StrOutputParser()

    parser = PydanticOutputParser(pydantic_object=Verdict)
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict, impartial judge. Score the assistant's "
                   "response 1-10 on relevance and grounding in the provided "
                   "reviews. Return ONLY JSON matching this schema:\n"
                   "{format_instructions}"),
        ("human", "Query: {query}\n\nReviews:\n{context}\n\nResponse:\n{response}"),
    ]).partial(format_instructions=parser.get_format_instructions())
    judge_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, groq_api_key=api_key)
    judge_chain = judge_prompt | judge_llm | parser

    scores = []
    for query in queries:
        embedding = embed_model.encode(query, convert_to_numpy=True).tolist()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=5,
            where={"$and": [{"weight": {"$gt": 0.5}}, {"is_spam": False}]},
        )
        docs = results.get("documents", [[]])[0]
        context = "\n\n".join(docs) if docs else "No credible reviews found."

        try:
            response = rag_chain.invoke({"context": context, "question": query})
            verdict = judge_chain.invoke({"query": query, "context": context, "response": response})
            scores.append(verdict.score)
        except Exception as exc:
            print(f"[JUDGE][WARN] Skipping query '{query}': {exc}", file=sys.stderr)

    if not scores:
        return {"error": "No queries were successfully judged."}

    return {
        "queries_judged": len(scores),
        "avg_score": float(np.mean(scores)),
        "min_score": int(min(scores)),
        "max_score": int(max(scores)),
        "pass_rate_score_gte_6": float(np.mean([s >= 6 for s in scores])),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(stage_name: str, metrics: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f" {stage_name.upper()}")
    print(f"{'=' * 70}")
    if "error" in metrics:
        print(f"  [SKIPPED] {metrics['error']}")
        return
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"      {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="TrueRating evaluation harness: authenticity, accuracy, performance metrics."
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database to evaluate")
    parser.add_argument(
        "--stage",
        choices=["data", "extraction", "credibility", "scoring", "rag", "all"],
        default="all",
    )
    parser.add_argument("--chroma-path", default="./chroma_db", help="ChromaDB path (for --stage rag)")
    parser.add_argument("--collection", default="truerating_reviews")
    parser.add_argument(
        "--judge-sample", type=int, default=0,
        help="How many test queries to score with the LLM judge (0 = skip, costs Groq calls)"
    )
    parser.add_argument("--output", default=None, help="Optional path to save the full report as JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    conn = load_conn(args.db)

    report = {}
    stages_to_run = (
        ["data", "extraction", "credibility", "scoring", "rag"]
        if args.stage == "all"
        else [args.stage]
    )

    try:
        if "data" in stages_to_run:
            report["data"] = evaluate_data_layer(conn)
            print_report("Data Layer (ingest.py)", report["data"])

        if "extraction" in stages_to_run:
            report["extraction"] = evaluate_extraction(conn)
            print_report("Aspect Extraction (extract.py)", report["extraction"])

        if "credibility" in stages_to_run:
            report["credibility"] = evaluate_credibility(conn)
            print_report("Credibility & Spam Detection (credibility.py)", report["credibility"])

        if "scoring" in stages_to_run:
            report["scoring"] = evaluate_scoring(conn)
            print_report("Scoring Accuracy vs. Real Ratings (score.py)", report["scoring"])

        if "rag" in stages_to_run:
            report["rag"] = evaluate_rag(
                conn, args.chroma_path, args.collection,
                judge_sample=args.judge_sample,
            )
            print_report("RAG Retrieval & Performance (rag.py)", report["rag"])

    finally:
        conn.close()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report saved to {args.output}")

    print(f"\n{'=' * 70}\nEvaluation complete.\n{'=' * 70}")


if __name__ == "__main__":
    main()
