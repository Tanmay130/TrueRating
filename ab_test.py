#!/usr/bin/env python3
"""
ab_test.py

Self-evaluating A/B testing engine for the TrueRating RAG assistant
(formerly phase5_ab_engine.py; builds on rag.py's ChromaDB + Groq
infrastructure). Dataset-agnostic: works against any database/ChromaDB
store built by ingest.py + credibility.py, regardless of which adapter
(Yelp, Zomato, ...) produced the underlying data.

Compares two retrieval strategies:
    Variant A: credibility weight > 0.5  (broad recall)
    Variant B: credibility weight > 0.8  (high-credibility focus)

For each test query, both variants' RAG responses are generated, then an
independent LLM "Judge" scores each response 1-10 on relevance, use of
evidence, and conciseness. The higher-scoring variant wins (ties allowed).
Every result is appended to ab_test_results.jsonl.

Usage:
    python ab_test.py --db truerating.db --chroma-path ./chroma_db
    python ab_test.py --db truerating_zomato.db --chroma-path ./chroma_db_zomato

Requires (already present from earlier stages):
    chromadb
    sentence-transformers
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
from typing import Dict, List, Optional

import chromadb
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Must match the model used to build embeddings in credibility.py.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_CHROMA_PATH = "./chroma_db"
DEFAULT_COLLECTION_NAME = "truerating_reviews"
DEFAULT_DB_FILE = "truerating.db"
DEFAULT_OUTPUT_FILE = "ab_test_results.jsonl"

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.0
N_RESULTS = 5

VARIANT_A_MIN_WEIGHT = 0.5
VARIANT_B_MIN_WEIGHT = 0.8

MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 3
CALL_SLEEP_SECONDS = 2.5  # throttle between Groq calls (30 RPM free tier)

TEST_QUERIES = [
    "What are the best cafes in Indiranagar?",
    "Which restaurants have poor service?",
    "Is there a good spot for spicy food with fast delivery?",
    "What do reviewers say about hygiene at Italian restaurants?",
    "Recommend a good value restaurant for a family dinner.",
]

RAG_SYSTEM_PROMPT = (
    "You are a helpful restaurant assistant. Use the following credible "
    "reviews to answer the user's question. If the information isn't in "
    "the reviews, say you don't know."
)
RAG_HUMAN_PROMPT = "Credible reviews:\n{context}\n\nQuestion: {question}"

JUDGE_SYSTEM_PROMPT = """You are an impartial judge evaluating an AI restaurant \
assistant's answer to a user's question.

Score the response from 1 (very poor) to 10 (excellent) based on:
  a) Relevance: Does it directly answer the user's query?
  b) Use of Evidence: Does it explicitly rely on the provided reviews rather \
than making things up?
  c) Conciseness: Is it helpful without unnecessary filler?

You are a strict data parser, not a creative writer. Return ONLY valid JSON \
matching this schema, with no extra commentary:

{format_instructions}
"""

JUDGE_HUMAN_PROMPT = (
    "User query:\n{query}\n\n"
    "Reviews given to the assistant:\n{context}\n\n"
    "Assistant's response:\n{response}\n\n"
    "Evaluate this response now."
)


# ---------------------------------------------------------------------------
# Pydantic schema for the Judge
# ---------------------------------------------------------------------------
class JudgeVerdict(BaseModel):
    score: int = Field(ge=1, le=10)
    explanation: str


# ---------------------------------------------------------------------------
# Module-level retrieval state (initialized once in init_retrieval())
# ---------------------------------------------------------------------------
_collection = None
_embedding_model: Optional[SentenceTransformer] = None
_restaurant_names: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Environment / API key
# ---------------------------------------------------------------------------
def load_api_key() -> str:
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
# Retrieval initialization
# ---------------------------------------------------------------------------
def init_retrieval(chroma_path: str, collection_name: str) -> None:
    global _collection, _embedding_model

    try:
        client = chromadb.PersistentClient(path=chroma_path)
        _collection = client.get_or_create_collection(name=collection_name)
        print(
            f"[CHROMA] Connected to '{chroma_path}', collection "
            f"'{collection_name}' ({_collection.count()} vectors)."
        )
    except Exception as exc:
        print(f"[CHROMA][ERROR] Failed to initialize collection: {exc}", file=sys.stderr)
        raise

    try:
        print(f"[MODEL] Loading embedding model '{EMBEDDING_MODEL_NAME}' on CPU...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        print("[MODEL] Ready.")
    except Exception as exc:
        print(f"[MODEL][ERROR] Failed to load embedding model: {exc}", file=sys.stderr)
        raise


def load_restaurant_names(db_path: str) -> Dict[str, str]:
    """Best-effort restaurant_id -> name lookup, purely for readable context."""
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT id, name FROM restaurants;").fetchall()
        conn.close()
        return {restaurant_id: name for restaurant_id, name in rows}
    except sqlite3.Error as exc:
        print(f"[DB][WARN] Could not load restaurant names from {db_path}: {exc}", file=sys.stderr)
        return {}


# ---------------------------------------------------------------------------
# Retrieval (parameterized by credibility threshold => Variant A vs B)
# ---------------------------------------------------------------------------
def get_recommendations(query: str, min_weight: float, n_results: int = N_RESULTS) -> List[dict]:
    """
    Embed `query` and retrieve up to `n_results` reviews from ChromaDB whose
    stored credibility `weight` exceeds `min_weight` and are not spam.
    """
    if _collection is None or _embedding_model is None:
        raise RuntimeError("Retrieval not initialized. Call init_retrieval(...) first.")

    if not query or not query.strip():
        return []

    try:
        query_embedding = _embedding_model.encode(query, convert_to_numpy=True).tolist()

        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={
                "$and": [
                    {"weight": {"$gt": min_weight}},
                    {"is_spam": False},
                ]
            },
        )
    except Exception as exc:
        print(f"[RETRIEVAL][ERROR] ChromaDB query failed (min_weight={min_weight}): {exc}",
              file=sys.stderr)
        return []

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    recommendations = []
    for doc_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        recommendations.append(
            {
                "id": doc_id,
                "text": text,
                "restaurant_id": metadata.get("restaurant_id"),
                "weight": metadata.get("weight"),
                "distance": distance,
            }
        )
    return recommendations


def build_context(recommendations: List[dict]) -> str:
    """Combine retrieved review texts into a single numbered context block."""
    if not recommendations:
        return "No credible reviews were found for this query."

    lines = []
    for i, rec in enumerate(recommendations, start=1):
        restaurant_name = _restaurant_names.get(rec["restaurant_id"], rec["restaurant_id"])
        weight = rec.get("weight")
        weight_str = f"{weight:.2f}" if isinstance(weight, (int, float)) else "N/A"
        lines.append(
            f"Review {i} (restaurant: {restaurant_name}, credibility: {weight_str}):\n"
            f"\"{rec['text']}\""
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# LangChain pipelines: RAG assistant + Judge
# ---------------------------------------------------------------------------
def build_rag_chain(model_name: str, temperature: float, api_key: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_HUMAN_PROMPT),
        ]
    )
    llm = ChatGroq(model=model_name, temperature=temperature, groq_api_key=api_key)
    return prompt | llm | StrOutputParser()


def build_judge_chain(model_name: str, api_key: str):
    parser = PydanticOutputParser(pydantic_object=JudgeVerdict)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", JUDGE_SYSTEM_PROMPT),
            ("human", JUDGE_HUMAN_PROMPT),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    # temperature=0.0: the Judge should be as deterministic/strict as possible.
    llm = ChatGroq(model=model_name, temperature=0.0, groq_api_key=api_key)
    return prompt | llm | parser


# ---------------------------------------------------------------------------
# Retry wrapper (shared by RAG + Judge invocations)
# ---------------------------------------------------------------------------
def invoke_with_retry(chain, inputs: dict, label: str):
    attempt = 0
    while True:
        try:
            return chain.invoke(inputs)
        except Exception as exc:
            attempt += 1
            print(f"[{label}][WARN] Attempt {attempt} failed ({type(exc).__name__}): {exc}")
            if attempt > MAX_RETRIES:
                print(f"[{label}][ERROR] Exceeded max retries ({MAX_RETRIES}). Giving up.")
                return None
            print(f"[{label}] Retrying in {RETRY_DELAY_SECONDS}s...")
            time.sleep(RETRY_DELAY_SECONDS)


# ---------------------------------------------------------------------------
# Per-variant + per-query execution
# ---------------------------------------------------------------------------
def run_variant(rag_chain, query: str, min_weight: float, label: str) -> Dict:
    """Retrieve + generate a RAG response for one variant. Returns dict with
    context (for judging) and the generated response text."""
    recs = get_recommendations(query, min_weight=min_weight)
    context = build_context(recs)

    response = invoke_with_retry(
        rag_chain, {"context": context, "question": query}, label=label
    )
    time.sleep(CALL_SLEEP_SECONDS)

    if response is None:
        response = "[No response generated after repeated failures.]"

    return {"context": context, "response": response, "num_retrieved": len(recs)}


def run_judge(judge_chain, query: str, context: str, response: str, label: str) -> JudgeVerdict:
    verdict = invoke_with_retry(
        judge_chain,
        {"query": query, "context": context, "response": response},
        label=label,
    )
    time.sleep(CALL_SLEEP_SECONDS)

    if verdict is None:
        # Fail safe: lowest score, transparent about the failure.
        verdict = JudgeVerdict(score=1, explanation="Judge failed to produce a valid verdict.")
    return verdict


def determine_winner(score_a: int, score_b: int) -> str:
    if score_a > score_b:
        return "A"
    if score_b > score_a:
        return "B"
    return "Tie"


def process_query(rag_chain, judge_chain, query: str) -> Dict:
    print(f"\n[Query] {query}")

    variant_a = run_variant(rag_chain, query, VARIANT_A_MIN_WEIGHT, label="VARIANT-A")
    print(f"[Query] {query} -> [Variant A] response generated "
          f"({variant_a['num_retrieved']} reviews retrieved).")

    variant_b = run_variant(rag_chain, query, VARIANT_B_MIN_WEIGHT, label="VARIANT-B")
    print(f"[Query] {query} -> [Variant B] response generated "
          f"({variant_b['num_retrieved']} reviews retrieved).")

    judge_a = run_judge(
        judge_chain, query, variant_a["context"], variant_a["response"], label="JUDGE-A"
    )
    print(f"[Query] {query} -> [Variant A] -> [Judge Score] {judge_a.score}/10")

    judge_b = run_judge(
        judge_chain, query, variant_b["context"], variant_b["response"], label="JUDGE-B"
    )
    print(f"[Query] {query} -> [Variant B] -> [Judge Score] {judge_b.score}/10")

    winner = determine_winner(judge_a.score, judge_b.score)
    print(f"[Query] {query} -> [Winner] Variant {winner}")

    return {
        "query": query,
        "variant_a_response": variant_a["response"],
        "variant_b_response": variant_b["response"],
        "judge_scores": {
            "variant_a": {"score": judge_a.score, "explanation": judge_a.explanation},
            "variant_b": {"score": judge_b.score, "explanation": judge_b.explanation},
        },
        "winner": winner,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def append_result(output_path: str, record: Dict) -> None:
    try:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"[DB][ERROR] Failed to append result to {output_path}: {exc}", file=sys.stderr)
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="TrueRating: self-evaluating A/B testing engine for the RAG assistant."
    )
    parser.add_argument("--db", default=DEFAULT_DB_FILE, help="Path to the target database (for restaurant names)")
    parser.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH, help="ChromaDB persistent store path")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="ChromaDB collection name")
    parser.add_argument("--model", default=DEFAULT_GROQ_MODEL, help="Groq model for both assistant and judge")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Assistant sampling temperature")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Path to ab_test_results.jsonl")
    return parser.parse_args()


def main():
    global _restaurant_names

    args = parse_args()
    api_key = load_api_key()

    try:
        init_retrieval(args.chroma_path, args.collection)
        _restaurant_names = load_restaurant_names(args.db)
        rag_chain = build_rag_chain(args.model, args.temperature, api_key)
        judge_chain = build_judge_chain(args.model, api_key)
    except Exception as exc:
        print(f"[FATAL] Failed to initialize the A/B engine: {exc}", file=sys.stderr)
        sys.exit(1)

    tally = {"A": 0, "B": 0, "Tie": 0}

    for query in TEST_QUERIES:
        try:
            record = process_query(rag_chain, judge_chain, query)
        except Exception as exc:
            print(f"[QUERY][ERROR] Skipping query '{query}' due to: {exc}", file=sys.stderr)
            continue

        append_result(args.output, record)
        tally[record["winner"]] += 1

    print(
        f"\nA/B test complete. Results appended to '{args.output}'.\n"
        f"Final tally -> Variant A wins: {tally['A']}, "
        f"Variant B wins: {tally['B']}, Ties: {tally['Tie']}."
    )


if __name__ == "__main__":
    main()
