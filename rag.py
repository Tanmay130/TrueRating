#!/usr/bin/env python3
"""
rag.py

RAG engine stage of the TrueRating platform (formerly phase5_rag.py).
Dataset-agnostic: works against any database/ChromaDB store built by
ingest.py + credibility.py, regardless of which adapter (Yelp, Zomato, ...)
produced the underlying data.

Retrieves credibility-weighted review passages from the ChromaDB collection
built in credibility.py (`truerating_reviews`) and uses them as grounding
context for a Groq/Llama chat model, via LangChain. Only reviews with a
credibility `weight` > 0.5 (and not flagged as spam) are eligible to be
retrieved, so low-trust or spammy reviews never reach the LLM as context.

Run it and ask questions interactively:
    python rag.py --db truerating.db --chroma-path ./chroma_db
    python rag.py --db truerating_zomato.db --chroma-path ./chroma_db_zomato

Requires (already present in requirements.txt / added in earlier stages):
    chromadb
    sentence-transformers
    langchain
    langchain-groq
    python-dotenv
"""

import argparse
import os
import sqlite3
import sys
from typing import Dict, List, Optional

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# NOTE: must match the model used to build the embeddings in credibility.py,
# otherwise query vectors and stored vectors won't live in the same
# embedding space.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_CHROMA_PATH = "./chroma_db"
DEFAULT_COLLECTION_NAME = "truerating_reviews"
DEFAULT_DB_FILE = "truerating.db"

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.0

DEFAULT_N_RESULTS = 5
MIN_CREDIBILITY_WEIGHT = 0.5

SYSTEM_PROMPT = (
    "You are a helpful restaurant assistant. Use the following credible "
    "reviews to answer the user's question. If the information isn't in "
    "the reviews, say you don't know."
)

HUMAN_PROMPT = "Credible reviews:\n{context}\n\nQuestion: {question}"


# ---------------------------------------------------------------------------
# Module-level retrieval state (initialized once in main() / init_retrieval())
# ---------------------------------------------------------------------------
_collection = None
_embedding_model: Optional[SentenceTransformer] = None
_restaurant_names: Dict[str, str] = {}


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
# Retrieval initialization
# ---------------------------------------------------------------------------
def init_retrieval(chroma_path: str, collection_name: str) -> None:
    """
    Initialize the ChromaDB client/collection and the sentence-transformer
    embedding model as module-level singletons, used by get_recommendations().
    """
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
    """
    Best-effort lookup of restaurant_id -> name from the target database,
    purely for nicer citations in the printed context. Retrieval still
    works fine without it.
    """
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT id, name FROM restaurants;").fetchall()
        conn.close()
        return {restaurant_id: name for restaurant_id, name in rows}
    except sqlite3.Error as exc:
        print(f"[DB][WARN] Could not load restaurant names from {db_path}: {exc}", file=sys.stderr)
        return {}


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def get_recommendations(query: str, n_results: int = DEFAULT_N_RESULTS) -> List[dict]:
    """
    Embed `query` with the same model used to build the credibility.py
    embeddings, then run a similarity search against the
    `truerating_reviews` ChromaDB collection, restricted to credible,
    non-spam reviews (weight > MIN_CREDIBILITY_WEIGHT and is_spam == False).

    Returns a list of dicts: {id, text, restaurant_id, weight, distance}.
    """
    if _collection is None or _embedding_model is None:
        raise RuntimeError(
            "Retrieval has not been initialized. Call init_retrieval(...) before "
            "get_recommendations()."
        )

    if not query or not query.strip():
        return []

    try:
        query_embedding = _embedding_model.encode(
            query, convert_to_numpy=True
        ).tolist()

        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={
                "$and": [
                    {"weight": {"$gt": MIN_CREDIBILITY_WEIGHT}},
                    {"is_spam": False},
                ]
            },
        )
    except Exception as exc:
        print(f"[RETRIEVAL][ERROR] ChromaDB query failed: {exc}", file=sys.stderr)
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


# ---------------------------------------------------------------------------
# Context synthesis
# ---------------------------------------------------------------------------
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
# LangChain pipeline
# ---------------------------------------------------------------------------
def build_chain(model_name: str, temperature: float, api_key: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )

    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        groq_api_key=api_key,
    )

    chain = prompt | llm | StrOutputParser()
    return chain


def answer_query(chain, query: str, n_results: int = DEFAULT_N_RESULTS) -> str:
    """Retrieve credible context for `query` and invoke the RAG chain."""
    recommendations = get_recommendations(query, n_results=n_results)
    context = build_context(recommendations)

    print(f"[RETRIEVAL] Found {len(recommendations)} credible review(s) "
          f"(weight > {MIN_CREDIBILITY_WEIGHT}).")

    try:
        return chain.invoke({"context": context, "question": query})
    except Exception as exc:
        print(f"[LLM][ERROR] Chain invocation failed: {exc}", file=sys.stderr)
        return "Sorry, I hit an error trying to answer that. Please try again."


# ---------------------------------------------------------------------------
# Terminal UI
# ---------------------------------------------------------------------------
def run_terminal_loop(chain, n_results: int) -> None:
    print("\nTrueRating RAG assistant. Ask a question about a restaurant.")
    print("Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        answer = answer_query(chain, query, n_results=n_results)
        print(f"\nAssistant: {answer}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="TrueRating: credibility-weighted RAG assistant."
    )
    parser.add_argument("--db", default=DEFAULT_DB_FILE, help="Path to the target database (for restaurant names)")
    parser.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH, help="ChromaDB persistent store path")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="ChromaDB collection name")
    parser.add_argument("--model", default=DEFAULT_GROQ_MODEL, help="Groq model name")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--n-results", type=int, default=DEFAULT_N_RESULTS, help="Reviews to retrieve per query")
    return parser.parse_args()


def main():
    global _restaurant_names

    args = parse_args()
    api_key = load_api_key()

    try:
        init_retrieval(args.chroma_path, args.collection)
        _restaurant_names = load_restaurant_names(args.db)
        chain = build_chain(args.model, args.temperature, api_key)
    except Exception as exc:
        print(f"[FATAL] Failed to initialize the RAG engine: {exc}", file=sys.stderr)
        sys.exit(1)

    run_terminal_loop(chain, n_results=args.n_results)


if __name__ == "__main__":
    main()
