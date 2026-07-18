"""
TrueRating Command Palette
---------------------------
A Raycast-style search UI over the TrueRating RAG pipeline.

Requirements:
    pip install streamlit langchain-groq langchain-core chromadb sentence-transformers \
                python-dotenv pandas plotly

Run:
    streamlit run app.py
"""

import csv
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import chromadb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Overridable via environment variables so you can point the app at a
# different dataset (e.g. the Zomato adapter) without editing this file:
#   TRUERATING_DB=truerating_zomato.db TRUERATING_CHROMA_PATH=./chroma_db_zomato streamlit run app.py
CHROMA_PATH = os.getenv("TRUERATING_CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("TRUERATING_COLLECTION", "truerating_reviews")
DB_FILE = os.getenv("TRUERATING_DB", "truerating.db")
FEEDBACK_FILE = "user_feedback.csv"

GROQ_MODEL = "llama-3.1-8b-instant"
TEMPERATURE = 0.0
N_RESULTS = 5
MIN_CREDIBILITY_WEIGHT = 0.5

ASPECT_COLUMNS = ["taste", "hygiene", "service", "value", "delivery"]
CHART_ACCENT = "#60a5fa"
CHART_ACCENT_DIM = "#1e3a8a"

# Distinct, high-contrast hues (not just shades of one color) so restaurants
# are easy to tell apart at a glance. Same restaurant gets the same color
# across every chart in the dashboard.
QUALITATIVE_PALETTE = [
    "#60a5fa",  # blue
    "#f472b6",  # pink
    "#34d399",  # green
    "#fbbf24",  # amber
    "#c084fc",  # purple
    "#fb7185",  # rose
    "#38bdf8",  # cyan
    "#a3e635",  # lime
]
CREDIBILITY_COLOR_MAP = {
    "High (>0.8)": "#60a5fa",
    "Moderate (0.5-0.8)": "#fbbf24",
}

SYSTEM_PROMPT = (
    "You are a sharp, concise restaurant assistant embedded in a command "
    "palette. Use the credible reviews below to answer the user's question "
    "in a few tight sentences. If the reviews don't cover it, say you don't "
    "know instead of guessing."
)
HUMAN_PROMPT = "Credible reviews:\n{context}\n\nQuestion: {question}"

# --- A/B testing (Variant A vs Variant B, judged by an LLM referee) ---------
VARIANT_A_MIN_WEIGHT = 0.5
VARIANT_B_MIN_WEIGHT = 0.8
AB_RESULTS_FILE = "ab_test_results.jsonl"
AB_JUDGE_MAX_RETRIES = 2
AB_JUDGE_RETRY_DELAY_SECONDS = 2
AB_VARIANT_COLORS = {"Variant A": "#60a5fa", "Variant B": "#f472b6"}

AB_JUDGE_SYSTEM_PROMPT = """You are an impartial judge evaluating an AI restaurant \
assistant's answer to a user's question.

Score the response from 1 (very poor) to 10 (excellent) on each of these:
  - relevance: does it directly answer the user's query?
  - evidence: does it explicitly rely on the provided reviews rather than \
making things up?
  - conciseness: is it helpful without unnecessary filler?

You are a strict data parser, not a creative writer. Return ONLY valid JSON \
matching this schema, with no extra commentary:

{format_instructions}
"""
AB_JUDGE_HUMAN_PROMPT = (
    "User query:\n{query}\n\n"
    "Reviews given to the assistant:\n{context}\n\n"
    "Assistant's response:\n{response}\n\n"
    "Evaluate this response now."
)

st.set_page_config(
    page_title="TrueRating",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling — Raycast-inspired command palette
# ---------------------------------------------------------------------------
PALETTE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --void: #000000;
    --deep: #0a1128;
    --glow: #60a5fa;
    --glow-soft: rgba(96, 165, 250, 0.35);
    --ink: #e7ecf7;
    --ink-dim: #9aa7c7;
}

#MainMenu, header, footer, [data-testid="stSidebar"], [data-testid="stToolbar"],
[data-testid="stDecoration"] {
    visibility: hidden !important;
    height: 0 !important;
}

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    font-size: 18px;
}

.stApp {
    background: radial-gradient(circle at 50% 0%, var(--deep) 0%, var(--void) 70%);
}

.block-container {
    max-width: 1180px;
    margin: 5vh auto 4rem auto;
    padding: 2.5rem 3.25rem 3rem 3.25rem;
    background: rgba(10, 17, 40, 0.72);
    border: 1px solid rgba(96, 165, 250, 0.25);
    border-radius: 24px;
    box-shadow: 0 0 0 1px rgba(96, 165, 250, 0.08),
                0 20px 60px rgba(0, 0, 0, 0.55),
                0 0 40px rgba(96, 165, 250, 0.10);
    backdrop-filter: blur(18px);
}

@keyframes logoPulse {
    from { opacity: 0; transform: translateY(-6px) scale(0.92); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

.palette-logo-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    animation: logoPulse 520ms cubic-bezier(0.22, 1, 0.36, 1);
}

.palette-logo-icon {
    filter: drop-shadow(0 0 10px rgba(96, 165, 250, 0.55))
            drop-shadow(0 0 26px rgba(96, 165, 250, 0.28));
    margin-bottom: 0.75rem;
}

.palette-title {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 1.9rem;
    letter-spacing: -0.02em;
    color: var(--ink);
    text-align: center;
    margin-bottom: 0.35rem;
}

.palette-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    color: var(--ink-dim);
    text-align: center;
    margin-bottom: 1.75rem;
}

div[data-testid="stTextInput"] input {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    color: var(--ink);
    background: rgba(0, 0, 0, 0.55);
    border: 1.5px solid rgba(96, 165, 250, 0.35);
    border-radius: 16px;
    padding: 0.95rem 1.2rem;
    transition: border-color 160ms ease, box-shadow 160ms ease;
}

div[data-testid="stTextInput"] input:focus {
    border-color: var(--glow);
    box-shadow: 0 0 0 4px var(--glow-soft), 0 0 24px rgba(96, 165, 250, 0.35);
    outline: none;
}

div[data-testid="stTextInput"] input::placeholder {
    color: var(--ink-dim);
    opacity: 0.8;
}

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

.response-card {
    margin-top: 1.75rem;
    padding: 1.5rem 1.65rem;
    background: rgba(96, 165, 250, 0.08);
    border: 1px solid rgba(96, 165, 250, 0.30);
    border-radius: 18px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    animation: fadeSlideUp 480ms cubic-bezier(0.22, 1, 0.36, 1);
}

.response-card p, .response-card div {
    font-family: 'Sora', sans-serif;
    font-size: 1.15rem;
    line-height: 1.65;
    color: var(--ink);
}

.response-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--glow);
    margin-bottom: 0.6rem;
}

.dashboard-card {
    margin-top: 1.5rem;
    padding: 1.4rem 1.5rem 0.6rem 1.5rem;
    background: rgba(96, 165, 250, 0.06);
    border: 1px solid rgba(96, 165, 250, 0.25);
    border-radius: 18px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    animation: fadeSlideUp 540ms cubic-bezier(0.22, 1, 0.36, 1);
}

div[data-testid="stPlotlyChart"] {
    margin-bottom: 0.4rem;
}

.mini-chart-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--ink);
    text-align: center;
    margin: 0.2rem 0 0.1rem 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    cursor: default;
}

div[data-testid="stExpander"] {
    margin-top: 1.1rem;
    border: 1px solid rgba(96, 165, 250, 0.20);
    border-radius: 14px;
    background: rgba(0, 0, 0, 0.35);
    animation: fadeSlideUp 560ms cubic-bezier(0.22, 1, 0.36, 1);
}

div[data-testid="stExpander"] summary {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    color: var(--ink-dim);
}

.stButton button {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    background: rgba(96, 165, 250, 0.10);
    border: 1px solid rgba(96, 165, 250, 0.35);
    border-radius: 12px;
    color: var(--ink);
    transition: all 140ms ease;
}

.stButton button:hover {
    border-color: var(--glow);
    box-shadow: 0 0 16px rgba(96, 165, 250, 0.35);
    color: var(--glow);
}

.source-row {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.92rem;
    color: var(--ink-dim);
    padding: 0.35rem 0;
    border-bottom: 1px solid rgba(96, 165, 250, 0.10);
}

.source-weight {
    color: var(--glow);
    font-weight: 600;
}

.ab-response-box {
    background: rgba(0, 0, 0, 0.35);
    border: 1px solid rgba(96, 165, 250, 0.15);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    font-family: 'Sora', sans-serif;
    font-size: 0.98rem;
    line-height: 1.55;
    color: var(--ink);
    max-height: 220px;
    overflow-y: auto;
}

.verdict-box {
    margin-top: 1.4rem;
    padding: 1.2rem 1.4rem;
    background: rgba(251, 191, 36, 0.08);
    border: 1px solid rgba(251, 191, 36, 0.35);
    border-radius: 16px;
    animation: fadeSlideUp 500ms cubic-bezier(0.22, 1, 0.36, 1);
}

.verdict-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: #fbbf24;
    margin-bottom: 0.5rem;
}

.verdict-explanation {
    font-family: 'Sora', sans-serif;
    font-size: 1rem;
    line-height: 1.6;
    color: var(--ink);
}
</style>
"""

st.markdown(PALETTE_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Logo — inline SVG so the app stays a single portable file (no image assets)
# ---------------------------------------------------------------------------
LOGO_HTML = """
<div class="palette-logo-wrap">
    <svg width="88" height="88" viewBox="0 0 52 52" fill="none"
         xmlns="http://www.w3.org/2000/svg" class="palette-logo-icon">
        <rect x="10" y="10" width="24" height="24" rx="8"
              transform="rotate(45 26 26)"
              fill="rgba(96,165,250,0.10)" stroke="#60a5fa" stroke-width="2"/>
        <g transform="translate(14,14)">
            <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"
                  fill="#60a5fa"/>
        </g>
    </svg>
    <div class="palette-title">TrueRating</div>
    <div class="palette-subtitle">ask anything about the restaurants you've reviewed</div>
</div>
"""


# ---------------------------------------------------------------------------
# Cached backend resources (loaded once per server process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")


@st.cache_resource(show_spinner=False)
def get_restaurant_names():
    if not Path(DB_FILE).exists():
        return {}
    try:
        conn = sqlite3.connect(DB_FILE)
        rows = conn.execute("SELECT id, name FROM restaurants;").fetchall()
        conn.close()
        return {rid: name for rid, name in rows}
    except sqlite3.Error:
        return {}


@st.cache_resource(show_spinner=False)
def get_rag_chain():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY is missing from your .env file.")
        st.stop()

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )
    llm = ChatGroq(model=GROQ_MODEL, temperature=TEMPERATURE, groq_api_key=api_key)
    return prompt | llm | StrOutputParser()


class AspectJudgeVerdict(BaseModel):
    """The referee's per-criterion scores for a single RAG response."""

    relevance: int = Field(ge=1, le=10)
    evidence: int = Field(ge=1, le=10)
    conciseness: int = Field(ge=1, le=10)
    explanation: str

    @property
    def overall(self) -> float:
        return round((self.relevance + self.evidence + self.conciseness) / 3, 2)


@st.cache_resource(show_spinner=False)
def get_judge_chain():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY is missing from your .env file.")
        st.stop()

    parser = PydanticOutputParser(pydantic_object=AspectJudgeVerdict)
    prompt = ChatPromptTemplate.from_messages(
        [("system", AB_JUDGE_SYSTEM_PROMPT), ("human", AB_JUDGE_HUMAN_PROMPT)]
    ).partial(format_instructions=parser.get_format_instructions())

    # temperature=0.0: the referee should be as consistent/strict as possible.
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.0, groq_api_key=api_key)
    return prompt | llm | parser


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def get_recommendations(
    query: str, n_results: int = N_RESULTS, min_weight: float = MIN_CREDIBILITY_WEIGHT
) -> list:
    collection = get_chroma_collection()
    model = get_embedding_model()

    query_embedding = model.encode(query, convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={
            "$and": [
                {"weight": {"$gt": min_weight}},
                {"is_spam": False},
            ]
        },
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    recs = []
    for doc_id, text, meta in zip(ids, documents, metadatas):
        recs.append(
            {
                "id": doc_id,
                "text": text,
                "restaurant_id": meta.get("restaurant_id"),
                "weight": meta.get("weight"),
            }
        )
    return recs


def build_context(recs: list) -> str:
    if not recs:
        return "No credible reviews were found for this query."

    names = get_restaurant_names()
    lines = []
    for i, rec in enumerate(recs, start=1):
        label = names.get(rec["restaurant_id"], rec["restaurant_id"])
        weight = rec.get("weight")
        weight_str = f"{weight:.2f}" if isinstance(weight, (int, float)) else "N/A"
        lines.append(f'Review {i} ({label}, credibility {weight_str}): "{rec["text"]}"')
    return "\n\n".join(lines)


def get_restaurant_scores(restaurant_ids: list) -> pd.DataFrame:
    """
    Pull each restaurant's Phase 4 aggregate scores (restaurant_scores table)
    for whichever restaurants were actually retrieved this query. Returns an
    empty DataFrame if the table doesn't exist yet or nothing matches — the
    dashboard just skips rendering in that case rather than erroring.
    """
    if not restaurant_ids or not Path(DB_FILE).exists():
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(DB_FILE)
        placeholders = ",".join("?" for _ in restaurant_ids)
        query = f"""
            SELECT restaurant_id, taste, hygiene, service, "value" AS value,
                   delivery, true_overall_rating, review_count
            FROM restaurant_scores
            WHERE restaurant_id IN ({placeholders});
        """
        df = pd.read_sql_query(query, conn, params=restaurant_ids)
        conn.close()
        return df
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()


def _style_chart(fig, height: int = 380, bottom_margin: int = 40, tickangle: int = 0,
                  tick_font_size: int = 12):
    """Shared transparent/dark styling so charts blend into the glass panel."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#e7ecf7", size=13),
        height=height,
        margin=dict(l=10, r=10, t=10, b=bottom_margin),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
    )
    fig.update_xaxes(tickangle=tickangle, tickfont=dict(size=tick_font_size))
    return fig


def _color_map_for(names: list) -> dict:
    """Assign each restaurant a fixed, distinct color so it stays the same
    hue across every chart in the dashboard (not just shades of one color)."""
    unique_names = sorted(set(names))
    return {
        name: QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)]
        for i, name in enumerate(unique_names)
    }


def _line_chart_for_restaurant(row: pd.Series, color: str) -> go.Figure:
    """
    One small line chart for a single restaurant's 5 aspect scores.
    Aspects with no data (NaN — never mentioned in any credible review) are
    left as real gaps in the line rather than plotted as 0, so a missing
    aspect never looks like a "neutral" score.
    """
    values = [row[aspect] for aspect in ASPECT_COLUMNS]

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dot", line_width=1, line_color="rgba(154, 167, 199, 0.35)")
    fig.add_trace(
        go.Scatter(
            x=ASPECT_COLUMNS,
            y=values,
            mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(color=color, size=9),
            connectgaps=False,
        )
    )
    fig.update_layout(template="plotly_dark", showlegend=False)
    fig.update_yaxes(range=[-1, 1], gridcolor="rgba(96, 165, 250, 0.12)")
    fig.update_xaxes(gridcolor="rgba(96, 165, 250, 0.08)")
    return fig


def render_dashboard(recs: list) -> None:
    """
    Build a small live dashboard for the restaurants behind this response:
    one mini line chart per restaurant (small multiples, so aspect scores
    stay readable even with 5+ restaurants), an overall-rating comparison
    bar, and a pie chart showing the credibility mix of the retrieved
    reviews. Renders nothing if there's no restaurant_scores data yet.
    """
    if not recs:
        return

    restaurant_ids = sorted({rec["restaurant_id"] for rec in recs if rec.get("restaurant_id")})
    scores_df = get_restaurant_scores(restaurant_ids)
    if scores_df.empty:
        return

    names = get_restaurant_names()
    scores_df["restaurant_name"] = scores_df["restaurant_id"].map(lambda rid: names.get(rid, rid))
    color_map = _color_map_for(scores_df["restaurant_name"])

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="response-label">Aspect Breakdown</div>', unsafe_allow_html=True)
    st.caption(
        "Each line: taste → hygiene → service → value → delivery, -1 to 1. "
        "A gap means that aspect was never mentioned in a credible review."
    )

    charts_per_row = 3
    restaurant_rows = [row for _, row in scores_df.iterrows()]
    for i in range(0, len(restaurant_rows), charts_per_row):
        row_chunk = restaurant_rows[i : i + charts_per_row]
        cols = st.columns(charts_per_row, gap="medium")
        for col, restaurant_row in zip(cols, row_chunk):
            with col:
                name = restaurant_row["restaurant_name"]
                st.markdown(
                    f'<div class="mini-chart-title" title="{name}">{name}</div>',
                    unsafe_allow_html=True,
                )
                color = color_map[name]
                line_fig = _line_chart_for_restaurant(restaurant_row, color)
                st.plotly_chart(
                    _style_chart(line_fig, height=260, bottom_margin=36),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<div class="response-label">Overall Rating</div>', unsafe_allow_html=True)
        rating_fig = px.bar(
            scores_df,
            x="restaurant_name",
            y="true_overall_rating",
            color="restaurant_name",
            range_y=[-1, 1],
            template="plotly_dark",
            color_discrete_map=color_map,
        )
        rating_fig.update_layout(showlegend=False)
        st.plotly_chart(
            _style_chart(rating_fig, height=420, bottom_margin=130, tickangle=-40, tick_font_size=11),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with col_b:
        st.markdown('<div class="response-label">Review Credibility Mix</div>', unsafe_allow_html=True)
        credibility_labels = [
            "High (>0.8)" if (rec.get("weight") or 0) > 0.8 else "Moderate (0.5-0.8)"
            for rec in recs
        ]
        credibility_counts = pd.Series(credibility_labels).value_counts().reset_index()
        credibility_counts.columns = ["credibility", "count"]
        pie_fig = px.pie(
            credibility_counts,
            names="credibility",
            values="count",
            hole=0.55,
            template="plotly_dark",
            color="credibility",
            color_discrete_map=CREDIBILITY_COLOR_MAP,
        )
        st.plotly_chart(
            _style_chart(pie_fig, height=360),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown("</div>", unsafe_allow_html=True)


def run_ab_variant(query: str, min_weight: float) -> dict:
    """Retrieve + generate a RAG response for one retrieval strategy."""
    chain = get_rag_chain()
    recs = get_recommendations(query, min_weight=min_weight)
    context = build_context(recs)
    response = chain.invoke({"context": context, "question": query})
    return {"response": response, "context": context, "recs": recs}


def run_ab_judge(query: str, context: str, response: str) -> AspectJudgeVerdict:
    """Score one response with the referee LLM, retrying on parse/API errors."""
    judge_chain = get_judge_chain()
    attempt = 0
    while True:
        try:
            return judge_chain.invoke({"query": query, "context": context, "response": response})
        except Exception:
            attempt += 1
            if attempt > AB_JUDGE_MAX_RETRIES:
                return AspectJudgeVerdict(
                    relevance=1,
                    evidence=1,
                    conciseness=1,
                    explanation="The referee failed to produce a valid verdict after retries.",
                )
            time.sleep(AB_JUDGE_RETRY_DELAY_SECONDS)


def run_ab_test(query: str) -> dict:
    """Run Variant A vs Variant B end-to-end and have the referee score both."""
    variant_a = run_ab_variant(query, VARIANT_A_MIN_WEIGHT)
    variant_b = run_ab_variant(query, VARIANT_B_MIN_WEIGHT)

    judge_a = run_ab_judge(query, variant_a["context"], variant_a["response"])
    judge_b = run_ab_judge(query, variant_b["context"], variant_b["response"])

    if judge_a.overall > judge_b.overall:
        winner = "A"
    elif judge_b.overall > judge_a.overall:
        winner = "B"
    else:
        winner = "Tie"

    return {
        "query": query,
        "variant_a": {"response": variant_a["response"], "judge": judge_a},
        "variant_b": {"response": variant_b["response"], "judge": judge_b},
        "winner": winner,
    }


def save_ab_result(result: dict) -> None:
    """Append one A/B test result to ab_test_results.jsonl for later review."""
    judge_a = result["variant_a"]["judge"]
    judge_b = result["variant_b"]["judge"]
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": result["query"],
        "variant_a_response": result["variant_a"]["response"],
        "variant_b_response": result["variant_b"]["response"],
        "judge_scores": {
            "variant_a": {**judge_a.model_dump(), "overall": judge_a.overall},
            "variant_b": {**judge_b.model_dump(), "overall": judge_b.overall},
        },
        "winner": result["winner"],
    }
    try:
        with open(AB_RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        st.warning(f"Couldn't save A/B result to {AB_RESULTS_FILE}: {exc}")


def render_ab_results(result: dict) -> None:
    """Render the two variant responses, a per-criterion comparison chart,
    and the referee's final verdict."""
    judge_a = result["variant_a"]["judge"]
    judge_b = result["variant_b"]["judge"]

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="response-label">A/B Test — Variant A (weight &gt; 0.5) '
        'vs Variant B (weight &gt; 0.8)</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown('<div class="mini-chart-title">Variant A response</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ab-response-box">{result["variant_a"]["response"]}</div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown('<div class="mini-chart-title">Variant B response</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ab-response-box">{result["variant_b"]["response"]}</div>',
            unsafe_allow_html=True,
        )

    comparison_df = pd.DataFrame(
        [
            {"criterion": "Relevance", "variant": "Variant A", "score": judge_a.relevance},
            {"criterion": "Relevance", "variant": "Variant B", "score": judge_b.relevance},
            {"criterion": "Evidence", "variant": "Variant A", "score": judge_a.evidence},
            {"criterion": "Evidence", "variant": "Variant B", "score": judge_b.evidence},
            {"criterion": "Conciseness", "variant": "Variant A", "score": judge_a.conciseness},
            {"criterion": "Conciseness", "variant": "Variant B", "score": judge_b.conciseness},
            {"criterion": "Overall", "variant": "Variant A", "score": judge_a.overall},
            {"criterion": "Overall", "variant": "Variant B", "score": judge_b.overall},
        ]
    )
    comparison_fig = px.bar(
        comparison_df,
        x="criterion",
        y="score",
        color="variant",
        barmode="group",
        range_y=[0, 10],
        template="plotly_dark",
        color_discrete_map=AB_VARIANT_COLORS,
    )
    st.plotly_chart(
        _style_chart(comparison_fig, height=380, bottom_margin=50),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    verdict_label = {
        "A": "Variant A wins",
        "B": "Variant B wins",
        "Tie": "It's a tie",
    }[result["winner"]]

    if result["winner"] == "A":
        explanation_html = f"<b>Variant A:</b> {judge_a.explanation}"
    elif result["winner"] == "B":
        explanation_html = f"<b>Variant B:</b> {judge_b.explanation}"
    else:
        explanation_html = (
            f"<b>Variant A:</b> {judge_a.explanation}<br><br>"
            f"<b>Variant B:</b> {judge_b.explanation}"
        )

    st.markdown('<div class="verdict-box">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="verdict-title">Referee\'s Verdict — {verdict_label}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="verdict-explanation">{explanation_html}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def save_feedback(query: str, response: str, feedback: str) -> None:
    is_new_file = not Path(FEEDBACK_FILE).exists()
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["timestamp", "query", "response", "feedback"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), query, response, feedback])


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for key, default in {
    "last_processed_query": "",
    "last_response": "",
    "last_sources": [],
    "feedback_submitted": False,
    "ab_result": None,
    "ab_result_query": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.markdown(LOGO_HTML, unsafe_allow_html=True)

query = st.text_input(
    "search",
    placeholder="e.g. which places have the fastest delivery?",
    label_visibility="collapsed",
    key="query_input",
)

is_new_query = bool(query) and query != st.session_state["last_processed_query"]

if is_new_query:
    chain = get_rag_chain()
    recs = get_recommendations(query)
    context = build_context(recs)

    st.markdown('<div class="response-card">', unsafe_allow_html=True)
    st.markdown('<div class="response-label">Answer</div>', unsafe_allow_html=True)

    def _stream():
        for chunk in chain.stream({"context": context, "question": query}):
            yield chunk

    full_response = st.write_stream(_stream)
    st.markdown("</div>", unsafe_allow_html=True)

    st.session_state["last_processed_query"] = query
    st.session_state["last_response"] = full_response
    st.session_state["last_sources"] = recs
    st.session_state["feedback_submitted"] = False
    st.session_state["ab_result"] = None
    st.session_state["ab_result_query"] = ""

elif st.session_state["last_response"] and query == st.session_state["last_processed_query"]:
    st.markdown('<div class="response-card">', unsafe_allow_html=True)
    st.markdown('<div class="response-label">Answer</div>', unsafe_allow_html=True)
    st.markdown(f"<div>{st.session_state['last_response']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state["last_response"] and query == st.session_state["last_processed_query"]:
    render_dashboard(st.session_state["last_sources"])

    with st.expander("Source reviews"):
        if not st.session_state["last_sources"]:
            st.write("No credible reviews were retrieved for this query.")
        else:
            names = get_restaurant_names()
            for rec in st.session_state["last_sources"]:
                label = names.get(rec["restaurant_id"], rec["restaurant_id"])
                weight = rec.get("weight")
                weight_str = f"{weight:.2f}" if isinstance(weight, (int, float)) else "N/A"
                st.markdown(
                    f'<div class="source-row">{label} '
                    f'<span class="source-weight">[{weight_str}]</span> — {rec["text"][:140]}</div>',
                    unsafe_allow_html=True,
                )

    col1, col2, _ = st.columns([1, 1, 4])
    if not st.session_state["feedback_submitted"]:
        with col1:
            if st.button("👍", key="thumbs_up"):
                save_feedback(query, st.session_state["last_response"], "up")
                st.session_state["feedback_submitted"] = True
                st.rerun()
        with col2:
            if st.button("👎", key="thumbs_down"):
                save_feedback(query, st.session_state["last_response"], "down")
                st.session_state["feedback_submitted"] = True
                st.rerun()
    else:
        st.caption("Thanks — feedback logged.")

    st.markdown(
        '<div class="response-label" style="margin-top:1.75rem;">A/B Testing</div>',
        unsafe_allow_html=True,
    )
    if st.button("⚡ Generate A/B Testing", key="ab_test_button"):
        with st.spinner("Running Variant A vs Variant B, then asking the referee..."):
            ab_result = run_ab_test(query)
            save_ab_result(ab_result)
        st.session_state["ab_result"] = ab_result
        st.session_state["ab_result_query"] = query
        st.rerun()

    if (
        st.session_state["ab_result"] is not None
        and st.session_state["ab_result_query"] == query
    ):
        render_ab_results(st.session_state["ab_result"])
