"""
dashboard/app.py — Regulatory RAG Query Dashboard

Streamlit interface for querying the regulatory RAG pipeline.
Runs retrieval and LLM synthesis directly — no external API needed.
Designed for Streamlit Cloud deployment.

Run with:
    streamlit run dashboard/app.py
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Regulatory Intelligence",
    page_icon  = "⚖️",
    layout     = "wide",
)

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"

SOURCES = {
    "All Sources":       None,
    "SEC EDGAR":         "SEC_EDGAR",
    "FDA Drug Labels":   "FDA_DRUG_LABELS",
    "OSHA Standards":    "OSHA_STANDARDS",
    "Federal Register":  "FEDERAL_REGISTER",
    "NIST AI Documents": "NIST",
}

EXAMPLE_QUERIES = [
    "What are the OSHA lockout tagout requirements?",
    "What are the drug interaction warnings for Naproxen?",
    "What does NIST say about AI risk management?",
    "What are SEC requirements for risk factor disclosure?",
    "What are OSHA requirements for respiratory protection?",
    "What are the FDA warnings for Mekinist?",
]

# ── Qdrant Initialisation ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def initialise_qdrant():
    """
    Build local Qdrant vector store from chunks on first run.

    On Streamlit Cloud, the qdrant_local directory does not persist across
    deployments. This function checks if the store exists and builds it
    from the committed chunk JSON files if not.

    Cached with st.cache_resource so it only runs once per session.

    Returns:
        True when the vector store is ready.
    """
    qdrant_path = Path("data/qdrant_local")

    if not qdrant_path.exists() or not any(qdrant_path.iterdir()):
        with st.spinner("Building vector index on first run — this takes a few minutes..."):
            from embedder.embedder import embed_and_upsert
            embed_and_upsert()

    return True


# ── Core Pipeline Functions ───────────────────────────────────────────────────

def run_retrieval(query: str, top_k: int, source_filter: str | None) -> list[dict]:
    """
    Run semantic retrieval against the local Qdrant store.

    Args:
        query:         User query string.
        top_k:         Number of results to return.
        source_filter: Optional source domain filter.

    Returns:
        List of ranked chunk dicts with text and metadata.
    """
    from retriever.retriever import retrieve
    return retrieve(query=query, top_k=top_k, source_filter=source_filter)


def synthesise_answer(query: str, chunks: list[dict]) -> str:
    """
    Call Groq LLM to generate a grounded, cited answer from retrieved chunks.

    Args:
        query:  User query string.
        chunks: Retrieved chunk dicts with text and metadata.

    Returns:
        LLM-generated answer string with inline [SOURCE N] citations.
    """
    context_parts = []
    total_chars   = 0

    for i, chunk in enumerate(chunks, 1):
        snippet = chunk["text"][:800]
        block   = (
            f"[SOURCE {i}]\n"
            f"Document: {chunk['document_name']}\n"
            f"Source: {chunk['source']}\n"
            f"Section: {chunk['section']}\n"
            f"Content: {snippet}\n"
        )
        if total_chars + len(block) > 6000:
            break
        context_parts.append(block)
        total_chars += len(block)

    prompt = f"""You are a regulatory compliance expert. Answer the user's question using ONLY the provided regulatory documents.

Rules:
- Cite sources inline using [SOURCE N] notation
- If the answer is not in the documents, say so clearly
- Be precise and factual — this is regulatory content
- Keep the answer concise but complete

REGULATORY DOCUMENTS:
{"---".join(context_parts)}

USER QUESTION: {query}

ANSWER:"""

    client   = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model       = GROQ_MODEL,
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0.1,
        max_tokens  = 1024,
    )
    return response.choices[0].message.content.strip()


def query_pipeline(
    query:         str,
    top_k:         int,
    source_filter: str | None,
) -> dict:
    """
    Run the full RAG pipeline: retrieve + synthesise.

    Args:
        query:         User query string.
        top_k:         Number of source chunks to retrieve.
        source_filter: Optional source domain filter.

    Returns:
        Dict with answer, citations list, and model name.
    """
    chunks = run_retrieval(query, top_k, source_filter)

    if not chunks:
        return {
            "answer":    "No relevant documents found for this query.",
            "citations": [],
            "model":     GROQ_MODEL,
        }

    answer = synthesise_answer(query, chunks)

    citations = [
        {
            "document_name": c["document_name"],
            "source":        c["source"],
            "section":       c["section"],
            "score":         c["score"],
            "text_snippet":  c["text"][:200],
        }
        for c in chunks
    ]

    return {
        "answer":    answer,
        "citations": citations,
        "model":     GROQ_MODEL,
    }


# ── Initialise Vector Store ───────────────────────────────────────────────────
initialise_qdrant()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp { background: #f7f9f7; }

    .dash-header {
        background: linear-gradient(135deg, #1a4a1a 0%, #2d7a2d 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .dash-header h1 { color: white; font-size: 1.8rem; margin: 0; font-weight: 600; }
    .dash-header p  { color: #a8d5a8; margin: 0.25rem 0 0 0; font-size: 0.9rem; }

    .answer-box {
        background: white;
        border: 1px solid #d4e8d4;
        border-left: 4px solid #2d7a2d;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        line-height: 1.7;
    }

    .score-badge {
        display: inline-block;
        background: #e6f4e6;
        color: #1a4a1a;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'DM Mono', monospace;
    }

    .source-tag {
        display: inline-block;
        background: #1a4a1a;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 500;
        margin-right: 6px;
    }

    [data-testid="stTextArea"] textarea {
        border: 2px solid #d4e8d4 !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #2d7a2d !important;
    }

    .stButton button {
        background: #1a4a1a !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 0.5rem 2rem !important;
    }

    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
    <h1>Regulatory Intelligence</h1>
    <p>Semantic search and Q&A over SEC filings · FDA drug labels · OSHA standards · AI policy documents</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Search Settings")

    source_label  = st.selectbox("Filter by Source", options=list(SOURCES.keys()))
    source_filter = SOURCES[source_label]

    top_k = st.slider("Number of Sources", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.markdown("### Example Queries")
    for example in EXAMPLE_QUERIES:
        label = example[:50] + "..." if len(example) > 50 else example
        if st.button(label, use_container_width=True, key=f"ex_{example[:20]}"):
            st.session_state["query_input"]   = example
            st.session_state["trigger_search"] = True
            st.rerun()

    st.markdown("---")
    st.markdown("### Data Sources")
    st.markdown("""
- **SEC EDGAR** — 10-K, 8-K, DEF 14A filings
- **FDA** — Drug labels and warnings
- **OSHA** — Workplace safety standards
- **AI Policy** — Federal Register + NIST AI RMF
    """)

    st.markdown("---")
    st.markdown("### Evaluation")
    st.markdown("""
| Metric | Score |
|---|---|
| Source Accuracy | **100%** |
| Keyword Hit Rate | **100%** |
| Avg Cosine Score | **0.663** |
| Avg Latency | **~105ms** |
    """)

# ── Main Query Interface ──────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_area(
        "Ask a regulatory question",
        value            = st.session_state.get("query_input", ""),
        height           = 100,
        placeholder      = "e.g. What are the OSHA requirements for hazard communication?",
        key              = "query_box",
        label_visibility = "collapsed",
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_clicked = st.button("Search", use_container_width=True)

# ── Query Execution ───────────────────────────────────────────────────────────
trigger = st.session_state.pop("trigger_search", False)

if (search_clicked or trigger) and query.strip():
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not set. Add it to your .env file or Streamlit secrets.")
    else:
        with st.spinner("Searching regulatory documents..."):
            try:
                data = query_pipeline(query, top_k, source_filter)

                # ── Answer ────────────────────────────────────────────────────
                st.markdown("### Answer")
                st.markdown(
                    f'<div class="answer-box">{data["answer"]}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Model: {data['model']} · Sources: {len(data['citations'])}"
                )

                # ── Citations ─────────────────────────────────────────────────
                if data["citations"]:
                    st.markdown("### Source Citations")
                    for i, citation in enumerate(data["citations"], 1):
                        with st.expander(
                            f"[{i}] {citation['document_name']} · Score: {citation['score']}",
                            expanded = i == 1,
                        ):
                            st.markdown(
                                f'<span class="source-tag">{citation["source"]}</span>'
                                f'<span class="score-badge">{citation["score"]}</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"**Section:** {citation['section']}")
                            st.markdown(f"**Snippet:** {citation['text_snippet']}...")

            except Exception as exc:
                st.error(f"Error: {exc}")

elif search_clicked and not query.strip():
    st.warning("Please enter a query.")

# ── Empty State ───────────────────────────────────────────────────────────────
if not search_clicked and not trigger:
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #888;">
        <div style="font-size: 3rem; color: #2d7a2d; font-weight: 600;">RAG</div>
        <div style="font-size: 1.1rem; margin-top: 1rem;">
            Ask any question about regulatory requirements
        </div>
        <div style="font-size: 0.85rem; margin-top: 0.5rem;">
            Try the example queries in the sidebar to get started
        </div>
    </div>
    """, unsafe_allow_html=True)