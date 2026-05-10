"""
dashboard/app.py — Regulatory RAG Query Dashboard

Streamlit interface for querying the regulatory RAG pipeline.
Connects to the FastAPI backend at localhost:8000.

Run with:
    streamlit run dashboard/app.py
"""

import requests
import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Regulatory Intelligence",
    page_icon  = "⚖️",
    layout     = "wide",
)

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

SOURCES = {
    "All Sources":        None,
    "SEC EDGAR":          "SEC_EDGAR",
    "FDA Drug Labels":    "FDA_DRUG_LABELS",
    "OSHA Standards":     "OSHA_STANDARDS",
    "Federal Register":   "FEDERAL_REGISTER",
    "NIST AI Documents":  "NIST",
}

EXAMPLE_QUERIES = [
    "What are the OSHA lockout tagout requirements?",
    "What are the drug interaction warnings for Naproxen?",
    "What does NIST say about AI risk management?",
    "What are SEC requirements for risk factor disclosure?",
    "What are OSHA requirements for respiratory protection?",
    "What are the FDA warnings for Mekinist?",
]

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

    .citation-card {
        background: white;
        border: 1px solid #e8f0e8;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
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
        if st.button(
            example[:50] + "..." if len(example) > 50 else example,
            use_container_width=True,
            key=f"example_{example[:20]}",
        ):
            st.session_state["query_input"] = example
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

# ── Main Query Interface ──────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_area(
        "Ask a regulatory question",
        value       = st.session_state.get("query_input", ""),
        height      = 100,
        placeholder = "e.g. What are the OSHA requirements for hazard communication?",
        key         = "query_box",
        label_visibility = "collapsed",
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_clicked = st.button("Search", use_container_width=True)

# ── API Health Check ──────────────────────────────────────────────────────────
def check_api() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# ── Query Execution ───────────────────────────────────────────────────────────
trigger = st.session_state.pop("trigger_search", False)
if (search_clicked or trigger) and query.strip():
    if not check_api():
        st.error("API is not running. Start it with: `uvicorn api.app:app --port 8000`")
    else:
        with st.spinner("Searching regulatory documents..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json    = {
                        "query":         query,
                        "top_k":         top_k,
                        "source_filter": source_filter,
                    },
                    timeout = 120,
                )
                response.raise_for_status()
                data = response.json()

                # ── Answer ────────────────────────────────────────────────────
                st.markdown("### Answer")
                st.markdown(
                    f'<div class="answer-box">{data["answer"]}</div>',
                    unsafe_allow_html=True,
                )

                st.caption(f"Model: {data['model']} · Sources: {len(data['citations'])}")

                # ── Citations ─────────────────────────────────────────────────
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

            except requests.exceptions.Timeout:
                st.error("Request timed out — the model is taking too long. Try a simpler query.")
            except Exception as exc:
                st.error(f"Error: {exc}")

elif search_clicked and not query.strip():
    st.warning("Please enter a query.")

# ── Empty State ───────────────────────────────────────────────────────────────
if not search_clicked:
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