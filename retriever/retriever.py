"""
retriever/retriever.py — Semantic Retrieval from Qdrant

Queries the local Qdrant collection using semantic search.
Returns ranked chunks with full provenance metadata.

Supports:
    - Free-text semantic search across all documents
    - Filtered search by source (SEC_EDGAR, FDA, OSHA, FEDERAL_REGISTER, NIST)
    - Configurable top-k results

Usage:
    python retriever/retriever.py     # standalone test
    retrieve(query)                   # called by api/app.py
"""

import logging
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import os

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("retriever")

# ── Config ────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "regulatory_docs"
MODEL_NAME      = "all-MiniLM-L6-v2"
TOP_K_DEFAULT   = 5
BGE_QUERY_PREFIX = "Represent this question for searching relevant passages: "

# ── Singletons — loaded once, reused across calls ─────────────────────────────
_model:  SentenceTransformer | None = None
_client: QdrantClient | None        = None


def _get_model() -> SentenceTransformer:
    """
    Load and cache the embedding model.

    Returns:
        Loaded SentenceTransformer model.
    """
    global _model
    if _model is None:
        logger.info("Loading retrieval model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Model loaded")
    return _model


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        qdrant_url     = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url and qdrant_api_key:
            _client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            logger.info("Connected to Qdrant Cloud: %s", qdrant_url)
        else:
            _client = QdrantClient(path="data/qdrant_local")
            logger.info("Connected to local Qdrant")
    return _client


# ── Core Retrieval ────────────────────────────────────────────────────────────

def retrieve(
    query:       str,
    top_k:       int = TOP_K_DEFAULT,
    source_filter: str | None = None,
) -> list[dict]:
    """
    Retrieve the most semantically relevant chunks for a query.

    Embeds the query with the same model used at indexing time.
    Optionally filters results to a specific regulatory source.

    Args:
        query:         Natural language query string.
        top_k:         Number of results to return (default 5).
        source_filter: Optional source to filter by. One of:
                       'SEC_EDGAR', 'FDA_DRUG_LABELS', 'OSHA_STANDARDS',
                       'FEDERAL_REGISTER', 'NIST'

    Returns:
        List of result dicts with keys:
            text, source, document_name, section,
            chunk_index, score, chunk_id
    """
    model  = _get_model()
    client = _get_client()

    # BGE-style query prefix — signals this is a search query not a passage
    query_text = BGE_QUERY_PREFIX + query

    query_vector = model.encode(
        query_text,
        normalize_embeddings = True,
    ).tolist()

    # Build optional source filter
    qdrant_filter = None
    if source_filter:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key   = "source",
                    match = MatchValue(value=source_filter),
                )
            ]
        )

    results = client.query_points(
        collection_name = COLLECTION_NAME,
        query           = query_vector,
        limit           = top_k,
        query_filter    = qdrant_filter,
        with_payload    = True,
    ).points

    hits = []
    for r in results:
        payload = r.payload or {}
        hits.append({
            "text":          payload.get("text", ""),
            "source":        payload.get("source", ""),
            "document_name": payload.get("document_name", ""),
            "section":       payload.get("section", ""),
            "chunk_index":   payload.get("chunk_index", 0),
            "score":         round(r.score, 4),
            "chunk_id":      payload.get("chunk_id", ""),
        })

    logger.info(
        "Query: '%s' | Source filter: %s | Results: %d",
        query[:60], source_filter or "none", len(hits),
    )
    return hits


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "What are the requirements for hazard communication in the workplace?",
        "What are the drug interaction warnings for Naproxen?",
        "What does the AI Act say about high-risk AI systems?",
        "What are the SEC requirements for risk factor disclosure?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print("="*60)
        results = retrieve(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] Score: {r['score']} | Source: {r['source']}")
            print(f"    Document: {r['document_name']}")
            print(f"    Section: {r['section']}")
            print(f"    Text: {r['text'][:200]}...")