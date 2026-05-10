"""
api/app.py — FastAPI Query Endpoint

Exposes the RAG pipeline as a REST API with two endpoints:
    GET  /health       — liveness check
    POST /query        — semantic search + LLM answer synthesis

The /query endpoint:
    1. Retrieves top-k relevant chunks from Qdrant
    2. Builds a grounded prompt with retrieved context
    3. Calls Groq (Llama 3 70B) to synthesise a cited answer
    4. Returns answer + source citations

Usage:
    uvicorn api.app:app --reload --port 8000
    curl -X POST http://localhost:8000/query \
         -H "Content-Type: application/json" \
         -d '{"query": "What are OSHA lockout tagout requirements?"}'
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel

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
logger = logging.getLogger("api")

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL      = "llama-3.3-70b-versatile"
TOP_K           = 5
MAX_CONTEXT_CHARS = 6000   # stay well within Llama 3's 8192 token context

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Regulatory RAG Pipeline API",
    description = "Semantic search and Q&A over SEC, FDA, OSHA, and AI policy documents",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Request / Response Models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:         str
    top_k:         int  = TOP_K
    source_filter: str | None = None


class SourceCitation(BaseModel):
    document_name: str
    source:        str
    section:       str
    score:         float
    text_snippet:  str


class QueryResponse(BaseModel):
    query:     str
    answer:    str
    citations: list[SourceCitation]
    model:     str


# ── Groq Client ───────────────────────────────────────────────────────────────

def _get_groq_client() -> Groq:
    """
    Initialise and return a Groq client.

    Raises:
        HTTPException: If GROQ_API_KEY is not set.
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    return Groq(api_key=GROQ_API_KEY)


# ── Answer Synthesis ──────────────────────────────────────────────────────────

def _build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build a grounded RAG prompt from retrieved chunks.

    Each chunk is presented with its source and section so the LLM
    can cite them accurately. The prompt instructs the model to
    answer only from provided context and cite sources inline.

    Args:
        query:  User query string.
        chunks: Retrieved chunk dicts from retriever.

    Returns:
        Formatted prompt string.
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
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(block)
        total_chars += len(block)

    context = "\n---\n".join(context_parts)

    return f"""You are a regulatory compliance expert. Answer the user's question using ONLY the provided regulatory documents.

Rules:
- Cite sources inline using [SOURCE N] notation
- If the answer is not in the documents, say so clearly
- Be precise and factual — this is regulatory content
- Keep the answer concise but complete

REGULATORY DOCUMENTS:
{context}

USER QUESTION: {query}

ANSWER:"""


def _synthesise_answer(query: str, chunks: list[dict]) -> str:
    """
    Call Groq LLM to synthesise a cited answer from retrieved chunks.

    Args:
        query:  User query string.
        chunks: Retrieved chunk dicts with text and metadata.

    Returns:
        LLM-generated answer string with inline citations.
    """
    client = _get_groq_client()
    prompt = _build_prompt(query, chunks)

    response = client.chat.completions.create(
        model       = GROQ_MODEL,
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0.1,   # low temperature for factual regulatory answers
        max_tokens  = 1024,
    )

    return response.choices[0].message.content.strip()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Liveness check — returns 200 if the API is running."""
    return {"status": "ok", "service": "regulatory-rag-api"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    Semantic search + LLM answer synthesis.

    Retrieves relevant regulatory chunks, synthesises a grounded
    answer using Groq Llama 3 70B, and returns citations.

    Args:
        request: QueryRequest with query, top_k, and optional source_filter.

    Returns:
        QueryResponse with answer text and source citations.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info("Query received: '%s'", request.query[:80])

    # Import here to avoid circular imports and use cached singletons
    from retriever.retriever import retrieve

    chunks = retrieve(
        query         = request.query,
        top_k         = request.top_k,
        source_filter = request.source_filter,
    )

    if not chunks:
        raise HTTPException(
            status_code = 404,
            detail      = "No relevant documents found for this query",
        )

    answer = _synthesise_answer(request.query, chunks)

    citations = [
        SourceCitation(
            document_name = c["document_name"],
            source        = c["source"],
            section       = c["section"],
            score         = c["score"],
            text_snippet  = c["text"][:200],
        )
        for c in chunks
    ]

    logger.info(
        "Answer synthesised — %d citations, model: %s",
        len(citations), GROQ_MODEL,
    )

    return QueryResponse(
        query     = request.query,
        answer    = answer,
        citations = citations,
        model     = GROQ_MODEL,
    )