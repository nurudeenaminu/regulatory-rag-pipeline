"""
embedder/embedder.py — Chunk Embedding and Qdrant Upsert

Loads all chunks from data/chunks/, embeds each using BAAI/bge-base-en-v1.5,
and upserts to a Qdrant collection with full metadata as payload.

Key design decisions:
    - Batch embedding for efficiency (32 chunks per batch)
    - Qdrant upsert is idempotent — safe to rerun
    - chunk_id used as Qdrant point ID (hashed to uint64)
    - All metadata stored as Qdrant payload for filtered retrieval
    - Progress logged every 100 chunks

Usage:
    python embedder/embedder.py     # standalone run
    embed_and_upsert()              # called by main.py
"""

import hashlib
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

load_dotenv()
import os

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
logger = logging.getLogger("embedder")

# ── Config ────────────────────────────────────────────────────────────────────
CHUNKS_DIR        = Path("data/chunks")
MODEL_NAME        = "all-MiniLM-L6-v2"
COLLECTION_NAME   = os.getenv("QDRANT_COLLECTION_NAME", "regulatory_docs")
QDRANT_URL        = os.getenv("QDRANT_URL")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
EMBEDDING_DIM     = 384   # all-MiniLM-L6-v2 output dimension
BATCH_SIZE        = 32    # chunks per embedding batch
BGE_PREFIX        = "Represent this sentence for searching relevant passages: "


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_id_to_uint64(chunk_id: str) -> int:
    """
    Convert a string chunk_id to a uint64 integer for Qdrant point ID.

    Qdrant requires integer or UUID point IDs. We hash the chunk_id
    string to a stable uint64 so the same chunk always maps to the
    same point ID — making upserts idempotent.

    Args:
        chunk_id: String chunk identifier e.g. 'doc_name_42'.

    Returns:
        Stable uint64 integer derived from SHA256 of the chunk_id.
    """
    hash_bytes = hashlib.sha256(chunk_id.encode()).digest()
    return int.from_bytes(hash_bytes[:8], byteorder="big")


def _ensure_collection(client: QdrantClient) -> None:
    """
    Create the Qdrant collection if it doesn't already exist.

    Uses cosine distance — appropriate for sentence embeddings which
    are L2-normalised. Collection is created once and reused across runs.

    Args:
        client: Authenticated QdrantClient instance.
    """
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name = COLLECTION_NAME,
            vectors_config  = VectorParams(
                size     = EMBEDDING_DIM,
                distance = Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection: '%s'", COLLECTION_NAME)
    else:
        logger.info("Using existing Qdrant collection: '%s'", COLLECTION_NAME)


# ── Main Embedding Function ───────────────────────────────────────────────────

def embed_and_upsert() -> int:
    """
    Embed all chunks and upsert to Qdrant.

    Process:
        1. Load all chunk JSON files from data/chunks/
        2. Initialise the embedding model (downloads on first run ~90MB)
        3. Connect to Qdrant Cloud
        4. Ensure collection exists
        5. Embed in batches of BATCH_SIZE
        6. Upsert each batch to Qdrant with full metadata payload

    BGE models perform best with a query prefix for retrieval tasks.
    During indexing we use the passage prefix to signal these are
    documents to be retrieved, not queries.

    Returns:
        Total number of points upserted to Qdrant.

    Raises:
        EnvironmentError: If QDRANT_URL or QDRANT_API_KEY are missing.
    """
    if not os.getenv("QDRANT_URL") and not Path("data/qdrant_local").exists():
        logger.warning(
            "No QDRANT_URL set and no local Qdrant found — "
            "will create local store at data/qdrant_local"
        )

    # ── Load all chunks ───────────────────────────────────────────────────────
    chunk_files = list(CHUNKS_DIR.rglob("*.json"))
    logger.info("Found %d chunk files in %s", len(chunk_files), CHUNKS_DIR)

    # Load checkpoint — skip already-upserted chunks on resume
    checkpoint_path = Path("data/embedded_ids.json")
    embedded_ids: set[str] = set()
    if checkpoint_path.exists():
        try:
            embedded_ids = set(json.loads(checkpoint_path.read_text(encoding="utf-8")))
            logger.info("Resuming — %d chunks already embedded", len(embedded_ids))
        except Exception:
            pass

    all_chunks = []
    for chunk_file in chunk_files:
        try:
            chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
            all_chunks.extend(chunks)
        except Exception as exc:
            logger.error("Failed to load %s: %s", chunk_file.name, exc)

    all_chunks = [c for c in all_chunks if c["chunk_id"] not in embedded_ids]
    logger.info("%d chunks remaining to embed", len(all_chunks))

    logger.info("Loaded %d total chunks for embedding", len(all_chunks))

    if not all_chunks:
        logger.error("No chunks found — run chunker first")
        return 0

    # ── Load embedding model ──────────────────────────────────────────────────
    logger.info("Loading embedding model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Model loaded successfully")

    # ── Connect to Qdrant ─────────────────────────────────────────────────────
    qdrant_url     = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if qdrant_url and qdrant_api_key:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        logger.info("Connected to Qdrant Cloud: %s", qdrant_url)
    else:
        client = QdrantClient(path="data/qdrant_local")
        logger.info("Using local Qdrant storage at data/qdrant_local")
    _ensure_collection(client)

    # ── Embed and upsert in batches ───────────────────────────────────────────
    total_upserted = 0

    for batch_start in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[batch_start: batch_start + BATCH_SIZE]

        # BGE models use a prefix for passage encoding
        texts = [BGE_PREFIX + chunk["text"] for chunk in batch]

        try:
            embeddings = model.encode(
                texts,
                normalize_embeddings = True,  # L2 normalise for cosine similarity
                show_progress_bar    = False,
                batch_size           = BATCH_SIZE,
            )
        except Exception as exc:
            logger.error(
                "Embedding failed for batch %d-%d: %s",
                batch_start, batch_start + len(batch), exc,
            )
            continue

        points = []
        for chunk, embedding in zip(batch, embeddings):
            point_id = _chunk_id_to_uint64(chunk["chunk_id"])
            payload  = {
                "chunk_id":      chunk["chunk_id"],
                "text":          chunk["text"],
                "source":        chunk.get("source", "UNKNOWN"),
                "document_name": chunk.get("document_name", ""),
                "section":       chunk.get("section", "General"),
                "chunk_index":   chunk.get("chunk_index", 0),
                "total_chunks":  chunk.get("total_chunks", 0),
                "char_count":    chunk.get("char_count", 0),
            }
            points.append(PointStruct(
                id      = point_id,
                vector  = embedding.tolist(),
                payload = payload,
            ))

        try:
            client.upsert(
                collection_name = COLLECTION_NAME,
                points          = points,
                wait            = True,
            )
            total_upserted += len(points)
            # Save checkpoint after every successful batch
            embedded_ids.update(chunk["chunk_id"] for chunk in batch)
            checkpoint_path.write_text(
                json.dumps(list(embedded_ids), ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error(
                "Qdrant upsert failed for batch %d-%d: %s",
                batch_start, batch_start + len(batch), exc,
            )
            continue

        if total_upserted % 500 == 0 or total_upserted == len(all_chunks):
            logger.info(
                "Progress: %d/%d chunks upserted",
                total_upserted, len(all_chunks),
            )

    logger.info(
        "Embedding complete — %d points upserted to collection '%s'",
        total_upserted, COLLECTION_NAME,
    )
    return total_upserted


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    embed_and_upsert()