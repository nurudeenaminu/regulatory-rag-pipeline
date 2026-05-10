"""
main.py — Pipeline Orchestrator

Runs all pipeline stages in sequence:
    1. scrape_sec()          — collector/sec_scraper.py
    2. scrape_fda()          — collector/fda_scraper.py
    3. scrape_osha()         — collector/osha_scraper.py
    4. scrape_eu_ai_act()    — collector/eu_ai_act_scraper.py
    5. parse_all_documents() — parser/document_parser.py
    6. chunk_all_documents() — chunker/section_chunker.py
    7. embed_and_upsert()    — embedder/embedder.py

Usage:
    python main.py
"""

import logging
import sys
import time
from pathlib import Path

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("main")


def run_pipeline() -> bool:
    """
    Execute all pipeline stages in sequence.

    Each stage is wrapped in try/except. Collection stages are
    non-fatal — if one source fails, others continue. Parsing,
    chunking, and embedding are hard dependencies — failure aborts.

    Returns:
        True if all stages completed, False if any critical stage failed.
    """
    start = time.time()
    logger.info("=" * 60)
    logger.info("Pipeline run started")
    logger.info("=" * 60)

    failed_stages = []

    # ── Stage 1: Collection ───────────────────────────────────────────────────
    from scrapers.sec_scraper import scrape_sec
    from scrapers.fda_scraper import scrape_fda
    from scrapers.osha_scraper import scrape_osha
    from scrapers.eu_ai_act_scraper import scrape_eu_ai_act

    for scraper_fn, name in [
        (scrape_sec,        "SEC EDGAR"),
        (scrape_fda,        "FDA"),
        (scrape_osha,       "OSHA"),
        (scrape_eu_ai_act,  "AI Policy"),
    ]:
        try:
            logger.info("Collecting: %s", name)
            count = scraper_fn()
            logger.info("Collected %d documents from %s", count, name)
        except Exception as exc:
            logger.error("Collection failed for %s: %s", name, exc, exc_info=True)
            failed_stages.append(f"collection_{name}")

    # ── Stage 2: Parsing ──────────────────────────────────────────────────────
    try:
        logger.info("Stage: Parsing")
        from parser.document_parser import parse_all_documents
        parsed = parse_all_documents()
        logger.info("Parsed %d documents", parsed)
    except Exception as exc:
        logger.error("Parsing failed: %s", exc, exc_info=True)
        logger.critical("Cannot continue without parsed documents")
        return False

    # ── Stage 3: Chunking ─────────────────────────────────────────────────────
    try:
        logger.info("Stage: Chunking")
        from chunker.section_chunker import chunk_all_documents
        chunks = chunk_all_documents()
        logger.info("Produced %d chunks", chunks)
    except Exception as exc:
        logger.error("Chunking failed: %s", exc, exc_info=True)
        logger.critical("Cannot embed without chunks")
        return False

    # ── Stage 4: Embedding ────────────────────────────────────────────────────
    try:
        logger.info("Stage: Embedding and upsert")
        from embedder.embedder import embed_and_upsert
        upserted = embed_and_upsert()
        logger.info("Upserted %d points to Qdrant", upserted)
    except Exception as exc:
        logger.error("Embedding failed: %s", exc, exc_info=True)
        return False

    elapsed = round(time.time() - start, 1)

    if failed_stages:
        logger.warning(
            "Pipeline finished with partial failures: %s (%.1fs)",
            ", ".join(failed_stages), elapsed,
        )
        return False

    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1fs — all stages succeeded", elapsed)
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)