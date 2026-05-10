"""
scrapers/eu_ai_act_scraper.py — AI Policy and Governance Document Scraper

Replaces EUR-Lex (WAF-blocked) with two reliable public sources:

    1. Federal Register API — US AI executive orders, proposed rules,
       and agency guidance on AI governance.
       Endpoint: https://www.federalregister.gov/api/v1/documents
       No API key required.

    2. NIST AI RMF documents — National Institute of Standards and
       Technology AI Risk Management Framework.
       Direct download from NIST public servers.

These are more actionable for US-focused compliance work and fully
accessible without bot detection issues.

Usage:
    python scrapers/eu_ai_act_scraper.py     # standalone test
    scrape_eu_ai_act()                       # called by main.py
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

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
logger = logging.getLogger("ai_policy_scraper")

# ── Config ────────────────────────────────────────────────────────────────────
RAW_OUTPUT_DIR = Path("data/raw/ai_policy")
REQUEST_DELAY  = 0.5
MAX_RETRIES    = 3
USER_AGENT     = "Nurudeen Aminu nurudeen.aminu41@gmail.com"

FEDERAL_REGISTER_URL = "https://www.federalregister.gov/api/v1/documents"

END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")  # 2 years

# Federal Register AI search targets
FR_SEARCHES = [
    {
        "term":        "artificial intelligence",
        "description": "AI governance and regulation",
        "per_page":    15,
    },
    {
        "term":        "machine learning algorithm",
        "description": "ML algorithm regulation",
        "per_page":    10,
    },
    {
        "term":        "automated decision system",
        "description": "Automated decision-making rules",
        "per_page":    10,
    },
    {
        "term":        "algorithmic accountability",
        "description": "Algorithmic accountability requirements",
        "per_page":    8,
    },
]

# NIST AI documents — direct stable URLs
NIST_DOCUMENTS = [
    {
        "url":         "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
        "title":       "NIST AI Risk Management Framework (AI RMF 1.0)",
        "doc_id":      "NIST_AI_100_1",
        "description": "Core AI risk management framework",
    },
    {
        "url":         "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf",
        "title":       "NIST Generative AI Profile (NIST AI 600-1)",
        "doc_id":      "NIST_AI_600_1",
        "description": "Generative AI risk management profile",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _request_with_retry(
    url: str,
    params: dict = None,
    return_text: bool = False,
    stream: bool = False,
) -> dict | str | requests.Response | None:
    """
    Make an HTTP GET request with exponential backoff retry.

    Args:
        url:         Target URL.
        params:      Query parameters.
        return_text: Return raw text instead of parsed JSON.
        stream:      Stream response for binary downloads.

    Returns:
        Parsed JSON, text, streaming Response, or None on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(REQUEST_DELAY)
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,application/json,*/*"},
                timeout=60,
                stream=stream,
            )

            if response.status_code == 429:
                wait = 2 ** attempt
                logger.warning("Rate limited — waiting %ds", wait)
                time.sleep(wait)
                continue

            if response.status_code == 404:
                logger.warning("404 Not Found: %s", url)
                return None

            response.raise_for_status()

            if stream:
                return response
            if return_text:
                return response.text
            return response.json()

        except requests.exceptions.Timeout:
            logger.warning("Timeout on attempt %d", attempt)
        except requests.exceptions.RequestException as exc:
            logger.error("Request error on attempt %d: %s", attempt, exc)

        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)

    logger.error("All %d attempts failed for %s", MAX_RETRIES, url)
    return None


# ── Source 1: Federal Register API ───────────────────────────────────────────

def _scrape_federal_register() -> int:
    """
    Fetch AI-related documents from the Federal Register API.

    Returns structured text files containing document metadata, abstract,
    and full body text where available. Each document includes agency,
    type (Rule/Proposed Rule/Notice), and publication date.

    Returns:
        Number of documents saved.
    """
    output_dir = RAW_OUTPUT_DIR / "federal_register"
    output_dir.mkdir(parents=True, exist_ok=True)
    total_saved  = 0
    seen_doc_ids = set()

    for target in FR_SEARCHES:
        logger.info(
            "Fetching Federal Register: '%s'", target["description"]
        )

        params = {
            "conditions[term]":            target["term"],
            "conditions[publication_date][gte]": START_DATE,
            "conditions[publication_date][lte]": END_DATE,
            "per_page":                    target["per_page"],
            "order":                       "relevance",
            "fields[]": [
                "title", "abstract", "body_html_url", "document_number",
                "publication_date", "type", "agencies", "action",
                "effective_on", "citation",
            ],
        }

        data = _request_with_retry(FEDERAL_REGISTER_URL, params=params)
        if not data:
            continue

        results = data.get("results", [])
        logger.info(
            "Found %d Federal Register documents for '%s'",
            len(results), target["term"],
        )

        for doc in results:
            doc_number = doc.get("document_number", "")
            if not doc_number or doc_number in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_number)

            # Fetch full body text if available
            body_text = ""
            body_url  = doc.get("body_html_url", "")
            if body_url:
                body_html = _request_with_retry(body_url, return_text=True)
                if body_html:
                    body_text = body_html[:50000]  # cap at 50k chars

            agencies = ", ".join(
                [a.get("name", "") for a in doc.get("agencies", [])]
            )

            content = (
                f"TITLE: {doc.get('title', '')}\n"
                f"DOCUMENT NUMBER: {doc_number}\n"
                f"TYPE: {doc.get('type', '')}\n"
                f"AGENCY: {agencies}\n"
                f"PUBLISHED: {doc.get('publication_date', '')}\n"
                f"EFFECTIVE: {doc.get('effective_on', '')}\n"
                f"CITATION: {doc.get('citation', '')}\n"
                f"ACTION: {doc.get('action', '')}\n"
                f"{'=' * 60}\n\n"
                f"ABSTRACT:\n{doc.get('abstract', 'No abstract available.')}\n\n"
                f"FULL TEXT:\n{body_text}\n"
            )

            safe_name   = f"{doc_number.replace('-', '_')}.txt"
            output_path = output_dir / safe_name

            meta = {
                "source":           "FEDERAL_REGISTER",
                "title":            doc.get("title", ""),
                "document_number":  doc_number,
                "type":             doc.get("type", ""),
                "agency":           agencies,
                "publication_date": doc.get("publication_date", ""),
                "search_term":      target["term"],
                "scraped_at":       datetime.now().isoformat(),
                "local_path":       str(output_path),
            }

            output_path.write_text(content, encoding="utf-8")
            output_path.with_suffix(".json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            logger.info("Saved: %s (%d chars)", safe_name, len(content))
            total_saved += 1

    return total_saved


# ── Source 2: NIST AI Documents ───────────────────────────────────────────────

def _scrape_nist_documents() -> int:
    """
    Download NIST AI framework PDFs from NIST public servers.

    NIST documents are direct PDF downloads — no WAF, no auth required.

    Returns:
        Number of documents saved.
    """
    output_dir = RAW_OUTPUT_DIR / "nist"
    output_dir.mkdir(parents=True, exist_ok=True)
    total_saved = 0

    for doc in NIST_DOCUMENTS:
        logger.info("Downloading NIST: %s", doc["title"])

        output_path = output_dir / f"{doc['doc_id']}.pdf"

        if output_path.exists():
            logger.info("Already downloaded: %s", output_path.name)
            total_saved += 1
            continue

        response = _request_with_retry(doc["url"], stream=True)
        if not response:
            continue

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        meta = {
            "source":      "NIST",
            "title":       doc["title"],
            "doc_id":      doc["doc_id"],
            "description": doc["description"],
            "url":         doc["url"],
            "scraped_at":  datetime.now().isoformat(),
            "local_path":  str(output_path),
        }

        output_path.with_suffix(".json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        logger.info(
            "Saved: %s (%d bytes)",
            output_path.name, output_path.stat().st_size,
        )
        total_saved += 1

    return total_saved


# ── Main Scraper Function ─────────────────────────────────────────────────────

def scrape_eu_ai_act() -> int:
    """
    Scrape AI policy and governance documents from Federal Register and NIST.

    Function name retained for compatibility with main.py orchestrator.

    Returns:
        Total number of documents saved.
    """
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting AI policy scrape — Source 1: Federal Register")
    fr_count = _scrape_federal_register()

    logger.info("Starting AI policy scrape — Source 2: NIST AI Documents")
    nist_count = _scrape_nist_documents()

    total = fr_count + nist_count
    logger.info(
        "AI policy scraping complete — %d Federal Register + %d NIST = %d total",
        fr_count, nist_count, total,
    )
    return total


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scrape_eu_ai_act()