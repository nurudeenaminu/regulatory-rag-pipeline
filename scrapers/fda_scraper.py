"""
scrapers/fda_scraper.py — FDA Regulatory Content Scraper

Two data sources:
    1. openFDA Drug Label API — drug warnings, indications, dosage requirements
       Endpoint: https://api.fda.gov/drug/label.json
       No API key required.

    2. Regulations.gov API — FDA guidance documents and rulemaking
       Endpoint: https://api.regulations.gov/v4/documents
       Requires free API key from open.regulations.gov/signup

Usage:
    python scrapers/fda_scraper.py     # standalone test
    scrape_fda()                       # called by main.py
"""

import json
import logging
import os
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
logger = logging.getLogger("fda_scraper")

# ── Config ────────────────────────────────────────────────────────────────────
RAW_OUTPUT_DIR      = Path("data/raw/fda")
REQUEST_DELAY       = 0.5
MAX_RETRIES         = 3
REGULATIONS_API_KEY = os.getenv("REGULATIONS_API_KEY", "")

OPENFDA_LABEL_URL   = "https://api.fda.gov/drug/label.json"
REGULATIONS_API_URL = "https://api.regulations.gov/v4/documents"

USER_AGENT = "Nurudeen Aminu nurudeen.aminu41@gmail.com"

END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

# Drug label search targets — each maps to a regulatory content area
LABEL_SEARCHES = [
    {"search": "warnings",              "description": "Drug warnings and contraindications", "limit": 10},
    {"search": "drug interactions",     "description": "Drug interaction requirements",       "limit": 10},
    {"search": "clinical pharmacology", "description": "Clinical pharmacology guidance",      "limit": 10},
    {"search": "dosage administration", "description": "Dosage and administration standards", "limit": 10},
]

# Regulations.gov FDA docket search targets
REGULATIONS_SEARCHES = [
    "FDA drug safety guidance",
    "FDA medical device regulation",
    "FDA food safety compliance",
]
REGULATIONS_PER_SEARCH = 8


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_headers(include_regulations_key: bool = False) -> dict:
    """
    Return HTTP headers for FDA or Regulations.gov requests.

    Args:
        include_regulations_key: If True, adds X-Api-Key header for Regulations.gov.

    Returns:
        Dict of HTTP headers.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept":     "application/json",
    }
    if include_regulations_key and REGULATIONS_API_KEY:
        headers["X-Api-Key"] = REGULATIONS_API_KEY
    return headers


def _request_with_retry(
    url: str,
    params: dict = None,
    use_regulations_key: bool = False,
) -> dict | None:
    """
    Make an HTTP GET request with exponential backoff retry.

    Args:
        url:                  Target URL.
        params:               Query parameters.
        use_regulations_key:  Whether to include Regulations.gov API key header.

    Returns:
        Parsed JSON dict, or None on total failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(REQUEST_DELAY)
            response = requests.get(
                url,
                params=params,
                headers=_get_headers(use_regulations_key),
                timeout=30,
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
            return response.json()

        except requests.exceptions.Timeout:
            logger.warning("Timeout on attempt %d", attempt)
        except requests.exceptions.RequestException as exc:
            logger.error("Request error on attempt %d: %s", attempt, exc)

        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)

    logger.error("All %d attempts failed for %s", MAX_RETRIES, url)
    return None


def _save_text_document(content: str, output_path: Path, meta: dict) -> bool:
    """
    Save extracted text content and metadata to disk.

    Args:
        content:     Text content to save.
        output_path: Path for the text file.
        meta:        Metadata dict to save as companion JSON.

    Returns:
        True always — text content never fails to save.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    output_path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    logger.info("Saved: %s (%d chars)", output_path.name, len(content))
    return True


# ── Source 1: openFDA Drug Labels ─────────────────────────────────────────────

def _scrape_drug_labels() -> int:
    """
    Fetch drug label regulatory content from the openFDA label API.

    Each label contains structured regulatory sections: warnings,
    contraindications, drug interactions, dosage requirements.
    These are extracted as text and saved for downstream chunking.

    Returns:
        Number of label documents saved.
    """
    output_dir = RAW_OUTPUT_DIR / "drug_labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    total_saved = 0

    for target in LABEL_SEARCHES:
        logger.info("Fetching FDA drug labels: %s", target["description"])

        params = {
            "search": target["search"],
            "limit":  target["limit"],
        }

        data = _request_with_retry(OPENFDA_LABEL_URL, params=params)
        if not data:
            continue

        results = data.get("results", [])
        logger.info(
            "Found %d drug label records for '%s'",
            len(results), target["search"],
        )

        for i, label in enumerate(results):
            openfda   = label.get("openfda", {})
            drug_name = (openfda.get("brand_name", ["unknown"])[0]).replace(" ", "_")[:40]
            safe_name = f"{drug_name}_{i}_{target['search'].replace(' ', '_')}.txt"

            # Extract all regulatory text sections present in this label
            sections = [
                "warnings", "contraindications", "drug_interactions",
                "dosage_and_administration", "indications_and_usage",
                "clinical_pharmacology", "mechanism_of_action",
                "adverse_reactions", "precautions",
            ]

            content_parts = [f"DRUG: {drug_name}\n{'='*60}\n"]
            for section in sections:
                value = label.get(section)
                if value:
                    text = value[0] if isinstance(value, list) else value
                    content_parts.append(f"\n## {section.upper()}\n{text}\n")

            if len(content_parts) <= 1:
                continue  # skip labels with no extractable content

            content = "\n".join(content_parts)
            meta = {
                "source":        "FDA_DRUG_LABELS",
                "drug_name":     drug_name,
                "search_topic":  target["description"],
                "manufacturer":  openfda.get("manufacturer_name", ["Unknown"])[0]
                                 if openfda.get("manufacturer_name") else "Unknown",
                "ndc":           openfda.get("product_ndc", [""])[0]
                                 if openfda.get("product_ndc") else "",
                "scraped_at":    datetime.now().isoformat(),
                "local_path":    str(output_dir / safe_name),
            }

            _save_text_document(content, output_dir / safe_name, meta)
            total_saved += 1

    return total_saved


# ── Source 2: Regulations.gov FDA Documents ───────────────────────────────────

def _scrape_regulations_gov() -> int:
    """
    Fetch FDA guidance and rulemaking documents from Regulations.gov API.

    Skipped gracefully if REGULATIONS_API_KEY is not set in .env.
    Documents are saved as text files with metadata JSON companions.

    Returns:
        Number of documents saved, or 0 if API key missing.
    """
    if not REGULATIONS_API_KEY:
        logger.warning(
            "REGULATIONS_API_KEY not set — skipping Regulations.gov scrape. "
            "Get a free key at open.regulations.gov/signup"
        )
        return 0

    output_dir = RAW_OUTPUT_DIR / "regulations_gov"
    output_dir.mkdir(parents=True, exist_ok=True)
    total_saved = 0

    for search_term in REGULATIONS_SEARCHES:
        logger.info("Fetching Regulations.gov: '%s'", search_term)

        params = {
            "filter[searchTerm]":    search_term,
            "filter[agencyId]":      "FDA",
            "filter[documentType]":  "Rule,Proposed Rule,Notice,Supporting & Related Material",
            "filter[postedDate][ge]": START_DATE,
            "page[size]":            REGULATIONS_PER_SEARCH,
            "sort":                  "-postedDate",
        }

        data = _request_with_retry(
            REGULATIONS_API_URL,
            params=params,
            use_regulations_key=True,
        )

        if not data:
            continue

        documents = data.get("data", [])
        logger.info(
            "Found %d Regulations.gov documents for '%s'",
            len(documents), search_term,
        )

        for doc in documents:
            attrs     = doc.get("attributes", {})
            title     = attrs.get("title", "Unknown")[:60]
            doc_id    = doc.get("id", "unknown")
            safe_name = f"{doc_id}_{title.replace(' ', '_').replace('/', '_')}.txt"

            content = (
                f"TITLE: {attrs.get('title', '')}\n"
                f"AGENCY: {attrs.get('agencyId', '')}\n"
                f"TYPE: {attrs.get('documentType', '')}\n"
                f"POSTED: {attrs.get('postedDate', '')}\n"
                f"DOCKET: {attrs.get('docketId', '')}\n"
                f"{'='*60}\n\n"
                f"{attrs.get('summary', 'No summary available.')}\n"
            )

            meta = {
                "source":        "REGULATIONS_GOV_FDA",
                "title":         attrs.get("title", ""),
                "document_type": attrs.get("documentType", ""),
                "agency":        attrs.get("agencyId", "FDA"),
                "docket_id":     attrs.get("docketId", ""),
                "posted_date":   attrs.get("postedDate", ""),
                "doc_id":        doc_id,
                "scraped_at":    datetime.now().isoformat(),
                "local_path":    str(output_dir / safe_name),
            }

            _save_text_document(content, output_dir / safe_name, meta)
            total_saved += 1

    return total_saved


# ── Main Scraper Function ─────────────────────────────────────────────────────

def scrape_fda() -> int:
    """
    Scrape FDA regulatory content from openFDA Drug Label API.

    Returns:
        Total number of drug label documents saved.
    """
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting FDA scrape — openFDA Drug Labels")
    total = _scrape_drug_labels()

    logger.info(
        "FDA scraping complete — %d documents saved to %s",
        total, RAW_OUTPUT_DIR,
    )
    return total


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scrape_fda()