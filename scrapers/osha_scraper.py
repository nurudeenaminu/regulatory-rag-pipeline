"""
scrapers/osha_scraper.py — OSHA Standards and Enforcement Scraper

Two data sources:
    1. OSHA eTools API — workplace safety standards content
    2. OSHA Enforcement API — inspection and citation data
       Endpoint: https://data.dol.gov/get/full_inspection
       No API key required.

Usage:
    python scrapers/osha_scraper.py     # standalone test
    scrape_osha()                       # called by main.py
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
logger = logging.getLogger("osha_scraper")

# ── Config ────────────────────────────────────────────────────────────────────
RAW_OUTPUT_DIR = Path("data/raw/osha")
REQUEST_DELAY  = 0.5
MAX_RETRIES    = 3
USER_AGENT     = "Nurudeen Aminu nurudeen.aminu41@gmail.com"

# OSHA enforcement data API — Department of Labor public dataset
OSHA_ENFORCEMENT_URL = "https://data.dol.gov/get/full_inspection"

# OSHA standards — curated list of high-value regulatory standards
# These are stable URLs to OSHA's most referenced standards
OSHA_STANDARDS = [
    {
        "url":         "https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.1200",
        "title":       "Hazard Communication Standard",
        "standard":    "29 CFR 1910.1200",
        "description": "HazCom — chemical hazard communication requirements",
    },
    {
        "url":         "https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147",
        "title":       "Control of Hazardous Energy (Lockout/Tagout)",
        "standard":    "29 CFR 1910.147",
        "description": "LOTO — energy isolation procedures",
    },
    {
        "url":         "https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.134",
        "title":       "Respiratory Protection Standard",
        "standard":    "29 CFR 1910.134",
        "description": "Respiratory protection program requirements",
    },
    {
        "url":         "https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.501",
        "title":       "Fall Protection in Construction",
        "standard":    "29 CFR 1926.501",
        "description": "Construction fall protection requirements",
    },
    {
        "url":         "https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.119",
        "title":       "Process Safety Management",
        "standard":    "29 CFR 1910.119",
        "description": "PSM of highly hazardous chemicals",
    },
    {
        "url":         "https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.1030",
        "title":       "Bloodborne Pathogens Standard",
        "standard":    "29 CFR 1910.1030",
        "description": "Bloodborne pathogens exposure control",
    },
    {
        "url":         "https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.132",
        "title":       "Personal Protective Equipment",
        "standard":    "29 CFR 1910.132",
        "description": "PPE general requirements",
    },
    {
        "url":         "https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.303",
        "title":       "Electrical Systems General Requirements",
        "standard":    "29 CFR 1910.303",
        "description": "Electrical safety general requirements",
    },
]

END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _request_with_retry(
    url: str,
    params: dict = None,
    return_text: bool = False,
) -> dict | str | None:
    """
    Make an HTTP GET request with exponential backoff retry.

    Args:
        url:         Target URL.
        params:      Query parameters.
        return_text: If True, return response text instead of parsed JSON.

    Returns:
        Parsed JSON dict, response text string, or None on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(REQUEST_DELAY)
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/json"},
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

            if return_text:
                return response.text
            return response.json()

        except requests.exceptions.Timeout:
            logger.warning("Timeout on attempt %d", attempt)
        except requests.exceptions.RequestException as exc:
            logger.error("Request error on attempt %d: %s", attempt, exc)
        except Exception as exc:
            logger.error("Unexpected error on attempt %d: %s", attempt, exc)

        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)

    logger.error("All %d attempts failed for %s", MAX_RETRIES, url)
    return None


def _save_document(content: str, output_path: Path, meta: dict) -> bool:
    """
    Save text content and metadata JSON to disk.

    Args:
        content:     Document text content.
        output_path: Path for the text file.
        meta:        Metadata dict saved as companion JSON.

    Returns:
        True on success.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    output_path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    logger.info("Saved: %s (%d chars)", output_path.name, len(content))
    return True


# ── Source 1: OSHA Standards Pages ───────────────────────────────────────────

def _scrape_osha_standards() -> int:
    """
    Download OSHA regulatory standards pages as text documents.

    Fetches the HTML content of each standard page and extracts readable
    text for downstream parsing. Metadata includes standard number and
    regulatory description.

    Returns:
        Number of standards documents saved.
    """
    output_dir = RAW_OUTPUT_DIR / "standards"
    output_dir.mkdir(parents=True, exist_ok=True)
    total_saved = 0

    for standard in OSHA_STANDARDS:
        logger.info("Fetching OSHA standard: %s", standard["standard"])

        html = _request_with_retry(standard["url"], return_text=True)
        if not html:
            continue

        safe_name = standard["standard"].replace(" ", "_").replace("/", "_") + ".html"
        output_path = output_dir / safe_name

        meta = {
            "source":      "OSHA_STANDARDS",
            "title":       standard["title"],
            "standard":    standard["standard"],
            "description": standard["description"],
            "url":         standard["url"],
            "scraped_at":  datetime.now().isoformat(),
            "local_path":  str(output_path),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        output_path.with_suffix(".json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        logger.info(
            "Saved: %s (%d chars)",
            output_path.name, len(html),
        )
        total_saved += 1

    return total_saved


# ── Source 2: OSHA Enforcement Data ──────────────────────────────────────────

def _scrape_osha_enforcement() -> int:
    """
    Fetch recent OSHA workplace inspection and citation records.

    Uses the Department of Labor public enforcement dataset API.
    Records are formatted as structured text documents for RAG chunking.

    Returns:
        Number of enforcement documents saved.
    """
    output_dir = RAW_OUTPUT_DIR / "enforcement"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching OSHA enforcement records")

    params = {
        "filters": f"open_date>={START_DATE}",
        "limit":   25,
    }

    data = _request_with_retry(OSHA_ENFORCEMENT_URL, params=params)

    if not data:
        logger.warning("No enforcement data returned — skipping")
        return 0

    records = data if isinstance(data, list) else data.get("data", [])
    logger.info("Found %d OSHA enforcement records", len(records))

    if not records:
        return 0

    # Save all records as one structured document for context-rich chunking
    content_parts = ["OSHA ENFORCEMENT INSPECTION RECORDS\n" + "=" * 60 + "\n"]

    for record in records:
        content_parts.append(
            f"\nESTABLISHMENT: {record.get('estab_name', 'Unknown')}\n"
            f"INSPECTION ID: {record.get('activity_nr', '')}\n"
            f"OPEN DATE: {record.get('open_date', '')}\n"
            f"CLOSE DATE: {record.get('close_case_date', '')}\n"
            f"INDUSTRY: {record.get('sic_desc', '')} (SIC: {record.get('sic', '')})\n"
            f"STATE: {record.get('site_state', '')}\n"
            f"INSPECTION TYPE: {record.get('insp_type', '')}\n"
            f"SCOPE: {record.get('insp_scope', '')}\n"
            f"VIOLATIONS: {record.get('total_viol', '0')}\n"
            f"PENALTY: ${record.get('current_penalty', '0')}\n"
            f"{'─' * 40}\n"
        )

    content   = "\n".join(content_parts)
    timestamp = datetime.now().strftime("%Y%m%d")
    output_path = output_dir / f"osha_enforcement_{timestamp}.txt"

    meta = {
        "source":       "OSHA_ENFORCEMENT",
        "record_count": len(records),
        "date_range":   f"{START_DATE} to {END_DATE}",
        "scraped_at":   datetime.now().isoformat(),
        "local_path":   str(output_path),
    }

    _save_document(content, output_path, meta)
    return 1


# ── Main Scraper Function ─────────────────────────────────────────────────────

def scrape_osha() -> int:
    """
    Scrape OSHA regulatory standards and enforcement data.

    Returns:
        Total number of documents saved.
    """
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting OSHA scrape — Source 1: Standards")
    standards_count = _scrape_osha_standards()

    logger.info("Starting OSHA scrape — Source 2: Enforcement")
    enforcement_count = _scrape_osha_enforcement()

    total = standards_count + enforcement_count
    logger.info(
        "OSHA scraping complete — %d standards + %d enforcement docs = %d total",
        standards_count, enforcement_count, total,
    )
    return total


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scrape_osha()