"""
scrapers/sec_scraper.py — SEC EDGAR Full-Text Search Scraper

Collects regulatory documents from SEC EDGAR covering:
    - 10-K annual reports (risk factors, business descriptions)
    - 33-Act filings (securities regulations)
    - SEC guidance documents and interpretive releases

Uses EDGAR's full-text search API — no API key required.
Requires a valid User-Agent header per SEC fair use policy.

Rate limit: max 10 requests/second — enforced via inter-request delay.

Usage:
    python scrapers/sec_scraper.py     # standalone test
    scrape_sec()                       # called by main.py
"""

import logging
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv
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
logger = logging.getLogger("sec_scraper")

# ── Config ────────────────────────────────────────────────────────────────────
USER_AGENT      = os.getenv("SEC_USER_AGENT", "Nurudeen Aminu nurudeen.aminu41@gmail.com")
RAW_OUTPUT_DIR  = Path("data/raw/sec")
REQUEST_DELAY   = 0.15   # 150ms between requests — well under 10/sec limit
MAX_RETRIES     = 3

EDGAR_SEARCH_URL    = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FILING_URL    = "https://www.sec.gov/cgi-bin/browse-edgar"
EDGAR_ARCHIVES_URL  = "https://www.sec.gov/Archives/edgar/data"

# Document types to collect and their regulatory relevance
FILING_TARGETS = [
    {
        "form":        "10-K",
        "query":       "risk factors regulatory compliance",
        "description": "Annual reports — risk and compliance sections",
        "max_results": 20,
    },
    {
        "form":        "DEF 14A",
        "query":       "regulatory risk governance compliance",
        "description": "Proxy statements — governance and compliance",
        "max_results": 10,
    },
    {
        "form":        "8-K",
        "query":       "regulatory action SEC enforcement",
        "description": "Current reports — regulatory events",
        "max_results": 15,
    },
]

# Date range — last 12 months of filings
END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_headers() -> dict:
    """
    Return HTTP headers required by SEC EDGAR fair use policy.

    EDGAR blocks requests without a descriptive User-Agent.
    The Accept header ensures JSON responses from the search API.

    Returns:
        Dict of HTTP headers.
    """
    return {
        "User-Agent":    USER_AGENT,
        "Accept":        "application/json",
        "Accept-Encoding": "gzip, deflate",
        "Host":          "efts.sec.gov",
    }


def _request_with_retry(url: str, params: dict = None, headers: dict = None) -> dict | None:
    """
    Make an HTTP GET request with exponential backoff retry.

    Respects EDGAR's rate limit by sleeping between every request.
    Returns None on total failure rather than raising.

    Args:
        url:     Target URL.
        params:  Query parameters dict.
        headers: HTTP headers dict.

    Returns:
        Parsed JSON dict, or None on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(REQUEST_DELAY)
            response = requests.get(
                url,
                params=params,
                headers=headers or _get_headers(),
                timeout=30,
            )

            if response.status_code == 429:
                wait = 2 ** attempt
                logger.warning("Rate limited by EDGAR — waiting %ds", wait)
                time.sleep(wait)
                continue

            if response.status_code == 403:
                logger.error("EDGAR blocked request (403) — check User-Agent header")
                return None

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            logger.warning("Request timed out on attempt %d — %s", attempt, url)
        except requests.exceptions.RequestException as exc:
            logger.error("Request error on attempt %d: %s", attempt, exc)

        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)

    logger.error("All %d attempts failed for %s", MAX_RETRIES, url)
    return None



def _search_filings(form: str, query: str, max_results: int) -> list[dict]:
    """
    Search EDGAR full-text search for filings matching form type and query.

    Parses the real EDGAR response structure:
        - CIK from _source.ciks[0]
        - Accession number from _source.adsh
        - Filename from _id (format: {adsh}:{filename})
        - Entity name from _source.display_names[0]

    Args:
        form:        SEC form type e.g. '10-K', '8-K'.
        query:       Full-text search query string.
        max_results: Maximum number of filing results to return.

    Returns:
        List of filing metadata dicts.
    """
    params = {
        "q":       query,
        "forms":   form,
        "dateRange": "custom",
        "startdt": START_DATE,
        "enddt":   END_DATE,
    }

    data = _request_with_retry(EDGAR_SEARCH_URL, params=params)

    if not data:
        return []

    hits = data.get("hits", {}).get("hits", [])
    results = []

    for hit in hits[:max_results]:
        source = hit.get("_source", {})
        hit_id = hit.get("_id", "")

        # _id format: "{adsh}:{filename}"
        parts    = hit_id.split(":", 1)
        adsh     = source.get("adsh", parts[0] if parts else "")
        filename = parts[1] if len(parts) > 1 else ""

        cik = ""
        ciks = source.get("ciks", [])
        if ciks:
            cik = ciks[0].lstrip("0")  # remove leading zeros for URL

        display_names = source.get("display_names", [])
        entity_name   = display_names[0].split("(")[0].strip() if display_names else "Unknown"

        if not cik or not adsh or not filename:
            continue

        # Build direct document URL — no index fetch needed
        acc_clean    = adsh.replace("-", "")
        document_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{filename}"

        results.append({
            "entity_name":      entity_name,
            "cik":              cik,
            "accession_number": adsh,
            "filename":         filename,
            "document_url":     document_url,
            "filed_at":         source.get("file_date", ""),
            "form_type":        source.get("form", form),
            "period_ending":    source.get("period_ending", ""),
            "description":      query,
        })

    logger.info("Found %d %s filings for query '%s'", len(results), form, query)
    return results
def _download_document(url: str, output_path: Path) -> bool:
    """
    Download a single EDGAR document and save to disk.

    Args:
        url:         Direct URL to the filing document.
        output_path: Path where the file should be saved.

    Returns:
        True if download succeeded, False otherwise.
    """
    if output_path.exists():
        logger.debug("Already downloaded: %s", output_path.name)
        return True

    time.sleep(REQUEST_DELAY)

    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=60,
            stream=True,
        )
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("Downloaded: %s (%d bytes)", output_path.name, output_path.stat().st_size)
        return True

    except requests.exceptions.RequestException as exc:
        logger.error("Failed to download %s: %s", url, exc)
        return False


# ── Main Scraper Function ─────────────────────────────────────────────────────

def scrape_sec() -> int:
    """
    Scrape SEC EDGAR filings across all defined FILING_TARGETS.

    For each target:
        1. Search EDGAR full-text search for matching filings
        2. Fetch document URLs from each filing's index
        3. Download primary documents to data/raw/sec/

    Also saves a metadata JSON file alongside each document recording
    entity name, form type, CIK, accession number, and filing date.
    This metadata is consumed by the parser and chunker stages.

    Returns:
        Total number of documents successfully downloaded.
    """
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_downloaded = 0
    metadata_records = []

    for target in FILING_TARGETS:
        logger.info(
            "Searching %s filings: %s",
            target["form"],
            target["description"],
        )

        filings = _search_filings(
            form=target["form"],
            query=target["query"],
            max_results=target["max_results"],
        )

        for filing in filings:
            output_path = RAW_OUTPUT_DIR / f"{filing['cik']}_{filing['accession_number'].replace('-', '_')}_{filing['filename']}"

            success = _download_document(filing["document_url"], output_path)

            if success:
                total_downloaded += 1
                meta = {
                    **filing,
                    "source":     "SEC_EDGAR",
                    "local_path": str(output_path),
                    "scraped_at": datetime.now().isoformat(),
                }
                metadata_path = output_path.with_suffix(".json")
                metadata_path.write_text(
                    json.dumps(meta, indent=2),
                    encoding="utf-8",
                )
                metadata_records.append(meta)

    # Write consolidated metadata index
    index_path = RAW_OUTPUT_DIR / "sec_index.json"
    index_path.write_text(
        json.dumps(metadata_records, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "SEC scraping complete — %d documents downloaded to %s",
        total_downloaded,
        RAW_OUTPUT_DIR,
    )
    return total_downloaded


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scrape_sec()