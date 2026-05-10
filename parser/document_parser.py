"""
parser/document_parser.py — Document Parser

Converts raw scraped documents into clean plain text for chunking.

Handles three file types:
    - HTML (.html, .htm) — BeautifulSoup extracts text, strips boilerplate
    - PDF (.pdf)         — pdfminer.six extracts text page by page
    - Text (.txt)        — read directly, minimal cleaning

Each parsed document is saved as a .txt file in data/parsed/ with its
source metadata JSON preserved alongside it.

Usage:
    python parser/document_parser.py     # standalone run
    parse_all_documents()                # called by main.py
"""

import json
import logging
import re
from io import StringIO
from pathlib import Path

from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from bs4 import XMLParsedAsHTMLWarning
import warnings

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
logger = logging.getLogger("document_parser")

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR    = Path("data/raw")
PARSED_DIR = Path("data/parsed")

# HTML elements to remove before text extraction — navigation, headers, scripts
BOILERPLATE_TAGS = [
    "nav", "header", "footer", "script", "style", "noscript",
    "aside", "advertisement", "cookie", "banner",
]

# Minimum text length to consider a document worth keeping
MIN_TEXT_LENGTH = 200


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_html(file_path: Path) -> str:
    """
    Extract clean text from an HTML file using BeautifulSoup.

    Removes navigation, scripts, styles and other boilerplate before
    extracting text. Collapses excessive whitespace.

    Args:
        file_path: Path to the HTML file.

    Returns:
        Clean plain text string, or empty string on failure.
    """
    try:
        html = file_path.read_text(encoding="utf-8", errors="ignore")
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")

        # Remove boilerplate elements
        for tag in BOILERPLATE_TAGS:
            for el in soup.find_all(tag):
                el.decompose()

        # Remove elements with boilerplate class/id names
        for el in soup.find_all(True):
            if not el.parent:  # already decomposed by a parent removal
                continue
            el_id    = (el.get("id") or "").lower()
            el_class = " ".join(el.get("class") or []).lower()
            if any(word in el_id or word in el_class
                   for word in ["nav", "menu", "footer", "header", "cookie", "banner", "ad-"]):
                el.decompose()

        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excessive newlines
        text = re.sub(r"[ \t]+", " ", text)      # collapse horizontal whitespace
        return text.strip()

    except Exception as exc:
        logger.error("HTML parse failed for %s: %s", file_path.name, exc)
        return ""


def _parse_pdf(file_path: Path) -> str:
    """
    Extract text from a PDF file using pdfminer.six.

    Uses LAParams tuned for regulatory documents — these tend to have
    multi-column layouts and dense paragraph text.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text string, or empty string on failure.
    """
    try:
        output = StringIO()
        laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            all_texts=True,
        )
        with open(file_path, "rb") as f:
            extract_text_to_fp(f, output, laparams=laparams)

        text = output.getvalue()
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    except Exception as exc:
        logger.error("PDF parse failed for %s: %s", file_path.name, exc)
        return ""


def _parse_text(file_path: Path) -> str:
    """
    Read and lightly clean a plain text file.

    Args:
        file_path: Path to the text file.

    Returns:
        Cleaned text string, or empty string on failure.
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    except Exception as exc:
        logger.error("Text parse failed for %s: %s", file_path.name, exc)
        return ""


def parse_document(file_path: Path) -> str:
    """
    Parse a single document file to plain text.

    Routes to the appropriate parser based on file extension.
    Unsupported extensions are logged and skipped.

    Args:
        file_path: Path to the document file.

    Returns:
        Extracted plain text, or empty string if unsupported/failed.
    """
    ext = file_path.suffix.lower()

    if ext in (".html", ".htm"):
        return _parse_html(file_path)
    if ext == ".pdf":
        return _parse_pdf(file_path)
    if ext == ".txt":
        return _parse_text(file_path)

    logger.debug("Unsupported file type: %s — skipping", file_path.suffix)
    return ""


# ── Main Pipeline Function ────────────────────────────────────────────────────

def parse_all_documents() -> int:
    """
    Parse all raw documents across all source directories.

    Walks data/raw/ recursively, finds all HTML/PDF/TXT files,
    parses each to plain text, and saves to data/parsed/ mirroring
    the source subdirectory structure.

    Skips:
        - JSON metadata files (companion files, not documents)
        - Files that produce less than MIN_TEXT_LENGTH characters
        - Files that have already been parsed (idempotent)

    Returns:
        Total number of documents successfully parsed.
    """
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    supported_extensions = {".html", ".htm", ".pdf", ".txt"}
    all_files = [
        f for f in RAW_DIR.rglob("*")
        if f.is_file()
        and f.suffix.lower() in supported_extensions
        and not f.name.endswith(".json")
    ]

    logger.info("Found %d documents to parse in %s", len(all_files), RAW_DIR)

    total_parsed   = 0
    total_skipped  = 0
    total_failed   = 0

    for file_path in all_files:
        # Mirror directory structure under data/parsed/
        relative     = file_path.relative_to(RAW_DIR)
        output_path  = PARSED_DIR / relative.with_suffix(".txt")

        if output_path.exists():
            logger.debug("Already parsed: %s — skipping", file_path.name)
            total_parsed += 1
            continue

        text = parse_document(file_path)

        if len(text) < MIN_TEXT_LENGTH:
            logger.warning(
                "Too short after parsing (%d chars): %s — skipping",
                len(text), file_path.name,
            )
            total_skipped += 1
            continue

        # Save parsed text
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

        # Copy companion metadata JSON if it exists
        meta_src = file_path.with_suffix(".json")
        if meta_src.exists():
            meta_dst = output_path.with_suffix(".json")
            meta_dst.write_text(
                meta_src.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        logger.info(
            "Parsed: %s → %s (%d chars)",
            file_path.name, output_path.name, len(text),
        )
        total_parsed += 1

    logger.info(
        "Parsing complete — %d parsed, %d skipped (too short), %d failed",
        total_parsed, total_skipped, total_failed,
    )
    return total_parsed


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parse_all_documents()