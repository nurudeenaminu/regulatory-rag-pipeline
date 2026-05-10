"""
chunker/section_chunker.py — Section-Aware Document Chunker

Splits parsed documents into chunks that preserve regulatory context.

Key design decisions:
    - Chunks split at logical section boundaries, not arbitrary token counts
    - Section headers are detected and preserved as metadata
    - Each chunk carries full provenance: source, document, section, position
    - Overlap between chunks prevents context loss at boundaries
    - Chunk size tuned for regulatory text (500-1000 tokens typical section)

Usage:
    python chunker/section_chunker.py     # standalone run
    chunk_all_documents()                 # called by main.py
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

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
logger = logging.getLogger("section_chunker")

# ── Paths ─────────────────────────────────────────────────────────────────────
PARSED_DIR = Path("data/parsed")
CHUNKS_DIR = Path("data/chunks")

# ── Config ────────────────────────────────────────────────────────────────────
MAX_CHUNK_CHARS  = 1500   # ~375 tokens at 4 chars/token — fits bge-base 512 limit
MIN_CHUNK_CHARS  = 100    # discard chunks too short to be meaningful
OVERLAP_CHARS    = 200    # overlap between consecutive chunks to preserve context


# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single text chunk with full provenance metadata.

    All fields are strings for direct serialisation to JSON and
    Qdrant payload without type conversion.
    """
    chunk_id:      str   # unique identifier: {doc_id}_{chunk_index}
    text:          str   # chunk text content
    source:        str   # originating data source (SEC_EDGAR, FDA, OSHA, etc.)
    document_name: str   # original filename without extension
    section:       str   # detected section heading, or 'General' if none
    chunk_index:   int   # position within the document
    total_chunks:  int   # total chunks in this document (set after all chunks created)
    char_count:    int   # character count of this chunk


# ── Section Detection ─────────────────────────────────────────────────────────

# Patterns that indicate a new section heading in regulatory text
SECTION_PATTERNS = [
    # Numbered sections: "1.", "1.1", "Section 1", "SECTION I"
    r"^(?:Section|SECTION|Article|ARTICLE|Part|PART)\s+[\dIVXivx]+",
    r"^\d+\.\s+[A-Z][A-Za-z\s]{3,}",
    r"^\d+\.\d+\s+[A-Z][A-Za-z\s]{3,}",
    # ALL CAPS headings (common in SEC filings and OSHA standards)
    r"^[A-Z][A-Z\s]{4,}$",
    # Title case headings followed by content
    r"^(?:[A-Z][a-z]+\s){2,}(?:[A-Z][a-z]+)$",
    # CFR section markers
    r"^\§\s*\d+\.\d+",
    # Common regulatory section names
    r"^(?:PURPOSE|SCOPE|DEFINITIONS|REQUIREMENTS|PROCEDURES|"
    r"WARNINGS|CONTRAINDICATIONS|DOSAGE|RISK FACTORS|"
    r"ITEM\s+\d+[A-Z]?\.?\s)",
]

COMPILED_PATTERNS = [re.compile(p, re.MULTILINE) for p in SECTION_PATTERNS]


def _detect_section(line: str) -> str | None:
    """
    Check if a line is a section heading.

    Args:
        line: Single line of text stripped of whitespace.

    Returns:
        The section heading string if detected, None otherwise.
    """
    if not line or len(line) < 3 or len(line) > 120:
        return None

    for pattern in COMPILED_PATTERNS:
        if pattern.match(line.strip()):
            return line.strip()

    return None


# ── Chunking Logic ────────────────────────────────────────────────────────────

def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split document text into (section_heading, section_content) pairs.

    Scans line by line for section heading patterns. Content between
    two headings belongs to the first heading.

    Args:
        text: Full document text.

    Returns:
        List of (heading, content) tuples. First heading is 'General'
        if the document starts with content before any detected heading.
    """
    lines   = text.split("\n")
    sections: list[tuple[str, str]] = []

    current_heading = "General"
    current_lines: list[str] = []

    for line in lines:
        heading = _detect_section(line)
        if heading and len(current_lines) > 0:
            # Save current section and start a new one
            content = "\n".join(current_lines).strip()
            if content:
                sections.append((current_heading, content))
            current_heading = heading
            current_lines   = []
        else:
            current_lines.append(line)

    # Save the last section
    content = "\n".join(current_lines).strip()
    if content:
        sections.append((current_heading, content))

    return sections if sections else [("General", text)]


def _chunk_section(
    section_heading: str,
    section_text:    str,
    doc_id:          str,
    source:          str,
    start_index:     int,
) -> list[Chunk]:
    """
    Split a single section into fixed-size overlapping chunks.

    Sections longer than MAX_CHUNK_CHARS are split at word boundaries
    with OVERLAP_CHARS overlap between consecutive chunks to preserve
    context across boundaries.

    Args:
        section_heading: Section title for metadata.
        section_text:    Full section text content.
        doc_id:          Document identifier for chunk_id construction.
        source:          Data source label (SEC_EDGAR, FDA, etc.).
        start_index:     Starting chunk index within the document.

    Returns:
        List of Chunk objects for this section.
    """
    chunks: list[Chunk] = []
    text   = section_text.strip()

    if len(text) <= MAX_CHUNK_CHARS:
        if len(text) >= MIN_CHUNK_CHARS:
            chunks.append(Chunk(
                chunk_id      = f"{doc_id}_{start_index}",
                text          = text,
                source        = source,
                document_name = doc_id,
                section       = section_heading,
                chunk_index   = start_index,
                total_chunks  = 0,  # set by caller
                char_count    = len(text),
            ))
        return chunks

    # Split long sections into overlapping windows
    pos         = 0
    chunk_index = start_index

    while pos < len(text):
        end = pos + MAX_CHUNK_CHARS

        if end < len(text):
            # Find nearest word boundary before the cut
            boundary = text.rfind(" ", pos, end)
            if boundary > pos:
                end = boundary

        chunk_text = text[pos:end].strip()

        if len(chunk_text) >= MIN_CHUNK_CHARS:
            chunks.append(Chunk(
                chunk_id      = f"{doc_id}_{chunk_index}",
                text          = chunk_text,
                source        = source,
                document_name = doc_id,
                section       = section_heading,
                chunk_index   = chunk_index,
                total_chunks  = 0,
                char_count    = len(chunk_text),
            ))
            chunk_index += 1

        # Move forward by chunk size minus overlap
        pos += MAX_CHUNK_CHARS - OVERLAP_CHARS
        if pos >= len(text):
            break

    return chunks


def chunk_document(
    text:      str,
    doc_id:    str,
    meta:      dict,
) -> list[Chunk]:
    """
    Chunk a full document into section-aware overlapping text chunks.

    Process:
        1. Split into sections by detected headings
        2. Chunk each section independently with overlap
        3. Set total_chunks on all chunks once count is known

    Args:
        text:   Full parsed document text.
        doc_id: Unique document identifier (filename without extension).
        meta:   Metadata dict from companion JSON file.

    Returns:
        List of Chunk objects ready for embedding.
    """
    source   = meta.get("source", "UNKNOWN")
    sections = _split_into_sections(text)

    all_chunks: list[Chunk] = []
    chunk_index = 0

    for heading, content in sections:
        section_chunks = _chunk_section(
            section_heading = heading,
            section_text    = content,
            doc_id          = doc_id,
            source          = source,
            start_index     = chunk_index,
        )
        all_chunks.extend(section_chunks)
        chunk_index += len(section_chunks)

    # Set total_chunks on all chunks now that we know the count
    for chunk in all_chunks:
        chunk.total_chunks = len(all_chunks)

    return all_chunks


# ── Main Pipeline Function ────────────────────────────────────────────────────

def chunk_all_documents() -> int:
    """
    Chunk all parsed documents and save to data/chunks/.

    For each parsed .txt file:
        1. Load companion metadata JSON if available
        2. Chunk the document using section-aware splitting
        3. Save chunks as a single JSON array per document

    Skips documents that have already been chunked (idempotent).

    Returns:
        Total number of chunks produced across all documents.
    """
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    parsed_files = list(PARSED_DIR.rglob("*.txt"))
    logger.info("Found %d parsed documents to chunk", len(parsed_files))

    total_chunks    = 0
    total_documents = 0

    for file_path in parsed_files:
        doc_id      = file_path.stem
        output_path = CHUNKS_DIR / file_path.relative_to(PARSED_DIR).with_suffix(".json")

        if output_path.exists():
            # Count existing chunks for the total
            try:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
                total_chunks    += len(existing)
                total_documents += 1
            except Exception:
                pass
            continue

        # Load text
        text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            logger.warning("Empty file: %s — skipping", file_path.name)
            continue

        # Load companion metadata
        meta_path = file_path.with_suffix(".json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Chunk the document
        chunks = chunk_document(text, doc_id, meta)

        if not chunks:
            logger.warning("No chunks produced for: %s", file_path.name)
            continue

        # Save chunks
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([asdict(c) for c in chunks], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(
            "Chunked: %s → %d chunks (avg %d chars)",
            file_path.name,
            len(chunks),
            sum(c.char_count for c in chunks) // len(chunks),
        )

        total_chunks    += len(chunks)
        total_documents += 1

    logger.info(
        "Chunking complete — %d chunks across %d documents",
        total_chunks, total_documents,
    )
    return total_chunks


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chunk_all_documents()