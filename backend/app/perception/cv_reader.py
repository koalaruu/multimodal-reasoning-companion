"""Extract plain text from an uploaded CV or job summary.

Supports PDF, DOCX, and plain text. Extraction is deliberately conservative:
unreadable or encrypted files raise a typed error rather than returning junk
that would then be sent to the LLM as if it were the candidate's real history.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

MAX_CV_BYTES = 5 * 1024 * 1024
MAX_CV_CHARS = 20_000

_SUPPORTED_SUFFIXES = (".pdf", ".docx", ".txt", ".md")


class CvExtractionError(Exception):
    """Raised when a document cannot be turned into usable text."""


@dataclass(frozen=True, slots=True)
class CvDocument:
    text: str
    filename: str
    characters: int


def _suffix(filename: str) -> str:
    lowered = filename.lower()
    for suffix in _SUPPORTED_SUFFIXES:
        if lowered.endswith(suffix):
            return suffix
    return ""


def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise CvExtractionError("PDF support is not installed") from error

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise CvExtractionError("The PDF is password protected")
        pages = [page.extract_text() or "" for page in reader.pages]
    except CvExtractionError:
        raise
    except Exception as error:
        raise CvExtractionError("The PDF could not be read") from error

    return "\n".join(pages)


def _extract_docx(data: bytes) -> str:
    try:
        import docx
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise CvExtractionError("DOCX support is not installed") from error

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as error:
        raise CvExtractionError("The DOCX file could not be read") from error

    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    return "\n".join(paragraphs)


def _extract_text(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("latin-1")
        except UnicodeDecodeError as error:
            raise CvExtractionError(
                "The text file is not valid UTF-8 or Latin-1"
            ) from error


def extract_cv(filename: str, data: bytes) -> CvDocument:
    """Return normalized CV text or raise CvExtractionError."""

    if not data:
        raise CvExtractionError("The uploaded file is empty")
    if len(data) > MAX_CV_BYTES:
        raise CvExtractionError(
            f"The file exceeds {MAX_CV_BYTES // (1024 * 1024)} MiB"
        )

    suffix = _suffix(filename)
    if not suffix:
        raise CvExtractionError(
            "Use a PDF, DOCX, TXT, or MD file"
        )

    if suffix == ".pdf":
        raw = _extract_pdf(data)
    elif suffix == ".docx":
        raw = _extract_docx(data)
    else:
        raw = _extract_text(data)

    normalized = "\n".join(
        line.strip() for line in raw.splitlines() if line.strip()
    ).strip()

    if not normalized:
        raise CvExtractionError(
            "No readable text was found. Scanned PDFs need OCR first."
        )

    if len(normalized) > MAX_CV_CHARS:
        normalized = normalized[:MAX_CV_CHARS]

    return CvDocument(
        text=normalized,
        filename=filename,
        characters=len(normalized),
    )
