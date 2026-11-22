"""Tests for CV text extraction."""

from __future__ import annotations

import io

import pytest

from backend.app.perception.cv_reader import (
    MAX_CV_BYTES,
    CvExtractionError,
    extract_cv,
)


def test_extracts_plain_text() -> None:
    data = "  Built a classifier.\n\n  Macro-F1 0.81.  \n".encode()

    doc = extract_cv("cv.txt", data)

    assert doc.filename == "cv.txt"
    assert doc.text == "Built a classifier.\nMacro-F1 0.81."
    assert doc.characters == len(doc.text)


def test_extracts_markdown() -> None:
    doc = extract_cv("summary.md", b"# Summary\n\nML intern candidate.")
    assert "ML intern candidate." in doc.text


def test_extracts_docx() -> None:
    import docx

    document = docx.Document()
    document.add_paragraph("Ticket classifier project")
    document.add_paragraph("Macro-F1 0.81 on held-out data")
    buffer = io.BytesIO()
    document.save(buffer)

    doc = extract_cv("cv.docx", buffer.getvalue())

    assert "Ticket classifier project" in doc.text
    assert "Macro-F1 0.81" in doc.text


def test_rejects_empty_file() -> None:
    with pytest.raises(CvExtractionError, match="empty"):
        extract_cv("cv.txt", b"")


def test_rejects_unsupported_extension() -> None:
    with pytest.raises(CvExtractionError, match="PDF, DOCX, TXT, or MD"):
        extract_cv("cv.exe", b"data")


def test_rejects_oversized_file() -> None:
    with pytest.raises(CvExtractionError, match="exceeds"):
        extract_cv("cv.txt", b"x" * (MAX_CV_BYTES + 1))


def test_rejects_whitespace_only_text() -> None:
    with pytest.raises(CvExtractionError, match="No readable text"):
        extract_cv("cv.txt", b"   \n\n\t  \n")


def test_truncates_long_documents() -> None:
    doc = extract_cv("cv.txt", b"a" * 30_000)
    assert doc.characters == 20_000


def test_rejects_corrupt_pdf() -> None:
    with pytest.raises(CvExtractionError, match="could not be read"):
        extract_cv("cv.pdf", b"%PDF-1.4 not really a pdf")
