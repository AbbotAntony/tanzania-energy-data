"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parsers.pdf_text import PageText, decode_cids
from pipeline.provenance import DocumentRef
from pipeline.storage import StoredDoc

FIXTURES = Path(__file__).parent / "fixtures"


def load_pages(name: str) -> list[PageText]:
    """Load a page-text fixture: JSON list of {"page_no", "raw_text"} from a real PDF."""
    items = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    pages = []
    for item in items:
        text, decoded = decode_cids(item["raw_text"])
        pages.append(PageText(item["page_no"], text, decoded))
    return pages


def make_doc(title: str, category: str, url: str = "https://www.ewura.go.tz/x.pdf") -> StoredDoc:
    ref = DocumentRef("ewura", url, title, category, "2026-01-01")
    return StoredDoc(ref, Path("x.pdf"), "0" * 64, "2026-10-15")


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
