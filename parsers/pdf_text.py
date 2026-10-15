"""PDF text helpers: page text extraction, glyph decoding and number parsing.

Several older EWURA reports embed fonts without a ToUnicode map, so
pdfplumber returns ``(cid:N)`` placeholders. In the documents surveyed the
glyph IDs are a constant offset from ASCII (``chr(N + 25)``; ``N == 3`` is a
space). Decoding is only accepted when the decoded page contains known words,
and callers flag values that came from decoded text.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from pipeline import config

log = logging.getLogger(__name__)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

TEXT_CACHE_DIR: Path = config.CACHE_DIR / "pdf_text"

_CID = re.compile(r"\(cid:(\d+)\)")
_CID_OFFSET = 25
_VALIDATION_WORDS = re.compile(r"\b(Table|Figure|Natural|Gas|MMscf|the|and)\b")

NUMBER = r"-?\(?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|-?\(?\d+(?:\.\d+)?\)?"
_NUMBER_RE = re.compile(rf"(?<![\w/])({NUMBER})(?![\w/])")
FY_RE = re.compile(r"(?:FY\s*)?(20\d\d)\s*/\s*(?:20)?(\d\d)\b")


@dataclass(frozen=True)
class PageText:
    """Text of one PDF page (1-based ``page_no``)."""

    page_no: int
    text: str
    cid_decoded: bool


def decode_cids(text: str) -> tuple[str, bool]:
    """Replace ``(cid:N)`` glyph codes with characters if the result validates."""
    if "(cid:" not in text:
        return text, False

    def repl(m: re.Match[str]) -> str:
        n = int(m.group(1))
        return " " if n == 3 else chr(n + _CID_OFFSET) if 4 <= n <= 101 else "�"

    # Validate on the glyph-coded parts only: they must decode to real words.
    cid_only = "".join(m.group(0) for m in _CID.finditer(text))
    if len(cid_only) > 200 and not _VALIDATION_WORDS.search(_CID.sub(repl, cid_only)):
        log.debug("cid decoding did not validate; leaving glyph codes in place")
        return text, False
    return _CID.sub(repl, text), True


def read_pages(path: Path, cache_dir: Path | None = TEXT_CACHE_DIR) -> list[PageText]:
    """Extract (and if needed glyph-decode) the text of every page.

    Raw extracted text is cached by file hash, because pdfplumber takes
    minutes on the larger reports; decoding is re-applied on every read.
    """
    raw_pages = _cached_raw_text(path, cache_dir)
    pages: list[PageText] = []
    for i, raw in enumerate(raw_pages, 1):
        text, decoded = decode_cids(raw)
        pages.append(PageText(i, text, decoded))
    return pages


def _cached_raw_text(path: Path, cache_dir: Path | None) -> list[str]:
    cache_file = None
    if cache_dir is not None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        cache_file = cache_dir / f"{digest}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
    with pdfplumber.open(path) as pdf:
        raw_pages = [page.extract_text() or "" for page in pdf.pages]
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(raw_pages, ensure_ascii=False), encoding="utf-8")
    return raw_pages


def parse_number(token: str) -> float | None:
    """Parse "28,432.11", "(1,234)" (negative) or "-" (missing) into a float."""
    t = token.strip()
    if t in {"", "-", "–", "NA", "N/A", "n/a"}:
        return None
    negative = t.startswith("(") and t.endswith(")")
    t = t.strip("()").replace(",", "")
    try:
        value = float(t)
    except ValueError:
        return None
    return -value if negative else value


def numbers_in(line: str) -> list[str]:
    """Return numeric tokens in a line (ignoring percentages and FY labels)."""
    cleaned = FY_RE.sub(" ", line)
    cleaned = re.sub(rf"({NUMBER})\s*%", " ", cleaned)
    return _NUMBER_RE.findall(cleaned)


def fy_label(start_year: int) -> str:
    """2023 -> "2023/24"."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def fiscal_years_in(text: str) -> list[str]:
    """Return FY labels in order of appearance, e.g. ["2020/21", "2021/22"]."""
    out: list[str] = []
    for start, end in FY_RE.findall(text):
        if (int(start) + 1) % 100 == int(end):
            out.append(fy_label(int(start)))
    return out
