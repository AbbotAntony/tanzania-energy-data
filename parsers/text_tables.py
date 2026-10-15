"""Reconstruct simple tables from extracted PDF text lines.

PDF tables in EWURA reports extract cleanly as one text line per row, but
row labels sometimes wrap onto neighbouring lines and blank cells are printed
as "-" or omitted. This module finds a captioned table block, tokenises each
row into numeric cells and joins wrapped labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from parsers.pdf_text import NUMBER, PageText, fiscal_years_in, parse_number

_TOC_LINE = re.compile(r"\.{5,}")
_BLOCK_END = re.compile(
    r"^(Source\s*:|Figure\s*\d|Table\s*\w{1,2}\s*[:.]|Jedwali|Annex\s*\d|\d+(\.\d+)+\s+[A-Z]"
    r"|.*Performance Report for|.*SUB-SECTOR REPORT)", re.I)
_SN_PREFIX = re.compile(r"^\s*\d{1,2}\.?\s+(?=\D)")
# Lookbehind excludes "-" so the "1" in "Jet A-1" is not read as a number.
_CELL = re.compile(rf"(?<![\w/.\-])({NUMBER})(\s*%)?(?![\w/])|(?<![\w-])([-–])(?![\w-])")
_FY = re.compile(r"(?:FY\s*)?20\d\d\s*/\s*(?:20)?\d\d\b")
_TABLE_NO = re.compile(r"^(?:Table|Annex)\s*(\d+)", re.I)


@dataclass(frozen=True)
class Cell:
    """A numeric cell; ``value`` is None for a printed blank such as "-"."""

    value: float | None
    printed: str
    is_pct: bool


@dataclass
class Row:
    label: str
    cells: list[Cell]
    line: str


@dataclass
class TableBlock:
    """A captioned table located in a PDF."""

    caption: str
    page_no: int
    table_no: int | None
    header_lines: list[str]
    rows: list[Row]
    cid_decoded: bool
    years: list[str] = field(default_factory=list)
    body: list[str] = field(default_factory=list)  # raw lines after the caption


def tokenise(line: str) -> tuple[str, list[Cell]]:
    """Split a row line into its text label and numeric cells."""
    body = _FY.sub(" ", _SN_PREFIX.sub("", line, count=1))
    cells: list[Cell] = []
    for m in _CELL.finditer(body):
        if m.group(3):
            cells.append(Cell(None, m.group(3), False))
        else:
            cells.append(Cell(parse_number(m.group(1)), m.group(0).strip(), bool(m.group(2))))
    label = _CELL.sub(" ", body)
    return " ".join(label.split()).strip(" ,"), cells


def find_tables(pages: list[PageText], caption_re: re.Pattern[str],
                max_lines: int = 30) -> list[TableBlock]:
    """Return every table whose caption line matches ``caption_re`` (TOC lines skipped)."""
    blocks: list[TableBlock] = []
    for page in pages:
        lines = page.text.splitlines()
        for i, line in enumerate(lines):
            if caption_re.search(line) and not _TOC_LINE.search(line):
                body = _take_block(lines[i + 1:i + 1 + max_lines])
                blocks.append(_build_block(line.strip(), page, body))
    return blocks


def _take_block(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        if _BLOCK_END.search(line.strip()) and out:
            break
        out.append(line)
    return out


def _build_block(caption: str, page: PageText, body: list[str]) -> TableBlock:
    """Split a table body into header lines and data rows.

    Wrapped-label rules: text-only lines seen after the first data row are
    "fragments". A fragment joins the *next* row when that row has no label
    or the fragment ends with "," / "and"; otherwise it joins the previous row.
    """
    rows: list[Row] = []
    header: list[str] = []
    fragments: list[str] = []
    for line in body:
        if _PAGE_NO.match(line):
            continue
        label, cells = tokenise(line)
        if rows and _is_prose(line, cells):
            break
        is_row = any(not c.is_pct for c in cells) and not _is_header(line)
        if not rows and not is_row:
            header.append(line)
            continue
        if not is_row:
            fragments.append(line.strip())
            continue
        if not label and fragments and _is_serial(cells):
            cells = cells[1:]  # "4 879.5 ..." under a wrapped label: 4 is the S/N
        if not rows and not label and header and not _FY.search(header[-1]) \
                and re.search(r"[A-Za-z]{3}", header[-1]):
            label = header.pop().strip()  # first row's label printed on the line above it
        if fragments:
            if not label or fragments[-1].endswith((",", " and")):
                label = " ".join([*fragments, label]).strip()
            else:
                rows[-1].label = " ".join([rows[-1].label, *fragments])
            fragments = []
        if label:
            rows.append(Row(label, cells, line))
    if fragments and rows:
        rows[-1].label = " ".join([rows[-1].label, *fragments])
    m = _TABLE_NO.search(caption)
    years = fiscal_years_in(" ".join([caption, *header]))
    return TableBlock(caption, page.page_no, int(m.group(1)) if m else None,
                      header, rows, page.cid_decoded, years, body)


_PAGE_NO = re.compile(r"^\s*\d{1,3}\s*$")


def _is_serial(cells: list[Cell]) -> bool:
    """True if the first cell looks like a row serial number (small bare integer)."""
    first = cells[0].printed if cells else ""
    return len(cells) >= 2 and first.isdigit() and int(first) <= 50


def _is_prose(line: str, cells: list[Cell]) -> bool:
    """A sentence (many words, at most one number) marks the end of a table."""
    return len(re.findall(r"[A-Za-z]{2,}", line)) >= 9 and len(cells) <= 1


def _is_header(line: str) -> bool:
    """Header lines carry FY labels but no real data values."""
    stripped = _FY.sub(" ", line)
    has_data = re.search(r"\d{1,3},\d{3}|\d+\.\d+", stripped)
    return bool(_FY.search(line)) and not has_data


def unique_years(years: list[str], order: str = "sorted") -> list[str]:
    """Distinct FY labels, either sorted ascending or in order of first appearance."""
    seen: list[str] = []
    for y in years:
        if y not in seen:
            seen.append(y)
    return sorted(seen) if order == "sorted" else seen
