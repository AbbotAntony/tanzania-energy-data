"""Provenance-carrying record types shared by all parsers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class DocumentRef:
    """A document discovered on a listing page (before download)."""

    source: str
    url: str
    title: str
    category: str = "other"
    pub_date: str | None = None  # ISO date or free text as published
    listing_url: str | None = None


@dataclass
class Record:
    """One tidy observation with full provenance.

    ``variable`` names what is measured, ``value`` is the number as published
    and ``unit`` is the unit as published (e.g. "MMscf", "litres", "TZS/litre").
    Units are never converted silently; any conversion is described in
    ``conversion_note``.
    """

    source: str
    url: str
    doc_title: str
    retrieved_date: str
    variable: str
    value: float | None
    unit: str
    pub_date: str | None = None
    pdf_page: int | None = None
    table_no: int | None = None
    year: int | None = None
    month: int | None = None
    fy: str | None = None  # Tanzanian fiscal year, e.g. "2023/24" (July-June)
    category: str | None = None  # e.g. customer category, product, region
    value_as_printed: str | None = None
    conversion_note: str = ""
    qa_flag: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        extra = row.pop("extra")
        row.update({f"x_{k}": v for k, v in extra.items()})
        return row


PROVENANCE_FIELDS: tuple[str, ...] = (
    "source", "url", "doc_title", "pub_date", "retrieved_date", "pdf_page", "table_no",
)
RECORD_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Record) if f.name != "extra")
