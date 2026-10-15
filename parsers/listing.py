"""Parse document listing pages into ``DocumentRef`` objects.

Most Tanzanian government sites (EWURA, TPDC, PURA, PBPA, Ministry of Energy,
GASCO) run the same e-Government CMS template: each document is an ``<a>``
card containing an ``h6.card-text`` title and a ``div.text-muted`` date.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Callable
from urllib.parse import parse_qs, urljoin, urlsplit

from bs4 import BeautifulSoup

from pipeline.provenance import DocumentRef

Classifier = Callable[[str, str], str]


def parse_card_date(text: str) -> str | None:
    """Parse dates like "14 Jun, 2024" to ISO "2024-06-14"; None if unparseable."""
    cleaned = " ".join(text.replace(",", " ").split())
    cleaned = re.sub(r"^(\d{1,2})(st|nd|rd|th)\b", r"\1", cleaned)  # "08th Jul 2026"
    for fmt in ("%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_egov_cards(html: str, base_url: str, source: str,
                     classify: Classifier | None = None) -> list[DocumentRef]:
    """Extract document cards (title, date, PDF link) from an e-Gov CMS listing page."""
    soup = BeautifulSoup(html, "lxml")
    refs: dict[str, DocumentRef] = {}
    for a in soup.select("a[href]"):
        title_el = a.select_one("h6.card-text")
        if title_el is None:
            continue
        url = _unwrap_viewer(urljoin(base_url, a["href"].strip()))
        if not url.lower().endswith(".pdf"):
            continue
        title = " ".join(title_el.get_text(" ").split())
        date_el = a.select_one("div.text-muted")
        pub = parse_card_date(date_el.get_text()) if date_el else None
        category = classify(title, url) if classify else "other"
        refs.setdefault(url, DocumentRef(source, url, title, category, pub, base_url))
    return list(refs.values())


def parse_publication_cards(html: str, base_url: str, source: str,
                            classify: Classifier | None = None) -> list[DocumentRef]:
    """Parse the newer e-Gov "publication-section" cards (used by TPDC).

    Each card has a title ``div``, a "Published Date:" span and an
    ``<a download href="...pdf">`` link.
    """
    soup = BeautifulSoup(html, "lxml")
    refs: dict[str, DocumentRef] = {}
    for a in soup.select("a[download][href]"):
        url = _unwrap_viewer(urljoin(base_url, a["href"].strip()))
        card = a.find_parent("div", class_=lambda c: c and any(k.startswith("col-") for k in c.split()))
        if card is None or not url.lower().endswith(".pdf"):
            continue
        title_el = card.select_one(".publication-section .ls-01") or card.select_one("img[alt]")
        title = " ".join((title_el.get_text(" ") if title_el.name != "img" else title_el["alt"]).split())
        date_el = card.select_one(".small-details span")
        pub = parse_card_date(date_el.get_text()) if date_el else None
        category = classify(title, url) if classify else "other"
        refs.setdefault(url, DocumentRef(source, url, title, category, pub, base_url))
    return list(refs.values())


def parse_article_cards(html: str, base_url: str, source: str,
                        classify: Classifier | None = None) -> list[DocumentRef]:
    """Parse ``<article>`` cards with an ``h5`` title and "Posted On <date>" (Ministry of Energy)."""
    soup = BeautifulSoup(html, "lxml")
    refs: dict[str, DocumentRef] = {}
    for art in soup.select("article"):
        a = next((x for x in art.select("a[href]") if ".pdf" in x["href"].lower()), None)
        title_el = art.select_one("h5, h6")
        if a is None or title_el is None:
            continue
        url = urljoin(base_url, a["href"].strip())
        title = " ".join(title_el.get_text(" ").split())
        date_el = art.select_one("p.text-muted")
        date_text = date_el.get_text(" ").replace("Posted On", "") if date_el else ""
        pub = parse_card_date(date_text.strip())
        category = classify(title, url) if classify else "other"
        refs.setdefault(url, DocumentRef(source, url, title, category, pub, base_url))
    return list(refs.values())


def parse_bot_publications(html: str, base_url: str, source: str,
                           category: str) -> list[DocumentRef]:
    """Parse a Bank of Tanzania ``/Publications/Filter/<n>`` table.

    Each row has a date cell and an ``<a download="<Series> - <period>">`` link.
    """
    soup = BeautifulSoup(html, "lxml")
    refs: dict[str, DocumentRef] = {}
    for a in soup.select("a[download][href]"):
        url = urljoin(base_url, a["href"].strip())
        if not url.lower().endswith(".pdf"):
            continue
        title = " ".join(a["download"].split())
        row = a.find_parent("tr")
        cells = [td.get_text(" ", strip=True) for td in row.find_all("td")] if row else []
        pub = next((parse_card_date(c) for c in cells if parse_card_date(c)), None)
        refs.setdefault(url, DocumentRef(source, url, title, category, pub, base_url))
    return list(refs.values())


def find_page_links(html: str, base_url: str) -> list[str]:
    """Return absolute ``?page=N`` pagination URLs for the same listing path."""
    soup = BeautifulSoup(html, "lxml")
    base_path = urlsplit(base_url).path
    pages: dict[int, str] = {}
    for a in soup.select("a[href*='page=']"):
        url = urljoin(base_url, a["href"].strip())
        parts = urlsplit(url)
        num = parse_qs(parts.query).get("page", [""])[0]
        if parts.path == base_path and num.isdigit():
            pages[int(num)] = url
    return [pages[k] for k in sorted(pages)]


def _unwrap_viewer(url: str) -> str:
    """TPDC wraps PDFs as ``/pdf-viewer?file=<url>``; return the inner URL."""
    parts = urlsplit(url)
    if parts.path.endswith("/pdf-viewer"):
        inner = parse_qs(parts.query).get("file", [""])[0]
        return inner or url
    return url


_WS = re.compile(r"\s+")


def normalise_title(title: str) -> str:
    return _WS.sub(" ", title).strip().lower()
