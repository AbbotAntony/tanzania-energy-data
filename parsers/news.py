"""Topic matching and records for energy news articles."""

from __future__ import annotations

import re
from html import unescape

from bs4 import BeautifulSoup

from pipeline.provenance import Record

# Exact phrases (case-insensitive) that make an article relevant to the study.
TOPICS: dict[str, re.Pattern[str]] = {
    "natural_gas": re.compile(r"natural gas|gesi asilia|\bLNG\b|\bCNG\b", re.I),
    "hfo": re.compile(r"\bHFO\b|heavy fuel oil|fuel oil|mafuta mazito", re.I),
    "pipeline": re.compile(r"gas pipeline|pipeline", re.I),
    "gasco": re.compile(r"\bGASCO\b|Gas Company \(Tanzania\)", re.I),
    "tpdc": re.compile(r"\bTPDC\b|Tanzania Petroleum Development Corporation", re.I),
    "songo_songo": re.compile(r"Songo\s*Songo|\bSongas\b", re.I),
    "mtwara_mnazi": re.compile(r"Mnazi Bay|Madimba|Mtwara", re.I),
    "jnhpp": re.compile(r"Julius Nyerere Hydro|\bJNHPP\b|Rufiji Hydro|Stiegler", re.I),
    "gas_power_plants": re.compile(r"Kinyerezi|Ubungo (I|II|III|gas)|gas[- ]fired", re.I),
}
# Mtwara and "pipeline" alone are too broad: require a gas/energy topic as well.
WEAK = {"pipeline", "mtwara_mnazi"}
SEARCH_TERMS: tuple[str, ...] = (
    "natural gas", "HFO", "heavy fuel oil", "gas pipeline", "GASCO", "TPDC", "Songo Songo",
    "Songas", "Mnazi Bay", "Julius Nyerere Hydropower", "JNHPP", "Kinyerezi",
)


def html_to_text(html: str) -> str:
    return " ".join(BeautifulSoup(unescape(html or ""), "lxml").get_text(" ").split())


def match_topics(text: str) -> list[str]:
    """Topics found in ``text``; weak-only matches (e.g. just "Mtwara") are dropped."""
    found = [name for name, rx in TOPICS.items() if rx.search(text)]
    return found if set(found) - WEAK else []


def article_record(source: str, url: str, title: str, date: str | None, retrieved: str,
                   text: str = "", topics: list[str] | None = None, **extra: object) -> Record:
    topics = topics or []
    return Record(
        source=source, url=url, doc_title=title, pub_date=date, retrieved_date=retrieved,
        variable="news_article", value=None, unit="text",
        year=int(date[:4]) if date else None, month=int(date[5:7]) if date else None,
        category=";".join(topics), value_as_printed=title,
        extra={"topics": ";".join(topics), "text": text, "word_count": len(text.split()), **extra},
    )
