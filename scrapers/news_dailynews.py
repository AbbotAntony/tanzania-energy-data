"""Daily News (dailynews.co.tz) via its public WordPress REST API.

robots.txt (Yoast): ``Disallow:`` (all allowed). Endpoint verified 2026-10-15:
  https://dailynews.co.tz/wp-json/wp/v2/posts?search=<term>&per_page=100&page=N
The API holds articles from about August 2021 (older stories were not
migrated). Search is word-based, so every article is re-checked against exact
topic phrases (parsers/news.py) and kept only if it matches.
"""

from __future__ import annotations

import json
import logging
from urllib.parse import quote

import requests

from parsers.news import SEARCH_TERMS, article_record, html_to_text, match_topics
from pipeline.http_client import RobotsDisallowed
from pipeline.provenance import DocumentRef, Record
from pipeline.storage import StoredDoc, store_raw
from scrapers.base import Scraper

log = logging.getLogger(__name__)

API = "https://dailynews.co.tz/wp-json/wp/v2/posts"
FIELDS = "id,date,modified,link,title,content,excerpt"
MAX_PAGES_PER_TERM = 20


class DailyNewsScraper(Scraper):
    """Daily News articles on gas, HFO, pipelines, TPDC/GASCO and JNHPP (full text)."""

    name = "dailynews"

    def list_documents(self) -> list[DocumentRef]:
        """Each API result page is one 'document' (stored raw as JSON)."""
        refs: list[DocumentRef] = []
        for term in SEARCH_TERMS:
            for page in range(1, MAX_PAGES_PER_TERM + 1):
                url = f"{API}?search={quote(term)}&per_page=100&page={page}&_fields={FIELDS}"
                try:
                    items = json.loads(self.client.get(url).text())
                except requests.HTTPError:
                    break  # WordPress answers 400 past the last page
                except (requests.RequestException, RobotsDisallowed, ValueError) as exc:
                    self._fail(url, exc)
                    break
                refs.append(DocumentRef(self.name, url, f"API search '{term}' page {page}",
                                        "api_page", None, API))
                if len(items) < 100:
                    break
        return refs

    def parse(self, docs: list[StoredDoc]) -> dict[str, list[Record]]:
        records: dict[int, Record] = {}
        for doc in docs:
            for item in json.loads(doc.path.read_text(encoding="utf-8")):
                if item["id"] in records:
                    continue
                title = html_to_text(item["title"]["rendered"])
                text = html_to_text(item["content"]["rendered"])
                topics = match_topics(f"{title} {text}")
                if topics:
                    records[item["id"]] = article_record(
                        "Daily News", item["link"], title, item["date"][:10], doc.retrieved_date,
                        text, topics, post_id=item["id"], modified=item.get("modified", "")[:10],
                        api_query=doc.ref.title)
        log.info("%d relevant Daily News articles", len(records))
        return {"news_dailynews": list(records.values())}
