"""Common scraper workflow: list documents -> download -> parse -> export."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import requests

from parsers.listing import find_page_links
from pipeline import config
from pipeline.export import export
from pipeline.http_client import HttpClient, RobotsDisallowed
from pipeline.provenance import DocumentRef, Record
from pipeline.storage import StoredDoc, store_raw

log = logging.getLogger(__name__)


class Scraper(ABC):
    """Base class. Subclasses implement ``list_documents`` and ``parse``."""

    name: str = ""
    max_pages: int = 30
    system_trust: bool = False  # verify TLS with the OS store (sites with incomplete chains)

    def __init__(self, client: HttpClient, raw_dir: Path = config.RAW_DIR,
                 out_dir: Path = config.PROCESSED_DIR) -> None:
        self.client = client
        self.raw_dir = raw_dir
        self.out_dir = out_dir
        self.failures: list[tuple[str, str]] = []

    # ---- steps --------------------------------------------------------
    @abstractmethod
    def list_documents(self) -> list[DocumentRef]:
        """Discover documents on the source's listing pages."""

    def wanted(self, ref: DocumentRef) -> bool:
        """Return True if a listed document should be downloaded."""
        return True

    def download(self, refs: list[DocumentRef]) -> list[StoredDoc]:
        """Fetch each wanted document and store it unchanged in data/raw."""
        stored: list[StoredDoc] = []
        for ref in refs:
            if not self.wanted(ref):
                continue
            try:
                fetched = self.client.get(ref.url)
                stored.append(store_raw(self.name, ref, fetched, self.raw_dir))
            except (requests.RequestException, RobotsDisallowed) as exc:
                self._fail(ref.url, exc)
        return stored

    @abstractmethod
    def parse(self, docs: list[StoredDoc]) -> dict[str, list[Record]]:
        """Parse stored documents; return ``{dataset_name: records}``."""

    def export(self, datasets: dict[str, list[Record]]) -> None:
        for name, records in datasets.items():
            if records:
                export(records, name, self.out_dir)
            else:
                log.warning("dataset %s is empty; nothing written", name)

    def run(self) -> dict[str, list[Record]]:
        refs = self.list_documents()
        log.info("%s: %d documents listed", self.name, len(refs))
        docs = self.download(refs)
        datasets = self.parse(docs)
        self.export(datasets)
        self._report_failures()
        return datasets

    # ---- helpers ------------------------------------------------------
    def crawl_listing(self, start_url: str, parse_page) -> list[DocumentRef]:
        """Follow ``?page=N`` links from ``start_url`` and parse every page."""
        seen: set[str] = set()
        queue = [start_url]
        refs: dict[str, DocumentRef] = {}
        while queue and len(seen) < self.max_pages:
            url = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)
            try:
                fetched = self.client.get(url)
                html = fetched.text()
                store_raw(self.name, DocumentRef(self.name, url, "listing page", "listing"),
                          fetched, self.raw_dir)
            except (requests.RequestException, RobotsDisallowed) as exc:
                self._fail(url, exc)
                continue
            for ref in parse_page(html, url):
                refs.setdefault(ref.url, ref)
            queue.extend(u for u in find_page_links(html, url) if u not in seen)
        return list(refs.values())

    def _fail(self, url: str, exc: Exception) -> None:
        log.warning("FAILED %s: %s: %s", url, type(exc).__name__, exc)
        self.failures.append((url, f"{type(exc).__name__}: {exc}"))

    def _report_failures(self) -> None:
        if self.failures:
            log.warning("%s: %d failed URL(s) (see log above)", self.name, len(self.failures))
