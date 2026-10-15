"""TANESCO (Tanzania Electric Supply Company) scraper.

Why Playwright: https://www.tanesco.co.tz/annual-report is an Angular
single-page app whose HTML is an empty ``<app-root>``. The document list is
loaded by the page's own JavaScript (``/api/v1/documents/allAnnualReports``),
which rejects direct requests without the browser session. We render the
public page in headless Chromium and read that response, exactly as a
visitor's browser does; no protected endpoint is called directly. The PDFs
themselves are public files and are downloaded with the normal polite client.
robots.txt: none published (HTML returned), treated as no rules.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from pipeline import config
from pipeline.provenance import DocumentRef, Record
from pipeline.storage import StoredDoc
from scrapers.base import Scraper

log = logging.getLogger(__name__)

LISTING_PAGE = "https://www.tanesco.co.tz/annual-report"
LIST_RESPONSE_MARKER = "allAnnualReports"


def refs_from_listing(data: dict, listing_url: str = LISTING_PAGE) -> list[DocumentRef]:
    """Turn the page's annual-report JSON into DocumentRefs."""
    items = (data.get("_embedded") or {}).get("documentResponseDTOList", [])
    refs = []
    for item in items:
        if not item.get("isActive", True):
            continue
        created = item.get("createdDate")
        pub = time.strftime("%Y-%m-%d", time.gmtime(created)) if created else None
        for url in item.get("attachmentUrlList") or []:
            if url.lower().endswith(".pdf"):
                refs.append(DocumentRef("tanesco", url, item.get("title", "").strip(),
                                        "annual_report", pub, listing_url))
    return refs


def refs_from_manifests(source_dir: Path) -> list[DocumentRef]:
    """DocumentRefs of PDFs recorded in earlier raw-data manifests (newest first)."""
    refs: dict[str, DocumentRef] = {}
    for manifest in sorted(source_dir.glob("*/manifest.jsonl"), reverse=True):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e["url"].lower().endswith(".pdf"):
                refs.setdefault(e["url"], DocumentRef("tanesco", e["url"], e["title"], e["category"],
                                                      e["pub_date"], e.get("listing_url")))
    return list(refs.values())


def render_listing(user_agent: str = config.USER_AGENT, timeout_ms: int = 90000,
                   attempts: int = 2) -> dict:
    """Open the public annual-report page in headless Chromium; return its list JSON.

    Waits only for the list response (the page's chat widget keeps the network
    busy, so "networkidle" can time out); retries once.
    """
    from playwright.sync_api import sync_playwright

    last_error: Exception | None = None
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            for attempt in range(1, attempts + 1):
                page = browser.new_page(user_agent=user_agent)
                try:
                    with page.expect_response(lambda r: LIST_RESPONSE_MARKER in r.url,
                                              timeout=timeout_ms) as resp:
                        page.goto(LISTING_PAGE, wait_until="domcontentloaded", timeout=timeout_ms)
                    log.info("rendered %s; list response HTTP %s", LISTING_PAGE, resp.value.status)
                    return resp.value.json()
                except Exception as exc:  # retry once, then give up
                    last_error = exc
                    log.warning("render attempt %d failed: %s", attempt, exc)
                    time.sleep(5)
                finally:
                    page.close()
        finally:
            browser.close()
    raise RuntimeError(f"could not render {LISTING_PAGE}") from last_error


class TanescoScraper(Scraper):
    """TANESCO annual reports (listing rendered with Playwright; PDFs via HTTP)."""

    name = "tanesco"
    system_trust = True  # server omits an intermediate certificate; see http_client

    def list_documents(self) -> list[DocumentRef]:
        try:
            return refs_from_listing(render_listing())
        except Exception as exc:  # browser or network failure: reuse what was stored before
            self._fail(LISTING_PAGE, exc)
            refs = refs_from_manifests(self.raw_dir / self.name)
            log.warning("using %d documents from earlier manifests (offline fallback)", len(refs))
            return refs

    def parse(self, docs: list[StoredDoc]) -> dict[str, list[Record]]:
        from parsers import tanesco_ar

        records: list[Record] = []
        for doc in docs:
            try:
                records += tanesco_ar.parse(doc)
            except Exception as exc:  # one bad PDF must not stop the run
                self._fail(doc.ref.url, exc)
        return {"tanesco_generation": records}
