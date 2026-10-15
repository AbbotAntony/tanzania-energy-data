"""Polite, cached HTTP client shared by every scraper.

Features: descriptive User-Agent, per-host 3-5 s delay, retries with
exponential backoff, timeouts, robots.txt enforcement and an on-disk cache so
pages are not re-downloaded during development.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pipeline import config

log = logging.getLogger(__name__)


class RobotsDisallowed(Exception):
    """Raised when robots.txt forbids fetching a URL."""


@dataclass(frozen=True)
class FetchResult:
    """A fetched resource and where its bytes live on disk."""

    url: str
    final_url: str
    status: int
    content_type: str
    path: Path
    from_cache: bool
    fetched_at: str  # ISO timestamp of the original network retrieval

    def read_bytes(self) -> bytes:
        return self.path.read_bytes()

    def text(self, encoding: str = "utf-8") -> str:
        return self.path.read_bytes().decode(encoding, errors="replace")


class _SystemTrustAdapter(HTTPAdapter):
    """Verify TLS with the OS certificate store (truststore).

    Some sites (e.g. TANESCO) omit an intermediate certificate; the Windows
    store completes the chain, as browsers do. Verification stays enabled.
    """

    def init_poolmanager(self, *args, **kwargs):
        import ssl

        import truststore

        kwargs["ssl_context"] = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return super().init_poolmanager(*args, **kwargs)


def _build_session(user_agent: str, system_trust: bool = False) -> requests.Session:
    retry = Retry(
        total=config.MAX_RETRIES,
        backoff_factor=config.BACKOFF_FACTOR,
        status_forcelist=(429, 500, 502, 503, 504, 520, 522, 524),
        allowed_methods=("GET", "HEAD"),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    adapter = (_SystemTrustAdapter if system_trust else HTTPAdapter)(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": user_agent, "Accept-Language": "en,sw;q=0.8"})
    return session


class HttpClient:
    """Cached, rate-limited, robots-aware HTTP GET client."""

    def __init__(
        self,
        cache_dir: Path = config.CACHE_DIR,
        user_agent: str = config.USER_AGENT,
        min_delay: float = config.MIN_DELAY_S,
        max_delay: float = config.MAX_DELAY_S,
        refresh: bool = False,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        system_trust: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self.user_agent = user_agent
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.refresh = refresh
        self.session = session or _build_session(user_agent, system_trust)
        self._sleep = sleep
        self._last_request: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser | None] = {}

    # ---- public API -------------------------------------------------
    def get(self, url: str, use_cache: bool = True) -> FetchResult:
        """GET ``url``, honouring robots.txt, delay and cache."""
        cached = self._cache_paths(url)
        if use_cache and not self.refresh and cached[1].exists():
            meta = json.loads(cached[1].read_text(encoding="utf-8"))
            log.debug("cache hit %s", url)
            return FetchResult(url, meta["final_url"], meta["status"],
                               meta["content_type"], cached[0], True, meta["fetched_at"])
        if not self.allowed(url):
            raise RobotsDisallowed(url)
        resp = self._request(url)
        return self._store(url, resp, *cached)

    def allowed(self, url: str) -> bool:
        """Return True if robots.txt for the URL's host permits fetching it."""
        parts = urlsplit(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host not in self._robots:
            self._robots[host] = self._load_robots(host)
        parser = self._robots[host]
        return True if parser is None else parser.can_fetch(self.user_agent, url)

    # ---- internals --------------------------------------------------
    def _load_robots(self, host: str) -> RobotFileParser | None:
        """Fetch and parse robots.txt; ``None`` means "no rules published"."""
        robots_url = f"{host}/robots.txt"
        try:
            resp = self._request(robots_url)
        except requests.RequestException as exc:
            log.warning("robots.txt unreachable for %s (%s); treating as no rules", host, exc)
            return None
        ctype = resp.headers.get("Content-Type", "")
        if resp.status_code >= 400 or "html" in ctype.lower():
            log.info("no robots.txt at %s (HTTP %s, %s); treating as no rules",
                     host, resp.status_code, ctype)
            return None
        parser = RobotFileParser()
        parser.parse(resp.text.splitlines())
        return parser

    def _throttle(self, host: str) -> None:
        last = self._last_request.get(host)
        if last is not None:
            wait = random.uniform(self.min_delay, self.max_delay) - (time.monotonic() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_request[host] = time.monotonic()

    def _request(self, url: str) -> requests.Response:
        self._throttle(urlsplit(url).netloc)
        log.info("GET %s", url)
        resp = self.session.get(
            url, timeout=(config.CONNECT_TIMEOUT_S, config.READ_TIMEOUT_S)
        )
        return resp

    def _cache_paths(self, url: str) -> tuple[Path, Path]:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.bin", self.cache_dir / f"{key}.json"

    def _store(self, url: str, resp: requests.Response, body: Path, meta: Path) -> FetchResult:
        resp.raise_for_status()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        body.write_bytes(resp.content)
        info = {
            "url": url,
            "final_url": resp.url,
            "status": resp.status_code,
            "content_type": resp.headers.get("Content-Type", ""),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        meta.write_text(json.dumps(info), encoding="utf-8")
        return FetchResult(url, resp.url, resp.status_code, info["content_type"],
                           body, False, info["fetched_at"])
