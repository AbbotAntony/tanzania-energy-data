"""HttpClient: robots.txt, caching, throttling (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from pipeline.http_client import HttpClient, RobotsDisallowed


class FakeResponse:
    def __init__(self, url: str, body: bytes, status: int = 200, ctype: str = "text/html"):
        self.url = url
        self.content = body
        self.text = body.decode()
        self.status_code = status
        self.headers = {"Content-Type": ctype}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Serves canned responses and records every requested URL."""

    def __init__(self, routes: dict[str, FakeResponse]):
        self.routes = routes
        self.calls: list[str] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, timeout: object = None) -> FakeResponse:
        self.calls.append(url)
        return self.routes.get(url, FakeResponse(url, b"not found", 404))


ROBOTS = b"User-agent: *\nDisallow: /private/\n"


def make_client(tmp_path: Path, routes: dict[str, FakeResponse], sleeps: list[float]) -> HttpClient:
    return HttpClient(cache_dir=tmp_path, session=FakeSession(routes),  # type: ignore[arg-type]
                      sleep=sleeps.append, min_delay=3, max_delay=5)


def test_robots_disallow_is_enforced(tmp_path: Path) -> None:
    routes = {"https://a.tz/robots.txt": FakeResponse("https://a.tz/robots.txt", ROBOTS, ctype="text/plain")}
    client = make_client(tmp_path, routes, [])
    with pytest.raises(RobotsDisallowed):
        client.get("https://a.tz/private/doc.pdf")


def test_html_robots_means_no_rules(tmp_path: Path) -> None:
    """BoT/TANESCO answer /robots.txt with their HTML app: treat as no rules."""
    routes = {
        "https://b.tz/robots.txt": FakeResponse("https://b.tz/robots.txt", b"<html></html>"),
        "https://b.tz/page": FakeResponse("https://b.tz/page", b"ok"),
    }
    client = make_client(tmp_path, routes, [])
    assert client.get("https://b.tz/page").text() == "ok"


def test_second_get_is_served_from_cache(tmp_path: Path) -> None:
    routes = {
        "https://a.tz/robots.txt": FakeResponse("https://a.tz/robots.txt", ROBOTS, ctype="text/plain"),
        "https://a.tz/list": FakeResponse("https://a.tz/list", b"<p>listing</p>"),
    }
    client = make_client(tmp_path, routes, [])
    first = client.get("https://a.tz/list")
    second = client.get("https://a.tz/list")
    assert not first.from_cache and second.from_cache
    assert client.session.calls.count("https://a.tz/list") == 1  # type: ignore[attr-defined]
    assert second.fetched_at == first.fetched_at


def test_requests_to_same_host_are_delayed(tmp_path: Path) -> None:
    routes = {
        "https://a.tz/robots.txt": FakeResponse("https://a.tz/robots.txt", ROBOTS, ctype="text/plain"),
        "https://a.tz/1": FakeResponse("https://a.tz/1", b"1"),
        "https://a.tz/2": FakeResponse("https://a.tz/2", b"2"),
    }
    sleeps: list[float] = []
    client = make_client(tmp_path, routes, sleeps)
    client.get("https://a.tz/1")
    client.get("https://a.tz/2")
    assert len(sleeps) == 2 and all(0 < s <= 5 for s in sleeps)


def test_http_error_is_raised_and_not_cached(tmp_path: Path) -> None:
    routes = {"https://a.tz/robots.txt": FakeResponse("https://a.tz/robots.txt", ROBOTS, ctype="text/plain")}
    client = make_client(tmp_path, routes, [])
    with pytest.raises(requests.HTTPError):
        client.get("https://a.tz/missing")
    assert not list(tmp_path.glob("*.json"))
