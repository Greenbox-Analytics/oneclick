"""Fetching subscribed .ics calendar feeds.

The URL here is supplied by the user and fetched BY THE SERVER, which makes this a
classic SSRF sink: without checks, "https://169.254.169.254/..." would have the
backend read Cloud Run's metadata endpoint (service-account tokens) and hand the
body back to the caller. Every hostname is therefore resolved and checked to be a
public address before a request is made, redirects are followed manually so each
hop is re-checked, and the response is size-capped while streaming.
"""

import ipaddress
import socket
from datetime import date
from urllib.parse import urlparse, urlunparse

import httpx
from cachetools import TTLCache

from boards import ics

# Feeds are re-fetched at most this often. Google/Apple publish on a slow cadence
# anyway, so a short cache keeps calendar paging instant without going stale.
_CACHE_TTL_SECONDS = 900
_MAX_BYTES = 5 * 1024 * 1024
_TIMEOUT_SECONDS = 10.0
_MAX_REDIRECTS = 3

# url -> raw ics text. Per-process; a cold start just refetches.
_feed_cache: TTLCache = TTLCache(maxsize=512, ttl=_CACHE_TTL_SECONDS)


class FeedError(Exception):
    """A subscribed feed could not be fetched or wasn't a calendar."""


def normalize_url(raw: str) -> str:
    """Accept what users actually paste. Calendar apps hand out webcal:// links,
    which are just https:// with a different scheme sticker."""
    url = (raw or "").strip()
    if url.lower().startswith("webcal://"):
        url = "https://" + url[len("webcal://") :]
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise FeedError("Calendar links must start with https:// or webcal://")
    if not parsed.hostname:
        raise FeedError("That doesn't look like a calendar link")
    return urlunparse(parsed)


def _assert_public_host(url: str) -> None:
    """Block loopback, private, link-local and other non-public targets.

    Checks EVERY address the hostname resolves to: a name resolving to both a
    public and a private address must not slip through on the public one.
    """
    host = urlparse(url).hostname
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise FeedError(f"Could not resolve {host}") from None

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            raise FeedError("That address isn't reachable from the public internet")


async def _read_capped(response: httpx.Response) -> str:
    """Read the body, aborting past the cap. Content-Length is advisory — a feed
    can lie about it or omit it — so the running total is what enforces the limit."""
    chunks, total = [], 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > _MAX_BYTES:
            raise FeedError("That calendar is too large to import")
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


async def fetch_feed(url: str, use_cache: bool = True) -> str:
    """Fetch a calendar feed's raw .ics text, following redirects one validated hop
    at a time (httpx's own redirect handling would skip the per-hop SSRF check)."""
    if use_cache and url in _feed_cache:
        return _feed_cache[url]

    current = url
    async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS, follow_redirects=False) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            _assert_public_host(current)
            try:
                request = client.build_request("GET", current, headers={"Accept": "text/calendar"})
                response = await client.send(request, stream=True)
            except httpx.HTTPError as exc:
                raise FeedError(f"Could not reach that calendar: {exc}") from exc

            try:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise FeedError("That calendar link redirects nowhere")
                    current = normalize_url(str(response.next_request.url) if response.next_request else location)
                    continue
                if response.status_code == 404:
                    raise FeedError("That calendar link no longer exists (404)")
                if response.status_code in (401, 403):
                    raise FeedError("That calendar is private — use the secret iCal address")
                if response.status_code >= 400:
                    raise FeedError(f"That calendar returned an error ({response.status_code})")
                text = await _read_capped(response)
            finally:
                await response.aclose()

            if "BEGIN:VCALENDAR" not in text:
                raise FeedError("That link isn't a calendar feed (.ics)")
            _feed_cache[url] = text
            return text

    raise FeedError("That calendar link redirects too many times")


async def load_events(url: str, start: date, end: date, timezone: str | None = None) -> list[dict]:
    """Fetch + parse one subscription into overlay events for the given window."""
    return ics.parse_ics(await fetch_feed(url), start, end, timezone)


def forget(url: str) -> None:
    """Drop a cached feed — used on manual refresh and on unsubscribe."""
    _feed_cache.pop(url, None)
