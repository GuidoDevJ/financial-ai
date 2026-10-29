"""News ingestion: RSS feeds -> one LlamaIndex Document per item -> vector store.

CLI: `uv run python -m app.rag.ingest_news`
"""

import calendar
import hashlib
import json
import logging
import random
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import feedparser
import httpx
import redis
from bs4 import BeautifulSoup
from llama_index.core.schema import Document

from app.core.audit import log_event, safe_url, sanitize
from app.core.bedrock_errors import is_transient_bedrock_error
from app.core.config import Settings, get_settings
from app.core.redis_client import get_redis_client
from app.rag.store import NewsStore, build_news_store
from app.rag.tickers import detect_tickers

logger = logging.getLogger(__name__)
# httpx logs every request at INFO with the full URL (query string included): feed URLs may carry
# API keys, so keep that logger quiet; per-feed outcomes are logged sanitized by this module.
logging.getLogger("httpx").setLevel(logging.WARNING)

INGEST_LOCK_KEY = "lock:ingest_news"
INGEST_LAST_RUN_KEY = "ingest:last_run"
INGEST_LAST_SUCCESS_KEY = "ingest:last_success"
MAX_TEXT_CHARS = 2000  # Cohere embed v3 limits input length; news ledes are far shorter.
RETRY_CAP_SECONDS = 20.0
MAX_ERROR_CHARS = 200
ACCEPT = "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.5"
_FEED_MARKERS = (b"<rss", b"<feed", b"<rdf:rdf")
_URL_IN_TEXT = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S+")


class InvalidFeed(Exception):
    """The response is not a usable RSS/Atom feed (HTML page, garbage)."""


class FeedTooLarge(Exception):
    """The response body exceeded NEWS_MAX_FEED_BYTES."""


@dataclass
class IngestOptions:
    """Tunables of one run. `sleep` / `jitter` are injectable so tests never wait."""

    batch_size: int = 10
    max_retries: int = 3
    retry_base_seconds: float = 1.0
    max_feed_bytes: int = 5_000_000
    user_agent: str = "financial-ai-news-ingest/1.0"
    sleep: Callable[[float], None] = time.sleep
    jitter: Callable[[], float] = random.random

    @classmethod
    def from_settings(cls, settings: Settings, **overrides) -> "IngestOptions":
        return cls(
            batch_size=settings.ingest_batch_size,
            max_retries=settings.ingest_max_retries,
            retry_base_seconds=settings.ingest_retry_base_seconds,
            max_feed_bytes=settings.news_max_feed_bytes,
            user_agent=settings.news_user_agent,
            **overrides,
        )


@dataclass(frozen=True)
class NewsItem:
    title: str
    summary: str
    url: str
    source: str
    published: datetime  # timezone-aware UTC


@dataclass
class FeedStat:
    """Per-feed outcome of one run. `url` is scheme+host+path; `error` is sanitized."""

    url: str
    status: str = "ok"  # ok | failed | empty
    fetched: int = 0
    new: int = 0  # new or changed items sent to the store
    unchanged: int = 0
    old: int = 0
    invalid: int = 0
    indexed: int = 0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class IngestReport:
    indexed: int = 0
    skipped_old: int = 0
    skipped_invalid: int = 0
    skipped_unchanged: int = 0
    purged: int = 0
    skipped_locked: bool = False
    failed_feeds: list[tuple[str, str]] = field(default_factory=list)
    feeds: list[FeedStat] = field(default_factory=list)
    purge_error: str | None = None

    @property
    def ok_feeds(self) -> int:
        """Feeds that were fetched and valid (a valid feed with no entries is healthy)."""
        return sum(1 for f in self.feeds if f.status != "failed")

    @property
    def failed_feed_count(self) -> int:
        return sum(1 for f in self.feeds if f.status == "failed")

    def feeds_as_dicts(self) -> list[dict]:
        return [asdict(f) for f in self.feeds]


def stable_id(url: str) -> str:
    """Deterministic document id: hash of the URL (dedupe / upsert key)."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


def content_hash(title: str, summary: str) -> str:
    """sha256 of whitespace-normalized title + summary: detects changed items."""
    normalized = f"{' '.join(title.split())}\n{' '.join(summary.split())}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def sanitize_error(text: str) -> str:
    """URLs reduced to scheme://host/path, secrets redacted, truncated to 200 chars."""
    text = _URL_IN_TEXT.sub(lambda m: safe_url(m.group(0)), text)
    return sanitize(text, max_len=MAX_ERROR_CHARS)


def _describe(exc: BaseException) -> str:
    return sanitize_error(f"{type(exc).__name__}: {exc}")


def _clean(html: str) -> str:
    return " ".join(BeautifulSoup(html or "", "html.parser").get_text(" ").split())


def _published(entry) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    return datetime.fromtimestamp(calendar.timegm(parsed), tz=UTC)


@dataclass(frozen=True)
class ParsedFeed:
    items: list[NewsItem]
    dropped: int  # entries without title, link or date
    total: int  # entries the parser found
    bozo: bool = False
    bozo_reason: str = ""


def parse_feed_detailed(xml_text: str | bytes, feed_url: str) -> ParsedFeed:
    """Parse RSS/Atom. Items lacking title, link or date are dropped (never guessed) but counted."""
    parsed = feedparser.parse(xml_text)
    source = _clean(parsed.feed.get("title", "")) or urlparse(feed_url).netloc
    items = []
    for entry in parsed.entries:
        title = _clean(entry.get("title", ""))
        url = (entry.get("link") or "").strip()
        published = _published(entry)
        if not (title and url and published):
            continue
        summary = _clean(entry.get("summary", ""))
        items.append(NewsItem(title, summary, url, source, published))
    bozo = bool(parsed.get("bozo"))
    reason = type(parsed.get("bozo_exception")).__name__ if bozo else ""
    return ParsedFeed(items, len(parsed.entries) - len(items), len(parsed.entries), bozo, reason)


def parse_feed(xml_text: str | bytes, feed_url: str) -> list[NewsItem]:
    """Items only; use `parse_feed_detailed` to also learn how many entries were dropped."""
    return parse_feed_detailed(xml_text, feed_url).items


def item_to_document(item: NewsItem) -> Document:
    text = f"{item.title}\n{item.summary}".strip()[:MAX_TEXT_CHARS]
    return Document(
        id_=stable_id(item.url),
        text=text,
        metadata={
            "title": item.title,
            "summary": item.summary,
            "source": item.source,
            "url": item.url,
            "published": item.published.isoformat(),
            "published_epoch": int(item.published.timestamp()),
            "tickers": ",".join(detect_tickers(f"{item.title} {item.summary}")),
            "content_hash": content_hash(item.title, item.summary),
        },
    )


# ------------------------------------------------------------------ retries


def with_retries(
    fn: Callable[[], object],
    options: IngestOptions,
    *,
    is_transient: Callable[[BaseException], bool],
    retry_after: Callable[[BaseException], float | None] = lambda exc: None,
):
    """Call `fn`; transient errors are retried with exponential backoff + jitter (capped)."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= options.max_retries or not is_transient(exc):
                raise
            delay = retry_after(exc)
            if delay is None:
                base = options.retry_base_seconds
                delay = base * (2**attempt) + options.jitter() * base
            options.sleep(min(float(delay), RETRY_CAP_SECONDS))
            attempt += 1


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def _is_transient_fetch(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, httpx.TimeoutException | httpx.TransportError)


def _fetch_retry_after(exc: BaseException) -> float | None:
    if isinstance(exc, httpx.HTTPStatusError):
        return _parse_retry_after(exc.response.headers.get("retry-after"))
    return None


# -------------------------------------------------------------------- fetch


@dataclass(frozen=True)
class FetchedFeed:
    body: bytes
    content_type: str


def fetch_feed(client: httpx.Client, url: str, options: IngestOptions | None = None) -> FetchedFeed:
    """GET one feed with a descriptive User-Agent, streaming and capping the body size."""
    options = options or IngestOptions()
    headers = {"User-Agent": options.user_agent, "Accept": ACCEPT}
    limit = options.max_feed_bytes
    with client.stream("GET", url, headers=headers) as resp:
        resp.raise_for_status()
        declared = resp.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            raise FeedTooLarge(f"response larger than {limit} bytes")
        body = bytearray()
        for chunk in resp.iter_bytes():
            body += chunk
            if len(body) > limit:
                raise FeedTooLarge(f"response larger than {limit} bytes")
        return FetchedFeed(bytes(body), resp.headers.get("content-type", ""))


def _parse_checked(fetched: FetchedFeed, url: str) -> ParsedFeed:
    """Parse and reject responses that are clearly not a feed (HTML page, garbage)."""
    content_type = fetched.content_type.split(";")[0].strip().lower()
    head = fetched.body[:4096].lower()
    if "html" in content_type and not any(marker in head for marker in _FEED_MARKERS):
        raise InvalidFeed(f"unexpected content-type {content_type!r} and no feed markers")
    parsed = parse_feed_detailed(fetched.body, url)
    if parsed.bozo and parsed.total == 0:
        raise InvalidFeed(f"response is not a valid feed ({parsed.bozo_reason or 'parse error'})")
    return parsed


# ---------------------------------------------------------------- ingestion


def _index_batches(
    docs: list[Document], store: NewsStore, options: IngestOptions
) -> tuple[int, list[str]]:
    """Upsert in small batches; a failing batch is recorded and the rest still run."""
    indexed, errors = 0, []
    size = max(1, options.batch_size)
    for start in range(0, len(docs), size):
        batch = docs[start : start + size]
        try:
            with_retries(
                lambda b=batch: store.upsert(b), options, is_transient=is_transient_bedrock_error
            )
            indexed += len(batch)
        except Exception as exc:
            errors.append(_describe(exc))
    return indexed, errors


def _known_hashes(store: NewsStore, ids: list[str]) -> dict[str, str | None]:
    try:
        return store.existing_hashes(ids)
    except Exception as exc:  # lookup is an optimisation: never block indexing on it
        logger.warning("hash lookup failed (%s); upserting every item", _describe(exc))
        return {}


def _ingest_feed(
    client: httpx.Client,
    url: str,
    store: NewsStore,
    cutoff: datetime,
    options: IngestOptions,
    report: IngestReport,
    stat: FeedStat,
) -> None:
    fetched = with_retries(
        lambda: fetch_feed(client, url, options),
        options,
        is_transient=_is_transient_fetch,
        retry_after=_fetch_retry_after,
    )
    parsed = _parse_checked(fetched, url)
    if parsed.bozo:
        logger.warning("feed_bozo url=%s reason=%s", stat.url, parsed.bozo_reason)
    fresh = [i for i in parsed.items if i.published >= cutoff]
    stat.fetched = parsed.total
    stat.invalid = parsed.dropped
    stat.old = len(parsed.items) - len(fresh)
    report.skipped_invalid += stat.invalid
    report.skipped_old += stat.old
    if parsed.total == 0:
        stat.status = "empty"
        logger.warning("feed_empty url=%s", stat.url)
        return
    docs = [item_to_document(i) for i in fresh]
    known = _known_hashes(store, [d.id_ for d in docs]) if docs else {}
    changed = [d for d in docs if known.get(d.id_) != d.metadata["content_hash"]]
    stat.unchanged = len(docs) - len(changed)
    stat.new = len(changed)
    report.skipped_unchanged += stat.unchanged
    indexed, errors = _index_batches(changed, store, options)
    stat.indexed = indexed
    report.indexed += indexed
    report.failed_feeds.extend((stat.url, error) for error in errors)
    if errors:
        stat.status, stat.error = "failed", errors[0]
        logger.warning(
            "indexing failed for %s: %d batch(es), first: %s", stat.url, len(errors), errors[0]
        )


def ingest_feeds(
    client: httpx.Client,
    urls: list[str],
    store: NewsStore,
    *,
    now: datetime | None = None,
    options: IngestOptions | None = None,
) -> IngestReport:
    """Fetch every feed; a failing feed is recorded and never stops the others."""
    now = now or datetime.now(UTC)
    options = options or IngestOptions()
    cutoff = now - timedelta(days=store.retention_days)
    report = IngestReport()
    for url in urls:
        stat = FeedStat(url=safe_url(url))
        started = time.perf_counter()
        try:
            _ingest_feed(client, url, store, cutoff, options, report, stat)
        except Exception as exc:
            stat.status, stat.error = "failed", _describe(exc)
            report.failed_feeds.append((stat.url, stat.error))
            logger.warning("feed failed: %s (%s)", stat.url, stat.error)
        stat.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        report.feeds.append(stat)
        log_event(
            "news",
            "news_feed",
            level=logging.WARNING if stat.status == "failed" else logging.INFO,
            url=stat.url,
            status=stat.status,
            fetched=stat.fetched,
            new=stat.new,
            unchanged=stat.unchanged,
            old=stat.old,
            invalid=stat.invalid,
            indexed=stat.indexed,
            latency_ms=stat.latency_ms,
            error_type=stat.error.split(":", 1)[0] if stat.error else None,
        )
    return report


def _release(redis_client, token: str) -> None:
    # Compare-then-delete so we never release a lock that expired and was re-taken.
    try:
        if redis_client.get(INGEST_LOCK_KEY) == token:
            redis_client.delete(INGEST_LOCK_KEY)
    except redis.RedisError:
        logger.warning("could not release ingest lock", exc_info=True)


def _record_run(redis_client, report: IngestReport, finished: datetime) -> None:
    """Best-effort `ingest:last_run` / `ingest:last_success` (no TTL); never raises."""
    try:
        ok = report.ok_feeds > 0
        payload = {
            "finished_at": finished.isoformat(),
            "indexed": report.indexed,
            "failed_feeds": report.failed_feed_count,
            "ok": ok,
        }
        redis_client.set(INGEST_LAST_RUN_KEY, json.dumps(payload))
        if ok:
            redis_client.set(INGEST_LAST_SUCCESS_KEY, finished.isoformat())
    except Exception:
        logger.warning("could not record ingest status", exc_info=True)


def run_ingestion(
    settings: Settings,
    *,
    redis_client,
    store: NewsStore,
    client: httpx.Client | None = None,
    now: datetime | None = None,
    options: IngestOptions | None = None,
) -> IngestReport:
    """One ingestion cycle guarded by a Redis lock (SET NX EX): one instance at a time."""
    token = uuid.uuid4().hex
    if not redis_client.set(INGEST_LOCK_KEY, token, nx=True, ex=settings.ingest_lock_ttl_seconds):
        logger.info("another instance is ingesting; skipping")
        return IngestReport(skipped_locked=True)
    try:
        options = options or IngestOptions.from_settings(settings)
        own_client = client is None
        client = client or httpx.Client(timeout=settings.news_http_timeout, follow_redirects=True)
        try:
            report = ingest_feeds(client, settings.news_rss_urls, store, now=now, options=options)
        finally:
            if own_client:
                client.close()
        try:
            report.purged = store.purge(now=now)
        except Exception as exc:  # the ingest results above must survive a purge failure
            report.purge_error = _describe(exc)
            logger.warning("purge failed: %s", report.purge_error, exc_info=True)
        _record_run(redis_client, report, now or datetime.now(UTC))
        return report
    finally:
        _release(redis_client, token)


def build_ingest_job(settings: Settings | None = None) -> Callable[[], IngestReport]:
    """Zero-argument job for the scheduler / CLI; builds its dependencies lazily per run."""
    settings = settings or get_settings()

    def job() -> IngestReport:
        report = run_ingestion(
            settings,
            redis_client=get_redis_client(settings),
            store=build_news_store(settings),
        )
        logger.info("news ingestion finished: %s", report)
        return report

    return job


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    if not settings.news_rss_urls:
        print("NEWS_RSS_URLS is empty: nothing to ingest.")
        return 1
    report = build_ingest_job(settings)()
    if report.skipped_locked:
        print("Skipped: another instance holds the ingestion lock.")
        return 0
    print(
        f"indexed={report.indexed} skipped_unchanged={report.skipped_unchanged} "
        f"skipped_old={report.skipped_old} skipped_invalid={report.skipped_invalid} "
        f"purged={report.purged} failed_feeds={len(report.failed_feeds)}"
    )
    if report.purge_error:
        print(f"purge_error={report.purge_error}")
    for url, error in report.failed_feeds:
        print(f"  FAILED {url}: {error}")
    return 2 if report.feeds and not report.ok_feeds else 0


if __name__ == "__main__":
    raise SystemExit(main())
