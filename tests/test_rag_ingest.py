import hashlib
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import fakeredis
import httpx
import respx
from llama_index.core.vector_stores import SimpleVectorStore

from app.core.config import Settings
from app.rag.ingest_news import (
    INGEST_LOCK_KEY,
    IngestOptions,
    detect_tickers,
    ingest_feeds,
    item_to_document,
    parse_feed,
    run_ingestion,
    stable_id,
)
from app.rag.store import NewsStore
from tests.fakes import BagOfWordsEmbedding

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
FAST = IngestOptions(sleep=lambda s: None, jitter=lambda: 0.0)  # retries never really wait


def rss(items: list[tuple[str, str, str, datetime | None]], title="Ambito Mercados") -> str:
    body = ""
    for item_title, link, summary, when in items:
        pub = f"<pubDate>{format_datetime(when)}</pubDate>" if when else ""
        body += (
            f"<item><title>{item_title}</title><link>{link}</link>"
            f"<description><![CDATA[{summary}]]></description>{pub}</item>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>'
        f"<title>{title}</title>{body}</channel></rss>"
    )


def make_store(retention_days=7) -> NewsStore:
    return NewsStore(SimpleVectorStore(), BagOfWordsEmbedding(), retention_days=retention_days)


def size(store: NewsStore) -> int:
    return len(store.vector_store.data.embedding_dict)


# ------------------------------------------------------------------ parsing


def test_parse_feed_extracts_fields_and_strips_html():
    xml = rss(
        [
            (
                "Galicia sube 3%",
                "https://x.com/a",
                "<p>El banco <b>subio</b> fuerte.</p>",
                NOW - timedelta(hours=2),
            )
        ]
    )
    [item] = parse_feed(xml, "https://x.com/rss")
    assert item.title == "Galicia sube 3%"
    assert item.summary == "El banco subio fuerte."
    assert item.url == "https://x.com/a"
    assert item.source == "Ambito Mercados"
    assert item.published == NOW - timedelta(hours=2)


def test_parse_feed_skips_items_without_date_or_link():
    xml = rss(
        [
            ("Sin fecha", "https://x.com/1", "s", None),
            ("Sin link", "", "s", NOW),
            ("Ok", "https://x.com/3", "s", NOW),
        ]
    )
    assert [i.title for i in parse_feed(xml, "https://x.com/rss")] == ["Ok"]


def test_source_falls_back_to_hostname():
    xml = rss([("t", "https://x.com/1", "s", NOW)], title="")
    [item] = parse_feed(xml, "https://www.cronista.com/rss")
    assert item.source == "www.cronista.com"


def test_detect_tickers_by_symbol_and_company_name():
    text = "Galicia y YPFD lideran; Banco Macro y MercadoLibre cotizan; Pampa Energia estable"
    assert detect_tickers(text) == ["BMA", "GGAL", "MELI", "PAMP", "YPF"]
    assert detect_tickers("El dolar sube") == []
    assert detect_tickers("GGALX no es un ticker") == []


# ----------------------------------------------------------------- document


def test_document_has_stable_id_and_metadata():
    xml = rss([("Galicia sube", "https://x.com/a", "resumen", NOW)])
    [item] = parse_feed(xml, "https://x.com/rss")
    doc = item_to_document(item)
    assert doc.id_ == hashlib.sha256(b"https://x.com/a").hexdigest()[:32] == stable_id(item.url)
    assert doc.text == "Galicia sube\nresumen"
    md = doc.metadata
    assert md["title"] == "Galicia sube" and md["url"] == "https://x.com/a"
    assert md["source"] == "Ambito Mercados" and md["summary"] == "resumen"
    assert md["published"] == NOW.isoformat()
    assert md["published_epoch"] == int(NOW.timestamp())
    assert md["tickers"] == "GGAL"


# ---------------------------------------------------------------- ingestion


@respx.mock
def test_ingest_isolates_failing_feeds():
    good = rss([("Galicia sube", "https://x.com/a", "s", NOW - timedelta(hours=1))])
    respx.get("https://bad.test/rss").mock(return_value=httpx.Response(500))
    respx.get("https://boom.test/rss").mock(side_effect=httpx.ConnectError("down"))
    respx.get("https://good.test/rss").mock(return_value=httpx.Response(200, text=good))
    store = make_store()
    report = ingest_feeds(
        httpx.Client(),
        ["https://bad.test/rss", "https://boom.test/rss", "https://good.test/rss"],
        store,
        now=NOW,
        options=FAST,
    )
    assert report.indexed == 1
    assert {url for url, _ in report.failed_feeds} == {
        "https://bad.test/rss",
        "https://boom.test/rss",
    }
    assert size(store) == 1


@respx.mock
def test_ingest_is_idempotent_and_skips_items_older_than_retention():
    xml = rss(
        [
            ("Fresca", "https://x.com/a", "s", NOW - timedelta(hours=1)),
            ("Vieja", "https://x.com/b", "s", NOW - timedelta(days=8)),
        ]
    )
    respx.get("https://good.test/rss").mock(return_value=httpx.Response(200, text=xml))
    store = make_store()
    client = httpx.Client()
    first = ingest_feeds(client, ["https://good.test/rss"], store, now=NOW)
    second = ingest_feeds(client, ["https://good.test/rss"], store, now=NOW)
    assert first.indexed == 1 and first.skipped_old == 1
    assert second.indexed == 0 and second.skipped_unchanged == 1  # unchanged: not re-embedded
    assert size(store) == 1  # upsert by stable id, no duplicates


# ---------------------------------------------------------------- retention


def doc_at(url: str, title: str, when: datetime):
    xml = rss([(title, url, "resumen", when)])
    return item_to_document(parse_feed(xml, "https://x.com/rss")[0])


def test_purge_removes_only_documents_older_than_retention():
    store = make_store(retention_days=7)
    store.upsert(
        [
            doc_at("https://x.com/old", "vieja", NOW - timedelta(days=8)),
            doc_at("https://x.com/new", "nueva", NOW - timedelta(days=1)),
        ]
    )
    assert size(store) == 2
    assert store.purge(now=NOW) == 1
    assert size(store) == 1
    assert store.purge(now=NOW) == 0


def test_retention_days_is_configurable():
    store = make_store(retention_days=1)
    store.upsert([doc_at("https://x.com/a", "a", NOW - timedelta(days=2))])
    assert store.purge(now=NOW) == 1


# --------------------------------------------------------------------- lock


def settings_for(urls):
    return Settings(_env_file=None, news_rss_urls=urls)


@respx.mock
def test_run_ingestion_takes_and_releases_lock_and_purges():
    xml = rss([("Galicia", "https://x.com/a", "s", NOW - timedelta(hours=1))])
    respx.get("https://good.test/rss").mock(return_value=httpx.Response(200, text=xml))
    r = fakeredis.FakeRedis(decode_responses=True)
    store = make_store()
    store.upsert([doc_at("https://x.com/old", "vieja", NOW - timedelta(days=9))])
    report = run_ingestion(
        settings_for(["https://good.test/rss"]), redis_client=r, store=store, now=NOW, options=FAST
    )
    assert not report.skipped_locked and report.indexed == 1 and report.purged == 1
    assert r.get(INGEST_LOCK_KEY) is None


def test_run_ingestion_skips_when_another_instance_holds_the_lock():
    r = fakeredis.FakeRedis(decode_responses=True)
    r.set(INGEST_LOCK_KEY, "other", nx=True, ex=60)
    store = make_store()
    report = run_ingestion(settings_for([]), redis_client=r, store=store, now=NOW)
    assert report.skipped_locked and report.indexed == 0
    assert r.get(INGEST_LOCK_KEY) == "other"  # not stolen nor released


def test_lock_released_even_if_ingestion_raises(monkeypatch):
    import app.rag.ingest_news as mod

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(mod, "ingest_feeds", boom)
    r = fakeredis.FakeRedis(decode_responses=True)
    try:
        run_ingestion(settings_for([]), redis_client=r, store=make_store(), now=NOW)
    except RuntimeError:
        pass
    assert r.get(INGEST_LOCK_KEY) is None
