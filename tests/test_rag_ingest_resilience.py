"""Ingestion hardening: skip unchanged, batching/retries, fetch hygiene, no silent failures."""

import hashlib
import json
import logging
from datetime import timedelta

import fakeredis
import httpx
import pytest
import redis
import respx
from botocore.exceptions import ClientError
from llama_index.core.vector_stores import SimpleVectorStore

from app.core.bedrock_errors import is_transient_bedrock_error
from app.core.config import Settings
from app.rag import ingest_news as mod
from app.rag.ingest_news import (
    INGEST_LAST_RUN_KEY,
    INGEST_LAST_SUCCESS_KEY,
    IngestOptions,
    content_hash,
    ingest_feeds,
    item_to_document,
    parse_feed,
    parse_feed_detailed,
    run_ingestion,
    stable_id,
)
from app.rag.redis_store import RedisNewsVectorStore
from app.rag.store import NewsStore
from tests.fakes import BagOfWordsEmbedding
from tests.test_rag_ingest import NOW, rss

GOOD = "https://good.test/rss"
RECENT = NOW - timedelta(hours=1)


class CountingEmbedding(BagOfWordsEmbedding):
    """Records every text it embeds; can raise queued errors or fail on a marker text."""

    texts: list = []
    errors: list = []
    fail_substring: str = ""

    def _get_text_embedding(self, text: str) -> list[float]:
        self.texts.append(text)
        if self.errors:
            raise self.errors.pop(0)
        if self.fail_substring and self.fail_substring in text:
            raise ValueError("bad input")
        return super()._get_text_embedding(text)


def make_store(**kw) -> NewsStore:
    return NewsStore(SimpleVectorStore(), CountingEmbedding(**kw), retention_days=7)


def aws_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "x"}}, "InvokeModel")


def opts(sleeps: list | None = None, **kw) -> IngestOptions:
    sleeps = sleeps if sleeps is not None else []
    return IngestOptions(sleep=sleeps.append, jitter=lambda: 0.0, **kw)


def items_xml(n: int, summary="s", prefix="Noticia") -> str:
    return rss([(f"{prefix} {i}", f"https://x.com/{i}", summary, RECENT) for i in range(n)])


def serve(url: str, text: str):
    respx.get(url).mock(return_value=httpx.Response(200, text=text))


def run(urls, store, options=None, client=None):
    return ingest_feeds(client or httpx.Client(), urls, store, now=NOW, options=options or opts())


def warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


# ------------------------------------------------------------ 1. skip unchanged


def test_content_hash_is_sha256_of_normalized_title_and_summary():
    expected = hashlib.sha256(b"Galicia sube\nresumen largo").hexdigest()
    assert content_hash("Galicia sube", "resumen largo") == expected
    assert content_hash("  Galicia   sube ", "resumen\n largo") == expected
    assert content_hash("Galicia sube", "otro") != expected


def test_document_carries_content_hash_excluded_from_embedding():
    [item] = parse_feed(rss([("Galicia", "https://x.com/a", "resumen", NOW)]), GOOD)
    doc = item_to_document(item)
    assert doc.metadata["content_hash"] == content_hash("Galicia", "resumen")
    from app.rag.store import document_to_node

    node = document_to_node(doc)
    assert "content_hash" in node.excluded_embed_metadata_keys
    assert node.get_content(metadata_mode="embed") == doc.text


@respx.mock
def test_unchanged_items_are_not_embedded_on_second_run():
    serve(GOOD, items_xml(3))
    store = make_store()
    first = run([GOOD], store)
    embedded_first = len(store.embed_model.texts)
    second = run([GOOD], store)
    assert (first.indexed, first.skipped_unchanged) == (3, 0)
    assert (second.indexed, second.skipped_unchanged) == (0, 3)
    assert len(store.embed_model.texts) == embedded_first  # zero embed calls on rerun
    assert second.feeds[0].unchanged == 3 and second.feeds[0].new == 0


@respx.mock
def test_changed_content_is_reembedded_and_unchanged_stay_skipped():
    store = make_store()
    serve(GOOD, items_xml(2))
    run([GOOD], store)
    before = len(store.embed_model.texts)
    changed = rss(
        [
            ("Noticia 0", "https://x.com/0", "s", RECENT),
            ("Noticia 1", "https://x.com/1", "resumen nuevo", RECENT),
        ]
    )
    respx.get(GOOD).mock(return_value=httpx.Response(200, text=changed))
    report = run([GOOD], store)
    assert (report.indexed, report.skipped_unchanged) == (1, 1)
    assert len(store.embed_model.texts) == before + 1
    assert len(store.vector_store.data.embedding_dict) == 2  # still upserted, no duplicates


@respx.mock
def test_hash_lookup_failure_falls_back_to_upserting_everything(monkeypatch, caplog):
    serve(GOOD, items_xml(2))
    store = make_store()
    run([GOOD], store)

    def boom(ids):
        raise redis.ConnectionError("redis://:pw@h down")

    monkeypatch.setattr(store, "existing_hashes", boom)
    caplog.set_level(logging.INFO)
    report = run([GOOD], store)
    assert (report.indexed, report.skipped_unchanged) == (2, 0)
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert "pw@h" not in "\n".join(r.getMessage() for r in caplog.records)


def test_simple_store_existing_hashes_reads_metadata():
    store = make_store()
    [item] = parse_feed(rss([("Galicia", "https://x.com/a", "r", NOW)]), GOOD)
    doc = item_to_document(item)
    store.upsert([doc])
    got = store.existing_hashes([doc.id_, "missing"])
    assert got == {doc.id_: content_hash("Galicia", "r"), "missing": None}


def test_existing_hashes_without_strategy_is_not_implemented():
    class Odd(SimpleVectorStore):
        pass

    store = NewsStore(Odd(), CountingEmbedding(), retention_days=7)
    store.vector_store = object()  # type: ignore[assignment]
    with pytest.raises(NotImplementedError):
        store.existing_hashes(["a"])


def test_redis_store_existing_hashes_uses_pipelined_hget():
    fake = fakeredis.FakeRedis()
    fake.hset("news/vector_a", mapping={"content_hash": "h1", "title": "x"})
    fake.hset("news/vector_nohash", mapping={"title": "x"})
    store = RedisNewsVectorStore(
        index_name="idx:news", index_prefix="news", dims=4, retention_seconds=60, redis_client=fake
    )
    got = store.existing_hashes(["a", "nohash", "gone"])
    assert got == {"a": "h1", "nohash": None, "gone": None}
    assert store.existing_hashes([]) == {}


def test_redis_store_existing_hashes_handles_decoded_strings():
    fake = fakeredis.FakeRedis(decode_responses=True)
    fake.hset("news/vector_a", mapping={"content_hash": "h1"})
    store = RedisNewsVectorStore(
        index_name="idx:news", index_prefix="news", dims=4, retention_seconds=60, redis_client=fake
    )
    assert store.existing_hashes(["a"]) == {"a": "h1"}


# --------------------------------------------------------- 2. batching + retries


@respx.mock
def test_one_failing_batch_does_not_lose_the_rest_of_the_feed():
    xml = rss(
        [
            (f"Noticia {i}", f"https://x.com/{i}", "BOOM" if i == 2 else "ok", RECENT)
            for i in range(5)
        ]
    )
    serve(GOOD, xml)
    store = make_store(fail_substring="BOOM")
    report = run([GOOD], store, opts(batch_size=2))
    # batches: [0,1] ok, [2,3] fails, [4] ok -> only successfully upserted items count
    assert report.indexed == 3
    assert len(store.vector_store.data.embedding_dict) == 3
    assert len(report.failed_feeds) == 1
    url, error = report.failed_feeds[0]
    assert url == GOOD and error.startswith("ValueError")
    stat = report.feeds[0]
    assert stat.status == "failed" and stat.indexed == 3 and stat.error.startswith("ValueError")


@respx.mock
def test_transient_bedrock_errors_are_retried_with_exponential_backoff():
    serve(GOOD, items_xml(1))
    errors = [aws_error("ThrottlingException"), aws_error("ServiceUnavailableException")]
    store = make_store(errors=errors)
    sleeps: list = []
    report = run([GOOD], store, opts(sleeps, max_retries=3, retry_base_seconds=1.0))
    assert report.indexed == 1 and not report.failed_feeds
    assert sleeps == [1.0, 2.0]


@respx.mock
def test_backoff_is_capped_and_gets_jitter():
    serve(GOOD, items_xml(1))
    store = make_store(errors=[aws_error("ModelTimeoutException")] * 3)
    sleeps: list = []
    options = IngestOptions(
        sleep=sleeps.append, jitter=lambda: 0.5, max_retries=3, retry_base_seconds=10.0
    )
    report = run([GOOD], store, options)
    assert report.indexed == 1
    assert sleeps == [10.0 + 5.0, 20.0, 20.0]  # base+jitter, then capped at 20s


@respx.mock
def test_non_transient_errors_are_not_retried():
    serve(GOOD, items_xml(1))
    store = make_store(errors=[aws_error("ValidationException")])
    sleeps: list = []
    report = run([GOOD], store, opts(sleeps))
    assert report.indexed == 0 and len(report.failed_feeds) == 1
    assert sleeps == [] and len(store.embed_model.texts) == 1


@respx.mock
def test_retries_stop_after_max_retries():
    serve(GOOD, items_xml(1))
    store = make_store(errors=[aws_error("InternalServerException")] * 10)
    sleeps: list = []
    report = run([GOOD], store, opts(sleeps, max_retries=2))
    assert report.indexed == 0 and len(report.failed_feeds) == 1
    assert len(store.embed_model.texts) == 3 and len(sleeps) == 2


def test_transient_classification_for_bedrock():
    for code in (
        "ThrottlingException",
        "ServiceUnavailableException",
        "ModelTimeoutException",
        "InternalServerException",
    ):
        assert is_transient_bedrock_error(aws_error(code))
    for code in ("ValidationException", "AccessDeniedException", "ExpiredTokenException"):
        assert not is_transient_bedrock_error(aws_error(code))
    assert not is_transient_bedrock_error(ValueError("dimension mismatch"))


@respx.mock
def test_fetch_retries_5xx_then_succeeds():
    route = respx.get(GOOD).mock(
        side_effect=[httpx.Response(503), httpx.Response(200, text=items_xml(1))]
    )
    sleeps: list = []
    report = run([GOOD], make_store(), opts(sleeps))
    assert route.call_count == 2 and sleeps == [1.0]
    assert report.indexed == 1 and not report.failed_feeds


@respx.mock
def test_fetch_retries_timeouts_and_connection_errors():
    respx.get(GOOD).mock(
        side_effect=[
            httpx.ReadTimeout("slow"),
            httpx.ConnectError("down"),
            httpx.Response(200, text=items_xml(1)),
        ]
    )
    sleeps: list = []
    report = run([GOOD], make_store(), opts(sleeps))
    assert report.indexed == 1 and sleeps == [1.0, 2.0]


@respx.mock
def test_fetch_honors_retry_after_on_429_with_cap():
    respx.get(GOOD).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(429, headers={"Retry-After": "9999"}),
            httpx.Response(200, text=items_xml(1)),
        ]
    )
    sleeps: list = []
    report = run([GOOD], make_store(), opts(sleeps))
    assert report.indexed == 1 and sleeps == [7.0, 20.0]


@respx.mock
def test_fetch_does_not_retry_4xx_other_than_429():
    route = respx.get(GOOD).mock(return_value=httpx.Response(404))
    sleeps: list = []
    report = run([GOOD], make_store(), opts(sleeps))
    assert route.call_count == 1 and sleeps == []
    assert report.feeds[0].status == "failed"


@respx.mock
def test_fetch_gives_up_after_max_retries():
    route = respx.get(GOOD).mock(return_value=httpx.Response(500))
    sleeps: list = []
    report = run([GOOD], make_store(), opts(sleeps, max_retries=2))
    assert route.call_count == 3 and len(sleeps) == 2
    assert report.feeds[0].status == "failed"


# ------------------------------------------------------------- fetch hygiene


@respx.mock
def test_user_agent_and_accept_headers_are_sent():
    route = respx.get(GOOD).mock(return_value=httpx.Response(200, text=items_xml(1)))
    run([GOOD], make_store(), opts(user_agent="my-agent/9"))
    headers = route.calls.last.request.headers
    assert headers["user-agent"] == "my-agent/9"
    assert "xml" in headers["accept"]


@respx.mock
def test_default_user_agent_is_descriptive():
    route = respx.get(GOOD).mock(return_value=httpx.Response(200, text=items_xml(1)))
    run([GOOD], make_store())
    assert route.calls.last.request.headers["user-agent"] == "financial-ai-news-ingest/1.0"


@respx.mock
def test_oversized_response_is_a_feed_failure_by_content_length():
    respx.get(GOOD).mock(return_value=httpx.Response(200, content=b"x" * 5000))
    report = run([GOOD], make_store(), opts(max_feed_bytes=1000))
    assert report.feeds[0].status == "failed" and "FeedTooLarge" in report.feeds[0].error


@respx.mock
def test_oversized_streamed_response_aborts_without_content_length():
    def chunks():
        for _ in range(10):
            yield b"<!-- " + b"x" * 400 + b" -->"

    respx.get(GOOD).mock(return_value=httpx.Response(200, content=chunks()))
    report = run([GOOD], make_store(), opts(max_feed_bytes=1000))
    assert report.feeds[0].status == "failed" and "FeedTooLarge" in report.feeds[0].error


# -------------------------------------------------------- 3. no silent failures


@respx.mock
def test_html_page_is_a_failed_feed_not_a_success():
    respx.get(GOOD).mock(
        return_value=httpx.Response(
            200,
            text="<html><body>Please log in</body></html>",
            headers={"content-type": "text/html"},
        )
    )
    report = run([GOOD], make_store())
    assert report.indexed == 0
    assert report.feeds[0].status == "failed" and report.feeds[0].error.startswith("InvalidFeed")
    assert report.failed_feeds and report.failed_feeds[0][1].startswith("InvalidFeed")


@respx.mock
def test_garbage_body_with_bozo_and_no_entries_is_failed():
    respx.get(GOOD).mock(
        return_value=httpx.Response(
            200, text="<not-a-feed><<<", headers={"content-type": "application/xml"}
        )
    )
    report = run([GOOD], make_store())
    assert report.feeds[0].status == "failed" and "InvalidFeed" in report.feeds[0].error


@respx.mock
def test_html_content_type_with_real_feed_markers_is_accepted():
    respx.get(GOOD).mock(
        return_value=httpx.Response(
            200, text=items_xml(1), headers={"content-type": "text/html; charset=utf-8"}
        )
    )
    assert run([GOOD], make_store()).indexed == 1


@respx.mock
def test_bozo_feed_with_entries_is_ok_but_warns(caplog):
    serve(GOOD, items_xml(2) + "<junk")
    caplog.set_level(logging.INFO)
    report = run([GOOD], make_store())
    assert report.indexed == 2 and report.feeds[0].status == "ok"
    assert any("feed_bozo" in r.getMessage() for r in warnings(caplog))


@respx.mock
def test_empty_valid_feed_logs_warning_feed_empty_and_is_not_a_failure(caplog):
    serve(GOOD, rss([]))
    caplog.set_level(logging.INFO)
    report = run([GOOD], make_store())
    assert report.feeds[0].status == "empty" and not report.failed_feeds
    assert any("feed_empty" in r.getMessage() for r in warnings(caplog))


def test_parse_feed_detailed_counts_dropped_entries_and_parse_feed_stays_compatible():
    xml = rss(
        [
            ("Sin fecha", "https://x.com/1", "s", None),
            ("", "https://x.com/2", "s", NOW),
            ("Sin link", "", "s", NOW),
            ("Ok", "https://x.com/4", "s", NOW),
        ]
    )
    parsed = parse_feed_detailed(xml, GOOD)
    assert [i.title for i in parsed.items] == ["Ok"]
    assert parsed.dropped == 3 and parsed.total == 4 and not parsed.bozo
    assert [i.title for i in parse_feed(xml, GOOD)] == ["Ok"]


@respx.mock
def test_skipped_invalid_is_counted_in_report_and_feed_stat():
    xml = rss(
        [
            ("Sin fecha", "https://x.com/1", "s", None),
            ("Ok", "https://x.com/2", "s", RECENT),
            ("Vieja", "https://x.com/3", "s", NOW - timedelta(days=9)),
        ]
    )
    serve(GOOD, xml)
    report = run([GOOD], make_store())
    assert report.skipped_invalid == 1 and report.skipped_old == 1 and report.indexed == 1
    stat = report.feeds[0]
    assert (stat.fetched, stat.invalid, stat.old, stat.new, stat.unchanged, stat.indexed) == (
        3,
        1,
        1,
        1,
        0,
        1,
    )


@respx.mock
def test_feed_stats_are_sanitized_and_error_truncated():
    url = "https://feed.test/rss?token=SECRET"
    leaky = "https://user:pw@feed.test/rss?token=SECRET#frag"
    respx.get(url).mock(side_effect=httpx.ConnectError("boom " + leaky + " " + "z" * 500))
    report = run([url], make_store(), opts(max_retries=0))
    stat = report.feeds[0]
    assert stat.url == "https://feed.test/rss"
    assert stat.status == "failed" and stat.latency_ms >= 0
    assert "SECRET" not in stat.error and "user:pw" not in stat.error
    assert len(stat.error) <= 260
    assert "SECRET" not in repr(report.failed_feeds)


@respx.mock
def test_per_feed_audit_event_has_counts_only_and_no_text_or_secrets(caplog):
    caplog.set_level(logging.INFO)
    url = "https://feed.test/rss?apikey=TOPSECRET"
    respx.get(url).mock(
        return_value=httpx.Response(
            200, text=rss([("Titulo CONFIDENCIAL", "https://x.com/1", "Resumen PRIVADO", RECENT)])
        )
    )
    run([url], make_store())
    events = [
        json.loads(r.getMessage())
        for r in caplog.records
        if r.name == "audit.tools.news" and "news_feed" in r.getMessage()
    ]
    assert len(events) == 1
    ev = events[0]
    assert (
        ev["event"] == "news_feed" and ev["status"] == "ok" and ev["url"] == "https://feed.test/rss"
    )
    assert (ev["fetched"], ev["new"], ev["unchanged"], ev["indexed"]) == (1, 1, 0, 1)
    everything = "\n".join(r.getMessage() for r in caplog.records)
    for leak in ("TOPSECRET", "CONFIDENCIAL", "PRIVADO"):
        assert leak not in everything


@respx.mock
def test_failure_logs_do_not_leak_query_secrets(caplog):
    caplog.set_level(logging.DEBUG)
    url = "https://feed.test/rss?apikey=TOPSECRET"
    respx.get(url).mock(return_value=httpx.Response(404))
    run([url], make_store())
    assert "TOPSECRET" not in "\n".join(r.getMessage() for r in caplog.records)


# -------------------------------------------------------- run_ingestion pieces


def settings_for(urls, **kw):
    return Settings(_env_file=None, news_rss_urls=urls, **kw)


@respx.mock
def test_purge_failure_is_recorded_and_report_is_kept(caplog):
    serve(GOOD, items_xml(1))
    store = make_store()

    def boom(now=None):
        raise redis.ConnectionError("purge down redis://:pw@h")

    store.purge = boom  # type: ignore[method-assign]
    r = fakeredis.FakeRedis(decode_responses=True)
    caplog.set_level(logging.INFO)
    report = run_ingestion(
        settings_for([GOOD]), redis_client=r, store=store, now=NOW, options=opts()
    )
    assert report.indexed == 1 and report.purged == 0
    assert report.purge_error and report.purge_error.startswith("ConnectionError")
    assert "pw@h" not in report.purge_error
    assert any(r_.levelno == logging.WARNING and r_.exc_info for r_ in caplog.records)


@respx.mock
def test_last_run_and_last_success_are_written_without_ttl():
    serve(GOOD, items_xml(2))
    respx.get("https://bad.test/rss").mock(return_value=httpx.Response(404))
    r = fakeredis.FakeRedis(decode_responses=True)
    report = run_ingestion(
        settings_for([GOOD, "https://bad.test/rss"]),
        redis_client=r,
        store=make_store(),
        now=NOW,
        options=opts(),
    )
    last = json.loads(r.get(INGEST_LAST_RUN_KEY))
    assert last == {
        "finished_at": NOW.isoformat(),
        "indexed": 2,
        "failed_feeds": 1,
        "ok": True,
    }
    assert r.ttl(INGEST_LAST_RUN_KEY) == -1
    assert r.get(INGEST_LAST_SUCCESS_KEY) == NOW.isoformat()
    assert r.ttl(INGEST_LAST_SUCCESS_KEY) == -1
    assert report.indexed == 2


@respx.mock
def test_all_feeds_failed_writes_last_run_not_ok_and_keeps_last_success():
    respx.get(GOOD).mock(return_value=httpx.Response(404))
    r = fakeredis.FakeRedis(decode_responses=True)
    r.set(INGEST_LAST_SUCCESS_KEY, "2026-01-01T00:00:00+00:00")
    run_ingestion(settings_for([GOOD]), redis_client=r, store=make_store(), now=NOW, options=opts())
    assert json.loads(r.get(INGEST_LAST_RUN_KEY))["ok"] is False
    assert r.get(INGEST_LAST_SUCCESS_KEY) == "2026-01-01T00:00:00+00:00"


@respx.mock
def test_recording_last_run_never_raises():
    serve(GOOD, items_xml(1))

    class Flaky(fakeredis.FakeRedis):
        def set(self, name, value, *a, **k):
            if name.startswith("ingest:"):
                raise redis.ConnectionError("down")
            return super().set(name, value, *a, **k)

    report = run_ingestion(
        settings_for([GOOD]),
        redis_client=Flaky(decode_responses=True),
        store=make_store(),
        now=NOW,
        options=opts(),
    )
    assert report.indexed == 1


def test_skipped_locked_run_does_not_write_last_run():
    r = fakeredis.FakeRedis(decode_responses=True)
    r.set(mod.INGEST_LOCK_KEY, "other")
    run_ingestion(settings_for([]), redis_client=r, store=make_store(), now=NOW)
    assert r.get(INGEST_LAST_RUN_KEY) is None


def test_settings_defaults_for_new_ingest_options():
    s = Settings(_env_file=None)
    assert s.ingest_batch_size == 10
    assert s.ingest_max_retries == 3
    assert s.ingest_retry_base_seconds == 1.0
    assert s.news_user_agent == "financial-ai-news-ingest/1.0"
    assert s.news_max_feed_bytes == 5_000_000
    o = IngestOptions.from_settings(Settings(_env_file=None, ingest_batch_size=4))
    assert o.batch_size == 4 and o.max_retries == 3


def test_stable_id_still_matches_redis_key_layout():
    # Redis keys are "<prefix>/vector_<id>"; existing_hashes relies on this layout.
    fake = fakeredis.FakeRedis()
    store = RedisNewsVectorStore(
        index_name="idx:news", index_prefix="news", dims=4, retention_seconds=60, redis_client=fake
    )
    fake.hset(f"news/vector_{stable_id('https://x.com/a')}", mapping={"content_hash": "h"})
    assert store.existing_hashes([stable_id("https://x.com/a")]) == {
        stable_id("https://x.com/a"): "h"
    }


# ------------------------------------------------------------------------ CLI


def _cli(monkeypatch, capsys, report):
    monkeypatch.setattr(mod, "get_settings", lambda: settings_for([GOOD]))
    monkeypatch.setattr(mod, "build_ingest_job", lambda settings=None: lambda: report)
    code = mod.main()
    return code, capsys.readouterr().out


def _stat(status):
    return mod.FeedStat(url=GOOD, status=status)


def test_cli_exit_0_when_any_feed_ok_and_prints_new_counters(monkeypatch, capsys):
    report = mod.IngestReport(
        indexed=1,
        skipped_unchanged=4,
        skipped_invalid=2,
        failed_feeds=[("https://b.test/rss", "HTTPStatusError: 404")],
        feeds=[_stat("ok"), _stat("failed")],
    )
    code, out = _cli(monkeypatch, capsys, report)
    assert code == 0
    assert "skipped_unchanged=4" in out and "skipped_invalid=2" in out
    assert "FAILED https://b.test/rss" in out


def test_cli_exit_2_when_all_feeds_failed(monkeypatch, capsys):
    report = mod.IngestReport(
        failed_feeds=[(GOOD, "InvalidFeed: html")], feeds=[_stat("failed"), _stat("failed")]
    )
    code, out = _cli(monkeypatch, capsys, report)
    assert code == 2


def test_cli_prints_purge_error(monkeypatch, capsys):
    report = mod.IngestReport(purge_error="ConnectionError: x", feeds=[_stat("ok")])
    code, out = _cli(monkeypatch, capsys, report)
    assert code == 0 and "purge_error=ConnectionError: x" in out
