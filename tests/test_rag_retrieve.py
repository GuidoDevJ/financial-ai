import json
import logging
from datetime import UTC, datetime, timedelta

import pytest
from llama_index.core.vector_stores import SimpleVectorStore

from app.core.config import Settings
from app.rag.ingest_news import NewsItem, item_to_document
from app.rag.retrieve import retrieve_news
from app.rag.store import NewsStore
from app.tools.errors import ToolError
from app.tools.search_news import make_search_news
from tests.fakes import BagOfWordsEmbedding

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


def doc(n: int, title: str, hours_ago: float, summary="resumen", source="Ambito"):
    item = NewsItem(
        title=title,
        summary=summary,
        url=f"https://x.com/{n}",
        source=source,
        published=NOW - timedelta(hours=hours_ago),
    )
    return item_to_document(item)


@pytest.fixture
def store() -> NewsStore:
    s = NewsStore(SimpleVectorStore(), BagOfWordsEmbedding(), retention_days=7)
    s.upsert(
        [
            doc(1, "Galicia sube tras balance", 2, "GGAL lidera el panel"),
            doc(2, "YPF anuncia inversion en Vaca Muerta", 5, "YPFD petroleo"),
            doc(3, "Galicia vieja noticia de balance", 100, "GGAL antiguo"),
            doc(4, "Dolar blue estable", 3, "mercado cambiario"),
            doc(5, "Pampa Energia gas balance", 4, "PAMP gas"),
        ]
    )
    return s


def test_returns_spec_fields_ordered_by_similarity(store):
    [first, *_] = retrieve_news(store, "Galicia balance", hours=72, top_k=4, now=NOW)
    assert set(first) == {"titulo", "resumen", "fuente", "url", "fecha"}
    assert first["titulo"] == "Galicia sube tras balance"
    assert first["resumen"] == "GGAL lidera el panel"
    assert first["fuente"] == "Ambito" and first["url"] == "https://x.com/1"
    assert first["fecha"] == (NOW - timedelta(hours=2)).isoformat()


def test_top_k_limits_results(store):
    assert len(retrieve_news(store, "balance", hours=72, top_k=2, now=NOW)) == 2


def test_time_window_excludes_older_news(store):
    titles = [
        r["titulo"] for r in retrieve_news(store, "Galicia balance", hours=72, top_k=5, now=NOW)
    ]
    assert "Galicia vieja noticia de balance" not in titles
    wide = [
        r["titulo"] for r in retrieve_news(store, "Galicia balance", hours=200, top_k=5, now=NOW)
    ]
    assert "Galicia vieja noticia de balance" in wide


def test_ticker_filter(store):
    results = retrieve_news(store, "balance", ticker="ggal", hours=72, top_k=5, now=NOW)
    assert [r["url"] for r in results] == ["https://x.com/1"]
    # broker-style suffix and alias resolve to the canonical ticker
    ypf = retrieve_news(store, "inversion", ticker="YPFD.BA", hours=72, top_k=5, now=NOW)
    assert [r["url"] for r in ypf] == ["https://x.com/2"]


def test_empty_results(store):
    assert retrieve_news(store, "x", ticker="MELI", hours=72, top_k=4, now=NOW) == []
    assert retrieve_news(store, "x", hours=0, top_k=4, now=NOW) == []


# ---------------------------------------------------------------- the tool


def tool(store):
    return make_search_news(store=store, settings=Settings(_env_file=None), now=lambda: NOW)


def test_tool_returns_list_of_news(store):
    out = tool(store)(query="Galicia balance", ticker="GGAL")
    assert isinstance(out, list) and out[0]["url"] == "https://x.com/1"
    assert len(out) <= 4  # settings.news_top_k default


def test_tool_defaults_to_72_hours(store):
    out = tool(store)(query="Galicia balance")
    assert "https://x.com/3" not in {r["url"] for r in out}


def test_tool_no_news_is_explicit_error_not_invented(store):
    with pytest.raises(ToolError) as exc:
        tool(store)(query="x", ticker="MELI")
    assert exc.value.code == "NO_NEWS_FOUND"


@pytest.mark.parametrize(
    "kwargs", [{"query": "  "}, {"query": "a", "horas": 0}, {"query": "a", "horas": -5}]
)
def test_tool_validates_params(store, kwargs):
    with pytest.raises(ToolError) as exc:
        tool(store)(**kwargs)
    assert exc.value.code == "INVALID_PARAMS"


def test_tool_store_failure_is_explicit_error():
    def broken():
        raise ConnectionError("redis down")

    fn = make_search_news(store_factory=broken, settings=Settings(_env_file=None))
    with pytest.raises(ToolError) as exc:
        fn(query="Galicia")
    assert exc.value.code == "NEWS_STORE_UNAVAILABLE"


def test_tool_is_audited(store, caplog):
    with caplog.at_level(logging.INFO, logger="audit.tools"):
        tool(store)(query="Galicia balance")
    record = json.loads(caplog.records[-1].message)
    assert record["tool"] == "search_news" and record["params"]["query"] == "Galicia balance"


# ------------------------------------------------------------- Redis result cache


class _CountingIndex:
    def __init__(self, index):
        self._index = index
        self.calls = 0

    def as_retriever(self, **kwargs):
        self.calls += 1
        return self._index.as_retriever(**kwargs)


def _counting(store):
    from types import SimpleNamespace

    index = _CountingIndex(store.index)
    return SimpleNamespace(index=index), index


def test_cached_retrieval_skips_the_embedding_and_vector_search(store):
    import fakeredis

    r = fakeredis.FakeRedis(decode_responses=True)
    spy, index = _counting(store)
    kw = dict(hours=72, top_k=4, now=NOW, redis_client=r, cache_ttl=300)
    first = retrieve_news(spy, "Galicia balance", **kw)
    second = retrieve_news(spy, "Galicia balance", **kw)
    assert first and second == first
    assert index.calls == 1
    (key,) = r.keys("news:retrieve:*")
    assert 0 < r.ttl(key) <= 300


def test_cache_key_depends_on_query_ticker_and_hours(store):
    import fakeredis

    r = fakeredis.FakeRedis(decode_responses=True)
    spy, index = _counting(store)
    kw = dict(top_k=4, now=NOW, redis_client=r, cache_ttl=300)
    retrieve_news(spy, "Galicia balance", hours=72, **kw)
    retrieve_news(spy, "Galicia balance", hours=24, **kw)
    retrieve_news(spy, "Galicia balance", hours=72, ticker="GGAL", **kw)
    retrieve_news(spy, "Dolar", hours=72, **kw)
    assert index.calls == 4


def test_cache_is_off_without_ttl_or_redis(store):
    import fakeredis

    r = fakeredis.FakeRedis(decode_responses=True)
    spy, index = _counting(store)
    retrieve_news(spy, "balance", hours=72, now=NOW, redis_client=r, cache_ttl=0)
    retrieve_news(spy, "balance", hours=72, now=NOW, redis_client=r, cache_ttl=0)
    retrieve_news(spy, "balance", hours=72, now=NOW)
    retrieve_news(spy, "balance", hours=72, now=NOW)
    assert index.calls == 4
    assert r.keys("*") == []


def test_empty_results_are_not_cached(store):
    import fakeredis

    r = fakeredis.FakeRedis(decode_responses=True)
    spy, index = _counting(store)
    kw = dict(hours=72, now=NOW, redis_client=r, cache_ttl=300)
    assert retrieve_news(spy, "zzzz", ticker="NOPE", **kw) == []
    retrieve_news(spy, "zzzz", ticker="NOPE", **kw)
    assert index.calls == 2


def test_redis_down_degrades_to_a_direct_search(store):
    import redis

    class Down:
        def get(self, *a, **k):
            raise redis.ConnectionError("down")

        setex = get

    spy, index = _counting(store)
    results = retrieve_news(
        spy, "Galicia balance", hours=72, now=NOW, redis_client=Down(), cache_ttl=300
    )
    assert results and index.calls == 1


def test_search_news_tool_uses_the_cache(store):
    import fakeredis

    r = fakeredis.FakeRedis(decode_responses=True)
    spy, index = _counting(store)
    tool = make_search_news(
        store=spy, settings=Settings(_env_file=None), redis_client=r, now=lambda: NOW
    )
    tool(query="Galicia balance")
    tool(query="Galicia balance")
    assert index.calls == 1
    off = make_search_news(
        store=spy,
        settings=Settings(_env_file=None, news_cache_ttl_seconds=0),
        redis_client=r,
        now=lambda: NOW,
    )
    off(query="Galicia balance")
    off(query="Galicia balance")
    assert index.calls == 3
