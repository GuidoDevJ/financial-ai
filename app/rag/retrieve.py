"""News retrieval: embed the query, top-k similarity with ticker and time-window filters."""

import hashlib
import json
import logging
import time
from datetime import UTC, datetime, timedelta

from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters

from app.core.audit import annotate, log_event
from app.rag.store import NewsStore
from app.rag.tickers import normalize_ticker

_CACHE_PREFIX = "news:retrieve:"


def _cache_key(query: str, symbol: str | None, hours: int, top_k: int) -> str:
    raw = json.dumps([query.strip().lower(), symbol, hours, top_k], ensure_ascii=False)
    return _CACHE_PREFIX + hashlib.sha256(raw.encode()).hexdigest()[:32]


def retrieve_news(
    store: NewsStore,
    query: str,
    *,
    ticker: str | None = None,
    hours: int = 72,
    top_k: int = 4,
    now: datetime | None = None,
    redis_client=None,
    cache_ttl: int = 0,
) -> list[dict]:
    """Return [{"titulo","resumen","fuente","url","fecha"}] ordered by similarity (may be empty).

    With `redis_client` and `cache_ttl` > 0 non-empty results are cached for `cache_ttl` seconds
    by (query, ticker, hours, top_k), skipping the query embedding and the vector search. A Redis
    failure degrades to a direct search.
    """
    now = now or datetime.now(UTC)
    since = int((now - timedelta(hours=hours)).timestamp())
    symbol = normalize_ticker(ticker) if ticker else None

    key = _cache_key(query, symbol, hours, top_k)
    if redis_client is not None and cache_ttl > 0:
        try:
            raw = redis_client.get(key)
            if raw is not None:
                cached = json.loads(raw)
                annotate(source="news_store", hits=len(cached), cache="hit")
                return cached
        except Exception as exc:
            log_event(
                "news",
                "news_cache_error",
                level=logging.WARNING,
                op="read",
                error_type=type(exc).__name__,
            )

    filters = [MetadataFilter(key="published_epoch", value=since, operator=FilterOperator.GTE)]
    if symbol:
        filters.append(
            MetadataFilter(key="tickers", value=symbol, operator=FilterOperator.CONTAINS)
        )
    retriever = store.index.as_retriever(
        similarity_top_k=top_k, filters=MetadataFilters(filters=filters)
    )

    started = time.perf_counter()
    candidates = retriever.retrieve(query)
    results = []
    scores = []
    for hit in candidates:
        md = hit.node.metadata
        # Defence in depth: never trust the store alone for the filters (e.g. substring match
        # semantics in the in-memory store).
        if md.get("published_epoch", 0) < since:
            continue
        if symbol and symbol not in str(md.get("tickers", "")).split(","):
            continue
        if getattr(hit, "score", None) is not None:
            scores.append(float(hit.score))
        results.append(
            {
                "titulo": md.get("title", ""),
                "resumen": md.get("summary", ""),
                "fuente": md.get("source", ""),
                "url": md.get("url", ""),
                "fecha": md.get("published", ""),
            }
        )
    score_min = round(min(scores), 4) if scores else None
    score_max = round(max(scores), 4) if scores else None
    annotate(source="news_store", hits=len(results))
    if redis_client is not None and cache_ttl > 0 and results:
        try:
            redis_client.setex(key, cache_ttl, json.dumps(results, ensure_ascii=False))
        except Exception as exc:
            log_event(
                "news",
                "news_cache_error",
                level=logging.WARNING,
                op="write",
                error_type=type(exc).__name__,
            )
    # Query text is truncated by the logger; titles/summaries are never logged.
    log_event(
        "news",
        "news_search",
        query=query,
        ticker=symbol,
        hours=hours,
        top_k=top_k,
        candidates=len(candidates),
        hits=len(results),
        score_min=score_min,
        score_max=score_max,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return results
