"""search_news tool: semantic search over the indexed financial news."""

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache

from app.core.audit import audited_tool, log_event
from app.core.config import Settings, get_settings
from app.rag.retrieve import retrieve_news
from app.rag.store import NewsStore, build_news_store
from app.tools.errors import ToolError

DEFAULT_HOURS = 72

_stores: dict[tuple, NewsStore] = {}
_stores_lock = threading.Lock()


def shared_news_store(settings: Settings) -> NewsStore:
    """Process-wide news store per settings: the chat tool, the brief tool and the startup
    warm-up share one Redis connection and one embedding client instead of building three."""
    key = (
        settings.redis_url,
        settings.news_index_name,
        settings.news_index_prefix,
        settings.news_retention_days,
        settings.embedding_provider,
        settings.embedding_model_id,
        settings.embedding_dimensions,
        settings.aws_region,
    )
    with _stores_lock:
        if key not in _stores:
            _stores[key] = build_news_store(settings)
        return _stores[key]


def warm_news_store(settings: Settings) -> None:
    """Build the shared news store now (startup warm-up) instead of on the first search."""
    shared_news_store(settings)


def make_search_news(
    *,
    store: NewsStore | None = None,
    store_factory: Callable[[], NewsStore] | None = None,
    settings: Settings | None = None,
    redis_client=None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
):
    """Build the audited `search_news(query, ticker=None, horas=72)` function.

    Pass `store` directly (tests) or a `store_factory`, which is called lazily on first use so a
    missing Redis Stack / embedding credentials only affects this tool, not app startup.
    With `redis_client`, results are cached for NEWS_CACHE_TTL_SECONDS (0 disables).
    """
    settings = settings or get_settings()
    if store is None:
        factory = store_factory or (lambda: shared_news_store(settings))
        get_store = lru_cache(maxsize=1)(factory)
    else:

        def get_store() -> NewsStore:
            return store

    @audited_tool
    def search_news(
        query: str, ticker: str | None = None, horas: int = DEFAULT_HOURS
    ) -> list[dict]:
        query_n = (query or "").strip()
        if not query_n:
            raise ToolError("INVALID_PARAMS", "query must not be empty.")
        if not isinstance(horas, int) or isinstance(horas, bool) or horas <= 0:
            raise ToolError("INVALID_PARAMS", "horas must be a positive integer.")
        try:
            results = retrieve_news(
                get_store(),
                query_n,
                ticker=(ticker or "").strip() or None,
                hours=horas,
                top_k=settings.news_top_k,
                now=now(),
                redis_client=redis_client,
                cache_ttl=settings.news_cache_ttl_seconds,
            )
        except ToolError:
            raise
        except Exception as exc:
            log_event(
                "news",
                "news_store_error",
                level=logging.ERROR,
                exc_info=True,
                error_type=type(exc).__name__,
            )
            raise ToolError(
                "NEWS_STORE_UNAVAILABLE", f"News search failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not results:
            raise ToolError(
                "NO_NEWS_FOUND",
                f"No news found for the query in the last {horas} hours"
                + (f" for ticker {ticker}." if ticker else "."),
            )
        return results

    return search_news
