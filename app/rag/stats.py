"""Summary of the news index state (aggregates only; never titles, summaries or URLs).

`news_index_summary` reads the whole index, not the last ingestion run. The Redis store answers
with server-side RediSearch aggregations (`index_aggregates`); the in-memory SimpleVectorStore used
by tests is scanned directly. CLI: `uv run python -m app.rag.stats [--json] [--top N]`.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import redis
from llama_index.core.vector_stores import SimpleVectorStore

from app.core.config import Settings, get_settings
from app.core.redis_client import get_redis_client
from app.rag.embeddings import embedding_dimensions, resolve_model_id
from app.rag.ingest_news import INGEST_LAST_RUN_KEY, INGEST_LAST_SUCCESS_KEY, sanitize_error
from app.rag.redis_store import RedisNewsVectorStore

logger = logging.getLogger(__name__)

DEFAULT_TOP = 20
MAX_TOP = 100
DAY = 86400


def _iso(epoch: int | None) -> str | None:
    return None if epoch is None else datetime.fromtimestamp(epoch, UTC).isoformat()


def _simple_aggregates(
    store: SimpleVectorStore, expiring_from: int, expiring_to: int
) -> dict[str, Any]:
    metadata = list(store.data.metadata_dict.values())
    epochs = [int(m["published_epoch"]) for m in metadata if m.get("published_epoch") is not None]
    return {
        "exists": True,
        "num_docs": len(metadata),
        "total": len(metadata),
        "min_epoch": min(epochs) if epochs else None,
        "max_epoch": max(epochs) if epochs else None,
        "expiring": sum(1 for e in epochs if expiring_from <= e < expiring_to),
        "sources": dict(Counter(m["source"] for m in metadata if m.get("source"))),
        "ticker_groups": dict(Counter(m["tickers"] for m in metadata if m.get("tickers"))),
        "days": dict(Counter(e // DAY for e in epochs)),
    }


def _aggregates(vector_store: Any, expiring_from: int, expiring_to: int) -> dict[str, Any]:
    if hasattr(vector_store, "index_aggregates"):
        return vector_store.index_aggregates(expiring_from=expiring_from, expiring_to=expiring_to)
    if isinstance(vector_store, SimpleVectorStore):
        return _simple_aggregates(vector_store, expiring_from, expiring_to)
    raise NotImplementedError(f"No index summary for {type(vector_store).__name__}.")


def _top(counts: dict[str, int], top: int) -> dict[str, int]:
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return dict(ranked[:top])


def _tickers(groups: dict[str, int], total: int, top: int) -> tuple[dict[str, int], int]:
    per_ticker: Counter[str] = Counter()
    tagged = 0
    for combo, n in groups.items():
        names = {t.strip() for t in combo.split(",") if t.strip()}
        if not names:
            continue
        tagged += n
        for name in names:
            per_ticker[name] += n
    return _top(dict(per_ticker), top), max(0, total - tagged)


def _by_day(days: dict[int, int], now: datetime, retention_days: int) -> dict[str, int]:
    today = int(now.timestamp()) // DAY
    first = today - retention_days + 1
    return {
        datetime.fromtimestamp(d * DAY, UTC).date().isoformat(): days.get(d, 0)
        for d in range(first, today + 1)
    }


def _last_ingestion(client: Any) -> dict[str, Any]:
    """Best-effort read of the ingestion status keys; anything unreadable is None."""
    result: dict[str, Any] = {"last_run": None, "last_success": None}
    if client is None:
        return result
    try:
        raw = client.get(INGEST_LAST_RUN_KEY)
        if raw:
            parsed = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            if isinstance(parsed, dict):
                result["last_run"] = parsed
    except Exception:
        logger.debug("could not read %s", INGEST_LAST_RUN_KEY, exc_info=True)
    try:
        raw = client.get(INGEST_LAST_SUCCESS_KEY)
        if raw:
            result["last_success"] = raw.decode() if isinstance(raw, bytes) else str(raw)
    except Exception:
        logger.debug("could not read %s", INGEST_LAST_SUCCESS_KEY, exc_info=True)
    return result


def _embedding(settings: Settings) -> tuple[dict[str, Any], int | None]:
    try:
        model_id: str | None = resolve_model_id(settings)
    except ValueError:
        model_id = settings.embedding_model_id
    try:
        dims: int | None = embedding_dimensions(settings)
    except ValueError:
        dims = None
    return {"provider": settings.embedding_provider, "model_id": model_id}, dims


def news_index_summary(
    store: Any,
    settings: Settings,
    now: datetime | None = None,
    *,
    redis_client: Any = None,
    top: int = DEFAULT_TOP,
) -> dict[str, Any]:
    """Aggregate view of everything currently indexed.

    `store` is a NewsStore or a bare vector store. A missing Redis index yields
    `index.exists = false` with zeros; other Redis errors propagate (redis.RedisError).
    """
    now = now or datetime.now(UTC)
    vector_store = getattr(store, "vector_store", store)
    retention_days = settings.news_retention_days
    cutoff = int((now - timedelta(days=retention_days)).timestamp())
    agg = _aggregates(vector_store, cutoff, cutoff + DAY)

    exists = bool(agg.get("exists"))
    total = int(agg.get("total", 0)) if exists else 0
    newest = agg.get("max_epoch") if exists else None
    by_ticker, untagged = _tickers(agg.get("ticker_groups", {}), total, top)
    embedding, dims = _embedding(settings)
    return {
        "generated_at": now.isoformat(),
        "total_documents": total,
        "oldest_published": _iso(agg.get("min_epoch")) if exists else None,
        "newest_published": _iso(newest),
        "newest_age_hours": (
            None if newest is None else round(max(0.0, now.timestamp() - newest) / 3600, 1)
        ),
        "retention_days": retention_days,
        "expiring_24h": int(agg.get("expiring", 0)) if exists else 0,
        "by_source": _top(agg.get("sources", {}), top) if exists else {},
        "by_day": _by_day(agg.get("days", {}), now, retention_days),
        "by_ticker": by_ticker if exists else {},
        "untagged": untagged if exists else 0,
        "index": {
            "name": settings.news_index_name,
            "prefix": settings.news_index_prefix,
            "dims": dims,
            "exists": exists,
            "num_docs": agg.get("num_docs") if exists else None,
        },
        "embedding": embedding,
        "last_ingestion": _last_ingestion(redis_client),
    }


def build_summary_job(settings: Settings | None = None) -> Callable[[int], dict[str, Any]]:
    """`top -> summary`; builds the Redis store lazily (no embedding model is needed)."""
    settings = settings or get_settings()

    def job(top: int = DEFAULT_TOP) -> dict[str, Any]:
        client = get_redis_client(settings)
        store = RedisNewsVectorStore(
            index_name=settings.news_index_name,
            index_prefix=settings.news_index_prefix,
            dims=embedding_dimensions(settings),
            retention_seconds=settings.news_retention_days * DAY,
            redis_client=client,
        )
        return news_index_summary(store, settings, redis_client=client, top=top)

    return job


# ------------------------------------------------------------------------ CLI


def _table(title: str, counts: dict, empty: str = "(none)") -> list[str]:
    lines = [title]
    if not counts:
        return [*lines, f"  {empty}"]
    width = max(len(str(k)) for k in counts)
    lines += [f"  {str(k):<{width}}  {v:>6}" for k, v in counts.items()]
    return lines


def format_text(summary: dict[str, Any]) -> str:
    idx, emb, last = summary["index"], summary["embedding"], summary["last_ingestion"]
    age = summary["newest_age_hours"]
    lines = [
        "News index summary",
        f"  Index            {idx['name']} (prefix {idx['prefix']}, dims {idx['dims']}, "
        f"exists={str(idx['exists']).lower()}, num_docs={idx['num_docs']})",
        f"  Embedding        {emb['provider']} / {emb['model_id']}",
        f"  Total documents  {summary['total_documents']}",
        f"  Oldest           {summary['oldest_published'] or '-'}",
        f"  Newest           {summary['newest_published'] or '-'}"
        + ("" if age is None else f" ({age} h ago)"),
        f"  Retention        {summary['retention_days']} days "
        f"({summary['expiring_24h']} expiring within 24h)",
        f"  Untagged         {summary['untagged']} (no ticker)",
        f"  Last success     {last['last_success'] or '-'}",
        f"  Last run         {json.dumps(last['last_run']) if last['last_run'] else '-'}",
        "",
        *_table("By source", summary["by_source"]),
        "",
        *_table("By ticker", summary["by_ticker"]),
        "",
        *_table("By day", summary["by_day"]),
    ]
    return "\n".join(lines)


def _top_arg(value: str) -> int:
    top = int(value)
    if not 1 <= top <= MAX_TOP:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_TOP}")
    return top


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.rag.stats", description=__doc__)
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.add_argument("--top", type=_top_arg, default=DEFAULT_TOP, help="top N sources/tickers")
    args = parser.parse_args(argv)
    try:
        summary = build_summary_job()(args.top)
    except redis.RedisError as exc:
        print(f"Redis unreachable or failing: {sanitize_error(str(exc))}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2) if args.json else format_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
