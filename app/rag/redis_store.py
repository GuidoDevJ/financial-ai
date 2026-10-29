"""RedisVectorStore specialised for news.

The stock llama-index RedisVectorStore (0.1.x) only indexes TAG metadata and only supports
exact-match filters. News needs a numeric `published_epoch` field (time-window filter) and a
purge by age, so this subclass:

* creates the RediSearch schema with TAG (`tickers`, `source`) and NUMERIC (`published_epoch`)
  fields plus a vector field of the configured dimension,
* translates MetadataFilters (tag match / numeric comparisons) to RediSearch syntax,
* sets a per-key TTL equal to the retention window on every write,
* exposes `purge_older_than` and treats "no results" as an empty result, not an error.

TODO(unverified): the RediSearch schema and queries were not run against a live Redis Stack in
this environment; see the `integration` test and run it with `docker compose up -d`.
"""

import logging
from typing import Any

import redis
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.vector_stores.redis.utils import (
    TokenEscaper,
    array_to_buffer,
    get_redis_query,
)
from redis.commands.search import reducers
from redis.commands.search.aggregation import AggregateRequest
from redis.commands.search.field import NumericField, TagField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

logger = logging.getLogger(__name__)

TAG_FIELDS = ("tickers", "source")
NUMERIC_FIELDS = ("published_epoch",)

_TAG_OPS = {FilterOperator.EQ, FilterOperator.CONTAINS, FilterOperator.TEXT_MATCH}
_escaper = TokenEscaper()


def build_redis_filter(filters: MetadataFilters | None) -> str:
    """MetadataFilters -> RediSearch pre-filter string ("*" when there is nothing to filter)."""
    if filters is None or not filters.filters:
        return "*"
    if filters.condition == FilterCondition.OR:
        raise ValueError("OR conditions are not supported by the news Redis store.")
    parts = []
    for flt in filters.filters:
        if isinstance(flt, MetadataFilters):
            raise ValueError("Nested filters are not supported by the news Redis store.")
        if flt.key in TAG_FIELDS and flt.operator in _TAG_OPS:
            parts.append(f"@{flt.key}:{{{_escaper.escape(str(flt.value))}}}")
        elif flt.key in NUMERIC_FIELDS and flt.operator in (
            FilterOperator.GT,
            FilterOperator.GTE,
            FilterOperator.LT,
            FilterOperator.LTE,
        ):
            v = flt.value
            ranges = {
                FilterOperator.GTE: f"[{v} +inf]",
                FilterOperator.GT: f"[({v} +inf]",
                FilterOperator.LTE: f"[-inf {v}]",
                FilterOperator.LT: f"[-inf ({v}]",
            }
            parts.append(f"@{flt.key}:{ranges[flt.operator]}")
        else:
            raise ValueError(f"Unsupported filter on {flt.key!r} with operator {flt.operator}.")
    return "(" + " ".join(parts) + ")"


def _is_missing_index(exc: Exception) -> bool:
    text = str(exc).lower()
    return "no such index" in text or "unknown index" in text


def _as_text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _overall_request() -> AggregateRequest:
    return AggregateRequest("*").group_by(
        [],
        reducers.count().alias("n"),
        reducers.min("@published_epoch").alias("min_epoch"),
        reducers.max("@published_epoch").alias("max_epoch"),
    )


def _group_request(field: str) -> AggregateRequest:
    request = AggregateRequest("*")
    if field == "day":
        request = request.apply(day="floor(@published_epoch / 86400)")
    return request.group_by(f"@{field}", reducers.count().alias("n"))


def _agg_rows(result: Any) -> list[dict[str, str]]:
    """FT.AGGREGATE rows (flat [name, value, ...] lists) -> list of dicts of text."""
    rows = []
    for raw in result.rows:
        items = [_as_text(x) for x in raw]
        rows.append(dict(zip(items[::2], items[1::2], strict=False)))
    return rows


def _agg_groups(result: Any, field: str) -> dict[str, int]:
    return {
        row[field]: int(row["n"])
        for row in _agg_rows(result)
        if row.get(field) not in (None, "") and "n" in row
    }


class RedisNewsVectorStore(RedisVectorStore):
    _dims: int = PrivateAttr()
    _retention_seconds: int = PrivateAttr()

    def __init__(
        self,
        *,
        index_name: str,
        index_prefix: str,
        dims: int,
        retention_seconds: int,
        redis_url: str | None = None,
        redis_client: Any = None,
        overwrite: bool = False,
    ) -> None:
        # The parent constructor connects and checks RediSearch eagerly; we only set its private
        # state so the store can be built with an injected client and connect lazily.
        if redis_client is None:
            redis_client = redis.Redis.from_url(redis_url or "redis://localhost:6379")
        self._redis_client = redis_client
        self._prefix = f"{index_prefix}/vector"
        self._index_name = index_name
        self._index_args = {"dims": dims}
        self._metadata_fields = [*TAG_FIELDS, *NUMERIC_FIELDS]
        self._overwrite = overwrite
        self._vector_field = "vector"
        self._vector_key = "vector"
        self._tokenizer = _escaper
        self._dims = dims
        self._retention_seconds = retention_seconds
        super(RedisVectorStore, self).__init__()

    # -------------------------------------------------------------- schema

    def _create_index(self) -> None:
        fields = [
            TextField("text", weight=1.0),
            TagField("doc_id", sortable=False),
            TagField("id", sortable=False),
            *(TagField(name, sortable=False) for name in TAG_FIELDS),
            *(NumericField(name, sortable=True) for name in NUMERIC_FIELDS),
            self._create_vector_field(self._vector_field, **self._index_args),
        ]
        logger.info("Creating RediSearch index %s (dims=%s)", self._index_name, self._dims)
        self._redis_client.ft(self._index_name).create_index(
            fields=fields,
            definition=IndexDefinition(prefix=[self._prefix], index_type=IndexType.HASH),
        )

    # --------------------------------------------------------------- write

    def add(self, nodes: list[BaseNode], **add_kwargs: Any) -> list[str]:
        for node in nodes:
            size = len(node.get_embedding())
            if size != self._dims:
                raise ValueError(
                    f"Embedding has {size} dimensions but the news index is configured for "
                    f"{self._dims}; check EMBEDDING_MODEL_ID / EMBEDDING_DIMENSIONS."
                )
        ids = super().add(nodes, **add_kwargs)
        for node_id in ids:
            self._redis_client.expire(f"{self._prefix}_{node_id}", self._retention_seconds)
        return ids

    def existing_hashes(self, ids: list[str]) -> dict[str, str | None]:
        """`content_hash` field of each `<prefix>_<id>` hash (None when absent), one round trip.

        llama-index stores flat node metadata as top-level hash fields, so HGET suffices.
        """
        if not ids:
            return {}
        pipe = self._redis_client.pipeline(transaction=False)
        for node_id in ids:
            pipe.hget(f"{self._prefix}_{node_id}", "content_hash")
        values = pipe.execute()
        return {
            node_id: value.decode() if isinstance(value, bytes) else value
            for node_id, value in zip(ids, values, strict=True)
        }

    def purge_older_than(self, cutoff_epoch: int) -> int:
        """Delete every stored item with published_epoch < cutoff_epoch."""
        query = Query(f"@published_epoch:[-inf ({cutoff_epoch}]").no_content().paging(0, 500)
        deleted = 0
        while True:
            try:
                docs = self._redis_client.ft(self._index_name).search(query).docs
            except redis.ResponseError:  # index does not exist yet: nothing to purge
                return deleted
            if not docs:
                return deleted
            self._redis_client.delete(*[doc.id for doc in docs])
            deleted += len(docs)

    # ---------------------------------------------------------------- stats

    def index_aggregates(self, *, expiring_from: int, expiring_to: int) -> dict[str, Any]:
        """Server-side aggregates of the whole index (no document text is transferred).

        One FT.INFO plus FT.AGGREGATE GROUPBY on `source`, `tickers` and a daily bucket of
        `published_epoch`, a global COUNT/MIN/MAX and an FT.SEARCH count of items whose
        `published_epoch` is in [expiring_from, expiring_to). A missing index is reported as
        `{"exists": False}`; any other Redis error propagates.
        """
        ft = self._redis_client.ft(self._index_name)
        try:
            info = ft.info()
            overall = _agg_rows(ft.aggregate(_overall_request()))
            sources = _agg_groups(ft.aggregate(_group_request("source")), "source")
            tickers = _agg_groups(ft.aggregate(_group_request("tickers")), "tickers")
            days = _agg_groups(ft.aggregate(_group_request("day")), "day")
            expiring = ft.search(
                Query(f"@published_epoch:[{expiring_from} ({expiring_to}]").paging(0, 0)
            ).total
        except redis.ResponseError as exc:
            if _is_missing_index(exc):
                return {"exists": False}
            raise
        row = overall[0] if overall else {}
        return {
            "exists": True,
            "num_docs": int(_as_text(info.get("num_docs", 0)) or 0),
            "total": int(row.get("n", 0)),
            "min_epoch": int(float(row["min_epoch"])) if "min_epoch" in row else None,
            "max_epoch": int(float(row["max_epoch"])) if "max_epoch" in row else None,
            "expiring": int(expiring),
            "sources": sources,
            "ticker_groups": tickers,
            "days": {int(float(k)): v for k, v in days.items()},
        }

    # ---------------------------------------------------------------- read

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if not query.query_embedding:
            raise ValueError("Query embedding is required for querying.")
        return_fields = ["id", "doc_id", "text", self._vector_key, "vector_score", "_node_content"]
        redis_query = get_redis_query(
            return_fields=return_fields,
            top_k=query.similarity_top_k,
            vector_field=self._vector_field,
            filters=build_redis_filter(query.filters),
        )
        try:
            results = self._redis_client.ft(self._index_name).search(
                redis_query, query_params={"vector": array_to_buffer(query.query_embedding)}
            )
        except redis.ResponseError as exc:
            if "no such index" in str(exc).lower():  # nothing ingested yet
                return VectorStoreQueryResult(nodes=[], ids=[], similarities=[])
            raise
        ids, nodes, scores = [], [], []
        for doc in results.docs:
            try:
                node = metadata_dict_to_node({"_node_content": doc._node_content})
                node.text = doc.text
            except Exception:
                node = TextNode(
                    text=doc.text,
                    id_=doc.id,
                    relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc.doc_id)},
                )
            ids.append(doc.id.removeprefix(f"{self._prefix}_"))
            nodes.append(node)
            scores.append(1 - float(doc.vector_score))
        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=scores)
