from unittest.mock import MagicMock

import pytest
import redis
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, SimpleVectorStore
from llama_index.core.vector_stores.types import FilterCondition, FilterOperator

from app.core.config import Settings
from app.rag.embeddings import (
    DEFAULT_HF_MODEL,
    build_embed_model,
    embedding_dimensions,
    resolve_model_id,
)
from app.rag.redis_store import RedisNewsVectorStore, build_redis_filter
from app.rag.store import NewsStore, build_news_store
from tests.fakes import BagOfWordsEmbedding


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


# -------------------------------------------------------------- embeddings


def test_embedding_defaults():
    s = settings()
    assert s.embedding_provider == "bedrock"
    assert s.embedding_model_id == "cohere.embed-multilingual-v3"
    assert embedding_dimensions(s) == 1024


def test_huggingface_provider_swaps_default_model_and_dims():
    s = settings(embedding_provider="huggingface")
    assert resolve_model_id(s) == DEFAULT_HF_MODEL
    assert DEFAULT_HF_MODEL.endswith("paraphrase-multilingual-MiniLM-L12-v2")
    assert embedding_dimensions(s) == 384


def test_explicit_dimensions_override():
    assert embedding_dimensions(settings(embedding_dimensions=768)) == 768


def test_unknown_model_without_dimensions_is_an_error():
    with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS"):
        embedding_dimensions(settings(embedding_model_id="amazon.titan-embed-text-v2:0"))


def test_bedrock_embedding_uses_explicit_credentials(monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr("app.rag.embeddings.BedrockEmbedding", Fake)
    build_embed_model(
        settings(
            aws_region="us-east-1",
            aws_access_key_id="AKIAEXAMPLE",
            aws_secret_access_key="s3cr3t",
            aws_session_token="tok",
        )
    )
    assert seen == {
        "model_name": "cohere.embed-multilingual-v3",
        "region_name": "us-east-1",
        "aws_access_key_id": "AKIAEXAMPLE",
        "aws_secret_access_key": "s3cr3t",
        "aws_session_token": "tok",
    }


def test_bedrock_embedding_without_explicit_credentials_uses_default_chain(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "app.rag.embeddings.BedrockEmbedding", lambda **kw: seen.update(kw) or object()
    )
    build_embed_model(settings())
    assert seen == {"model_name": "cohere.embed-multilingual-v3", "region_name": "us-east-1"}


def test_huggingface_embedding_built_lazily(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "app.rag.embeddings._load_huggingface", lambda model_id: seen.setdefault("m", model_id)
    )
    build_embed_model(settings(embedding_provider="huggingface"))
    assert seen["m"] == DEFAULT_HF_MODEL


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="embedding_provider"):
        build_embed_model(settings(embedding_provider="openai"))


# ------------------------------------------------------------ store factory


def test_factory_accepts_injected_vector_store_and_embeddings():
    vs = SimpleVectorStore()
    store = build_news_store(settings(), vector_store=vs, embed_model=BagOfWordsEmbedding())
    assert isinstance(store, NewsStore) and store.vector_store is vs
    assert store.retention_days == 7


def test_factory_builds_redis_store_by_default_with_configured_dims(monkeypatch):
    captured = {}

    class FakeRedisStore(SimpleVectorStore):
        def __init__(self, **kw):
            super().__init__()
            captured.update(kw)

    monkeypatch.setattr("app.rag.store.RedisNewsVectorStore", FakeRedisStore)
    build_news_store(settings(news_retention_days=3), embed_model=BagOfWordsEmbedding())
    assert captured["dims"] == 1024
    assert captured["retention_seconds"] == 3 * 86400
    assert captured["redis_url"].startswith("redis://")


# ------------------------------------------------------- redis filter string


def f(key, op, value):
    return MetadataFilter(key=key, value=value, operator=op)


def test_filter_string_combines_tag_and_numeric():
    flt = MetadataFilters(
        filters=[
            f("tickers", FilterOperator.CONTAINS, "GGAL"),
            f("published_epoch", FilterOperator.GTE, 1760000000),
        ]
    )
    assert build_redis_filter(flt) == "(@tickers:{GGAL} @published_epoch:[1760000000 +inf])"


def test_filter_string_numeric_operators_and_none():
    assert build_redis_filter(None) == "*"
    gt = MetadataFilters(filters=[f("published_epoch", FilterOperator.GT, 5)])
    assert build_redis_filter(gt) == "(@published_epoch:[(5 +inf])"
    lt = MetadataFilters(filters=[f("published_epoch", FilterOperator.LT, 5)])
    assert build_redis_filter(lt) == "(@published_epoch:[-inf (5])"


def test_filter_string_rejects_unsupported():
    with pytest.raises(ValueError):
        build_redis_filter(MetadataFilters(filters=[f("title", FilterOperator.EQ, "x")]))
    with pytest.raises(ValueError):
        build_redis_filter(
            MetadataFilters(
                filters=[f("tickers", FilterOperator.EQ, "A")], condition=FilterCondition.OR
            )
        )


# ------------------------------------------- redis vector store (mock client)


def make_redis_store(dims=1024, client=None):
    client = client or MagicMock()
    client.execute_command.return_value = []  # FT._LIST -> no index yet
    return (
        RedisNewsVectorStore(
            index_name="idx:news",
            index_prefix="news",
            dims=dims,
            retention_seconds=7 * 86400,
            redis_client=client,
        ),
        client,
    )


def node(i: str, dims: int, epoch: int = 1760000000) -> TextNode:
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

    return TextNode(
        id_=i,
        text="titulo\nresumen",
        embedding=[0.1] * dims,
        metadata={"tickers": "GGAL", "source": "Ambito", "published_epoch": epoch},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=i)},
    )


def test_index_schema_has_tag_numeric_and_configured_vector_dims():
    store, client = make_redis_store(dims=1024)
    store.add([node("a", 1024)])
    fields = client.ft.return_value.create_index.call_args.kwargs["fields"]
    by_name = {fld.name: fld for fld in fields}
    assert type(by_name["tickers"]).__name__ == "TagField"
    assert type(by_name["source"]).__name__ == "TagField"
    assert type(by_name["published_epoch"]).__name__ == "NumericField"
    assert by_name["vector"].args[by_name["vector"].args.index("DIM") + 1] == 1024
    client.ft.assert_any_call("idx:news")


def test_add_sets_per_key_ttl_for_retention():
    store, client = make_redis_store(dims=4)
    store.add([node("a", 4)])
    key = "news/vector_a"
    client.hset.assert_called_once()
    assert client.hset.call_args.args[0] == key
    client.expire.assert_called_once_with(key, 7 * 86400)


def test_add_rejects_embedding_dimension_mismatch():
    store, client = make_redis_store(dims=1024)
    with pytest.raises(ValueError, match="1024"):
        store.add([node("a", 384)])
    client.hset.assert_not_called()


def test_purge_older_than_deletes_matching_keys():
    store, client = make_redis_store(dims=4)
    first = MagicMock(docs=[MagicMock(id="news/vector_a"), MagicMock(id="news/vector_b")])
    client.ft.return_value.search.side_effect = [first, MagicMock(docs=[])]
    assert store.purge_older_than(1760000000) == 2
    client.delete.assert_called_once_with("news/vector_a", "news/vector_b")
    query = client.ft.return_value.search.call_args_list[0].args[0]
    assert "@published_epoch:[-inf (1760000000]" in query.query_string()


def test_empty_result_is_not_an_error():
    store, client = make_redis_store(dims=4)
    client.ft.return_value.search.return_value = MagicMock(docs=[])
    from llama_index.core.vector_stores.types import VectorStoreQuery

    res = store.query(VectorStoreQuery(query_embedding=[0.1] * 4, similarity_top_k=3))
    assert res.nodes == [] and res.ids == []


# --------------------------------------------------- integration (real Redis)


@pytest.mark.integration
def test_real_redis_stack_roundtrip():
    """Needs Redis Stack (docker compose up -d). Run with: uv run pytest -m integration"""
    from datetime import UTC, datetime, timedelta

    from app.rag.ingest_news import NewsItem, item_to_document
    from app.rag.retrieve import retrieve_news

    s = settings(news_index_name="idx:news-test", news_index_prefix="news-test")
    client = redis.Redis.from_url(s.redis_url)
    try:
        client.ping()
    except redis.RedisError:
        pytest.skip("Redis is not reachable")
    try:
        client.ft(s.news_index_name).dropindex(delete_documents=True)
    except redis.ResponseError:
        pass
    embed = BagOfWordsEmbedding(dim=1024)
    store = build_news_store(s, embed_model=embed)
    now = datetime.now(UTC)
    store.upsert(
        [
            item_to_document(NewsItem("Galicia sube", "GGAL", "https://x/1", "Ambito", now)),
            item_to_document(
                NewsItem("Galicia vieja", "GGAL", "https://x/2", "Ambito", now - timedelta(days=9))
            ),
        ]
    )
    res = retrieve_news(store, "Galicia", ticker="GGAL", hours=72, top_k=4, now=now)
    assert [r["url"] for r in res] == ["https://x/1"]
    assert store.purge(now=now) == 1
    client.ft(s.news_index_name).dropindex(delete_documents=True)
