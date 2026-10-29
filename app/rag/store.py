"""News vector store: a LlamaIndex VectorStoreIndex over an injectable vector store.

Production uses RedisNewsVectorStore (Redis Stack / RediSearch). Tests inject LlamaIndex's
in-memory SimpleVectorStore and a fake embedding, so no Redis, AWS or model is needed.
"""

import logging
from datetime import UTC, datetime, timedelta

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import Document, NodeRelationship, TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from app.core.config import Settings, get_settings
from app.rag.embeddings import build_embed_model, embedding_dimensions
from app.rag.redis_store import RedisNewsVectorStore

logger = logging.getLogger(__name__)


def document_to_node(doc: Document) -> TextNode:
    """One Document -> exactly one node (no chunking), keeping the stable id.

    Reusing the document id as node id makes re-ingesting the same URL an overwrite (upsert).
    Metadata is excluded from the embedded/LLM text: only title + lede are embedded.
    """
    keys = list(doc.metadata)
    return TextNode(
        id_=doc.id_,
        text=doc.text,
        metadata=dict(doc.metadata),
        excluded_embed_metadata_keys=keys,
        excluded_llm_metadata_keys=keys,
        relationships={NodeRelationship.SOURCE: doc.as_related_node_info()},
    )


class NewsStore:
    def __init__(
        self,
        vector_store: BasePydanticVectorStore,
        embed_model: BaseEmbedding,
        *,
        retention_days: int = 7,
    ) -> None:
        self.vector_store = vector_store
        self.embed_model = embed_model
        self.retention_days = retention_days
        # from_vector_store() refuses stores that keep no text (SimpleVectorStore), so build the
        # index explicitly with an in-memory docstore that those stores rely on.
        self.index = VectorStoreIndex(
            nodes=[],
            storage_context=StorageContext.from_defaults(vector_store=vector_store),
            embed_model=embed_model,
        )

    def upsert(self, documents: list[Document]) -> int:
        nodes = [document_to_node(d) for d in documents]
        if nodes:
            self.index.insert_nodes(nodes)
        return len(nodes)

    def existing_hashes(self, ids: list[str]) -> dict[str, str | None]:
        """Stored `content_hash` per node id (None when missing or without a hash).

        Raises on lookup failure (e.g. Redis down); the caller decides the fallback.
        """
        if not ids:
            return {}
        if hasattr(self.vector_store, "existing_hashes"):
            return self.vector_store.existing_hashes(ids)
        if isinstance(self.vector_store, SimpleVectorStore):
            metadata = self.vector_store.data.metadata_dict
            return {i: (metadata.get(i) or {}).get("content_hash") for i in ids}
        raise NotImplementedError(f"No hash lookup for {type(self.vector_store).__name__}.")

    def purge(self, now: datetime | None = None) -> int:
        """Delete news older than the retention window; returns how many were removed."""
        now = now or datetime.now(UTC)
        cutoff = int((now - timedelta(days=self.retention_days)).timestamp())
        if hasattr(self.vector_store, "purge_older_than"):
            return self.vector_store.purge_older_than(cutoff)
        if isinstance(self.vector_store, SimpleVectorStore):
            stale = [
                node_id
                for node_id, md in self.vector_store.data.metadata_dict.items()
                if md.get("published_epoch", cutoff) < cutoff
            ]
            if stale:
                self.index.delete_nodes(stale, delete_from_docstore=True)
            return len(stale)
        raise NotImplementedError(f"No purge strategy for {type(self.vector_store).__name__}.")


def build_news_store(
    settings: Settings | None = None,
    *,
    vector_store: BasePydanticVectorStore | None = None,
    embed_model: BaseEmbedding | None = None,
) -> NewsStore:
    """Factory. Pass `vector_store` / `embed_model` to inject test doubles."""
    settings = settings or get_settings()
    if embed_model is None:
        embed_model = build_embed_model(settings)
    if vector_store is None:
        vector_store = RedisNewsVectorStore(
            index_name=settings.news_index_name,
            index_prefix=settings.news_index_prefix,
            dims=embedding_dimensions(settings),
            retention_seconds=settings.news_retention_days * 86400,
            redis_url=settings.redis_url,
        )
    return NewsStore(vector_store, embed_model, retention_days=settings.news_retention_days)
