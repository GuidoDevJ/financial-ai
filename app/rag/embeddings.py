"""Embedding model factory: Amazon Bedrock (default) or local HuggingFace."""

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.bedrock import BedrockEmbedding

from app.core.aws import explicit_credentials
from app.core.config import Settings, get_settings

DEFAULT_BEDROCK_MODEL = "cohere.embed-multilingual-v3"
DEFAULT_HF_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Output vector sizes of the supported models.
KNOWN_DIMENSIONS = {
    "cohere.embed-multilingual-v3": 1024,
    "cohere.embed-english-v3": 1024,
    DEFAULT_HF_MODEL: 384,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
}

PROVIDERS = ("bedrock", "huggingface")


def _check_provider(settings: Settings) -> None:
    if settings.embedding_provider not in PROVIDERS:
        raise ValueError(
            f"embedding_provider must be one of {list(PROVIDERS)}, "
            f"got {settings.embedding_provider!r}."
        )


def resolve_model_id(settings: Settings) -> str:
    """Model id for the provider; a Bedrock (cohere.*) id is not valid for HuggingFace."""
    _check_provider(settings)
    model_id = settings.embedding_model_id
    if settings.embedding_provider == "huggingface" and model_id.startswith("cohere."):
        return DEFAULT_HF_MODEL
    return model_id


def embedding_dimensions(settings: Settings) -> int:
    """Vector size for the Redis index; must match what the embedding model returns."""
    if settings.embedding_dimensions:
        return settings.embedding_dimensions
    model_id = resolve_model_id(settings)
    try:
        return KNOWN_DIMENSIONS[model_id]
    except KeyError:
        raise ValueError(
            f"Unknown vector size for embedding model {model_id!r}; set EMBEDDING_DIMENSIONS."
        ) from None


def _load_huggingface(model_id: str) -> BaseEmbedding:
    # Lazy import: pulls in torch / sentence-transformers, which Bedrock deployments don't need.
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name=model_id)


def build_embed_model(settings: Settings | None = None) -> BaseEmbedding:
    settings = settings or get_settings()
    model_id = resolve_model_id(settings)
    if settings.embedding_provider == "huggingface":
        return _load_huggingface(model_id)
    return BedrockEmbedding(
        model_name=model_id,
        region_name=settings.aws_region,
        **explicit_credentials(settings),
    )
