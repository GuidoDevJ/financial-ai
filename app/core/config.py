import json
import re
from functools import lru_cache
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _parse_api_keys(raw: str) -> dict[str, str]:
    """Parse AUTH_API_KEYS. Error messages never include the raw value (it holds secrets)."""
    problem = "AUTH_API_KEYS must be a JSON object mapping api keys to cliente_id"
    try:
        data = json.loads(raw) if raw.strip() else {}
    except ValueError:
        raise ValueError(f"{problem} (not valid JSON).") from None
    if not isinstance(data, dict):
        raise ValueError(f"{problem} (got a {type(data).__name__}).")
    for key, cliente_id in data.items():
        if not key.strip():
            raise ValueError(f"{problem} (empty api key).")
        if not isinstance(cliente_id, str) or not _CLIENT_ID_RE.fullmatch(cliente_id):
            raise ValueError(f"{problem}; each cliente_id must match [A-Za-z0-9_-]{{1,64}}.")
    return dict(data)


def _parse_admin_keys(raw: str) -> list[str]:
    """Parse ADMIN_API_KEYS (comma-separated or a JSON list). Errors never echo the value."""
    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            data = json.loads(text)
        except ValueError:
            raise ValueError("ADMIN_API_KEYS is not a valid JSON list.") from None
        if not isinstance(data, list) or not all(isinstance(k, str) for k in data):
            raise ValueError("ADMIN_API_KEYS must be a JSON list of strings.")
    else:
        data = text.split(",")
    return [key.strip() for key in data if key.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
        hide_input_in_errors=True,  # validation errors never echo values (they may be secrets)
    )

    # Amazon Bedrock. Explicit credentials (from .env / environment, never committed). When
    # both key id and secret are unset the default boto3 chain (env, IAM role) is used.
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: SecretStr | None = None
    aws_session_token: SecretStr | None = None  # only for temporary credentials
    bedrock_model_id: str = "us.anthropic.claude-sonnet-5-5"
    # Converse inferenceConfig / request options.
    bedrock_max_tokens: int = 1024
    bedrock_temperature: float = 0.0
    # Optional faster/cheaper model for the chat graph (tool routing + answer). Empty = use
    # BEDROCK_MODEL_ID, so behaviour is unchanged unless it is set. The brief never uses it.
    bedrock_chat_model_id: str = ""
    # Chat client timeouts, sized to fit inside CHAT_TIMEOUT_SECONDS (read timeout x attempts).
    # The brief client keeps its own longer limits (30s read, 3 retries).
    bedrock_chat_read_timeout_seconds: float = Field(default=18.0, gt=0)
    bedrock_chat_max_attempts: int = Field(default=2, ge=1, le=5)  # first try + retries
    # Converse prompt caching (cachePoint on system/messages/tools) for the chat client.
    bedrock_prompt_caching: bool = False

    # System prompts live in Amazon Bedrock Prompt Management (no local-file fallback), same
    # naming as ia-comafi-infra: BEDROCK_PROMPT_<NAME>_ID per prompt (id or ARN, required by the
    # feature that uses it) and one global BEDROCK_PROMPT_VERSION ("DRAFT" or a number; use a
    # numbered version in production).
    bedrock_prompt_chat_id: str = ""
    bedrock_prompt_brief_id: str = ""
    bedrock_prompt_version: str = "DRAFT"
    prompt_cache_ttl_seconds: int = Field(default=300, ge=0)
    # Value of the {{banco}} prompt variable. The default is an assumption: set BANK_NAME.
    bank_name: str = "Banco Comafi"

    # Tags sent as Converse requestMetadata (visible in model invocation logs).
    app_name: str = "financial-ai"
    app_env: str = "dev"

    iol_user: str = ""
    iol_password: str = ""
    iol_base_url: str = "https://api.invertironline.com"

    price_cache_ttl: int = 900
    # IOL (BYMA) HTTP timeout: kept short so the yfinance fallback starts quickly.
    iol_timeout_seconds: float = Field(default=3.5, gt=0)
    macro_cache_ttl: int = 300

    news_rss_urls: Annotated[list[str], NoDecode] = []
    news_retention_days: int = 7
    news_top_k: int = 4
    # Redis cache of news retrieval results keyed by (query, ticker, hours, top_k); 0 disables.
    news_cache_ttl_seconds: int = Field(default=300, ge=0)
    news_http_timeout: float = 10.0
    # Feed fetch hygiene: identifying User-Agent and a hard cap on the response size.
    news_user_agent: str = "financial-ai-news-ingest/1.0"
    news_max_feed_bytes: int = Field(default=5_000_000, gt=0)
    # Indexing batch size and retry policy (exponential backoff + jitter, capped at 20s).
    ingest_batch_size: int = Field(default=10, ge=1)
    ingest_max_retries: int = Field(default=3, ge=0)
    ingest_retry_base_seconds: float = Field(default=1.0, ge=0)
    # RediSearch index for news (needs Redis Stack). Keys live under `<prefix>/vector_<id>`.
    news_index_name: str = "idx:news"
    news_index_prefix: str = "news"

    # Embeddings. "bedrock" (default) or "huggingface" (local, CPU). Reuses the AWS_* settings.
    embedding_provider: str = "bedrock"
    embedding_model_id: str = "cohere.embed-multilingual-v3"
    # Vector size of the Redis index; derived from the model when unset (Cohere v3 = 1024).
    embedding_dimensions: int | None = None

    # Ingestion scheduler: off unless ENABLE_SCHEDULER=true (one instance should enable it).
    enable_scheduler: bool = False
    ingest_interval_minutes: int = 90
    ingest_lock_ttl_seconds: int = 900

    redis_url: str = "redis://localhost:6379/0"
    # Per-command Redis socket timeout (a hung Redis must not stall a request).
    redis_socket_timeout: float = Field(default=5.0, gt=0)

    # Simulated brief delivery. With SMTP_HOST unset nothing is sent (only "email skipped" is
    # logged). SMTP_TO is a single test mailbox for every client (MVP).
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: SecretStr | None = None
    smtp_from: str = ""
    smtp_to: str = ""
    smtp_starttls: bool = True

    # Daily brief job (runs only when ENABLE_SCHEDULER=true).
    brief_lock_ttl_seconds: int = 600

    # Chat request limits. The timeout covers the whole turn (LLM + tools) and yields 504
    # CHAT_TIMEOUT; the cap bounds model->tools rounds in one turn before a graceful message.
    chat_timeout_seconds: float = Field(default=25.0, gt=0)
    chat_max_tool_iterations: int = Field(default=6, ge=1, le=20)
    # History sent to the model each round: keep at most this many trailing messages
    # (0 = send everything, the previous behaviour). Tool call/result pairs are never split.
    chat_history_max_messages: int = Field(default=0, ge=0)
    # POST /chat/stream: also emit the model's text deltas as `token` events. Off by default
    # because the guardrail runs after the model: the final `done` event is the authoritative
    # reply and may replace what the tokens showed.
    chat_stream_tokens: bool = False

    # Warm the chat graph, brief dependencies, news store and prompts at startup (failures are
    # logged and the lazy path stays as fallback). Each step is bounded by the timeout.
    warmup_on_startup: bool = True
    warmup_timeout_seconds: float = Field(default=30.0, gt=0)

    # API authentication: JSON object {"<api key>": "<cliente_id>"}. Unset/empty is only
    # accepted when APP_ENV=dev (the body's cliente_id is then trusted, with a startup warning).
    auth_api_keys: SecretStr | None = None

    # Admin keys (comma-separated or JSON list) for global operations such as POST /news/ingest.
    # Unset/empty is only accepted when APP_ENV=dev (startup warning); otherwise 503.
    admin_api_keys: SecretStr | None = None

    # Stale-if-error copies (seconds) served, flagged stale, when every provider fails.
    price_stale_ttl_seconds: int = Field(default=604800, ge=0)  # 7 days
    macro_stale_ttl_seconds: int = Field(default=604800, ge=0)  # 7 days
    tasa_bcra_stale_ttl_seconds: int = Field(default=2592000, ge=0)  # 30 days (moves slowly)

    @field_validator("bedrock_prompt_version", mode="after")
    @classmethod
    def _valid_prompt_version(cls, value: str) -> str:
        version = value.strip()
        if version == "DRAFT" or (
            version.isascii() and version.isdigit() and 1 <= int(version) <= 99999
        ):
            return version
        raise ValueError(
            f"BEDROCK_PROMPT_VERSION must be 'DRAFT' or an integer in 1..99999, got {value!r}."
        )

    @field_validator("auth_api_keys", mode="after")
    @classmethod
    def _valid_api_keys(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None or not value.get_secret_value().strip():
            return None
        _parse_api_keys(value.get_secret_value())  # fail fast; never echoes the value
        return value

    def api_key_map(self) -> dict[str, str]:
        """`{api_key: cliente_id}` (empty when authentication is not configured)."""
        if self.auth_api_keys is None:
            return {}
        return _parse_api_keys(self.auth_api_keys.get_secret_value())

    @field_validator("admin_api_keys", mode="after")
    @classmethod
    def _valid_admin_keys(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None or not _parse_admin_keys(value.get_secret_value()):
            return None
        return value

    def admin_key_list(self) -> list[str]:
        """Configured admin API keys (empty when none)."""
        if self.admin_api_keys is None:
            return []
        return _parse_admin_keys(self.admin_api_keys.get_secret_value())

    @field_validator("news_rss_urls", mode="before")
    @classmethod
    def _split_csv(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
