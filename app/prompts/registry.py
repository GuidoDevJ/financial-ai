"""System prompts live in Amazon Bedrock Prompt Management; the app only reads them.

Same registration pattern as the sibling project `ia-comafi-infra`: a logical prompt name maps to
the env var `BEDROCK_PROMPT_<NAME>_ID` (`chat`, `brief`), one global `BEDROCK_PROMPT_VERSION`
(default `DRAFT`) selects the version, and `get_prompt_registry().get("chat")` returns a
`PromptVersion` (version, content, hash) so audit records can say which prompt produced an
answer. The cached template is raw; `PromptVersion.render(**variables)` fills the `{{variables}}`
at use time (strict: an unresolved variable raises `PROMPT_VARIABLES_MISSING`).
`docs/prompts/registry.yaml` is the hand-maintained source of truth / changelog; it is never read
at runtime.

The text is fetched with `bedrock-agent` `GetPrompt` (never used as a Converse prompt ARN, which
Converse forbids together with `system` / `toolConfig` / `inferenceConfig`). There is no
local-file fallback.

GetPrompt response (botocore `bedrock-agent` 2023-06-05): `variants[]` (max 1), each with
`templateType` (`TEXT` | `CHAT`) and `templateConfiguration.text.text`. `promptVersion` is
`DRAFT` or a number.

Resilience (kept from the original design, beyond the sibling): text is cached in memory per
(id, version) for `PROMPT_CACHE_TTL_SECONDS`; if a refresh fails the last good text is served
(stale-if-error). Nothing is fetched at import time and boto3 is called off the event loop.
"""

import asyncio
import hashlib
import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from app.core.aws import explicit_credentials, missing_credential_vars
from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_SECONDS = 5
READ_TIMEOUT_SECONDS = 10
MAX_RETRIES = 3
# After a failed refresh the stale text is served and Bedrock is not retried for this long.
STALE_RETRY_SECONDS = 30
DEFAULT_VERSION = "DRAFT"

# `{{variable}}` placeholder (Amazon Bedrock Prompt Management convention).
VARIABLE_PATTERN = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")

# Logical prompt name -> Settings attribute holding the Bedrock prompt id (env var
# BEDROCK_PROMPT_<NAME>_ID). Add a prompt here, in Settings and in docs/prompts/registry.yaml.
_PROMPT_ID_MAP: dict[str, str] = {
    "chat": "bedrock_prompt_chat_id",
    "brief": "bedrock_prompt_brief_id",
}


def prompt_env_var(name: str) -> str:
    return f"BEDROCK_PROMPT_{name.upper()}_ID"


class PromptError(Exception):
    """A prompt could not be obtained. Carries the HTTP status and API error code."""

    def __init__(self, code: str, message: str, status: int = 502):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


AWS_MESSAGE_MAX_CHARS = 500


def _log_get_prompt_failure(exc: ClientError, name: str, prompt_id: str, version: str) -> None:
    """One warning with the AWS detail that the mapped PromptError would otherwise discard."""
    error = exc.response.get("Error", {})
    meta = exc.response.get("ResponseMetadata", {})
    logger.warning(
        "GetPrompt failed: prompt=%s id=%s version=%s aws_code=%s http_status=%s "
        "request_id=%s aws_message=%s",
        name,
        prompt_id,
        version,
        error.get("Code", ""),
        meta.get("HTTPStatusCode", ""),
        meta.get("RequestId", ""),
        str(error.get("Message", ""))[:AWS_MESSAGE_MAX_CHARS],
    )


def _fetch_error(aws_code: str, label: str, prompt_id: str) -> tuple[int, str, str] | None:
    where = f"{label}={prompt_id!r}"
    if aws_code == "AccessDeniedException":
        return (
            502,
            "PROMPT_ACCESS_DENIED",
            f"Access denied reading the prompt ({where}). The server log has the AWS error "
            "detail (message and request id). Grant bedrock:GetPrompt on the prompt ARN to the "
            "credentials in use; other usual causes: the prompt is in a different account/region "
            "than the credentials (AWS_REGION), an explicit deny (SCP, permissions boundary or "
            "session policy), or a KMS customer-managed key without kms:Decrypt.",
        )
    if aws_code == "ResourceNotFoundException":
        return (
            502,
            "PROMPT_NOT_FOUND",
            f"Prompt not found ({where}). Check {label}, BEDROCK_PROMPT_VERSION and AWS_REGION.",
        )
    if aws_code in ("ThrottlingException", "ServiceQuotaExceededException"):
        return (503, "PROMPT_UNAVAILABLE", f"Bedrock is throttling prompt reads ({where}); retry.")
    if aws_code in ("InternalServerException", "ServiceUnavailableException"):
        return (503, "PROMPT_UNAVAILABLE", f"Bedrock Prompt Management is unavailable ({where}).")
    return None


class PromptVersion:
    """One concrete version of a prompt: text plus the identity stored in audit records."""

    def __init__(self, version: str, content: str, name: str = "") -> None:
        self.version = version
        self.content = content.strip()
        self.name = name
        self.hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]

    def variables(self) -> list[str]:
        """Sorted names of the `{{variables}}` used in the template."""
        return sorted(set(VARIABLE_PATTERN.findall(self.content)))

    def render(self, **variables: Any) -> str:
        """Fill the `{{variables}}` of the template (strict, single pass).

        Raises `PromptError` `PROMPT_VARIABLES_MISSING` if the template uses a variable that
        was not given; extra variables are ignored. Values are inserted verbatim and are never
        scanned again, so a value that contains `{{x}}` (e.g. untrusted news text) is not
        substituted.
        """
        missing: set[str] = set()

        def substitute(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in variables:
                missing.add(name)
                return match.group(0)
            return str(variables[name])

        rendered = VARIABLE_PATTERN.sub(substitute, self.content)
        if missing:
            listing = ", ".join("{{" + name + "}}" for name in sorted(missing))
            raise PromptError(
                "PROMPT_VARIABLES_MISSING",
                f"Prompt '{self.name or 'prompt'}' v{self.version} uses variables the app does "
                f"not provide: {listing}. Declare only supported variables in the Bedrock "
                "prompt (see docs/prompts/README.md).",
            )
        return rendered

    def metadata(self) -> dict[str, str]:
        """`{name, version, hash}` for audit logs and stored payloads."""
        return {"name": self.name, "version": self.version, "hash": self.hash}

    def __repr__(self) -> str:
        return f"<PromptVersion name={self.name} v={self.version} hash={self.hash}>"


class PromptSource(Protocol):
    """Where a graph gets its system prompt from, on every run."""

    def get(self) -> PromptVersion: ...

    async def aget(self) -> PromptVersion: ...


class StaticPrompt:
    """Fixed text. For tests and injection only; production uses `RegistryPrompt`."""

    def __init__(self, text: str, name: str = "static", version: str = "static"):
        self.text = text
        self._version = PromptVersion(version, text, name)

    def get(self) -> PromptVersion:
        return self._version

    async def aget(self) -> PromptVersion:
        return self._version


class PromptProvider:
    """Fetches prompts from Bedrock Prompt Management with a TTL cache (thread-safe)."""

    def __init__(
        self,
        client_factory: Callable[[], Any],
        ttl_seconds: float = 300,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._client_factory = client_factory
        self._client: Any = None
        self._ttl = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        # key -> (prompt, expires_at)
        self._cache: dict[tuple[str, str], tuple[PromptVersion, float]] = {}

    def fetch(
        self,
        prompt_id: str,
        version: str = DEFAULT_VERSION,
        *,
        label: str = "prompt id",
        name: str = "",
    ) -> PromptVersion:
        version = (version or "").strip() or DEFAULT_VERSION
        key = (prompt_id, version)
        with self._lock:
            now = self._clock()
            cached = self._cache.get(key)
            if cached is not None and now < cached[1]:
                return cached[0]
            try:
                prompt = self._fetch(prompt_id, version, label, name)
            except Exception as exc:
                if cached is None:
                    raise
                logger.warning(
                    "prompt %s (version %s) refresh failed (%s: %s); serving stale text",
                    prompt_id,
                    version,
                    type(exc).__name__,
                    getattr(exc, "code", exc),
                )
                self._cache[key] = (cached[0], now + min(self._ttl, STALE_RETRY_SECONDS))
                return cached[0]
            self._cache[key] = (prompt, now + self._ttl)
            return prompt

    def get(self, prompt_id: str, version: str = DEFAULT_VERSION, *, label: str = "prompt id"):
        """Prompt text only."""
        return self.fetch(prompt_id, version, label=label).content

    def _fetch(self, prompt_id: str, version: str, label: str, name: str) -> PromptVersion:
        if self._client is None:
            self._client = self._client_factory()
        try:
            response = self._client.get_prompt(promptIdentifier=prompt_id, promptVersion=version)
        except ClientError as exc:
            _log_get_prompt_failure(exc, name or label, prompt_id, version)
            mapped = _fetch_error(exc.response.get("Error", {}).get("Code", ""), label, prompt_id)
            if mapped is None:
                raise  # e.g. ExpiredToken: handled by the shared Bedrock error mapping
            status, code, message = mapped
            raise PromptError(code, message, status) from exc
        except NoCredentialsError:
            raise  # handled by the shared Bedrock error mapping (AWS_CREDENTIALS_MISSING)
        except BotoCoreError as exc:
            raise PromptError(
                "PROMPT_UNAVAILABLE",
                f"Could not reach Bedrock Prompt Management ({type(exc).__name__}).",
                503,
            ) from exc
        text = _template_text(response, label, prompt_id)
        prompt = PromptVersion(str(response.get("version") or version), text, name)
        logger.info(
            "prompt '%s' v%s loaded from Bedrock (hash=%s)",
            name or prompt_id,
            prompt.version,
            prompt.hash,
        )
        return prompt


def _template_text(response: dict, label: str, prompt_id: str) -> str:
    variants = response.get("variants") or []
    chosen = next((v for v in variants if v.get("name") == response.get("defaultVariant")), None)
    variant = chosen or (variants[0] if variants else None)
    text = ((variant or {}).get("templateConfiguration") or {}).get("text", {}).get("text")
    if not text:
        raise PromptError(
            "PROMPT_INVALID",
            f"Prompt {label}={prompt_id!r} has no TEXT template "
            "(variants[0].templateConfiguration.text.text). Create it as a text prompt.",
        )
    return text


# ----------------------------------------------------------------- registry


class PromptRegistry:
    """Logical name -> `PromptVersion`, resolved from env configuration through a provider."""

    def __init__(self, settings: Settings | None = None, provider: PromptProvider | None = None):
        self._settings = settings
        self._provider = provider

    @property
    def settings(self) -> Settings:
        return self._settings or get_settings()

    @property
    def provider(self) -> PromptProvider:
        return self._provider or default_provider(self.settings)

    def prompt_id(self, name: str) -> str:
        """The configured Bedrock prompt id/ARN for `name` ('' when not configured)."""
        if name not in _PROMPT_ID_MAP:
            raise KeyError(f"Unknown prompt '{name}'. Available prompts: {sorted(_PROMPT_ID_MAP)}")
        return (getattr(self.settings, _PROMPT_ID_MAP[name]) or "").strip()

    def _require_id(self, name: str) -> str:
        prompt_id = self.prompt_id(name)
        if not prompt_id:
            label = prompt_env_var(name)
            raise PromptError(
                "PROMPT_ID_MISSING",
                f"{label} is not set. Create the prompt in Amazon Bedrock Prompt Management "
                f"and set {label} (see docs/prompts/README.md).",
                503,
            )
        return prompt_id

    def get(self, name: str) -> PromptVersion:
        """The active version (BEDROCK_PROMPT_VERSION) of prompt `name`."""
        prompt_id = self._require_id(name)
        return self.provider.fetch(
            prompt_id,
            self.settings.bedrock_prompt_version,
            label=prompt_env_var(name),
            name=name,
        )

    async def aget(self, name: str) -> PromptVersion:
        # boto3 is blocking: keep it off the event loop.
        return await asyncio.to_thread(self.get, name)

    def source(self, name: str) -> "RegistryPrompt":
        """A `PromptSource` for `name`; raises PROMPT_ID_MISSING now, makes no network call."""
        self._require_id(name)
        return RegistryPrompt(self, name)

    def get_version_metadata(self, name: str) -> dict[str, str]:
        """`{version, hash}` for audit records."""
        v = self.get(name)
        return {"version": v.version, "hash": v.hash}

    def get_all_current_versions(self) -> dict[str, str]:
        """`{name: version}` for every configured prompt that can be fetched."""
        result: dict[str, str] = {}
        for name in _PROMPT_ID_MAP:
            if not self.prompt_id(name):
                continue
            try:
                result[name] = self.get(name).version
            except Exception as exc:
                logger.warning("could not get the version of prompt '%s': %s", name, exc)
        return result

    def list_prompts(self) -> list[dict]:
        """Metadata of every configured prompt (errors are reported per entry)."""
        result = []
        for name in _PROMPT_ID_MAP:
            prompt_id = self.prompt_id(name)
            if not prompt_id:
                continue
            try:
                pv = self.get(name)
                result.append(
                    {
                        "name": name,
                        "prompt_id": prompt_id,
                        "current_version": pv.version,
                        "current_hash": pv.hash,
                    }
                )
            except Exception as exc:
                result.append({"name": name, "prompt_id": prompt_id, "error": str(exc)})
        return result


class RegistryPrompt:
    """`PromptSource` backed by a registry entry (read through the cache on every call)."""

    def __init__(self, registry: PromptRegistry, name: str):
        self.registry = registry
        self.name = name

    def get(self) -> PromptVersion:
        return self.registry.get(self.name)

    async def aget(self) -> PromptVersion:
        return await self.registry.aget(self.name)


# ------------------------------------------------------------------- wiring


def build_agent_client(settings: Settings):
    """`bedrock-agent` client with the same explicit credentials/region as the other clients."""
    if missing := missing_credential_vars(settings):
        raise PromptError(
            "AWS_CREDENTIALS_MISSING",
            f"Incomplete AWS credentials: {', '.join(missing)} is not set.",
            503,
        )
    return boto3.client(
        "bedrock-agent",
        region_name=settings.aws_region,
        config=Config(
            connect_timeout=CONNECT_TIMEOUT_SECONDS,
            read_timeout=READ_TIMEOUT_SECONDS,
            retries={"max_attempts": MAX_RETRIES, "mode": "standard"},
        ),
        **explicit_credentials(settings),
    )


_default_provider: PromptProvider | None = None
_registry: PromptRegistry | None = None
_default_lock = threading.Lock()


def default_provider(settings: Settings | None = None) -> PromptProvider:
    """Process-wide provider (shared cache); builds no client until the first fetch."""
    global _default_provider
    with _default_lock:
        if _default_provider is None:
            settings = settings or get_settings()
            _default_provider = PromptProvider(
                lambda: build_agent_client(settings), settings.prompt_cache_ttl_seconds
            )
        return _default_provider


def get_prompt_registry() -> PromptRegistry:
    """Module-level singleton registry (settings from the environment)."""
    global _registry
    with _default_lock:
        if _registry is None:
            _registry = PromptRegistry()
        return _registry


def reset_prompt_registry() -> None:
    """Drop the singleton and the prompt cache to force a reload from Bedrock."""
    global _registry, _default_provider
    with _default_lock:
        _registry = None
        _default_provider = None


def chat_prompt_source(settings: Settings, provider: PromptProvider | None = None) -> PromptSource:
    return PromptRegistry(settings, provider or default_provider(settings)).source("chat")


def brief_prompt_source(settings: Settings, provider: PromptProvider | None = None) -> PromptSource:
    return PromptRegistry(settings, provider or default_provider(settings)).source("brief")
