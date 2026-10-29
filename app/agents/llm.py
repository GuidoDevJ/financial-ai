"""Amazon Bedrock chat model factory and configuration checks."""

import re

import boto3
from botocore.config import Config
from langchain_aws import ChatBedrockConverse

from app.core.aws import explicit_credentials, missing_credential_vars
from app.core.config import Settings

READ_TIMEOUT_SECONDS = 30
CONNECT_TIMEOUT_SECONDS = 5
MAX_RETRIES = 3


class LLMConfigError(Exception):
    """The Bedrock LLM is not usable with the current configuration."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _resolve_credentials():
    """Credentials from the default boto3 chain (env vars, shared files, IAM role)."""
    return boto3.Session().get_credentials()


def ensure_llm_configured(settings: Settings) -> None:
    """Raise LLMConfigError if the model id, region or AWS credentials are missing.

    Credentials rule: both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY set -> used explicitly;
    exactly one set -> error; neither set -> the default boto3 chain must find some.
    """
    if not settings.bedrock_model_id:
        raise LLMConfigError("BEDROCK_MODEL_ID_MISSING", "BEDROCK_MODEL_ID is not set.")
    if not settings.aws_region:
        raise LLMConfigError("AWS_REGION_MISSING", "AWS_REGION is not set.")
    if missing := missing_credential_vars(settings):
        raise LLMConfigError(
            "AWS_CREDENTIALS_MISSING",
            f"Incomplete AWS credentials: {', '.join(missing)} is not set "
            "(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set together).",
        )
    if explicit_credentials(settings):
        return
    try:
        credentials = _resolve_credentials()
    except Exception as exc:  # malformed shared config, etc.
        raise LLMConfigError(
            "AWS_CREDENTIALS_MISSING", f"AWS credentials lookup failed: {type(exc).__name__}"
        ) from exc
    if credentials is None:
        raise LLMConfigError(
            "AWS_CREDENTIALS_MISSING",
            "No AWS credentials found: set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
            "(plus AWS_SESSION_TOKEN for temporary credentials) or use an IAM role.",
        )


_METADATA_FORBIDDEN = re.compile(r"[^a-zA-Z0-9\s:_@$#=/+,.\-]")
_METADATA_MAX_LEN = 256


def _clean_metadata_value(value: str) -> str:
    """Bedrock requestMetadata values allow a restricted charset and at most 256 chars."""
    return _METADATA_FORBIDDEN.sub("_", str(value))[:_METADATA_MAX_LEN]


def build_request_metadata(
    base: dict[str, str], cliente_id: str, session_id: str
) -> dict[str, str]:
    """Converse requestMetadata: app tags plus clienteId/sessionId only (no other PII).

    Keys are camelCase on purpose: langchain-aws snake->camel converts the keys of per-call
    kwargs, which would silently rename `cliente_id` to `clienteId` anyway.
    """
    metadata = {**base, "clienteId": cliente_id, "sessionId": session_id}
    return {k: _clean_metadata_value(v) for k, v in metadata.items()}


def build_chat_model(
    settings: Settings, client=None, *, for_chat: bool = False
) -> ChatBedrockConverse:
    """ChatBedrockConverse with configured inferenceConfig, timeouts and retries.

    `client` lets tests inject a fake `bedrock-runtime` client (no AWS access needed).
    `for_chat` selects the interactive profile: the optional BEDROCK_CHAT_MODEL_ID plus a
    tighter read timeout and fewer attempts so a turn fits in CHAT_TIMEOUT_SECONDS. The brief
    keeps the longer defaults.
    """
    kwargs = dict(explicit_credentials(settings))
    if client is not None:
        kwargs["client"] = client
    if for_chat:
        model_id = settings.bedrock_chat_model_id or settings.bedrock_model_id
        read_timeout = settings.bedrock_chat_read_timeout_seconds
        # botocore counts the first try, so max_attempts here is "total attempts".
        retries = {"total_max_attempts": settings.bedrock_chat_max_attempts, "mode": "standard"}
    else:
        model_id = settings.bedrock_model_id
        read_timeout = READ_TIMEOUT_SECONDS
        retries = {"max_attempts": MAX_RETRIES, "mode": "standard"}
    return ChatBedrockConverse(
        model=model_id,
        region_name=settings.aws_region,
        temperature=settings.bedrock_temperature,
        max_tokens=settings.bedrock_max_tokens,
        config=Config(
            read_timeout=read_timeout,
            connect_timeout=CONNECT_TIMEOUT_SECONDS,
            retries=retries,
        ),
        **kwargs,
    )
