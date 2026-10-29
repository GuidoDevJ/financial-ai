"""Mapping of Bedrock / AWS failures to HTTP errors, shared by /chat and /brief/preview."""

import logging

from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import HTTPException

from app.prompts.registry import PromptError

logger = logging.getLogger(__name__)

THROTTLE_RETRY_AFTER_SECONDS = 5

_EXPIRED_MESSAGE = (
    "The AWS temporary credentials have expired. Refresh AWS_ACCESS_KEY_ID, "
    "AWS_SECRET_ACCESS_KEY and AWS_SESSION_TOKEN and restart the service."
)

# Bedrock ClientError code -> (HTTP status, API error code, message)
BEDROCK_ERRORS: dict[str, tuple[int, str, str]] = {
    "AccessDeniedException": (
        403,
        "BEDROCK_ACCESS_DENIED",
        "Access denied by Bedrock. Check model access for BEDROCK_MODEL_ID in this region "
        "and the IAM permission bedrock:InvokeModel.",
    ),
    "ExpiredTokenException": (503, "AWS_CREDENTIALS_EXPIRED", _EXPIRED_MESSAGE),
    "ExpiredToken": (503, "AWS_CREDENTIALS_EXPIRED", _EXPIRED_MESSAGE),
    "ThrottlingException": (429, "BEDROCK_THROTTLED", "Bedrock is throttling requests; retry."),
    "ValidationException": (502, "BEDROCK_VALIDATION", "Bedrock rejected the request."),
    "ModelNotReadyException": (503, "BEDROCK_UNAVAILABLE", "The Bedrock model is not ready."),
    "ServiceUnavailableException": (503, "BEDROCK_UNAVAILABLE", "Bedrock is unavailable."),
    "ModelTimeoutException": (504, "BEDROCK_TIMEOUT", "The Bedrock model timed out."),
}


# Codes worth retrying with backoff (the call may succeed a moment later).
TRANSIENT_BEDROCK_CODES = frozenset(
    {
        "ThrottlingException",
        "ServiceUnavailableException",
        "ModelTimeoutException",
        "InternalServerException",
    }
)


def is_transient_bedrock_error(exc: BaseException) -> bool:
    """True for a botocore ClientError whose code is transient (also through `__cause__`)."""
    seen = 0
    while exc is not None and seen < 5:
        if isinstance(exc, ClientError):
            return exc.response.get("Error", {}).get("Code", "") in TRANSIENT_BEDROCK_CODES
        exc = exc.__cause__ or exc.__context__
        seen += 1
    return False


def bedrock_http_error(exc: ClientError) -> HTTPException:
    code = exc.response.get("Error", {}).get("Code", "")
    status, error, message = BEDROCK_ERRORS.get(
        code, (502, "LLM_ERROR", f"{type(exc).__name__}: {exc}")
    )
    headers = {"Retry-After": str(THROTTLE_RETRY_AFTER_SECONDS)} if status == 429 else None
    detail = {"error": error, "message": message}
    if code and error != "LLM_ERROR":
        detail["bedrock_code"] = code
    return HTTPException(status_code=status, detail=detail, headers=headers)


def llm_exception_to_http(exc: Exception) -> HTTPException:
    """Map any failure raised while calling the LLM to the API error contract."""
    if isinstance(exc, PromptError):
        logger.warning("Prompt Management failure: %s %s", exc.code, exc.message)
        return HTTPException(
            status_code=exc.status, detail={"error": exc.code, "message": exc.message}
        )
    if isinstance(exc, NoCredentialsError):
        return HTTPException(
            status_code=503, detail={"error": "AWS_CREDENTIALS_MISSING", "message": str(exc)}
        )
    if isinstance(exc, ClientError):
        logger.warning("Bedrock call failed: %s", exc)
        return bedrock_http_error(exc)
    logger.error("LLM call failed", exc_info=exc)
    return HTTPException(
        status_code=502,
        detail={"error": "LLM_ERROR", "message": f"{type(exc).__name__}: {exc}"},
    )
