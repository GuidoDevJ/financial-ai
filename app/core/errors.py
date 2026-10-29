"""Consistent error contract: every non-2xx response is

    {"error": {"code": "...", "message": "...", "request_id": "..."}}

with the same id in the `X-Request-ID` header and in the audit logs.
"""

import logging
import re
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.audit import request_context, request_id_var

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_STATUS_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHENTICATED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    422: "VALIDATION_ERROR",
    429: "TOO_MANY_REQUESTS",
}
MAX_VALIDATION_DETAILS = 3


def api_error(
    status: int, code: str, message: str, headers: dict[str, str] | None = None, **extra: Any
) -> HTTPException:
    """HTTPException carrying the error contract (rendered by `install_error_handling`)."""
    return HTTPException(
        status_code=status, detail={"error": code, "message": message, **extra}, headers=headers
    )


def _request_id(request: Request) -> str:
    return (
        request_id_var.get() or getattr(request.state, "request_id", None) or uuid.uuid4().hex[:16]
    )


def _response(
    request: Request,
    status: int,
    code: str,
    message: str,
    headers: dict[str, str] | None = None,
    **extra: Any,
) -> JSONResponse:
    rid = _request_id(request)
    body = {"error": {"code": code, "message": message, "request_id": rid, **extra}}
    return JSONResponse(
        body, status_code=status, headers={**(headers or {}), REQUEST_ID_HEADER: rid}
    )


def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        extra = {k: v for k, v in detail.items() if k not in ("error", "message")}
        return _response(
            request,
            exc.status_code,
            str(detail["error"]),
            str(detail.get("message", "")),
            exc.headers,
            **extra,
        )
    code = _STATUS_CODES.get(exc.status_code, f"HTTP_{exc.status_code}")
    return _response(request, exc.status_code, code, str(detail), exc.headers)


def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Field paths and messages only: never the submitted values (they may be private text).
    problems = [
        f"{'.'.join(str(p) for p in err['loc'] if p != 'body')}: {err['msg']}"
        for err in exc.errors()[:MAX_VALIDATION_DETAILS]
    ]
    return _response(request, 422, "VALIDATION_ERROR", "Invalid request: " + "; ".join(problems))


def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error", exc_info=exc)
    return _response(request, 500, "INTERNAL_ERROR", "Internal server error.")


def install_error_handling(app: FastAPI) -> None:
    """Request-id middleware plus handlers for every error path."""

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        rid = incoming if _VALID_REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex[:16]
        request.state.request_id = rid
        with request_context(request_id=rid):
            response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = rid
        return response

    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(Exception, _unhandled_handler)
