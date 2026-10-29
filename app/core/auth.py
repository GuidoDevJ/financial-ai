"""API-key authentication that binds each request to one client (`cliente_id`).

`AUTH_API_KEYS` maps API keys to a cliente_id. The `X-API-Key` header selects the client; a
`cliente_id` in the request body must match it (403 CLIENT_MISMATCH otherwise) and tools read the
client from the request context, never from model-supplied arguments.

Without configured keys: APP_ENV=dev trusts the body's cliente_id (a startup warning says so);
any other environment answers 503 AUTH_NOT_CONFIGURED instead of running unauthenticated.
"""

import hmac
import logging
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header

from app.core.audit import log_event
from app.core.config import Settings, get_settings
from app.core.errors import api_error

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Principal:
    """The caller. `cliente_id` is None only in unauthenticated dev mode."""

    cliente_id: str | None
    authenticated: bool


def startup_warning(settings: Settings) -> str | None:
    """Text of the startup warning for an unsafe authentication setup (None when configured)."""
    if settings.api_key_map():
        return None
    if settings.app_env == "dev":
        return (
            "AUTH_API_KEYS is not set: authentication is DISABLED and the cliente_id in the "
            "request body is trusted (allowed only because APP_ENV=dev)."
        )
    return (
        f"AUTH_API_KEYS is not set and APP_ENV={settings.app_env}: /chat and /brief endpoints "
        "will answer 503 AUTH_NOT_CONFIGURED."
    )


def admin_startup_warning(settings: Settings) -> str | None:
    """Startup warning for the admin endpoints (None when ADMIN_API_KEYS is configured)."""
    if settings.admin_key_list():
        return None
    if settings.app_env == "dev":
        return (
            "ADMIN_API_KEYS is not set: admin endpoints (POST /news/ingest) accept any caller "
            "that passes client authentication (allowed only because APP_ENV=dev)."
        )
    return (
        f"ADMIN_API_KEYS is not set and APP_ENV={settings.app_env}: admin endpoints "
        "(POST /news/ingest) will answer 503 ADMIN_NOT_CONFIGURED."
    )


def _match_client(keys: dict[str, str], presented: str) -> str | None:
    """Constant-time lookup: every configured key is compared, with no early exit."""
    candidate = presented.encode("utf-8")
    matched: str | None = None
    for key, cliente_id in keys.items():
        if hmac.compare_digest(candidate, key.encode("utf-8")):
            matched = cliente_id
    return matched


def authenticate(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> Principal:
    settings = get_settings()
    keys = settings.api_key_map()
    if not keys:
        if settings.app_env == "dev":
            return Principal(cliente_id=None, authenticated=False)
        raise api_error(
            503,
            "AUTH_NOT_CONFIGURED",
            "Authentication is not configured on the server (AUTH_API_KEYS).",
        )
    if not x_api_key:
        raise api_error(401, "UNAUTHENTICATED", "Missing X-API-Key header.")
    cliente_id = _match_client(keys, x_api_key)
    if cliente_id is None:
        log_event("security", "invalid_api_key", level=logging.WARNING)
        raise api_error(401, "INVALID_API_KEY", "Invalid API key.")
    return Principal(cliente_id=cliente_id, authenticated=True)


def require_admin(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Gate for global (not per-client) operations: only `ADMIN_API_KEYS` holders pass.

    With no admin keys: APP_ENV=dev falls back to the regular client authentication, any other
    environment answers 503 ADMIN_NOT_CONFIGURED. A valid client key that is not an admin
    key gets 403 ADMIN_REQUIRED.
    """
    settings = get_settings()
    admin_keys = settings.admin_key_list()
    if not admin_keys:
        if settings.app_env == "dev":
            authenticate(x_api_key)
            return
        raise api_error(
            503,
            "ADMIN_NOT_CONFIGURED",
            "Admin access is not configured on the server (ADMIN_API_KEYS).",
        )
    if not x_api_key:
        raise api_error(401, "UNAUTHENTICATED", "Missing X-API-Key header.")
    presented = x_api_key.encode("utf-8")
    is_admin = False
    for key in admin_keys:  # compare every key: no early exit
        if hmac.compare_digest(presented, key.encode("utf-8")):
            is_admin = True
    if is_admin:
        return
    if _match_client(settings.api_key_map(), x_api_key) is not None:
        log_event("security", "admin_required", level=logging.WARNING)
        raise api_error(403, "ADMIN_REQUIRED", "This operation requires an admin API key.")
    log_event("security", "invalid_api_key", level=logging.WARNING)
    raise api_error(401, "INVALID_API_KEY", "Invalid API key.")


CurrentPrincipal = Annotated[Principal, Depends(authenticate)]


def resolve_cliente_id(principal: Principal, body_cliente_id: str | None) -> str:
    """The client this request acts for: the authenticated one, or the body's in dev mode."""
    if principal.cliente_id is None:
        if not body_cliente_id:
            raise api_error(422, "CLIENT_ID_REQUIRED", "cliente_id is required.")
        return body_cliente_id
    if body_cliente_id is not None and body_cliente_id != principal.cliente_id:
        log_event(
            "security",
            "client_mismatch",
            level=logging.WARNING,
            authenticated_cliente_id=principal.cliente_id,
            requested_cliente_id=body_cliente_id,
        )
        raise api_error(
            403, "CLIENT_MISMATCH", "cliente_id does not match the authenticated client."
        )
    return principal.cliente_id
