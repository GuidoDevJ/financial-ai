"""get_last_brief tool: latest (or dated) daily brief stored for a client."""

import logging

import redis

from app.brief.store import BriefStore, validate_client_id, validate_fecha
from app.core.audit import annotate, audited_tool, log_event
from app.tools.errors import ToolError


def make_get_last_brief(*, store: BriefStore):
    @audited_tool
    def get_last_brief(cliente_id: str, fecha: str | None = None) -> dict:
        try:
            validate_client_id(cliente_id)
            if fecha:
                validate_fecha(fecha)
        except ValueError as exc:
            log_event(
                "brief_tool",
                "last_brief",
                level=logging.WARNING,
                cliente_id=cliente_id,
                found=None,
                reason="invalid_params",
            )
            raise ToolError("INVALID_PARAMS", str(exc)) from exc
        try:
            brief = store.get(cliente_id, fecha) if fecha else store.latest(cliente_id)
        except redis.RedisError as exc:
            log_event(
                "brief_tool",
                "storage_error",
                level=logging.ERROR,
                exc_info=True,
                cliente_id=cliente_id,
                error_type=type(exc).__name__,
            )
            raise ToolError("STORAGE_UNAVAILABLE", f"Brief storage unavailable: {exc}") from exc
        annotate(source="redis")
        if brief is None:
            log_event(
                "brief_tool",
                "last_brief",
                level=logging.WARNING,
                cliente_id=cliente_id,
                fecha=fecha,
                found=False,
            )
            when = f"for {fecha}" if fecha else "yet"
            raise ToolError("NO_BRIEF_FOUND", f"No brief generated {when} for client {cliente_id}.")
        assets = brief["activos_mencionados"]
        annotate(result_size=len(assets))
        # Ids, dates and sizes only: the brief text itself is never logged.
        log_event(
            "brief_tool",
            "last_brief",
            cliente_id=cliente_id,
            fecha=brief["fecha"],
            found=True,
            assets=len(assets),
            text_chars=len(brief["texto_brief"]),
        )
        return {
            "fecha": brief["fecha"],
            "texto_brief": brief["texto_brief"],
            "activos_mencionados": brief["activos_mencionados"],
        }

    return get_last_brief
