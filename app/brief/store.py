"""Brief storage on Redis. The brief job writes here; get_last_brief reads from here.

Layout:
  brief:{cliente_id}:{fecha}   JSON {"fecha","texto_brief","activos_mencionados"} plus the
                               optional "html", "degraded", "missing_inputs", "generated_at",
                               "prompt" ({"name","version","hash"} of the system prompt used)
  briefidx:{cliente_id}        sorted set, member = fecha, score = date ordinal (latest lookup)
"""

import json
import re
from datetime import date

_CLIENT_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_client_id(cliente_id: str) -> str:
    if not isinstance(cliente_id, str) or not _CLIENT_RE.fullmatch(cliente_id):
        raise ValueError("cliente_id must match [A-Za-z0-9_-]{1,64}.")
    return cliente_id


def validate_fecha(fecha: str) -> date:
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", fecha):
            raise ValueError
        return date.fromisoformat(fecha)
    except (TypeError, ValueError):
        raise ValueError("fecha must be a valid YYYY-MM-DD date.") from None


class BriefStore:
    def __init__(self, redis_client) -> None:
        self.redis = redis_client

    @staticmethod
    def _key(cliente_id: str, fecha: str) -> str:
        return f"brief:{cliente_id}:{fecha}"

    @staticmethod
    def _index(cliente_id: str) -> str:
        return f"briefidx:{cliente_id}"

    def save(
        self,
        cliente_id: str,
        fecha: str,
        texto_brief: str,
        activos_mencionados: list[str],
        *,
        html: str | None = None,
        degraded: bool | None = None,
        missing_inputs: list[str] | None = None,
        generated_at: str | None = None,
        prompt: dict[str, str] | None = None,
    ) -> None:
        validate_client_id(cliente_id)
        day = validate_fecha(fecha)
        payload: dict = {
            "fecha": fecha,
            "texto_brief": texto_brief,
            "activos_mencionados": list(activos_mencionados),
        }
        optional = {
            "html": html,
            "degraded": degraded,
            "missing_inputs": list(missing_inputs) if missing_inputs is not None else None,
            "generated_at": generated_at,
            "prompt": dict(prompt) if prompt is not None else None,
        }
        payload.update({k: v for k, v in optional.items() if v is not None})
        self.redis.set(self._key(cliente_id, fecha), json.dumps(payload, ensure_ascii=False))
        self.redis.zadd(self._index(cliente_id), {fecha: day.toordinal()})

    def get(self, cliente_id: str, fecha: str) -> dict | None:
        validate_client_id(cliente_id)
        validate_fecha(fecha)
        raw = self.redis.get(self._key(cliente_id, fecha))
        return json.loads(raw) if raw is not None else None

    def exists(self, cliente_id: str, fecha: str) -> bool:
        validate_client_id(cliente_id)
        validate_fecha(fecha)
        return bool(self.redis.exists(self._key(cliente_id, fecha)))

    def latest(self, cliente_id: str) -> dict | None:
        """Most recent brief by date; index entries whose key vanished are skipped."""
        validate_client_id(cliente_id)
        for fecha in self.redis.zrevrange(self._index(cliente_id), 0, -1):
            brief = self.get(cliente_id, fecha)
            if brief is not None:
                return brief
        return None
