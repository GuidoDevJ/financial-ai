"""Output guardrail for /chat: the assistant must never emit an explicit buy/sell recommendation.

The system prompt asks for it; this deterministic check enforces it on the final reply. A
violating reply is replaced by a fixed refusal before it reaches the client or the conversation
history. Detection reuses the brief validator's phrase list (Spanish conjugations, voseo and tuteo
imperatives, disguised and English phrasings); phrases inside a clause that denies giving advice
("No puedo decirte que te conviene comprar") are treated as refusals and allowed.
"""

import logging
from dataclasses import dataclass

from app.brief.validate import find_recommendations_outside_refusals
from app.core.audit import log_event

SAFE_REFUSAL = (
    "No puedo dar asesoramiento personalizado ni recomendarte comprar o vender activos. "
    "Puedo mostrarte datos de mercado, tu cartera y el último brief, o derivarte a un "
    "asesor humano."
)
BLOCKED_BY_GUARDRAIL = "recommendation_blocked"


@dataclass(frozen=True)
class GuardrailVerdict:
    blocked: bool
    matches: tuple[str, ...] = ()


def check_reply(text: str) -> GuardrailVerdict:
    matches = tuple(dict.fromkeys(find_recommendations_outside_refusals(text or "")))
    return GuardrailVerdict(blocked=bool(matches), matches=matches)


def log_blocked(verdict: GuardrailVerdict, *, reply_chars: int) -> None:
    """Audit WARNING for a replaced reply (matched phrases and size, never the full reply)."""
    log_event(
        "guardrail",
        BLOCKED_BY_GUARDRAIL,
        level=logging.WARNING,
        matches=list(verdict.matches[:5]),
        reply_chars=reply_chars,
    )
