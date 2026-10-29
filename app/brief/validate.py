"""Deterministic validation of the generated narrative (no LLM involved).

Rules: every number must be traceable to the data given to the model, no buy/sell
recommendation phrases, bold-labelled items within size limits, no table or disclaimer written
by the model, and (on the final text) the fixed disclaimer must close the brief.
"""

import re
import unicodedata
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from app.brief.render import DISCLAIMER, parse_items

MIN_ITEMS, MAX_ITEMS = 1, 2
ITEM_MIN_CHARS, ITEM_MAX_CHARS, ITEM_MAX_LINES = 40, 700, 4
FINAL_MAX_CHARS = 5000

_LABEL_OK = re.compile(r"^[A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ ]{1,29}$")

# es-AR grouped numbers (1.612 / 1.612,50) or plain decimals (3,1 / 3.1) not glued to letters.
_NUMBER_RE = re.compile(r"(?<!\w)(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)(?!\w)")
_THOUSANDS_DOT = re.compile(r"\d{1,3}(\.\d{3})+")
_THOUSANDS_COMMA = re.compile(r"\d{1,3}(,\d{3})+")

# Matched on lower-cased text. `_ACCENTED` keeps accents (voseo imperatives); `_PHRASES` runs on
# accent-stripped text. Neutral uses such as "la compra de dólares" must not match.
_ACCENTED = [
    r"\bcomprá\b",
    r"\bvendé\b",
    r"\bmantené\b",
    r"\bcomprate\b",
    r"\bvendete\b",
    r"\baprovechá\b",
    r"\bliquidá\b",
    r"\bsalí de\b",
    r"\btomá ganancias\b",
    r"\bdolarizá\b",
    r"\bpesificá\b",
    r"\b(?:aumentá|reducí|incrementá|disminuí)\s+tu\s+(?:exposición|posición|tenencia)\b",
]
_ACTION = (
    r"(?:comprar|vender|mantener|aumentar|reducir|rotar|liquidar|acumular|incorporar|"
    r"tomar ganancias|salir de|entrar en|dolarizar|pesificar)(?:l[oa]s?)?"
)
_PHRASES = [
    rf"\b(?:conviene|recomendable|aconsejable|hay que|habria que)\s+{_ACTION}\b",
    r"\bte (?:recomiendo|recomendamos|sugiero|sugerimos|aconsejo|aconsejamos|conviene)\b",
    rf"\b(?:recomendamos|sugerimos|aconsejamos|recomiendo|sugiero|aconsejo)\s+{_ACTION}\b",
    rf"\b(?:deberias|deberia|debes|debe|tenes que|tienes que|tendrias que|podrias)\s+"
    rf"(?:considerar\s+|evaluar\s+)?{_ACTION}\b",
    r"\bmomento\s+(?:para|de)\s+(?:comprar|vender|mantener|compra|venta|entrar|salir)\b",
    r"\boportunidad\s+de\s+(?:compra|venta|comprar|vender)\b",
    r"\brecomendacion\s+de\s+(?:compra|venta)\b",
    r"\b(?:sobreponderar|infraponderar|subponderar)\b",
    r"\b(?:precio objetivo|target price)\b",
    # Conjugations, disguised and role-play phrasings (Spanish).
    r"\bmi (?:recomendacion|consejo|sugerencia) es\b",
    r"\ben tu lugar\b[^.!?\n]{0,40}"
    r"\b(?:comprar\w*|vender\w*|mantendr\w*|liquidar\w*|vendo|compro)\b",
    r"\byo (?:\w+ ){0,2}(?:comprari\w+|venderi\w+|mantendri\w+|liquidari\w+|invertiri\w+)\b",
    rf"\baprovech\w*\s+(?:para\s+)?{_ACTION}\b",
    r"\bvale la pena\s+(?:comprar|vender|mantener|invertir en)\b",
    r"\blo mejor es\s+(?:comprar|vender|mantener)\b",
    r"\b(?:compral[oa]s?|vendel[oa]s?|mantenel[oa]s?)\b",
    r"\bno\s+(?:vendas|compres|vendan|compren)\b",
    r"\b(?:dolarizate|pesificate|cubrite|pasate a)\b",
    # English.
    r"\byou should (?:buy|sell|hold|purchase|dump|short|invest)\b",
    r"\bi (?:would |will |'d )?(?:recommend|suggest|advise)\b[^.!?\n]{0,30}"
    r"\b(?:buy|sell|hold|purchas\w*|buying|selling|holding)\b",
    r"\b(?:buy|sell)\s+(?:now|today|immediately)\b",
    r"\bi(?:'d| would) (?:buy|sell|hold)\b",
    r"\bconsider (?:buying|selling)\b",
    r"\bgood (?:time|moment|opportunity) to (?:buy|sell)\b",
    r"\b(?:smart|wise|better|good idea) to (?:buy|sell|hold)\b",
]

# A recommendation phrase inside a clause that denies the ability to give one ("no puedo decirte
# que te conviene comprar", "esto no es una recomendación de compra") is a refusal, not advice.
_DENIAL = re.compile(
    r"\b(?:no|nunca|jamas|tampoco)\s+"
    r"(?:(?:es|son|constituye|constituyen|implica|representa|puedo|podemos|podria|estoy|estamos|"
    r"tengo|tenemos|me|nos|doy|damos|brindo|brindamos|ofrezco|ofrecemos|emito|emitimos|"
    r"realizo|realizamos)\b|una?\s+recomendaci)"
    r"|\b(?:cannot|can't|can not|won't|will not|am not (?:able|allowed|authorized)|"
    r"unable to|not (?:able|allowed|permitted|authorized) to|do not|don't|is not|isn't|not a)\b"
)
_CLAUSE_BREAK = re.compile(r"[.;:!?\n,]")


@dataclass(frozen=True)
class NumberToken:
    raw: str
    candidates: tuple[tuple[float, int], ...]  # (value, written decimals)


def _candidates(raw: str) -> tuple[tuple[float, int], ...]:
    has_dot, has_comma = "." in raw, "," in raw
    if has_dot and has_comma:  # es-AR "1.612,50"
        return ((float(raw.replace(".", "").replace(",", ".")), len(raw.split(",")[1])),)
    if has_comma:
        decimal = (float(raw.replace(",", ".")), len(raw.split(",")[1]))
        if _THOUSANDS_COMMA.fullmatch(raw):  # "1,234" may be en-US thousands
            return ((float(raw.replace(",", "")), 0), decimal)
        return (decimal,)
    if has_dot:
        if _THOUSANDS_DOT.fullmatch(raw):
            thousands = (float(raw.replace(".", "")), 0)
            if raw.count(".") == 1:  # "3.100" may be a decimal too
                return (thousands, (float(raw), len(raw.split(".")[1])))
            return (thousands,)
        return ((float(raw), len(raw.split(".")[1])),)
    return ((float(raw), 0),)


def extract_numbers(text: str) -> list[NumberToken]:
    return [NumberToken(m.group(0), _candidates(m.group(0))) for m in _NUMBER_RE.finditer(text)]


def allowed_values(data) -> set[float]:
    """Absolute values of every number in the data, including numbers inside strings."""
    found: set[float] = set()

    def walk(node) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, int | float):
            found.add(abs(float(node)))
        elif isinstance(node, str):
            for token in extract_numbers(node):
                found.update(value for value, _ in token.candidates)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list | tuple):
            for value in node:
                walk(value)

    walk(data)
    return found


def _traceable(token: NumberToken, allowed: set[float]) -> bool:
    for value, decimals in token.candidates:
        quantum = Decimal(1).scaleb(-decimals)
        for a in allowed:
            rounded = Decimal(str(a)).quantize(quantum, rounding=ROUND_HALF_UP)
            if abs(float(rounded) - value) < 1e-9:
                return True
    return False


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def find_recommendations(text: str) -> list[str]:
    lowered = text.lower()
    hits = [m.group(0) for p in _ACCENTED for m in re.finditer(p, lowered)]
    plain = _strip_accents(lowered)
    hits += [m.group(0) for p in _PHRASES for m in re.finditer(p, plain)]
    return hits


def find_recommendations_outside_refusals(text: str) -> list[str]:
    """Like `find_recommendations`, ignoring phrases inside a clause that denies giving advice.

    Used on free-form chat replies, where "No puedo decirte que te conviene comprar" is a valid
    refusal. A comma, period or other clause break ends the denial, so "No puedo asegurarlo,
    pero comprá GGAL" is still caught.
    """
    lowered = text.lower()
    plain = _strip_accents(lowered)
    spans = [(m.start(), m.group(0)) for p in _ACCENTED for m in re.finditer(p, lowered)]
    spans += [(m.start(), m.group(0)) for p in _PHRASES for m in re.finditer(p, plain)]
    hits = []
    for start, phrase in sorted(spans):
        breaks = list(_CLAUSE_BREAK.finditer(plain, 0, start))
        # The clause up to and including the phrase: "no una recomendacion de compra" denies.
        clause = plain[breaks[-1].end() if breaks else 0 : start + len(phrase)]
        if not _DENIAL.search(clause):
            hits.append(phrase)
    return hits


def untraceable_numbers(text: str, data) -> list[str]:
    """Numbers written in `text` that equal no number in `data` (same rule as the brief).

    Single-digit integers are ignored: they are list numbering or counts, not market data.
    """
    allowed = allowed_values(data)
    found: list[str] = []
    for token in extract_numbers(text):
        small_int = token.raw.isdigit() and len(token.raw) == 1
        if token.raw not in found and not small_int and not _traceable(token, allowed):
            found.append(token.raw)
    return found


def validate_narrative(narrative: str, data) -> list[str]:
    """Return the list of violations (empty = valid)."""
    text = (narrative or "").strip()
    if not text:
        return ["narrative is empty"]
    violations: list[str] = []
    if DISCLAIMER in text:
        violations.append("narrative must not include the disclaimer (it is added by the system)")
    if re.search(r"^\s*\|", text, re.MULTILINE):
        violations.append("narrative must not include a table (it is added by the system)")

    preamble, items = parse_items(text)
    if preamble:
        violations.append("text found before the first bold label; every item needs a **LABEL**")
    if not (MIN_ITEMS <= len(items) <= MAX_ITEMS):
        violations.append(
            f"expected {MIN_ITEMS}-{MAX_ITEMS} items with a bold label, got {len(items)}"
        )
    for label, body in items:
        if not _LABEL_OK.fullmatch(label):
            violations.append(f"label '{label}' must be upper case letters (e.g. TU CARTERA)")
        lines = [ln for ln in body.splitlines() if ln.strip()]
        if len(body) < ITEM_MIN_CHARS:
            violations.append(
                f"item '{label}' is too short ({len(body)} chars, min {ITEM_MIN_CHARS})"
            )
        if len(body) > ITEM_MAX_CHARS:
            violations.append(
                f"item '{label}' is too long ({len(body)} chars, max {ITEM_MAX_CHARS})"
            )
        if len(lines) > ITEM_MAX_LINES:
            violations.append(f"item '{label}' has {len(lines)} lines (max {ITEM_MAX_LINES})")

    allowed = allowed_values(data)
    seen: set[str] = set()
    for token in extract_numbers(text):
        if token.raw not in seen and not _traceable(token, allowed):
            seen.add(token.raw)
            violations.append(f"number not traceable to the input data: {token.raw}")

    for phrase in dict.fromkeys(find_recommendations(text)):
        violations.append(f"buy/sell recommendation phrase not allowed: '{phrase}'")
    return violations


def validate_final_text(text: str) -> list[str]:
    violations = []
    if text.count(DISCLAIMER) != 1 or not text.rstrip().endswith(DISCLAIMER):
        violations.append("the fixed disclaimer must appear exactly once, at the end")
    if len(text) > FINAL_MAX_CHARS:
        violations.append(f"brief is too long ({len(text)} chars, max {FINAL_MAX_CHARS})")
    return violations
