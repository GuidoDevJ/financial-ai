"""Deterministic rendering of the daily brief: es-AR formatting, price table, HTML.

The LLM only writes the narrative. The table and the disclaimer are built here from the
resolved data, so no number in the table can ever be hallucinated.
"""

import re
from dataclasses import dataclass
from html import escape

DISCLAIMER = "Información generada automáticamente, no constituye una recomendación de inversión."
NA = "n/d"
TABLE_HEADER = ("Ticker", "Precio", "1D", "5D", "Mes")

_LABEL_RE = re.compile(r"^\s*\*\*([^*\n]+)\*\*[ \t]*", re.MULTILINE)


@dataclass(frozen=True)
class AssetRow:
    """One held asset with its resolved market data (None = unavailable)."""

    ticker: str
    moneda: str  # "USD" | "ARS"
    mercado: str  # "nyse" | "byma"
    peso_pct: float | None
    precio: float | None
    fecha_precio: str | None
    fuente: str | None
    var_1d: float | None
    var_5d: float | None
    var_mes: float | None


def fmt_number(value: float, decimals: int = 2) -> str:
    """es-AR: '.' for thousands, ',' for decimals."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".")


def fmt_pct(value: float | None) -> str:
    if value is None:
        return NA
    sign = "+" if value > 0 else ""
    return f"{sign}{fmt_number(value)}%"


def fmt_price(value: float | None, moneda: str) -> str:
    if value is None:
        return NA
    prefix = "US$" if moneda == "USD" else "$"
    return f"{prefix} {fmt_number(value)}"


def render_table(rows: list[AssetRow]) -> str:
    lines = [
        "| " + " | ".join(TABLE_HEADER) + " |",
        "| " + " | ".join("---" for _ in TABLE_HEADER) + " |",
    ]
    for r in rows:
        cells = (
            r.ticker,
            fmt_price(r.precio, r.moneda),
            fmt_pct(r.var_1d),
            fmt_pct(r.var_5d),
            fmt_pct(r.var_mes),
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compose_text(narrative: str, rows: list[AssetRow]) -> str:
    parts = [narrative.strip(), render_table(rows), DISCLAIMER]
    return "\n\n".join(p for p in parts if p)


def parse_items(narrative: str) -> tuple[str, list[tuple[str, str]]]:
    """Split '**LABEL** body' items. Returns (text before the first label, [(label, body)])."""
    text = narrative.strip()
    matches = list(_LABEL_RE.finditer(text))
    if not matches:
        return text, []
    preamble = text[: matches[0].start()].strip()
    items = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        items.append((m.group(1).strip(), text[m.end() : end].strip()))
    return preamble, items


_STYLE = (
    "body{font-family:Arial,Helvetica,sans-serif;color:#1a1a1a;max-width:640px;margin:0 auto;"
    "padding:16px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;"
    "padding:6px 8px;text-align:right}th:first-child,td:first-child{text-align:left}"
    "p{line-height:1.45}.disclaimer{color:#666;font-size:12px;margin-top:24px}"
)


def render_html(fecha: str, narrative: str, rows: list[AssetRow], degraded: bool) -> str:
    """Self-contained HTML (inline CSS, no external assets). Every dynamic string is escaped."""
    preamble, items = parse_items(narrative)
    body = []
    if preamble:
        body.append(f"<p>{escape(preamble)}</p>")
    for label, text in items:
        body.append(
            f"<p><strong>{escape(label)}</strong> {escape(text).replace(chr(10), '<br>')}</p>"
        )
    if degraded:
        body.append("<p><em>Resumen narrativo no disponible; se muestran solo los datos.</em></p>")
    head = "".join(f"<th>{escape(h)}</th>" for h in TABLE_HEADER)
    trs = []
    for r in rows:
        cells = (
            r.ticker,
            fmt_price(r.precio, r.moneda),
            fmt_pct(r.var_1d),
            fmt_pct(r.var_5d),
            fmt_pct(r.var_mes),
        )
        trs.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in cells) + "</tr>")
    return (
        '<!doctype html><html lang="es"><head><meta charset="utf-8">'
        f"<title>Brief diario {escape(fecha)}</title><style>{_STYLE}</style></head><body>"
        f"<h1>Brief diario - {escape(fecha)}</h1>"
        + "".join(body)
        + f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(trs)}</tbody></table>"
        f'<p class="disclaimer">{escape(DISCLAIMER)}</p></body></html>'
    )
