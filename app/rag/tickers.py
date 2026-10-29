"""Ticker alias map used to tag news with the assets they mention.

Canonical symbol -> aliases (ticker variants and company names). Matching is case-insensitive
on whole words. Extend this map to cover more assets.
"""

import re

TICKER_ALIASES: dict[str, tuple[str, ...]] = {
    "GGAL": ("GGAL", "Galicia", "Grupo Financiero Galicia"),
    "YPF": ("YPF", "YPFD", "YPF S.A."),
    "PAMP": ("PAMP", "Pampa", "Pampa Energia", "Pampa Energía"),
    "BMA": ("BMA", "Banco Macro"),
    "MELI": ("MELI", "MercadoLibre", "Mercado Libre"),
    "BBAR": ("BBAR", "BBVA Argentina", "Banco Frances", "Banco Francés"),
    "SUPV": ("SUPV", "Supervielle", "Grupo Supervielle"),
    "CEPU": ("CEPU", "Central Puerto"),
    "TXAR": ("TXAR", "Ternium Argentina"),
    "VIST": ("VIST", "Vista Energy"),
}


def _compile(aliases: tuple[str, ...]) -> re.Pattern[str]:
    parts = sorted((re.escape(a) for a in aliases), key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)", re.IGNORECASE)


_PATTERNS = {symbol: _compile(aliases) for symbol, aliases in TICKER_ALIASES.items()}
_SYMBOL_BY_ALIAS = {
    alias.upper(): symbol for symbol, aliases in TICKER_ALIASES.items() for alias in aliases
}


def detect_tickers(text: str) -> list[str]:
    """Canonical symbols mentioned in `text`, sorted."""
    return sorted(symbol for symbol, pattern in _PATTERNS.items() if pattern.search(text))


def normalize_ticker(raw: str) -> str:
    """Map user/LLM input (ggal, YPFD.BA, 'Galicia') to the canonical symbol used in metadata."""
    value = (raw or "").strip().upper().removesuffix(".BA")
    return _SYMBOL_BY_ALIAS.get(value, value)
