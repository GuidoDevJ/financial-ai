import pytest

from app.brief.render import DISCLAIMER
from app.brief.validate import (
    extract_numbers,
    validate_final_text,
    validate_narrative,
)

DATA = {
    "fecha": "2026-10-15",
    "cartera": [
        {"ticker": "GGAL", "precio": 12.3456, "var_1d_pct": 3.1, "var_5d_pct": -0.4},
        {"ticker": "YPFD.BA", "precio": "no disponible"},
    ],
    "macro": {
        "dolar_ccl": {"valor": 1612.0, "fecha": "2026-10-14"},
        "riesgo_pais": {"valor": 597.0},
    },
    "noticias": {"GGAL": [{"titulo": "Depósitos privados crecen 8%", "resumen": ""}]},
}

OK = (
    "**TU CARTERA** GGAL subió 3,1% y cerró a 12,35 dólares en la última rueda del 14.\n"
    "Sobre su variación semanal, bajó 0,4%.\n\n"
    "**MACRO** El dólar CCL está en $1.612 y el riesgo país en 597 puntos."
)


def test_valid_narrative_passes_with_es_ar_and_en_formats():
    assert validate_narrative(OK, DATA) == []
    assert (
        validate_narrative("**MACRO** El CCL está en 1612,0 y GGAL 3.1% arriba, pero sigue.", DATA)
        == []
    )


@pytest.mark.parametrize(
    "text,fragment",
    [
        ("**MACRO** El CCL subió 9,9% hoy y sigue estable en todo el día.", "9,9"),
        ("**MACRO** El riesgo país está en 598 puntos, dato de la rueda.", "598"),
        ("**TU CARTERA** YPFD.BA cerró a 1.500 pesos en la rueda de hoy.", "1.500"),
    ],
)
def test_untraceable_numbers_are_reported(text, fragment):
    violations = validate_narrative(text, DATA)
    assert any("number" in v and fragment in v for v in violations)


def test_written_number_must_equal_the_data_rounded_to_its_own_precision():
    ok = "**TU CARTERA** GGAL a {} dólares aproximadamente en la rueda."
    assert validate_narrative(ok.format("12,35"), DATA) == []
    assert validate_narrative(ok.format("12,3"), DATA) == []  # coarser rounding is fine
    assert validate_narrative(ok.format("12,34"), DATA)  # 12.3456 rounds to 12,35
    assert validate_narrative(ok.format("12,36"), DATA)


def test_numbers_from_news_and_dates_are_traceable():
    text = "**TU CARTERA** Los depósitos crecieron 8% según la nota del 14 de octubre de 2026."
    assert validate_narrative(text, DATA) == []


def test_sign_is_ignored_when_matching():
    assert (
        validate_narrative("**MACRO** GGAL cedió 3,1% en la rueda de ayer, según datos.", DATA)
        == []
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "Comprá GGAL antes de que suba.",
        "Vendé YPF hoy.",
        "Conviene comprar acciones del banco.",
        "Te recomiendo mantener la posición.",
        "Es un buen momento para comprar GGAL.",
        "Conviene vender YPF.",
        "Deberías vender la posición.",
    ],
)
def test_recommendation_phrases_are_rejected(phrase):
    text = f"**TU CARTERA** {phrase} Esto es una frase de relleno para el largo."
    assert any("recommendation" in v for v in validate_narrative(text, DATA))


def test_neutral_use_of_buy_sell_words_is_allowed():
    text = "**MACRO** La compra de dólares por parte del BCRA continúa en el mercado local."
    assert validate_narrative(text, DATA) == []


def test_structure_rules():
    assert any("empty" in v for v in validate_narrative("", DATA))
    assert any("label" in v for v in validate_narrative("Texto sin etiqueta alguna.", DATA))
    three = "\n\n".join(
        f"**TEMA{c}** texto de relleno suficientemente largo para pasar." for c in "ABC"
    )
    assert any("items" in v for v in validate_narrative(three, DATA))
    assert any("short" in v for v in validate_narrative("**MACRO** corto", DATA))
    assert any("long" in v for v in validate_narrative("**MACRO** " + "palabra " * 200, DATA))
    many_lines = "**MACRO** " + "\n".join(["linea de texto de relleno"] * 5)
    assert any("lines" in v for v in validate_narrative(many_lines, DATA))


def test_narrative_must_not_include_table_or_disclaimer():
    base = "**MACRO** El dólar CCL está en $1.612 y el riesgo país en 597 puntos."
    assert any("table" in v for v in validate_narrative(base + "\n| a | b |", DATA))
    assert any("disclaimer" in v for v in validate_narrative(base + "\n" + DISCLAIMER, DATA))


def test_final_text_requires_trailing_disclaimer_and_length_limit():
    assert validate_final_text("algo\n\n" + DISCLAIMER) == []
    assert any("disclaimer" in v for v in validate_final_text("algo sin disclaimer"))
    assert any("disclaimer" in v for v in validate_final_text(DISCLAIMER + "\n\nalgo"))
    assert any("long" in v for v in validate_final_text("x" * 6000 + "\n\n" + DISCLAIMER))


def test_extract_numbers_ignores_units_glued_to_letters():
    raw = [t.raw for t in extract_numbers("1D 5D 3,1% $1.612 y 597. Luego 2026-10-15")]
    assert raw == ["3,1", "1.612", "597", "2026", "10", "15"]
