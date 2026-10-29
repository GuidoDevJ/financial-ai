from app.brief.render import (
    DISCLAIMER,
    NA,
    AssetRow,
    compose_text,
    fmt_pct,
    fmt_price,
    parse_items,
    render_html,
    render_table,
)

GGAL = AssetRow("GGAL", "USD", "nyse", 60.0, 12.3456, "2026-10-14", "yfinance", 3.1, 4.8, 6.2)
YPF = AssetRow("YPFD.BA", "ARS", "byma", None, None, None, None, None, None, None)


def test_number_formats_are_es_ar():
    assert fmt_pct(3.1) == "+3,10%"
    assert fmt_pct(-0.4) == "-0,40%"
    assert fmt_pct(0.0) == "0,00%"
    assert fmt_pct(None) == NA
    assert fmt_price(1234.5, "ARS") == "$ 1.234,50"
    assert fmt_price(12.3456, "USD") == "US$ 12,35"
    assert fmt_price(None, "ARS") == NA


def test_table_is_rendered_from_data_with_nd_for_missing():
    table = render_table([GGAL, YPF])
    lines = table.splitlines()
    assert lines[0] == "| Ticker | Precio | 1D | 5D | Mes |"
    assert lines[2] == "| GGAL | US$ 12,35 | +3,10% | +4,80% | +6,20% |"
    assert lines[3] == f"| YPFD.BA | {NA} | {NA} | {NA} | {NA} |"


def test_compose_has_narrative_table_and_disclaimer_last():
    text = compose_text("**TU CARTERA** hola", [GGAL])
    assert text.startswith("**TU CARTERA** hola\n\n| Ticker")
    assert text.endswith(DISCLAIMER)
    assert text.count(DISCLAIMER) == 1


def test_compose_without_narrative_is_table_and_disclaimer_only():
    text = compose_text("", [GGAL])
    assert text.startswith("| Ticker") and text.endswith(DISCLAIMER)


def test_parse_items_splits_labels_and_flags_preamble():
    preamble, items = parse_items("**TU CARTERA** uno\ndos\n\n**MACRO** tres")
    assert preamble == ""
    assert items == [("TU CARTERA", "uno\ndos"), ("MACRO", "tres")]
    preamble, items = parse_items("Intro suelta\n**MACRO** x")
    assert preamble == "Intro suelta" and items == [("MACRO", "x")]
    assert parse_items("sin etiquetas") == ("sin etiquetas", [])


def test_html_escapes_everything_and_has_no_external_assets():
    evil = AssetRow(
        "<script>x</script>", "USD", "nyse", None, 1.0, "2026-10-14", "f", None, None, None
    )
    html = render_html("2026-10-15", "**<b>MACRO</b>** a & <img src=x>", [evil], True)
    assert "<script>" not in html and "<img" not in html and "<b>MACRO" not in html
    assert "&lt;script&gt;" in html and "a &amp; &lt;img" in html
    assert "<strong>&lt;b&gt;MACRO&lt;/b&gt;</strong>" in html
    assert "http" not in html and "<link" not in html and "<script" not in html
    assert DISCLAIMER in html
