"""docs/prompts/registry.yaml: structure, Bedrock shape and the offline export helper."""

import json
import re

import boto3
import pytest
from botocore.validate import validate_parameters

from app.prompts import export, reference

REGISTRY = reference.load_reference()
NAMES = ["chat", "brief"]


def test_registry_has_both_prompts():
    assert sorted(REGISTRY) == sorted(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_current_version_exists_and_text_is_bedrock_shaped(name):
    entry = REGISTRY[name]
    assert reference.current_version(entry) in entry["versions"]
    assert entry["name"] and entry["description"]
    for version in entry["versions"].values():
        assert version["changelog"]
        for variant in version["variants"]:
            assert variant["templateType"] == "TEXT"
    assert entry["defaultVariant"] in [v["name"] for v in reference.current_variants(entry)]
    assert reference.current_text(entry)


@pytest.mark.parametrize("name", NAMES)
def test_no_undeclared_placeholders_in_any_version(name):
    for version in REGISTRY[name]["versions"].values():
        for variant in version["variants"]:
            text_cfg = variant["templateConfiguration"]["text"]
            declared = sorted(v["name"] for v in text_cfg["inputVariables"])
            assert reference.placeholders(text_cfg["text"]) == declared


EXPECTED_VARIABLES = {
    "chat": ["banco", "fecha_actual"],
    "brief": ["banco", "cartera", "fecha", "macro", "noticias", "precios"],
}


CURRENT_VERSIONS = {"chat": "4", "brief": "2"}


def test_current_versions_keep_the_history():
    for name in NAMES:
        entry = REGISTRY[name]
        current = CURRENT_VERSIONS[name]
        assert reference.current_version(entry) == current
        assert set(entry["versions"]) == {str(n) for n in range(1, int(current) + 1)}
        assert "rewrite" in entry["versions"]["2"]["changelog"].lower()


def test_chat_v3_tells_the_model_to_check_the_portfolio_and_use_valid_markets():
    text = reference.current_text(REGISTRY["chat"])
    assert "get_portfolio" in text and "Nunca le preguntes al cliente si tiene" in text
    assert '"byma"' in text and '"nyse"' in text and 'nunca uses "merval"' in text


@pytest.mark.parametrize("name", NAMES)
def test_current_version_declares_exactly_the_variables_used_in_the_text(name):
    entry = REGISTRY[name]
    [variant] = reference.current_variants(entry)
    declared = sorted(v["name"] for v in variant["templateConfiguration"]["text"]["inputVariables"])
    assert declared == reference.placeholders(reference.current_text(entry))
    assert declared == EXPECTED_VARIABLES[name]


@pytest.mark.parametrize("name", NAMES)
def test_variants_carry_only_what_create_prompt_needs(name):
    # modelId / inferenceConfiguration are ignored by the app and risk CreatePrompt rejection.
    for version in REGISTRY[name]["versions"].values():
        for variant in version["variants"]:
            assert set(variant) == {"name", "templateType", "templateConfiguration"}
            assert set(variant["templateConfiguration"]) == {"text"}


def test_chat_prompt_is_a_structured_spanish_template_with_all_the_rules():
    chat = reference.current_text(REGISTRY["chat"])
    for tag in ("rol", "reglas", "ejemplos"):
        assert f"<{tag}>" in chat and f"</{tag}>" in chat
    assert "Banco Comafi" not in chat  # the bank is a variable, not a hardcoded assumption
    for fragment in (
        "SIEMPRE las herramientas",
        "NUNCA",
        "asesor humano",
        "search_news",
        "medio y la fecha",
        "lenguaje simple",
        "no lo estimes",
    ):
        assert fragment in chat, fragment


def test_brief_prompt_is_a_structured_spanish_template_with_the_data_variables():
    brief = " ".join(reference.current_text(REGISTRY["brief"]).split())  # unwrap lines
    for tag in ("rol", "datos", "reglas", "formato", "ejemplo"):
        assert f"<{tag}>" in brief and f"</{tag}>" in brief
    for variable in EXPECTED_VARIABLES["brief"]:
        assert "{{" + variable + "}}" in brief
    assert "los datos que te paso a continuación" not in brief
    for fragment in (
        "NO los escribas",
        "no disponible",
        "NUNCA recomiendes comprar, vender o mantener",
        "No tenés herramientas",
        "$1.612",
        "2-4 líneas",
        "ejemplo",
    ):
        assert fragment in brief, fragment


def test_text_hash_matches_prompt_version_hash():
    from app.prompts.registry import PromptVersion

    text = reference.current_text(REGISTRY["chat"])
    assert reference.text_hash(text) == PromptVersion("2", text, "chat").hash


SAMPLE = {
    "banco": "Banco Demo",
    "fecha_actual": "2026-10-16",
    "fecha": "2026-10-16",
    "cartera": '[{"ticker":"GGAL","moneda":"USD","peso_pct":60.0}]',
    "precios": '[{"ticker":"GGAL","precio":12.35,"var_1d_pct":3.1}]',
    "macro": '{"dolar_ccl":{"valor":1612.0,"fecha":"2026-10-15","fuente":"d"}}',
    "noticias": '{"GGAL":"sin noticias en las últimas 24 horas"}',
}


@pytest.mark.parametrize("name", NAMES)
def test_every_version_renders_with_realistic_values(name):
    from app.prompts.registry import PromptVersion

    for number, version in REGISTRY[name]["versions"].items():
        [variant] = version["variants"]
        pv = PromptVersion(number, reference.variant_text(variant), name)
        rendered = pv.render(**SAMPLE)
        assert "{{" not in rendered
        if "{{banco}}" in pv.content:
            assert "Banco Demo" in rendered


def test_test_variables_doc_documents_every_variable():
    doc = (reference.REGISTRY_PATH.parent / "test-variables.md").read_text(encoding="utf-8")
    for name, variables in EXPECTED_VARIABLES.items():
        section = doc.split(f"## Prompt `{name}`", 1)[1].split("\n## ", 1)[0]
        for variable in variables:
            assert f"`{variable}`" in section, (name, variable)


# ------------------------------------------------------------------- export


def create_prompt_model():
    client = boto3.client(
        "bedrock-agent",
        region_name="us-east-1",
        aws_access_key_id="a",
        aws_secret_access_key="b",
    )
    return client.meta.service_model.operation_model("CreatePrompt").input_shape


@pytest.mark.parametrize("name", NAMES)
def test_export_payload_passes_botocore_input_validation(name):
    payload = export.build_payload(REGISTRY[name])
    # raises ParamValidationError if the shape is wrong (no network involved)
    validate_parameters(payload, create_prompt_model())
    assert payload["variants"][0]["templateType"] == "TEXT"
    assert payload["defaultVariant"] == payload["variants"][0]["name"]
    text_cfg = payload["variants"][0]["templateConfiguration"]["text"]
    assert sorted(v["name"] for v in text_cfg["inputVariables"]) == EXPECTED_VARIABLES[name]
    assert set(payload) == {"name", "description", "defaultVariant", "variants"}


def test_export_rejects_mismatched_input_variables():
    entry = json.loads(json.dumps(REGISTRY["chat"]))
    version = entry["versions"][entry["current_version"]]
    version["variants"][0]["templateConfiguration"]["text"]["text"] += "\n{{nueva}}"
    with pytest.raises(ValueError, match="nueva"):
        export.build_payload(entry)


def test_export_cli_prints_json_and_never_calls_aws(capsys, monkeypatch):
    monkeypatch.setattr(boto3, "client", lambda *a, **k: pytest.fail("AWS client created"))
    assert export.main(["--name", "brief"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["name"] == REGISTRY["brief"]["name"]
    assert "create-prompt-version" in captured.err
    assert export.main([]) == 0  # all prompts


def _doc_values(prompt: str) -> dict[str, str]:
    """Fenced values under each `#### `var`` heading of the given prompt section."""
    doc = (reference.REGISTRY_PATH.parent / "test-variables.md").read_text(encoding="utf-8")
    section = doc.split(f"## Prompt `{prompt}`", 1)[1].split("\n## ", 1)[0]
    pattern = re.compile(r"#### `(\w+)`\s+```\n(.*?)\n```", re.DOTALL)
    return dict(pattern.findall(section))


@pytest.mark.parametrize("name", NAMES)
def test_documented_test_variables_render_the_current_prompt(name):
    from app.prompts.registry import PromptVersion

    values = _doc_values(name)
    assert sorted(values) == EXPECTED_VARIABLES[name]
    text = reference.current_text(REGISTRY[name])
    rendered = PromptVersion("2", text, name).render(**values)
    assert "{{" not in rendered
    if name == "brief":
        for key in ("cartera", "precios", "macro", "noticias"):
            json.loads(values[key])  # the sample blocks are valid JSON like the app's


def test_chat_v4_tells_the_model_how_to_explain_portfolio_moves():
    text = reference.current_text(REGISTRY["chat"])
    assert "get_portfolio_performance" in text
    assert "no las sumes" in text and '"coincide' in text


def test_chat_v4_frames_simulations_as_hypotheses():
    text = reference.current_text(REGISTRY["chat"])
    assert "simulate_scenario" in text and "hipótesis de cálculo" in text
    assert "ni una recomendación" in text
