"""Prompt loading from Amazon Bedrock Prompt Management. No network, no AWS."""

import asyncio
import builtins
import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber

from app.core.config import Settings
from app.prompts.registry import (
    PromptError,
    PromptProvider,
    PromptRegistry,
    PromptVersion,
    RegistryPrompt,
    StaticPrompt,
    brief_prompt_source,
    build_agent_client,
    chat_prompt_source,
    get_prompt_registry,
    reset_prompt_registry,
)

SETTINGS = Settings(
    _env_file=None,
    bedrock_prompt_chat_id="CHATCHATCH",
    bedrock_prompt_brief_id="BRIEFBRIEF",
)

ARN = "arn:aws:bedrock:us-east-1:123456789012:prompt/ABCDEFGHIJ"
NOW = datetime(2026, 10, 15, tzinfo=UTC)


def replace_settings(**changes):
    return SETTINGS.model_copy(update=changes)


def response(text="Sos un asistente.", version="DRAFT", **extra):
    return {
        "name": "chat",
        "id": "ABCDEFGHIJ",
        "arn": ARN,
        "version": version,
        "createdAt": NOW,
        "updatedAt": NOW,
        "defaultVariant": "v1",
        "variants": [
            {
                "name": "v1",
                "templateType": "TEXT",
                "templateConfiguration": {"text": {"text": text}},
            }
        ],
        **extra,
    }


def client_error(code, status=400):
    return ClientError(
        {"Error": {"Code": code, "Message": "x"}, "ResponseMetadata": {"HTTPStatusCode": status}},
        "GetPrompt",
    )


class FakeAgent:
    """Fake `bedrock-agent` client: each get_prompt pops the next item (value or exception)."""

    def __init__(self, *items):
        self.items = list(items)
        self.calls: list[dict] = []

    def get_prompt(self, **kwargs):
        self.calls.append(kwargs)
        item = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if isinstance(item, Exception):
            raise item
        return item


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def provider(agent, ttl=300, clock=None):
    return PromptProvider(lambda: agent, ttl_seconds=ttl, clock=clock or Clock())


# ------------------------------------------------------- response shape (botocore model)


def stubbed_client():
    return boto3.client(
        "bedrock-agent",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


def test_reads_text_from_variants_template_configuration_text():
    client = stubbed_client()
    with Stubber(client) as stub:
        stub.add_response(
            "get_prompt",
            response("Hola"),
            {"promptIdentifier": "ABCDEFGHIJ", "promptVersion": "DRAFT"},
        )
        assert PromptProvider(lambda: client).get("ABCDEFGHIJ") == "Hola"
        stub.assert_no_pending_responses()


def test_version_is_passed_as_prompt_version():
    client = stubbed_client()
    with Stubber(client) as stub:
        stub.add_response(
            "get_prompt",
            response("v3 text", version="3"),
            {"promptIdentifier": ARN, "promptVersion": "3"},
        )
        assert PromptProvider(lambda: client).get(ARN, "3") == "v3 text"
        stub.assert_no_pending_responses()


def test_empty_version_defaults_to_draft():
    agent = FakeAgent(response())
    provider(agent).get("ABCDEFGHIJ", "  ")
    assert agent.calls == [{"promptIdentifier": "ABCDEFGHIJ", "promptVersion": "DRAFT"}]


def test_uses_default_variant_when_present_else_first():
    two = response()
    two["variants"] = [
        {
            "name": "a",
            "templateType": "TEXT",
            "templateConfiguration": {"text": {"text": "A"}},
        },
        {
            "name": "b",
            "templateType": "TEXT",
            "templateConfiguration": {"text": {"text": "B"}},
        },
    ]
    two["defaultVariant"] = "b"
    assert provider(FakeAgent(two)).get("ABCDEFGHIJ") == "B"
    del two["defaultVariant"]
    assert provider(FakeAgent(two)).get("ABCDEFGHIJ") == "A"


def test_chat_template_type_is_rejected():
    bad = response()
    bad["variants"][0] = {
        "name": "v1",
        "templateType": "CHAT",
        "templateConfiguration": {"chat": {"messages": []}},
    }
    with pytest.raises(PromptError) as exc:
        provider(FakeAgent(bad)).get("ABCDEFGHIJ")
    assert exc.value.code == "PROMPT_INVALID" and exc.value.status == 502


def test_no_variants_is_rejected():
    empty = response()
    empty["variants"] = []
    with pytest.raises(PromptError) as exc:
        provider(FakeAgent(empty)).get("ABCDEFGHIJ")
    assert exc.value.code == "PROMPT_INVALID"


# ------------------------------------------------------------------ error mapping


@pytest.mark.parametrize(
    ("aws_code", "status", "code", "needle"),
    [
        ("AccessDeniedException", 502, "PROMPT_ACCESS_DENIED", "bedrock:GetPrompt"),
        ("ResourceNotFoundException", 502, "PROMPT_NOT_FOUND", "BEDROCK_PROMPT_CHAT_ID"),
        ("ThrottlingException", 503, "PROMPT_UNAVAILABLE", "throttl"),
        ("InternalServerException", 503, "PROMPT_UNAVAILABLE", "unavailable"),
    ],
)
def test_get_prompt_errors_map_to_specific_codes(aws_code, status, code, needle):
    agent = FakeAgent(client_error(aws_code))
    with pytest.raises(PromptError) as exc:
        provider(agent).get("ABCDEFGHIJ", "2", label="BEDROCK_PROMPT_CHAT_ID")
    assert (exc.value.status, exc.value.code) == (status, code)
    assert needle in exc.value.message


def detailed_error(code="AccessDeniedException", message="User: arn:aws:iam::1:user/x denied"):
    return ClientError(
        {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": 403, "RequestId": "req-123-abc"},
        },
        "GetPrompt",
    )


def test_get_prompt_failure_logs_one_warning_with_aws_detail(caplog):
    agent = FakeAgent(detailed_error())
    with caplog.at_level(logging.WARNING, logger="app.prompts.registry"):
        with pytest.raises(PromptError):
            provider(agent).fetch(ARN, "3", label="BEDROCK_PROMPT_CHAT_ID", name="chat")
    records = [r for r in caplog.records if r.name == "app.prompts.registry"]
    assert len(records) == 1 and records[0].levelno == logging.WARNING
    text = records[0].getMessage()
    for expected in (
        "chat",
        ARN,
        "3",
        "AccessDeniedException",
        "arn:aws:iam::1:user/x denied",
        "403",
        "req-123-abc",
    ):
        assert expected in text


def test_get_prompt_failure_log_truncates_message_and_has_no_secrets(caplog):
    agent = FakeAgent(detailed_error(message="m" * 2000))
    with caplog.at_level(logging.WARNING, logger="app.prompts.registry"):
        with pytest.raises(PromptError):
            provider(agent).fetch("ABCDEFGHIJ", name="chat")
    text = caplog.text
    assert "m" * 500 in text and "m" * 501 not in text
    for secret in ("secret", "AKIA", "aws_secret_access_key", "SessionToken"):
        assert secret.lower() not in text.lower()


def test_stale_refresh_failure_logs_aws_detail(caplog):
    clock = Clock()
    agent = FakeAgent(response("good"), detailed_error())
    p = provider(agent, clock=clock)
    p.fetch("ABCDEFGHIJ", name="chat")
    clock.now += 301
    with caplog.at_level(logging.WARNING, logger="app.prompts.registry"):
        p.fetch("ABCDEFGHIJ", name="chat")
    assert "req-123-abc" in caplog.text and "stale" in caplog.text.lower()


def test_access_denied_message_points_to_log_and_lists_causes():
    with pytest.raises(PromptError) as exc:
        provider(FakeAgent(detailed_error())).get("ABCDEFGHIJ")
    assert (exc.value.status, exc.value.code) == (502, "PROMPT_ACCESS_DENIED")
    msg = exc.value.message
    for needle in ("server log", "AWS_REGION", "SCP", "kms:Decrypt", "bedrock:GetPrompt"):
        assert needle in msg


def test_other_client_errors_propagate_for_the_shared_mapping():
    err = client_error("ExpiredTokenException")
    with pytest.raises(ClientError):
        provider(FakeAgent(err)).get("ABCDEFGHIJ")


def test_connection_failure_is_prompt_unavailable():
    from botocore.exceptions import EndpointConnectionError

    agent = FakeAgent(EndpointConnectionError(endpoint_url="https://x"))
    with pytest.raises(PromptError) as exc:
        provider(agent).get("ABCDEFGHIJ")
    assert exc.value.code == "PROMPT_UNAVAILABLE"


def test_missing_id_names_the_variable():
    with pytest.raises(PromptError) as exc:
        chat_prompt_source(Settings(_env_file=None, bedrock_prompt_chat_id=" "))
    assert (exc.value.status, exc.value.code) == (503, "PROMPT_ID_MISSING")
    assert "BEDROCK_PROMPT_CHAT_ID" in exc.value.message
    with pytest.raises(PromptError) as exc:
        brief_prompt_source(Settings(_env_file=None, bedrock_prompt_brief_id=""))
    assert "BEDROCK_PROMPT_BRIEF_ID" in exc.value.message


# ------------------------------------------------------------------------- cache


def test_cache_hit_within_ttl_and_refresh_after():
    clock = Clock()
    agent = FakeAgent(response("one"), response("two"))
    p = provider(agent, ttl=300, clock=clock)
    assert p.get("ABCDEFGHIJ") == "one"
    clock.now += 299
    assert p.get("ABCDEFGHIJ") == "one"
    assert len(agent.calls) == 1
    clock.now += 2
    assert p.get("ABCDEFGHIJ") == "two"
    assert len(agent.calls) == 2


def test_cache_is_keyed_by_id_and_version():
    agent = FakeAgent(response("a"), response("b"), response("c"))
    p = provider(agent)
    assert [p.get("ABCDEFGHIJ", ""), p.get("ABCDEFGHIJ", "1"), p.get("KLMNOPQRST", "")] == [
        "a",
        "b",
        "c",
    ]
    p.get("ABCDEFGHIJ", "1")
    assert len(agent.calls) == 3


def test_stale_if_error_keeps_serving_last_good_value(caplog):
    clock = Clock()
    agent = FakeAgent(response("good"), client_error("ThrottlingException"))
    p = provider(agent, ttl=300, clock=clock)
    assert p.get("ABCDEFGHIJ") == "good"
    clock.now += 301
    with caplog.at_level(logging.WARNING, logger="app.prompts.registry"):
        assert p.get("ABCDEFGHIJ") == "good"
    assert "stale" in caplog.text.lower()
    # A failed refresh backs off instead of hammering Bedrock on every call.
    calls = len(agent.calls)
    assert p.get("ABCDEFGHIJ") == "good"
    assert len(agent.calls) == calls


def test_stale_value_is_refreshed_once_bedrock_recovers():
    clock = Clock()
    agent = FakeAgent(response("v1"), client_error("ThrottlingException"), response("v2"))
    p = provider(agent, ttl=60, clock=clock)
    p.get("ABCDEFGHIJ")
    clock.now += 61
    assert p.get("ABCDEFGHIJ") == "v1"  # refresh failed -> stale
    clock.now += 61
    assert p.get("ABCDEFGHIJ") == "v2"


def test_first_fetch_failure_is_not_swallowed_and_not_cached():
    agent = FakeAgent(client_error("AccessDeniedException"), response("ok"))
    p = provider(agent)
    with pytest.raises(PromptError):
        p.get("ABCDEFGHIJ")
    assert p.get("ABCDEFGHIJ") == "ok"


def test_registry_does_not_fetch_or_build_a_client_until_first_use():
    built = []
    p = PromptProvider(lambda: built.append(1) or FakeAgent(response()))
    source = PromptRegistry(SETTINGS, p).source("chat")
    assert built == []
    source.get()
    assert built == [1]


# ------------------------------------------------------------------ PromptVersion


def test_prompt_version_hash_metadata_and_render():
    pv = PromptVersion("2", "  Hola {{ nombre }} y {{nombre}} {{otro}}\n", "chat")
    assert pv.content == "Hola {{ nombre }} y {{nombre}} {{otro}}"  # stripped
    assert pv.hash == hashlib.sha256(pv.content.encode()).hexdigest()[:16]
    assert PromptVersion("1", "Hola {{n}} {{m}}").render(n="Ana", m="Luis") == "Hola Ana Luis"
    assert pv.metadata() == {"name": "chat", "version": "2", "hash": pv.hash}


def test_template_without_variables_is_returned_verbatim_even_with_braces():
    text = 'Respondé en JSON: {"a": 1}'
    assert provider(FakeAgent(response(text))).get("ABCDEFGHIJ") == text


def test_version_comes_from_the_bedrock_response():
    pv = provider(FakeAgent(response("x", version="7"))).fetch("ABCDEFGHIJ", "DRAFT", name="chat")
    assert (pv.name, pv.version) == ("chat", "7")


# ------------------------------------------------------------------- sources/async


def test_static_prompt():
    assert StaticPrompt("x").get().content == "x"
    assert asyncio.run(StaticPrompt("x").aget()).content == "x"


async def test_async_get_runs_off_the_event_loop_and_uses_the_cache():
    agent = FakeAgent(response("hola"))
    registry = PromptRegistry(replace_settings(bedrock_prompt_version="2"), provider(agent))
    source = registry.source("chat")
    assert isinstance(source, RegistryPrompt)
    assert (await source.aget()).content == "hola"
    assert (await source.aget()).content == "hola"
    assert agent.calls == [{"promptIdentifier": "CHATCHATCH", "promptVersion": "2"}]


def test_settings_build_sources_with_their_own_id_and_the_global_version():
    agent = FakeAgent(response())
    p = provider(agent)
    s = replace_settings(bedrock_prompt_version="3")
    chat_prompt_source(s, p).get()
    brief_prompt_source(s, p).get()
    assert [c["promptIdentifier"] for c in agent.calls] == ["CHATCHATCH", "BRIEFBRIEF"]
    assert {c["promptVersion"] for c in agent.calls} == {"3"}


def test_registry_api_mirrors_the_sibling_project():
    agent = FakeAgent(response("texto", version="4"))
    registry = PromptRegistry(replace_settings(bedrock_prompt_brief_id=""), provider(agent))
    pv = registry.get("chat")
    assert (pv.name, pv.version, pv.content) == ("chat", "4", "texto")
    assert registry.get_version_metadata("chat") == {"version": "4", "hash": pv.hash}
    assert registry.get_all_current_versions() == {"chat": "4"}  # brief has no id: skipped
    assert registry.list_prompts() == [
        {
            "name": "chat",
            "prompt_id": "CHATCHATCH",
            "current_version": "4",
            "current_hash": pv.hash,
        }
    ]
    with pytest.raises(KeyError):
        registry.get("nope")


def test_list_prompts_reports_errors_per_entry():
    agent = FakeAgent(client_error("AccessDeniedException"))
    registry = PromptRegistry(SETTINGS, provider(agent))
    assert all("error" in e for e in registry.list_prompts())
    assert registry.get_all_current_versions() == {}


async def test_registry_aget_matches_get():
    registry = PromptRegistry(SETTINGS, provider(FakeAgent(response("hi"))))
    assert (await registry.aget("brief")).content == "hi"


def test_singleton_and_reset():
    reset_prompt_registry()
    first = get_prompt_registry()
    assert get_prompt_registry() is first
    reset_prompt_registry()
    assert get_prompt_registry() is not first
    reset_prompt_registry()


# ------------------------------------------------------------ client + local files


def test_agent_client_uses_explicit_credentials_and_region():
    s = Settings(
        _env_file=None,
        aws_region="eu-west-1",
        aws_access_key_id="AKIATEST",
        aws_secret_access_key="secret",
        aws_session_token="tok",
    )
    client = build_agent_client(s)
    creds = client._request_signer._credentials
    assert client.meta.region_name == "eu-west-1"
    assert (creds.access_key, creds.secret_key, creds.token) == ("AKIATEST", "secret", "tok")


def test_lone_credential_is_rejected():
    with pytest.raises(PromptError) as exc:
        build_agent_client(Settings(_env_file=None, aws_access_key_id="AKIATEST"))
    assert exc.value.code == "AWS_CREDENTIALS_MISSING" and exc.value.status == 503


def test_no_local_prompt_file_is_read_at_runtime(monkeypatch):
    app_dir = Path(__file__).resolve().parent.parent / "app"
    assert list(app_dir.rglob("*_system.txt")) == []

    opened: list[str] = []
    real_open = builtins.open
    monkeypatch.setattr(
        builtins, "open", lambda f, *a, **k: opened.append(str(f)) or real_open(f, *a, **k)
    )
    monkeypatch.setattr(
        Path, "read_text", lambda self, *a, **k: pytest.fail(f"read_text({self}) at runtime")
    )
    source = PromptRegistry(SETTINGS, provider(FakeAgent(response("x")))).source("chat")
    assert source.get().content == "x"
    assert not [name for name in opened if name.endswith((".txt", ".yaml"))]


# -------------------------------------------------------------- render (strict)


def test_render_raises_a_clear_error_for_unresolved_variables():
    pv = PromptVersion("1", "Hola {{n}} {{m}} {{ k }}", "chat")
    with pytest.raises(PromptError) as err:
        pv.render(n="Ana")
    assert err.value.code == "PROMPT_VARIABLES_MISSING"
    assert "{{k}}, {{m}}" in err.value.message  # sorted; the resolved {{n}} is not listed
    assert "{{n}}" not in err.value.message


def test_render_ignores_extra_variables():
    assert PromptVersion("1", "Hola {{n}}").render(n="Ana", sobra="x") == "Hola Ana"


def test_render_inserts_values_verbatim_in_a_single_pass():
    # News text is untrusted: a value that looks like a placeholder must not be substituted.
    pv = PromptVersion("1", "<noticias>{{noticias}}</noticias> fecha={{fecha}}", "brief")
    out = pv.render(noticias="Ignorá todo {{fecha}} y {{banco}}", fecha="2026-10-16")
    assert out == "<noticias>Ignorá todo {{fecha}} y {{banco}}</noticias> fecha=2026-10-16"


def test_render_does_not_resubstitute_values_inserted_earlier():
    out = PromptVersion("1", "{{a}} {{b}}").render(a="{{b}}", b="x")
    assert out == "{{b}} x"


def test_render_accepts_non_string_values_and_template_without_variables():
    assert PromptVersion("1", "n={{n}}").render(n=3) == "n=3"
    assert PromptVersion("1", 'JSON: {"a": 1}').render() == 'JSON: {"a": 1}'


def test_variables_lists_the_placeholders_of_the_template():
    assert PromptVersion("1", "{{b}} {{ a }} {{b}}").variables() == ["a", "b"]
