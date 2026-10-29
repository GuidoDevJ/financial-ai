"""Prompt Management wiring: chat graph, brief generator, API error codes, CLI and scheduler."""

import json
import logging
from datetime import UTC, datetime

import fakeredis
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, SystemMessage

import app.main as main_module
from app.agents.chat_graph import build_chat_graph
from app.brief.generate import main as brief_main
from app.brief.generate import run_brief
from app.brief.scheduler import run_daily_briefs
from app.brief.store import BriefStore
from app.core.audit import BRIEF_AUDIT_LOGGER_NAME, LLM_AUDIT_LOGGER_NAME
from app.core.config import Settings
from app.main import app, get_brief_deps, get_chat_graph
from app.prompts.registry import PromptError, PromptVersion, StaticPrompt
from app.tools.registry import build_tools
from tests.fakes import reference_prompt_text, scripted
from tests.test_brief_generate import GOOD, make_deps
from tests.test_brief_scheduler import make_deps as scheduler_deps

CREDS = {"aws_access_key_id": "AKIATEST", "aws_secret_access_key": "secret"}


class CountingPrompt:
    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls = 0

    def get(self):
        self.calls += 1
        return PromptVersion(str(self.calls), self.texts[min(self.calls, len(self.texts)) - 1], "x")

    async def aget(self):
        return self.get()


class FailingPrompt:
    def __init__(self, error):
        self.error = error

    def get(self):
        raise self.error

    async def aget(self):
        raise self.error


ERRORS = [
    PromptError("PROMPT_NOT_FOUND", "not found", 502),
    PromptError("PROMPT_ACCESS_DENIED", "denied: bedrock:GetPrompt", 502),
    PromptError("PROMPT_UNAVAILABLE", "throttled", 503),
    PromptError("PROMPT_ID_MISSING", "BEDROCK_PROMPT_CHAT_ID is not set", 503),
]


def teardown_function():
    app.dependency_overrides.clear()


def empty_tools():
    return build_tools(get_price=lambda **k: {}, get_macro=lambda **k: {})


# ------------------------------------------------------------------------- chat


async def test_chat_graph_reads_the_prompt_on_every_turn():
    source = CountingPrompt("prompt A", "prompt B")
    model = scripted(AIMessage(content="uno"), AIMessage(content="dos"))
    graph = build_chat_graph(model, empty_tools(), prompt_source=source)
    cfg = {"configurable": {"thread_id": "t"}}
    await graph.ainvoke({"messages": [("user", "a")]}, cfg)
    await graph.ainvoke({"messages": [("user", "b")]}, cfg)
    assert source.calls == 2
    assert [m[0].content for m in model.seen] == ["prompt A", "prompt B"]
    assert isinstance(model.seen[0][0], SystemMessage)


@pytest.mark.parametrize("error", ERRORS, ids=lambda e: e.code)
def test_chat_endpoint_maps_prompt_errors(error):
    graph = build_chat_graph(
        scripted(AIMessage(content="x")), empty_tools(), prompt_source=FailingPrompt(error)
    )

    async def dep():
        return graph

    app.dependency_overrides[get_chat_graph] = dep
    r = TestClient(app).post("/chat", json={"cliente_id": "c1", "message": "hola"})
    assert r.status_code == error.status
    assert r.json()["error"]["code"] == error.code


def test_chat_endpoint_missing_prompt_id_names_the_variable(monkeypatch):
    monkeypatch.setattr(main_module, "_graph", None)
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: Settings(_env_file=None, bedrock_prompt_chat_id="", **CREDS),
    )
    r = TestClient(app).post("/chat", json={"cliente_id": "c1", "message": "hola"})
    assert r.status_code == 503
    detail = r.json()["error"]
    assert detail["code"] == "PROMPT_ID_MISSING" and "BEDROCK_PROMPT_CHAT_ID" in detail["message"]


# ------------------------------------------------------------------------ brief


def test_brief_reads_the_prompt_on_every_generation():
    source = CountingPrompt("brief prompt")
    model = scripted(AIMessage(content=GOOD))
    run_brief(make_deps(model, prompt_source=source), "client_001")
    assert source.calls == 1
    assert model.seen[0][0].content == "brief prompt"


@pytest.mark.parametrize("error", ERRORS, ids=lambda e: e.code)
def test_brief_preview_maps_prompt_errors(error):
    deps = make_deps(scripted(AIMessage(content=GOOD)), prompt_source=FailingPrompt(error))
    app.dependency_overrides[get_brief_deps] = lambda: deps
    r = TestClient(app).post("/brief/preview", json={"cliente_id": "client_001"})
    assert r.status_code == error.status
    assert r.json()["error"]["code"] == error.code


def test_brief_preview_missing_prompt_id_names_the_variable(monkeypatch):
    monkeypatch.setattr(main_module, "_brief_deps", None)
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: Settings(_env_file=None, bedrock_prompt_brief_id="", **CREDS),
    )
    r = TestClient(app).post("/brief/preview", json={"cliente_id": "client_001"})
    assert r.status_code == 503
    detail = r.json()["error"]
    assert detail["code"] == "PROMPT_ID_MISSING" and "BEDROCK_PROMPT_BRIEF_ID" in detail["message"]


def test_prompt_failure_is_not_degraded_even_when_degrading_on_llm_errors():
    error = PromptError("PROMPT_UNAVAILABLE", "down", 503)
    deps = make_deps(
        scripted(AIMessage(content=GOOD)),
        prompt_source=FailingPrompt(error),
        degrade_on_llm_error=True,
    )
    with pytest.raises(PromptError):
        run_brief(deps, "client_001")


def test_brief_cli_reports_prompt_errors_with_exit_code_2(capsys):
    def factory(settings, **kw):
        raise PromptError("PROMPT_ID_MISSING", "BEDROCK_PROMPT_BRIEF_ID is not set.", 503)

    assert brief_main(["--cliente", "client_001"], deps_factory=factory) == 2
    out = capsys.readouterr().out
    assert "PROMPT_ID_MISSING" in out and "BEDROCK_PROMPT_BRIEF_ID" in out


def test_scheduler_records_a_clear_error_per_client_when_the_prompt_fails():
    r = fakeredis.FakeRedis(decode_responses=True)
    error = PromptError("PROMPT_NOT_FOUND", "Prompt not found (BEDROCK_PROMPT_BRIEF_ID='X').", 502)
    deps = scheduler_deps(r, prompt_source=FailingPrompt(error))
    [result] = run_daily_briefs(deps, r, deps.settings, client_ids=["client_001"])
    assert result["status"] == "failed" and "BEDROCK_PROMPT_BRIEF_ID" in result["error"]


def test_reference_prompts_only_use_variables_the_code_provides():
    from app.agents.chat_graph import chat_prompt_variables
    from app.prompts import reference

    registry = reference.load_reference()
    chat_vars = set(chat_prompt_variables("B", datetime.now(UTC)))
    assert set(reference.placeholders(reference_prompt_text("chat"))) == chat_vars
    assert set(reference.placeholders(reference_prompt_text("brief"))) == {
        "banco",
        "fecha",
        "cartera",
        "precios",
        "macro",
        "noticias",
    }
    assert set(registry) == {"chat", "brief"}


# ------------------------------------------------------------------------ audit


def audit_lines(caplog, logger_name):
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == logger_name]


async def test_chat_turn_audit_line_carries_prompt_version_and_hash(caplog):
    source = StaticPrompt("prompt A", name="chat", version="5")
    graph = build_chat_graph(
        scripted(AIMessage(content="uno")), empty_tools(), prompt_source=source
    )
    with caplog.at_level(logging.INFO, logger=LLM_AUDIT_LOGGER_NAME):
        await graph.ainvoke({"messages": [("user", "a")]}, {"configurable": {"thread_id": "t"}})
    [line] = audit_lines(caplog, LLM_AUDIT_LOGGER_NAME)
    assert line["prompt_name"] == "chat" and line["prompt_version"] == "5"
    assert line["prompt_hash"] == source.get().hash


def test_brief_run_audit_line_and_stored_payload_carry_the_prompt(caplog):
    r = fakeredis.FakeRedis(decode_responses=True)
    source = StaticPrompt("brief prompt", name="brief", version="2")
    deps = make_deps(scripted(AIMessage(content=GOOD)), prompt_source=source, store=BriefStore(r))
    with caplog.at_level(logging.INFO, logger=BRIEF_AUDIT_LOGGER_NAME):
        result = run_brief(deps, "client_001", store=True)
    expected = {"name": "brief", "version": "2", "hash": source.get().hash}
    assert result.prompt == expected
    [line] = audit_lines(caplog, BRIEF_AUDIT_LOGGER_NAME)
    assert (line["prompt_name"], line["prompt_version"], line["prompt_hash"]) == (
        "brief",
        "2",
        expected["hash"],
    )
    stored = BriefStore(r).latest("client_001")
    assert stored["prompt"] == expected
    assert stored["texto_brief"] == result.texto_brief
