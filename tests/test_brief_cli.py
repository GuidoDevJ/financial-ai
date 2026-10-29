import json

import fakeredis
from langchain_core.messages import AIMessage

from app.brief.generate import main
from app.brief.render import DISCLAIMER
from app.brief.store import BriefStore
from tests.fakes import ScriptedChatModel, scripted
from tests.test_brief_generate import GOOD, Tools, make_deps


def factory(deps, seen_kwargs=None):
    def build(settings, **kw):
        if seen_kwargs is not None:
            seen_kwargs.update(kw)
        return deps

    return build


def test_cli_prints_brief_without_storing_by_default(capsys):
    store = BriefStore(fakeredis.FakeRedis(decode_responses=True))
    kw = {}
    deps = make_deps(scripted(AIMessage(content=GOOD)), store=store)
    code = main(["--cliente", "client_001"], deps_factory=factory(deps, kw))
    out = capsys.readouterr().out
    assert code == 0 and GOOD in out and DISCLAIMER in out
    summary = json.loads(out.strip().splitlines()[-1])
    assert summary["stored"] is False and summary["degraded"] is False
    assert store.get("client_001", "2026-10-15") is None
    assert kw == {"degrade_on_llm_error": False}


def test_cli_store_and_send_flags(capsys):
    store = BriefStore(fakeredis.FakeRedis(decode_responses=True))
    sent = []
    kw = {}
    deps = make_deps(
        scripted(AIMessage(content=GOOD)),
        store=store,
        send_email=lambda s, **k: sent.append(k) or {"status": "sent"},
    )
    code = main(
        ["--cliente", "client_001", "--fecha", "2026-10-09", "--store", "--send"],
        deps_factory=factory(deps, kw),
    )
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0 and summary["stored"] is True and summary["delivery"] == {"status": "sent"}
    assert store.get("client_001", "2026-10-09") is not None and len(sent) == 1
    assert kw == {"degrade_on_llm_error": True}  # a stored brief should survive an LLM outage


def test_cli_errors_have_distinct_exit_codes(capsys):
    unknown = make_deps(scripted(AIMessage(content=GOOD)), Tools(rows=None))
    assert main(["--cliente", "nope"], deps_factory=factory(unknown)) == 1
    assert "No portfolio" in capsys.readouterr().out

    ok = make_deps(scripted(AIMessage(content=GOOD)))
    assert main(["--cliente", "client_001", "--fecha", "bad"], deps_factory=factory(ok)) == 1

    class Boom(ScriptedChatModel):
        def _generate(self, *a, **k):
            raise RuntimeError("bedrock down")

    broken = make_deps(Boom(responses=[AIMessage(content="x")], seen=[]))
    assert main(["--cliente", "client_001"], deps_factory=factory(broken)) == 2
    assert "bedrock down" in capsys.readouterr().out
