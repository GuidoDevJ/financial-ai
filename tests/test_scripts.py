"""The opt-in live scripts, exercised offline against the in-process app."""

import importlib.util
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from app.main import app
from tests.test_adversarial_recommendations import chat_graph
from tests.test_chat_endpoint import client_for, graph_with

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def teardown_function():
    app.dependency_overrides.clear()


def test_eval_script_scores_a_compliant_assistant_at_100_percent():
    module = load("eval_adversarial")
    item = module.load_items()[0]
    client = client_for(chat_graph(item["compliant_reply"]))
    report = module.evaluate(client, [item])
    assert report["total"] == 1 and report["pass_rate"] == 1.0
    assert report["model_native_rate"] == 1.0 and report["failures"] == []


def test_eval_script_counts_server_side_blocks_as_not_model_native():
    module = load("eval_adversarial")
    item = module.load_items()[0]
    client = client_for(chat_graph(item["violating_reply"]))
    report = module.evaluate(client, [item])
    assert report["pass_rate"] == 1.0 and report["model_native_rate"] == 0.0
    assert report["refusal_rate"] == 1.0  # the replacement is itself a refusal


def test_eval_script_reports_http_failures():
    module = load("eval_adversarial")

    class Boom:
        async def ainvoke(self, *a, **k):
            raise RuntimeError("down")

    report = module.evaluate(client_for(Boom()), module.load_items()[:1])
    assert report["pass_rate"] == 0.0 and report["failures"][0]["http_status"] == 502


def test_eval_script_exit_code_follows_min_pass_rate(monkeypatch):
    module = load("eval_adversarial")
    monkeypatch.setattr(module, "evaluate", lambda *a, **k: {
        "total": 2, "pass_rate": 0.5, "model_native_rate": 0.5, "refusal_rate": 1.0,
        "failures": [{"id": "x", "http_status": 200, "reply": "comprá"}],
    })  # fmt: skip
    assert module.main(["--min-pass-rate", "1.0"]) == 1
    assert module.main(["--min-pass-rate", "0.4"]) == 0


def test_bench_script_reports_percentiles():
    module = load("bench_chat")
    client = client_for(graph_with(AIMessage(content="ok")))
    report = module.run_bench(client, 10, concurrency=1, cliente="c1")
    assert report["requests"] == 10 and report["ok"] == 10 and report["errors"] == 0
    assert report["p50"] <= report["p95"] <= report["p99"] <= report["max"]


@pytest.mark.parametrize(("p95", "budget", "code"), [(1.0, 5.0, 0), (6.0, 5.0, 1)])
def test_bench_script_exit_code_uses_the_budget(monkeypatch, p95, budget, code):
    module = load("bench_chat")
    monkeypatch.setattr(
        module, "run_bench", lambda *a, **k: {"requests": 1, "ok": 1, "errors": 0, "p95": p95}
    )
    assert module.main(["--budget", str(budget)]) == code
