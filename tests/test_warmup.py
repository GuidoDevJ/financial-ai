"""Startup warm-up: build the lazy singletons in the lifespan; failures never crash startup."""

import asyncio
import logging

from fastapi.testclient import TestClient

from app import main
from app.core.config import Settings
from app.tools import search_news as search_news_module


def patch_steps(monkeypatch, calls, *, fail=()):
    def step(name):
        def run(*_args):
            calls.append(name)
            if name in fail:
                raise RuntimeError(f"{name} boom")

        return run

    async def graph():
        step("graph")()

    async def prompts(settings):
        step("prompts")()

    monkeypatch.setattr(main, "get_chat_graph", graph)
    monkeypatch.setattr(main, "get_brief_deps", step("brief"))
    monkeypatch.setattr(main, "warm_news_store", step("news"))
    monkeypatch.setattr(main, "_warm_prompts", prompts)


async def test_warm_up_builds_graph_brief_deps_news_store_and_prompts(monkeypatch):
    calls = []
    patch_steps(monkeypatch, calls)
    await main.warm_up(Settings(_env_file=None))
    assert sorted(calls) == ["brief", "graph", "news", "prompts"]


async def test_a_failing_step_is_logged_and_does_not_stop_the_others(monkeypatch, caplog):
    calls = []
    patch_steps(monkeypatch, calls, fail={"graph", "news"})
    with caplog.at_level(logging.WARNING, logger="app.main"):
        await main.warm_up(Settings(_env_file=None))
    assert sorted(calls) == ["brief", "graph", "news", "prompts"]
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "graph" in text and "news" in text


async def test_a_hung_step_is_bounded_by_the_warmup_timeout(monkeypatch, caplog):
    calls = []
    patch_steps(monkeypatch, calls)

    async def hang():
        await asyncio.sleep(30)

    monkeypatch.setattr(main, "get_chat_graph", hang)
    with caplog.at_level(logging.WARNING, logger="app.main"):
        await asyncio.wait_for(
            main.warm_up(Settings(_env_file=None, warmup_timeout_seconds=0.1)), timeout=5
        )
    assert "timed out" in " ".join(r.getMessage() for r in caplog.records)


def test_lifespan_runs_the_warm_up_when_enabled(monkeypatch, set_env):
    set_env(WARMUP_ON_STARTUP="true")
    calls = []
    patch_steps(monkeypatch, calls)
    with TestClient(main.app):
        assert sorted(calls) == ["brief", "graph", "news", "prompts"]


def test_lifespan_skips_the_warm_up_when_disabled(monkeypatch, set_env):
    set_env(WARMUP_ON_STARTUP="false")
    calls = []
    patch_steps(monkeypatch, calls)
    with TestClient(main.app):
        assert calls == []


def test_startup_survives_every_step_failing(monkeypatch, set_env):
    set_env(WARMUP_ON_STARTUP="true")
    calls = []
    patch_steps(monkeypatch, calls, fail={"graph", "brief", "news", "prompts"})
    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200
    assert len(calls) == 4


def test_the_news_store_is_shared_between_tools_and_the_warm_up(monkeypatch):
    built = []
    monkeypatch.setattr(search_news_module, "_stores", {})
    monkeypatch.setattr(
        search_news_module, "build_news_store", lambda settings: built.append(1) or object()
    )
    settings = Settings(_env_file=None)
    first = search_news_module.shared_news_store(settings)
    assert search_news_module.shared_news_store(settings) is first
    assert built == [1]
    main.warm_news_store(settings)
    assert built == [1]
