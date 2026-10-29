import fakeredis
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.brief.generate import BriefDeps
from app.brief.scheduler import (
    BRIEF_JOB_ID,
    lock_key,
    mock_client_ids,
    register_brief_job,
    run_daily_briefs,
)
from app.brief.store import BriefStore
from app.core.config import Settings, get_settings
from app.main import app
from app.tools.errors import ToolError
from app.tools.get_portfolio import make_get_portfolio
from tests.fakes import BRIEF_PROMPT, ScriptedChatModel
from tests.test_brief_generate import NOW

FECHA = "2026-10-15"


class Narrator(ScriptedChatModel):
    """Always answers with a valid narrative (a script that never runs out)."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=NARRATIVE))])


# Valid for any portfolio: it cites no numbers.
NARRATIVE = (
    "**TU CARTERA** Tu cartera se muestra en la tabla con los últimos cierres disponibles.\n"
    "**MACRO** Las variables macro se informan con la fuente y la fecha de cada dato."
)


def price_for(ticker, mercado):
    if ticker == "BMA":
        raise ToolError("PROVIDER_ERROR", "boom")
    return {
        "ticker": ticker,
        "fecha": "2026-10-14",
        "precio_cierre": 10.0,
        "variacion_dia": 1.0,
        "variacion_semana": 2.0,
        "variacion_mes": 3.0,
        "fuente": "fake",
    }


def macro_for(variable):
    return {"variable": variable, "valor": 1000.0, "fecha": "2026-10-14", "fuente": "fake"}


def make_deps(redis_client, model=None, sent=None, **kw):
    sent = sent if sent is not None else []
    return BriefDeps(
        get_portfolio=make_get_portfolio(get_price=price_for, get_macro=macro_for),
        get_price=price_for,
        get_macro=macro_for,
        search_news=lambda **k: (_ for _ in ()).throw(ToolError("NO_NEWS_FOUND", "none")),
        model=model or Narrator(responses=[AIMessage(content="x")], seen=[]),
        settings=Settings(_env_file=None),
        prompt_source=kw.pop("prompt_source", BRIEF_PROMPT),
        store=BriefStore(redis_client),
        send_email=lambda settings, **k: sent.append(k) or {"status": "sent"},
        now=lambda: NOW,
        **kw,
    )


def redis_():
    return fakeredis.FakeRedis(decode_responses=True)


# ---------------------------------------------------------------- registration


def test_job_is_a_weekday_0730_buenos_aires_cron():
    sched = BackgroundScheduler()
    sched.start(paused=True)
    try:
        register_brief_job(sched, lambda: None)
        register_brief_job(sched, lambda: None)  # idempotent
        (job,) = [j for j in sched.get_jobs() if j.id == BRIEF_JOB_ID]
        fields = {f.name: str(f) for f in job.trigger.fields}
        assert fields["day_of_week"] == "mon-fri"
        assert fields["hour"] == "7" and fields["minute"] == "30"
        assert str(job.trigger.timezone) == "America/Argentina/Buenos_Aires"
        assert job.max_instances == 1 and job.coalesce is True
    finally:
        sched.shutdown(wait=False)


def test_mock_clients_come_from_the_portfolios_file():
    assert mock_client_ids() == ["client_001", "client_002", "client_003"]


# ------------------------------------------------------------------ the run


def test_acceptance_three_mock_portfolios_generate_without_error():
    r = redis_()
    sent = []
    results = run_daily_briefs(make_deps(r, sent=sent), r, Settings(_env_file=None), fecha=FECHA)
    assert [x["cliente_id"] for x in results] == ["client_001", "client_002", "client_003"]
    assert {x["status"] for x in results} == {"generated"}
    store = BriefStore(r)
    for cid in mock_client_ids():
        brief = store.get(cid, FECHA)
        assert brief and brief["texto_brief"] and "html" in brief
    assert len(sent) == 3
    # BMA price fails in this fixture: recorded, not fatal.
    assert "price:BMA" in store.get("client_002", FECHA)["missing_inputs"]


def test_one_failing_client_does_not_stop_the_others():
    r = redis_()
    deps = make_deps(r)
    real = deps.get_portfolio

    def flaky(cliente_id):
        if cliente_id == "client_001":
            raise RuntimeError("portfolio backend down")
        return real(cliente_id=cliente_id)

    deps.get_portfolio = flaky
    results = run_daily_briefs(deps, r, Settings(_env_file=None), fecha=FECHA)
    by_client = {x["cliente_id"]: x for x in results}
    assert by_client["client_001"]["status"] == "failed"
    assert "portfolio backend down" in by_client["client_001"]["error"]
    assert by_client["client_002"]["status"] == by_client["client_003"]["status"] == "generated"


def test_idempotent_skip_when_already_stored_unless_forced():
    r = redis_()
    store = BriefStore(r)
    store.save("client_001", FECHA, "viejo", ["GGAL"])
    deps = make_deps(r)
    results = run_daily_briefs(
        deps, r, Settings(_env_file=None), fecha=FECHA, client_ids=["client_001"]
    )
    assert results[0]["status"] == "skipped_exists"
    assert store.get("client_001", FECHA)["texto_brief"] == "viejo"
    assert deps.model.seen == []  # the LLM was not even called

    forced = run_daily_briefs(
        deps, r, Settings(_env_file=None), fecha=FECHA, client_ids=["client_001"], force=True
    )
    assert forced[0]["status"] == "generated"
    assert store.get("client_001", FECHA)["texto_brief"] != "viejo"


def test_lock_held_by_another_instance_skips_and_lock_is_released_after_run():
    r = redis_()
    settings = Settings(_env_file=None)
    r.set(lock_key("client_001", FECHA), "other-instance", nx=True, ex=60)
    deps = make_deps(r)
    results = run_daily_briefs(deps, r, settings, fecha=FECHA, client_ids=["client_001"])
    assert results[0]["status"] == "skipped_locked" and deps.model.seen == []
    assert r.get(lock_key("client_001", FECHA)) == "other-instance"  # not stolen

    r.delete(lock_key("client_001", FECHA))
    run_daily_briefs(deps, r, settings, fecha=FECHA, client_ids=["client_001"])
    assert r.get(lock_key("client_001", FECHA)) is None  # released
    assert BriefStore(r).exists("client_001", FECHA)


def test_lock_is_released_even_if_the_client_run_fails():
    r = redis_()
    deps = make_deps(r)
    deps.get_portfolio = lambda cliente_id: (_ for _ in ()).throw(RuntimeError("x"))
    run_daily_briefs(deps, r, Settings(_env_file=None), fecha=FECHA, client_ids=["client_001"])
    assert r.get(lock_key("client_001", FECHA)) is None


def test_two_instances_generate_a_brief_only_once():
    r = redis_()
    deps_a, deps_b = make_deps(r), make_deps(r)
    first = run_daily_briefs(
        deps_a, r, Settings(_env_file=None), fecha=FECHA, client_ids=["client_001"]
    )
    second = run_daily_briefs(
        deps_b, r, Settings(_env_file=None), fecha=FECHA, client_ids=["client_001"]
    )
    assert first[0]["status"] == "generated" and second[0]["status"] == "skipped_exists"
    assert deps_b.model.seen == []


# --------------------------------------------------------------- app wiring


def test_scheduler_does_not_start_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_SCHEDULER", raising=False)
    get_settings.cache_clear()
    try:
        with TestClient(app):
            assert app.state.scheduler is None
    finally:
        get_settings.cache_clear()


def test_lifespan_registers_ingest_and_brief_jobs_when_enabled(monkeypatch):
    monkeypatch.setenv("ENABLE_SCHEDULER", "true")
    get_settings.cache_clear()
    monkeypatch.setattr("app.main.build_ingest_job", lambda settings: lambda: None)
    monkeypatch.setattr("app.main.build_brief_job", lambda settings: lambda: None)
    try:
        with TestClient(app):
            ids = {j.id for j in app.state.scheduler.get_jobs()}
            assert {"ingest_news", BRIEF_JOB_ID} <= ids
    finally:
        get_settings.cache_clear()
        app.state.scheduler = None


def test_build_brief_job_is_lazy_and_never_raises(monkeypatch):
    from app.brief import scheduler as mod

    calls = []

    def boom(settings, **kw):
        calls.append(kw)
        raise RuntimeError("no AWS config")

    monkeypatch.setattr(mod, "build_default_deps", boom)
    job = mod.build_brief_job(Settings(_env_file=None))
    assert calls == []  # nothing is built until the job fires
    job()  # logged, not raised
    assert calls == [{"degrade_on_llm_error": True}]
