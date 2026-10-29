import json
import logging

import pytest

from app.core.audit import AUDIT_LOGGER_NAME, audited_tool


def records(caplog):
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == AUDIT_LOGGER_NAME]


def test_sync_success_logged(caplog):
    @audited_tool
    def get_price(ticker, mercado="nyse"):
        return {"ticker": ticker}

    with caplog.at_level(logging.INFO, logger=AUDIT_LOGGER_NAME):
        assert get_price("GGAL", mercado="byma") == {"ticker": "GGAL"}
    [rec] = records(caplog)
    assert rec["tool"] == "get_price"
    assert rec["status"] == "ok"
    assert rec["params"] == {"ticker": "GGAL", "mercado": "byma"}
    assert rec["duration_ms"] >= 0
    assert "timestamp" in rec


def test_sync_error_logged_and_reraised(caplog):
    @audited_tool
    def boom(x):
        raise ValueError("nope")

    with caplog.at_level(logging.INFO, logger=AUDIT_LOGGER_NAME), pytest.raises(ValueError):
        boom(1)
    [rec] = records(caplog)
    assert rec["status"] == "error"
    assert rec["error"] == "ValueError: nope"
    assert rec["params"] == {"x": 1}


async def test_async_success_logged(caplog):
    @audited_tool
    async def get_macro(variable):
        return {"variable": variable}

    with caplog.at_level(logging.INFO, logger=AUDIT_LOGGER_NAME):
        assert await get_macro("riesgo_pais") == {"variable": "riesgo_pais"}
    [rec] = records(caplog)
    assert rec["tool"] == "get_macro"
    assert rec["status"] == "ok"
    assert rec["params"] == {"variable": "riesgo_pais"}


async def test_async_error_logged(caplog):
    @audited_tool
    async def boom():
        raise RuntimeError("x")

    with caplog.at_level(logging.INFO, logger=AUDIT_LOGGER_NAME), pytest.raises(RuntimeError):
        await boom()
    [rec] = records(caplog)
    assert rec["status"] == "error"


def test_non_serializable_params_do_not_break(caplog):
    @audited_tool
    def f(obj):
        return 1

    with caplog.at_level(logging.INFO, logger=AUDIT_LOGGER_NAME):
        assert f(object()) == 1
    assert len(records(caplog)) == 1
