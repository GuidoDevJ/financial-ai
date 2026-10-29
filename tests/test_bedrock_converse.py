"""Converse API payload shape, metadata and audit (no network, no AWS)."""

import json
import logging

import fakeredis

from app.agents.chat_graph import build_chat_graph
from app.agents.llm import build_chat_model, build_request_metadata
from app.core.audit import LLM_AUDIT_LOGGER_NAME, log_llm_call
from app.core.config import Settings
from app.tools.get_macro import make_get_macro
from app.tools.registry import build_tools
from tests.fakes import CHAT_PROMPT


def settings(**kw) -> Settings:
    base = {"aws_region": "us-east-1", "bedrock_model_id": "us.anthropic.claude-test"}
    return Settings(_env_file=None, **{**base, **kw})


def converse_response(text="hola", stop="end_turn", **usage):
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": stop,
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15, **usage},
        "metrics": {"latencyMs": 123},
    }


class FakeBedrockClient:
    """Stands in for the boto3 bedrock-runtime client; records every `converse` call."""

    def __init__(self, *responses):
        self.responses = list(responses) or [converse_response()]
        self.calls: list[dict] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def tools():
    r = fakeredis.FakeRedis(decode_responses=True)
    return build_tools(
        get_price=lambda ticker, mercado: {},
        get_macro=make_get_macro(redis_client=r, providers={}),
    )


async def run(client, s=None, **graph_kw):
    s = s or settings()
    model = build_chat_model(s, client=client)
    graph = build_chat_graph(
        model,
        tools(),
        prompt_source=graph_kw.pop("prompt_source", CHAT_PROMPT),
        request_metadata_base={"app": "financial-ai", "env": "test"},
        **graph_kw,
    )
    cfg = {
        "configurable": {"thread_id": "c1:s1", "cliente_id": "c1", "session_id": "s1"},
        "recursion_limit": 12,
    }
    return await graph.ainvoke({"messages": [("user", "hola")]}, cfg)


# ---------------------------------------------------------------- payload shape


async def test_system_prompt_goes_in_converse_system_field():
    client = FakeBedrockClient()
    await run(client)
    [call] = client.calls
    assert call["system"] == [{"text": CHAT_PROMPT.text}]
    assert call["messages"] == [{"role": "user", "content": [{"text": "hola"}]}]
    assert call["modelId"] == "us.anthropic.claude-test"


async def test_inference_config_and_tool_config():
    client = FakeBedrockClient()
    await run(client, settings(bedrock_max_tokens=256, bedrock_temperature=0.2))
    [call] = client.calls
    assert call["inferenceConfig"]["maxTokens"] == 256
    assert call["inferenceConfig"]["temperature"] == 0.2
    names = {t["toolSpec"]["name"] for t in call["toolConfig"]["tools"]}
    assert {"get_price", "get_macro"} <= names
    assert "serviceTier" not in call
    assert "guardrailConfig" not in call
    assert "cachePoint" not in json.dumps(call)


async def test_request_metadata_has_ids_and_no_extra_pii():
    client = FakeBedrockClient()
    await run(client)
    assert client.calls[0]["requestMetadata"] == {
        "app": "financial-ai",
        "env": "test",
        "clienteId": "c1",
        "sessionId": "s1",
    }


def test_request_metadata_sanitised():
    md = build_request_metadata({"app": "x"}, "ana@mail.com ñ", "s" * 400)
    assert md["clienteId"] == "ana@mail.com _"
    assert len(md["sessionId"]) == 256


# ---------------------------------------------------------------- audit


def test_log_llm_call_fields(caplog):
    from langchain_core.messages import AIMessage

    msg = AIMessage(
        content="x",
        usage_metadata={
            "input_tokens": 15,
            "output_tokens": 5,
            "total_tokens": 20,
            "input_token_details": {"cache_read": 3, "cache_creation": 2},
        },
        response_metadata={"stopReason": "max_tokens", "metrics": {"latencyMs": [123]}},
    )
    with caplog.at_level(logging.INFO, logger=LLM_AUDIT_LOGGER_NAME):
        log_llm_call(msg, cliente_id="c1", session_id="s1")
    [rec] = [r for r in caplog.records if r.name == LLM_AUDIT_LOGGER_NAME]
    data = json.loads(rec.getMessage())
    assert data["cliente_id"] == "c1" and data["session_id"] == "s1"
    assert data["stop_reason"] == "max_tokens"
    assert data["input_tokens"] == 10  # Converse inputTokens (uncached)
    assert data["output_tokens"] == 5 and data["total_tokens"] == 20
    assert data["cache_read_input_tokens"] == 3
    assert data["cache_write_input_tokens"] == 2
    assert data["latency_ms"] == 123
    assert "timestamp" in data


async def test_graph_logs_llm_call_with_real_converse_response(caplog):
    client = FakeBedrockClient(
        converse_response(cacheReadInputTokens=4, cacheWriteInputTokens=1, totalTokens=20)
    )
    with caplog.at_level(logging.INFO, logger=LLM_AUDIT_LOGGER_NAME):
        await run(client)
    [rec] = [r for r in caplog.records if r.name == LLM_AUDIT_LOGGER_NAME]
    data = json.loads(rec.getMessage())
    assert data["cliente_id"] == "c1" and data["session_id"] == "s1"
    assert data["stop_reason"] == "end_turn"
    assert data["input_tokens"] == 10
    assert data["cache_read_input_tokens"] == 4
    assert data["cache_write_input_tokens"] == 1
    assert data["latency_ms"] == 123
