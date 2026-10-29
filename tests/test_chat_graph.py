import json
from datetime import UTC, date, datetime

import fakeredis
import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from app.agents.chat_graph import build_chat_graph, chat_prompt_variables, thread_id_for
from app.prompts.registry import PromptError, PromptVersion, StaticPrompt
from app.tools.errors import ToolError
from app.tools.get_macro import make_get_macro
from app.tools.get_price import PriceSeries, make_get_price
from app.tools.registry import build_tools
from tests.fakes import CHAT_PROMPT, reference_prompt_text, scripted, tool_call_msg


def tools_with(price_provider, macro_providers=None):
    r = fakeredis.FakeRedis(decode_responses=True)
    return build_tools(
        get_price=make_get_price(redis_client=r, providers={"nyse": price_provider}),
        get_macro=make_get_macro(redis_client=r, providers=macro_providers or {}),
    )


def good_price(ticker):
    return PriceSeries("fake", [(date(2026, 10, 13), 10.0), (date(2026, 10, 14), 11.0)])


def config(tid="c1:s1"):
    return {"configurable": {"thread_id": tid}}


def tool_msgs(result):
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


def test_thread_id():
    assert thread_id_for("c1", "s9") == "c1:s9"


def test_chat_prompt_variables_use_the_buenos_aires_date():
    # 01:30 UTC on the 17th is still the 16th in Buenos Aires (UTC-3).
    now = datetime(2026, 10, 17, 1, 30, tzinfo=UTC)
    assert chat_prompt_variables("Banco Demo", now) == {
        "banco": "Banco Demo",
        "fecha_actual": "2026-10-16",
    }


async def test_system_prompt_is_rendered_with_bank_and_date_on_every_turn():
    source = StaticPrompt("Sos el asistente de {{banco}}. Hoy es {{fecha_actual}}.")
    model = scripted(AIMessage(content="uno"), AIMessage(content="dos"))
    days = iter([datetime(2026, 10, 16, 15, tzinfo=UTC), datetime(2026, 10, 17, 15, tzinfo=UTC)])
    graph = build_chat_graph(
        model,
        tools_with(good_price),
        prompt_source=source,
        bank_name="Banco Demo",
        now=lambda: next(days),
    )
    await graph.ainvoke({"messages": [("user", "a")]}, config("r1"))
    await graph.ainvoke({"messages": [("user", "b")]}, config("r1"))
    assert model.seen[0][0].content == "Sos el asistente de Banco Demo. Hoy es 2026-10-16."
    assert model.seen[1][0].content == "Sos el asistente de Banco Demo. Hoy es 2026-10-17."


async def test_unresolved_variable_in_the_chat_prompt_fails_clearly():
    graph = build_chat_graph(
        scripted(AIMessage(content="x")),
        tools_with(good_price),
        prompt_source=StaticPrompt("Hola {{desconocida}}"),
    )
    with pytest.raises(PromptError) as err:
        await graph.ainvoke({"messages": [("user", "a")]}, config("r2"))
    assert err.value.code == "PROMPT_VARIABLES_MISSING"


def test_reference_chat_prompt_renders_with_the_graph_variables():
    text = reference_prompt_text("chat")
    rendered = PromptVersion("2", text, "chat").render(
        **chat_prompt_variables("Banco Demo", datetime(2026, 10, 16, 15, tzinfo=UTC))
    )
    assert "{{" not in rendered and "Banco Demo" in rendered and "2026-10-16" in rendered


def test_reference_prompt_file_has_the_chat_rules():
    # docs/prompts/registry.yaml is the source text published in Bedrock Prompt Management.
    text = reference_prompt_text("chat")
    assert "asistente" in text and "NUNCA" in text


async def test_tool_call_then_final_answer():
    model = scripted(
        tool_call_msg("get_price", {"ticker": "GGAL", "mercado": "nyse"}),
        AIMessage(content="GGAL cerro en 11."),
    )
    graph = build_chat_graph(model, tools_with(good_price), prompt_source=CHAT_PROMPT)
    result = await graph.ainvoke({"messages": [("user", "precio GGAL")]}, config())
    [tm] = tool_msgs(result)
    payload = json.loads(tm.content)
    assert payload["precio_cierre"] == 11.0 and payload["ticker"] == "GGAL"
    assert result["messages"][-1].content == "GGAL cerro en 11."
    # first model call saw the system prompt; second saw the tool result
    assert isinstance(model.seen[0][0], SystemMessage)
    assert any(isinstance(m, ToolMessage) for m in model.seen[1])


async def test_tool_error_surfaced_as_explicit_payload():
    def broken(ticker):
        raise ToolError("NO_DATA", "no price for ZZZ")

    model = scripted(
        tool_call_msg("get_price", {"ticker": "ZZZ", "mercado": "nyse"}),
        AIMessage(content="No hay dato disponible."),
    )
    graph = build_chat_graph(model, tools_with(broken), prompt_source=CHAT_PROMPT)
    result = await graph.ainvoke({"messages": [("user", "precio ZZZ")]}, config())
    [tm] = tool_msgs(result)
    assert json.loads(tm.content) == {"error": "NO_DATA", "message": "no price for ZZZ"}
    assert "precio_cierre" not in tm.content


async def test_invalid_args_surface_as_error_not_crash():
    model = scripted(
        tool_call_msg("get_macro", {"variable": "bitcoin"}), AIMessage(content="No puedo.")
    )
    graph = build_chat_graph(model, tools_with(good_price), prompt_source=CHAT_PROMPT)
    result = await graph.ainvoke({"messages": [("user", "x")]}, config())
    [tm] = tool_msgs(result)
    assert "error" in json.loads(tm.content)


async def test_checkpointer_keeps_thread_history():
    model = scripted(AIMessage(content="uno"), AIMessage(content="dos"))
    graph = build_chat_graph(model, tools_with(good_price), prompt_source=CHAT_PROMPT)
    await graph.ainvoke({"messages": [("user", "a")]}, config("t1"))
    await graph.ainvoke({"messages": [("user", "b")]}, config("t1"))
    assert len(model.seen[1]) == 4  # system + user a + ai uno + user b


@pytest.mark.parametrize("name", ["get_price", "get_macro"])
def test_registry_exposes_both_tools(name):
    assert name in {t.name for t in tools_with(good_price)}


# ------------------------------------------------------- latency: trimming and caching


async def test_history_is_trimmed_to_the_configured_limit_keeping_the_current_turn():
    model = scripted(*[AIMessage(content=f"r{i}") for i in range(4)])
    graph = build_chat_graph(
        model, tools_with(good_price), prompt_source=CHAT_PROMPT, history_max_messages=3
    )
    for i in range(4):
        await graph.ainvoke({"messages": [("user", f"pregunta {i}")]}, config())
    shown = model.seen[-1]
    assert isinstance(shown[0], SystemMessage)
    assert [m.content for m in shown[1:]] == ["pregunta 2", "r2", "pregunta 3"]
    # the checkpointed state keeps the whole conversation (only what is sent is trimmed)
    state = await graph.aget_state(config())
    assert len(state.values["messages"]) == 8


async def test_history_trim_never_orphans_tool_results_or_drops_the_current_turn():
    model = scripted(
        tool_call_msg("get_price", {"ticker": "GGAL", "mercado": "nyse"}, "p1"),
        AIMessage(content="GGAL 11"),
    )
    graph = build_chat_graph(
        model, tools_with(good_price), prompt_source=CHAT_PROMPT, history_max_messages=1
    )
    await graph.ainvoke({"messages": [("user", "precio GGAL")]}, config())
    second_round = model.seen[1]
    kinds = [type(m).__name__ for m in second_round]
    assert kinds == ["SystemMessage", "HumanMessage", "AIMessage", "ToolMessage"]


async def test_history_is_not_trimmed_by_default():
    model = scripted(*[AIMessage(content=f"r{i}") for i in range(3)])
    graph = build_chat_graph(model, tools_with(good_price), prompt_source=CHAT_PROMPT)
    for i in range(3):
        await graph.ainvoke({"messages": [("user", f"pregunta {i}")]}, config())
    assert len(model.seen[-1]) == 1 + 5  # system + 2 full earlier turns + current question


async def test_prompt_caching_passes_cache_control_only_when_enabled():
    from tests.fakes import ScriptedChatModel

    class Recording(ScriptedChatModel):
        kwargs_seen: list = []

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.kwargs_seen.append(kwargs)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    for enabled in (True, False):
        model = Recording(responses=[AIMessage(content="ok")], seen=[], kwargs_seen=[])
        graph = build_chat_graph(
            model, tools_with(good_price), prompt_source=CHAT_PROMPT, prompt_caching=enabled
        )
        await graph.ainvoke({"messages": [("user", "hola")]}, config())
        assert ("cache_control" in model.kwargs_seen[0]) is enabled
