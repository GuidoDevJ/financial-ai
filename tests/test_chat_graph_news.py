import json
from datetime import UTC, datetime, timedelta

import fakeredis
from langchain_core.messages import AIMessage, ToolMessage
from llama_index.core.vector_stores import SimpleVectorStore

from app.agents.chat_graph import build_chat_graph
from app.brief.store import BriefStore
from app.core.audit import request_context
from app.core.config import Settings
from app.rag.ingest_news import NewsItem, item_to_document
from app.rag.store import NewsStore
from app.tools.get_last_brief import make_get_last_brief
from app.tools.get_portfolio import make_get_portfolio
from app.tools.registry import build_tools
from app.tools.search_news import make_search_news
from tests.fakes import (
    CHAT_PROMPT,
    BagOfWordsEmbedding,
    reference_prompt_text,
    scripted,
    tool_call_msg,
)

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


def all_tools():
    store = NewsStore(SimpleVectorStore(), BagOfWordsEmbedding())
    store.upsert(
        [
            item_to_document(
                NewsItem(
                    "Galicia sube tras balance",
                    "GGAL lidera",
                    "https://x.com/1",
                    "Ambito",
                    NOW - timedelta(hours=3),
                )
            )
        ]
    )
    briefs = BriefStore(fakeredis.FakeRedis(decode_responses=True))
    briefs.save("c1", "2026-10-14", "TU CARTERA ...", ["GGAL"])

    def price(ticker, mercado):
        return {"precio_cierre": 10.0}

    return build_tools(
        get_price=price,
        get_macro=lambda variable: {"valor": 1000.0},
        search_news=make_search_news(
            store=store, settings=Settings(_env_file=None), now=lambda: NOW
        ),
        get_portfolio=make_get_portfolio(
            get_price=price,
            get_macro=lambda variable: {"valor": 1000.0},
            portfolios={"c1": [{"ticker": "GGAL", "cantidad": 3, "mercado": "nyse"}]},
        ),
        get_last_brief=make_get_last_brief(store=briefs),
    )


def cfg():
    return {"configurable": {"thread_id": "c1:s1"}}


def test_registry_exposes_all_five_tools():
    assert {t.name for t in all_tools()} == {
        "get_price",
        "get_macro",
        "search_news",
        "get_portfolio",
        "get_last_brief",
    }


def test_search_news_schema_matches_spec():
    tool = next(t for t in all_tools() if t.name == "search_news")
    schema = tool.args_schema.model_json_schema()
    assert schema["required"] == ["query"]
    assert set(schema["properties"]) == {"query", "ticker", "horas"}


async def test_graph_calls_search_news_and_answers_with_it():
    model = scripted(
        tool_call_msg("search_news", {"query": "Galicia balance", "ticker": "GGAL"}),
        AIMessage(content="Segun Ambito (2026-10-15), Galicia sube tras balance."),
    )
    graph = build_chat_graph(model, all_tools(), prompt_source=CHAT_PROMPT)
    result = await graph.ainvoke({"messages": [("user", "que paso con Galicia?")]}, cfg())
    [tm] = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    payload = json.loads(tm.content)
    assert payload[0]["titulo"] == "Galicia sube tras balance"
    assert payload[0]["fuente"] == "Ambito" and payload[0]["url"] == "https://x.com/1"
    assert "Ambito" in result["messages"][-1].content


async def test_graph_no_news_surfaces_explicit_error():
    model = scripted(
        tool_call_msg("search_news", {"query": "meli", "ticker": "MELI"}),
        AIMessage(content="No encontre noticias."),
    )
    graph = build_chat_graph(model, all_tools(), prompt_source=CHAT_PROMPT)
    result = await graph.ainvoke({"messages": [("user", "noticias de MELI")]}, cfg())
    [tm] = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert json.loads(tm.content)["error"] == "NO_NEWS_FOUND"


async def test_graph_portfolio_and_brief_tools():
    model = scripted(
        tool_call_msg("get_portfolio", {}, "a"),
        tool_call_msg("get_last_brief", {}, "b"),
        AIMessage(content="ok"),
    )
    graph = build_chat_graph(model, all_tools(), prompt_source=CHAT_PROMPT)
    # The client comes from the authenticated session, not from the model's arguments.
    with request_context(cliente_id="c1"):
        result = await graph.ainvoke({"messages": [("user", "mi cartera y brief")]}, cfg())
    payloads = [json.loads(m.content) for m in result["messages"] if isinstance(m, ToolMessage)]
    assert payloads[0][0]["valorizacion"] == 30.0
    assert payloads[1]["fecha"] == "2026-10-14"


def test_prompt_requires_citing_news_source_and_date():
    prompt = reference_prompt_text("chat")
    assert "search_news" in prompt and "medio" in prompt
