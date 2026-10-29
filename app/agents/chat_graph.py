"""LangGraph ReAct-style chat graph: agent -> tools -> agent loop."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from app.agents.llm import build_chat_model, build_request_metadata
from app.brief.store import BriefStore
from app.core.audit import log_event, log_llm_call
from app.core.config import Settings, get_settings
from app.core.guardrail import BLOCKED_BY_GUARDRAIL, SAFE_REFUSAL, check_reply, log_blocked
from app.core.redis_client import get_redis_client
from app.prompts.registry import PromptSource, chat_prompt_source
from app.tools.get_ccl_history import make_get_ccl_at
from app.tools.get_last_brief import make_get_last_brief
from app.tools.get_macro import default_macro_providers, make_get_macro
from app.tools.get_portfolio import make_get_portfolio
from app.tools.get_portfolio_performance import make_get_portfolio_performance
from app.tools.get_price import make_get_price
from app.tools.registry import build_tools
from app.tools.search_news import make_search_news
from app.tools.simulate_scenario import make_simulate_scenario

logger = logging.getLogger(__name__)

CHAT_TZ = ZoneInfo("America/Argentina/Buenos_Aires")
DEFAULT_BANK_NAME = "Banco Comafi"


def chat_prompt_variables(bank_name: str, now: datetime) -> dict[str, str]:
    """Values of the chat prompt's `{{variables}}` for one turn."""
    return {"banco": bank_name, "fecha_actual": now.astimezone(CHAT_TZ).date().isoformat()}


DEFAULT_MAX_TOOL_ITERATIONS = 6
LOOP_LIMIT_MESSAGE = (
    "No pude completar la consulta: se alcanzó el máximo de consultas de datos permitidas "
    "para una sola respuesta. Probá con una pregunta más acotada."
)


def thread_id_for(cliente_id: str, session_id: str) -> str:
    return f"{cliente_id}:{session_id}"


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block) for block in content
    )


def _tool_rounds_this_turn(messages: list) -> int:
    """Model->tools rounds requested since the last user message (including the latest)."""
    rounds = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage) and message.tool_calls:
            rounds += 1
    return rounds


def trim_history(messages: list, max_messages: int) -> list:
    """Messages sent to the model: the current turn in full plus as much recent history as fits.

    `max_messages` <= 0 disables trimming. The current turn (from the last user message on,
    including its tool calls and results) is never cut, so a tool result cannot lose its call;
    older history is trimmed to start on a user message.
    """
    if max_messages <= 0 or len(messages) <= max_messages:
        return messages
    turn_start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    history, current = messages[:turn_start], messages[turn_start:]
    budget = max_messages - len(current)
    if budget <= 0 or not history:
        return current
    kept = trim_messages(
        history,
        max_tokens=budget,
        token_counter=len,  # one "token" per message
        strategy="last",
        start_on="human",
    )
    return [*kept, *current]


def build_chat_graph(
    model,
    tools: list,
    checkpointer=None,
    *,
    prompt_source: PromptSource,
    request_metadata_base: dict[str, str] | None = None,
    bank_name: str = DEFAULT_BANK_NAME,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    history_max_messages: int = 0,
    prompt_caching: bool = False,
):
    """Compile the chat graph. `model` is any chat model supporting `bind_tools`.

    The system prompt template comes from `prompt_source` (Bedrock Prompt Management through a
    TTL cache) on every turn and is rendered per turn with `banco` and `fecha_actual` (Buenos
    Aires date). It is a SystemMessage, which ChatBedrockConverse sends as the Converse
    `system` field. Per-call kwargs (`request_metadata`) are forwarded to the
    Converse request; fake chat models used in tests ignore them. `history_max_messages`
    (0 = off) bounds the history sent each round; `prompt_caching` asks Bedrock to place
    cachePoints on the system prompt, tools and messages.
    """
    model_with_tools = model.bind_tools(tools)

    async def agent(state: MessagesState, config: RunnableConfig) -> dict:
        configurable = config.get("configurable", {})
        cliente_id = configurable.get("cliente_id")
        session_id = configurable.get("session_id")
        call_kwargs: dict = {}
        if cliente_id and session_id:
            call_kwargs["request_metadata"] = build_request_metadata(
                request_metadata_base or {}, cliente_id, session_id
            )
        if prompt_caching:
            call_kwargs["cache_control"] = {"ttl": "5m"}
        prompt_version = await prompt_source.aget()
        prompt = SystemMessage(
            content=prompt_version.render(**chat_prompt_variables(bank_name, now()))
        )
        response = await model_with_tools.ainvoke(
            [prompt, *trim_history(state["messages"], history_max_messages)], config, **call_kwargs
        )
        log_llm_call(response, cliente_id=cliente_id, session_id=session_id, prompt=prompt_version)
        return {"messages": [response]}

    def route_after_agent(state: MessagesState) -> str:
        if not getattr(state["messages"][-1], "tool_calls", None):
            return "guardrail"
        return (
            "tools" if _tool_rounds_this_turn(state["messages"]) <= max_tool_iterations else "limit"
        )

    def limit(state: MessagesState) -> dict:
        """Stop a runaway tool loop: replace the dangling tool request by a plain message."""
        last = state["messages"][-1]
        log_event(
            "chat",
            "tool_loop_limit",
            level=logging.WARNING,
            max_tool_iterations=max_tool_iterations,
        )
        return {
            "messages": [
                AIMessage(
                    content=LOOP_LIMIT_MESSAGE,
                    id=last.id,
                    response_metadata={"stopReason": "tool_loop_limit"},
                )
            ]
        }

    def guardrail(state: MessagesState) -> dict:
        """Replace a final reply that contains an explicit buy/sell recommendation."""
        last = state["messages"][-1]
        text = _text(last.content)
        verdict = check_reply(text)
        if not verdict.blocked:
            return {}
        log_blocked(verdict, reply_chars=len(text))
        meta = {**last.response_metadata, "guardrail": BLOCKED_BY_GUARDRAIL}
        return {"messages": [AIMessage(content=SAFE_REFUSAL, id=last.id, response_metadata=meta)]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("guardrail", guardrail)
    builder.add_node("limit", limit)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "guardrail": "guardrail", "limit": "limit"},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("guardrail", END)
    builder.add_edge("limit", END)
    return builder.compile(checkpointer=checkpointer or MemorySaver())


async def make_checkpointer(settings: Settings):
    """Redis checkpointer (needs Redis Stack / RediSearch); MemorySaver fallback."""
    try:
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver

        saver = AsyncRedisSaver(redis_url=settings.redis_url)
        await saver.asetup()
        return saver
    except Exception as exc:
        logger.warning(
            "Redis checkpointer unavailable; using in-memory MemorySaver.", exc_info=True
        )
        log_event(
            "chat",
            "checkpointer_fallback",
            level=logging.WARNING,
            fallback="MemorySaver",
            error_type=type(exc).__name__,
        )
        return MemorySaver()


async def build_default_chat_graph(settings: Settings | None = None):
    """Production wiring: Bedrock model + real tools + Redis cache/checkpointer."""
    settings = settings or get_settings()
    prompt_source = chat_prompt_source(settings)  # PROMPT_ID_MISSING before any heavy wiring
    redis_client = get_redis_client(settings)
    http = httpx.Client(timeout=8, follow_redirects=True)
    get_price = make_get_price(redis_client=redis_client, settings=settings)
    get_macro = make_get_macro(
        redis_client=redis_client,
        settings=settings,
        providers=default_macro_providers(http),
    )
    tools = build_tools(
        get_price=get_price,
        get_macro=get_macro,
        # The news store (Redis Stack + embeddings) is built lazily on first search_news call.
        search_news=make_search_news(settings=settings, redis_client=redis_client),
        get_portfolio=make_get_portfolio(get_price=get_price, get_macro=get_macro),
        get_portfolio_performance=make_get_portfolio_performance(
            get_price=get_price,
            get_macro=get_macro,
            get_ccl_at=make_get_ccl_at(redis_client=redis_client, client=http),
        ),
        simulate_scenario=make_simulate_scenario(get_price=get_price, get_macro=get_macro),
        get_last_brief=make_get_last_brief(store=BriefStore(redis_client)),
    )
    model = build_chat_model(settings, for_chat=True)
    return build_chat_graph(
        model,
        tools,
        await make_checkpointer(settings),
        prompt_source=prompt_source,
        request_metadata_base={"app": settings.app_name, "env": settings.app_env},
        bank_name=settings.bank_name,
        max_tool_iterations=settings.chat_max_tool_iterations,
        history_max_messages=settings.chat_history_max_messages,
        prompt_caching=settings.bedrock_prompt_caching,
    )
