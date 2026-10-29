"""LangChain tool wrappers around the framework-free tool functions."""

import json
import logging
from collections.abc import Callable
from typing import Annotated

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.core.audit import audited_tool, cliente_id_var, log_event
from app.tools.errors import ToolError
from app.tools.get_macro import VARIABLES
from app.tools.get_price import MERCADOS

SESSION_BOUND = frozenset(
    {"get_portfolio", "get_portfolio_performance", "simulate_scenario", "get_last_brief"}
)


# Enums are declared via json_schema_extra so the LLM sees them, while an out-of-range value
# reaches our function and comes back as an explicit INVALID_PARAMS payload (not a framework error).
class PriceArgs(BaseModel):
    ticker: str = Field(description="Ej. GGAL, YPFD.BA")
    mercado: Annotated[
        str,
        Field(
            description="'byma' para acciones argentinas locales (ej. PAMP); 'nyse' para ADRs.",
            json_schema_extra={"enum": list(MERCADOS)},
        ),
    ]


class MacroArgs(BaseModel):
    variable: Annotated[str, Field(json_schema_extra={"enum": list(VARIABLES)})]


class NewsArgs(BaseModel):
    query: str
    ticker: str | None = Field(default=None, description="opcional")
    horas: int = Field(default=72, description="ventana de tiempo, default 72")


# The client is never a model argument: get_portfolio / get_last_brief read it from the
# authenticated session (`cliente_id_var`, bound by /chat). The schemas therefore declare no
# `cliente_id`, and `SessionBoundTool` drops any value the model sends anyway, logging
# `cliente_id_override_attempt`.
class PortfolioArgs(BaseModel):
    pass


class PerformanceArgs(BaseModel):
    periodo: Annotated[
        str,
        Field(
            description="Ventana: 'dia' (último cierre), 'semana' (7 días) o 'mes' (30 días).",
            json_schema_extra={"enum": ["dia", "semana", "mes"]},
        ),
    ]


class ScenarioArgs(BaseModel):
    ccl_pct: float = Field(default=0, description="Variación hipotética del dólar CCL en %, ej. 10")
    precio_pct: float = Field(
        default=0, description="Variación hipotética del precio de todos los activos en %, ej. -5"
    )


class BriefArgs(BaseModel):
    fecha: str | None = Field(default=None, description="opcional, YYYY-MM-DD; default el último")


class SessionBoundTool(StructuredTool):
    """Tool whose `cliente_id` comes from the session, never from the model's arguments."""

    def _to_args_and_kwargs(self, tool_input, tool_call_id):
        if isinstance(tool_input, dict) and "cliente_id" in tool_input:
            attempted = tool_input["cliente_id"]
            tool_input = {k: v for k, v in tool_input.items() if k != "cliente_id"}
            session_client = cliente_id_var.get()
            if attempted != session_client:
                log_event(
                    "security",
                    "cliente_id_override_attempt",
                    level=logging.WARNING,
                    tool=self.name,
                    attempted_cliente_id=attempted,
                    session_cliente_id=session_client,
                )
        return super()._to_args_and_kwargs(tool_input, tool_call_id)


def _session_bound(func: Callable[..., object]) -> Callable[..., object]:
    """Call `func` with the session's `cliente_id` (error payload when there is none)."""

    def run(**kwargs):
        session_client = cliente_id_var.get()
        if not session_client:
            raise ToolError(
                "NO_SESSION_CLIENT", "No authenticated client is bound to this request."
            )
        return func(cliente_id=session_client, **kwargs)

    run.__name__ = func.__name__
    return run


# Every tool is wrapped with `audited_tool` here, so one uniform audit record per call is
# guaranteed even for plain functions; already-audited factory functions are left untouched.
def _safe(func: Callable[..., dict]) -> Callable[..., str]:
    def run(**kwargs) -> str:
        try:
            return json.dumps(func(**kwargs), ensure_ascii=False)
        except ToolError as exc:
            return json.dumps(exc.to_payload(), ensure_ascii=False)
        except Exception as exc:
            err = ToolError("INTERNAL_ERROR", f"{type(exc).__name__}: {exc}")
            return json.dumps(err.to_payload(), ensure_ascii=False)

    return run


def build_tools(
    *,
    get_price: Callable[..., dict],
    get_macro: Callable[..., dict],
    search_news: Callable[..., list] | None = None,
    get_portfolio: Callable[..., list] | None = None,
    get_portfolio_performance: Callable[..., dict] | None = None,
    simulate_scenario: Callable[..., dict] | None = None,
    get_last_brief: Callable[..., dict] | None = None,
) -> list:
    tools = [
        StructuredTool.from_function(
            func=_safe(audited_tool(get_price, name="get_price")),
            name="get_price",
            description=(
                "Precio de cierre y variación de un activo (acción argentina en BYMA o ADR)."
            ),
            args_schema=PriceArgs,
        ),
        StructuredTool.from_function(
            func=_safe(audited_tool(get_macro, name="get_macro")),
            name="get_macro",
            description="Valor de una variable macro argentina.",
            args_schema=MacroArgs,
        ),
    ]
    optional = [
        (
            search_news,
            "search_news",
            "Busca noticias financieras recientes, opcionalmente filtradas por activo.",
            NewsArgs,
        ),
        (
            get_portfolio,
            "get_portfolio",
            "Cartera del cliente: tenencias, cantidad, valorización y peso de cada activo.",
            PortfolioArgs,
        ),
        (
            get_portfolio_performance,
            "get_portfolio_performance",
            "Explica por qué se movió la cartera del cliente en un período: retorno total, efecto "
            "precio vs. efecto tipo de cambio (CCL) y contribución de cada activo, en puntos "
            "porcentuales. Asume la tenencia actual durante todo el período.",
            PerformanceArgs,
        ),
        (
            simulate_scenario,
            "simulate_scenario",
            "Simula cuánto valdría la cartera del cliente si el dólar CCL y/o los precios variaran "
            "un porcentaje hipotético. Es un cálculo, no una predicción.",
            ScenarioArgs,
        ),
        (
            get_last_brief,
            "get_last_brief",
            "Último brief diario generado para el cliente, para que el chat lo referencie "
            "o responda preguntas sobre él.",
            BriefArgs,
        ),
    ]
    for func, name, description, args in optional:
        if func is not None:
            audited = audited_tool(func, name=name)
            tool_class = StructuredTool
            if name in SESSION_BOUND:
                audited = _session_bound(audited)
                tool_class = SessionBoundTool
            tools.append(
                tool_class.from_function(
                    func=_safe(audited),
                    name=name,
                    description=description,
                    args_schema=args,
                )
            )
    return tools
