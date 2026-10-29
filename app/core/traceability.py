"""Audit check: every number in the final chat answer must come from a tool call.

Acceptance criterion "100% of quantitative data comes from a tool call". The numbers of the final
reply are compared (same rounding rules as the brief validator) with every number in the tool
outputs of the conversation plus the numbers the user wrote. It is an audit signal, not a filter:
untraced numbers are logged as `untraced_numbers` WARNING records.
"""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.brief.validate import untraceable_numbers
from app.core.audit import log_event

CONTEXT_RADIUS = 40


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block) for block in content
    )


def untraced_numbers(messages: list) -> list[str]:
    """Numbers in the last model reply that appear in no tool output or user message."""
    final = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None
    )
    if final is None:
        return []
    evidence: list = []
    for message in messages:
        if isinstance(message, ToolMessage):
            raw = _content_text(message.content)
            try:
                evidence.append(json.loads(raw))
            except ValueError:
                evidence.append(raw)
        elif isinstance(message, HumanMessage):
            evidence.append(_content_text(message.content))
    return untraceable_numbers(_content_text(final.content), evidence)


def number_contexts(text: str, numbers: list[str]) -> dict[str, str]:
    """Short snippet of `text` around the first occurrence of each number, to diagnose it."""
    contexts: dict[str, str] = {}
    for number in numbers:
        index = text.find(number)
        if index >= 0:
            start = max(0, index - CONTEXT_RADIUS)
            contexts[number] = text[start : index + len(number) + CONTEXT_RADIUS]
    return contexts


def _final_reply_text(messages: list) -> str:
    final = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None
    )
    return _content_text(final.content) if final else ""


def audit_traceability(messages: list) -> list[str]:
    """Log and return the untraced numbers of the final reply (empty list = fully traced)."""
    numbers = untraced_numbers(messages)
    if numbers:
        logged = numbers[:20]
        log_event(
            "chat",
            "untraced_numbers",
            level=logging.WARNING,
            numbers=logged,
            count=len(numbers),
            contexts=number_contexts(_final_reply_text(messages), logged),
        )
    return numbers
