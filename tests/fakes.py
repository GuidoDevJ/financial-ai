import hashlib
import math
import re
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from llama_index.core.base.embeddings.base import BaseEmbedding

from app.prompts.registry import StaticPrompt

ROOT = Path(__file__).resolve().parent.parent


def reference_prompt_text(name: str) -> str:
    """Current text of a prompt in docs/prompts/registry.yaml (the Bedrock source text)."""
    from app.prompts import reference

    return reference.current_text(reference.load_reference()[name])


# Static stand-ins for the prompts stored in Bedrock Prompt Management.
CHAT_PROMPT = StaticPrompt(
    "Sos un asistente de información de inversiones. Para noticias, usá search_news y citá "
    "siempre el medio y la fecha."
)
# The real brief template (all {{variables}}), so tests exercise the published text.
BRIEF_PROMPT = StaticPrompt(reference_prompt_text("brief"), name="brief", version="2")


class ScriptedChatModel(FakeMessagesListChatModel):
    """Fake chat model that replays scripted AIMessages and records what it was shown."""

    seen: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def scripted(*messages: AIMessage) -> ScriptedChatModel:
    return ScriptedChatModel(responses=list(messages), seen=[])


def tool_call_msg(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class BagOfWordsEmbedding(BaseEmbedding):
    """Deterministic offline embedding: hashed bag of words, L2-normalised.

    Texts sharing words get a high cosine similarity, so retrieval tests are meaningful
    without any model or network (MockEmbedding returns a constant vector).
    """

    dim: int = 256

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"\w+", text.lower()):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)
