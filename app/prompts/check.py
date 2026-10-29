"""Verify that both prompts load from Amazon Bedrock Prompt Management.

    uv run python -m app.prompts.check

Fetches the chat and brief prompts and prints name, id, version, hash, length, drift status and
the first 80 characters (never credentials). Drift compares the hash of the published text with
the `current_version` text in docs/prompts/registry.yaml: MATCH, DRIFT (warning only) or
NO_REFERENCE. Exit code 0 if both load (even on drift), 1 if any fetch fails.
"""

import logging
import sys

import yaml
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import Settings, get_settings
from app.prompts import reference
from app.prompts.registry import PromptError, PromptProvider, PromptRegistry

PROMPT_NAMES = ("chat", "brief")
PREVIEW_CHARS = 80


def drift_status(name: str, published_hash: str, ref: dict | None) -> str:
    """MATCH / DRIFT against registry.yaml, or NO_REFERENCE when it has no usable entry."""
    try:
        expected = reference.text_hash(reference.current_text((ref or {})[name]))
    except (KeyError, TypeError, IndexError):
        return "NO_REFERENCE"
    return "MATCH" if expected == published_hash else "DRIFT"


def _load_reference() -> dict | None:
    try:
        return reference.load_reference()
    except (OSError, yaml.YAMLError):
        return None


def _check(name: str, registry: PromptRegistry, ref: dict | None) -> bool:
    try:
        source = registry.source(name)
        prompt = source.get()
    except PromptError as exc:
        print(f"FAIL {name}: {exc.code}: {exc.message}")
        return False
    except (ClientError, BotoCoreError) as exc:
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        return False
    drift = drift_status(name, prompt.hash, ref)
    preview = " ".join(prompt.content.split())[:PREVIEW_CHARS]
    print(
        f"OK   {name}: id={registry.prompt_id(name)} version={prompt.version} "
        f"hash={prompt.hash} length={len(prompt.content)} drift={drift} text={preview!r}"
    )
    if drift == "DRIFT":
        print(
            f"WARN {name}: the published prompt differs from docs/prompts/registry.yaml "
            f"(version {reference.current_version(ref[name])}); update the file or republish."
        )
    return True


def main(settings: Settings | None = None, provider: PromptProvider | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    settings = settings or get_settings()
    registry = PromptRegistry(settings, provider)
    ref = _load_reference()
    results = [_check(name, registry, ref) for name in PROMPT_NAMES]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
