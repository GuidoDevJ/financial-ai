"""Reader for docs/prompts/registry.yaml, the hand-maintained source of truth.

Used only by offline tooling (`app.prompts.check` drift detection, `app.prompts.export`) and
tests. The running application never reads this file.
"""

import hashlib
from pathlib import Path
from typing import Any

import yaml

from app.prompts.registry import VARIABLE_PATTERN

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "docs" / "prompts" / "registry.yaml"


def load_reference(path: Path | None = None) -> dict[str, Any]:
    with (path or REGISTRY_PATH).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def current_version(entry: dict) -> str:
    return str(entry["current_version"])


def current_variants(entry: dict) -> list[dict]:
    """The Bedrock `variants` of the entry's `current_version`."""
    return entry["versions"][current_version(entry)]["variants"]


def default_variant(entry: dict) -> dict:
    variants = current_variants(entry)
    wanted = entry.get("defaultVariant")
    return next((v for v in variants if v["name"] == wanted), variants[0])


def variant_text(variant: dict) -> str:
    return variant["templateConfiguration"]["text"]["text"]


def current_text(entry: dict) -> str:
    return variant_text(default_variant(entry)).strip()


def placeholders(text: str) -> list[str]:
    return sorted(set(VARIABLE_PATTERN.findall(text)))


def text_hash(text: str) -> str:
    """Same hash as `PromptVersion.hash` (sha256 of the stripped text, first 16 hex chars)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]
