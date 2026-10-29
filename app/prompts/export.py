"""Print the `aws bedrock-agent create-prompt --cli-input-json` payload for a prompt.

    uv run python -m app.prompts.export [--name chat|brief]

Generated offline from docs/prompts/registry.yaml (`current_version`); makes no AWS call and
creates nothing. Creating the resources is a manual step (see docs/prompts/README.md).
"""

import argparse
import json
import sys

from app.prompts import reference

NEXT_STEPS = (
    "# Manual steps (nothing was created):\n"
    "#   1. aws bedrock-agent create-prompt --cli-input-json file://<payload>.json\n"
    "#   2. aws bedrock-agent create-prompt-version --prompt-identifier <id from step 1>\n"
    "#   3. set BEDROCK_PROMPT_<NAME>_ID=<id> and BEDROCK_PROMPT_VERSION=<version number>"
)


def build_payload(entry: dict) -> dict:
    """CreatePrompt request body for the entry's current version."""
    payload = {
        "name": entry["name"],
        "description": entry["description"],
        "defaultVariant": entry["defaultVariant"],
        "variants": reference.current_variants(entry),
    }
    for variant in payload["variants"]:
        declared = {v["name"] for v in variant["templateConfiguration"]["text"]["inputVariables"]}
        used = set(reference.placeholders(reference.variant_text(variant)))
        if declared != used:
            raise ValueError(
                f"variant '{variant['name']}': inputVariables {sorted(declared)} "
                f"do not match the {{{{variables}}}} in the text {sorted(used)}"
            )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.prompts.export")
    parser.add_argument("--name", choices=["chat", "brief"], help="prompt to export (default: all)")
    args = parser.parse_args(argv)
    registry = reference.load_reference()
    names = [args.name] if args.name else list(registry)
    for name in names:
        version = reference.current_version(registry[name])
        print(f"# {name}: create-prompt payload (registry.yaml version {version})", file=sys.stderr)
        print(json.dumps(build_payload(registry[name]), ensure_ascii=False, indent=2))
    print(NEXT_STEPS, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
