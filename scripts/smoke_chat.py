"""Manual smoke test: POST a sample question to a locally running server (not run in tests).

Usage: uv run python scripts/smoke_chat.py ["question"]
Requires `uv run uvicorn app.main:app` running and AWS credentials available
(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in .env, env vars or an IAM role)
with access to BEDROCK_MODEL_ID.
"""

import json
import sys

import httpx

URL = "http://localhost:8000/chat"
DEFAULT_QUESTION = "Cuanto cerro GGAL en NYSE y a cuanto esta el riesgo pais?"


def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    resp = httpx.post(URL, json={"cliente_id": "demo", "message": question}, timeout=60)
    print(resp.status_code)
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
    return 0 if resp.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
