"""Live latency benchmark for POST /chat (not run in tests).

Usage:
    uv run python scripts/bench_chat.py [--url http://localhost:8000] [--n 100]
        [--concurrency 1] [--api-key KEY] [--cliente demo] [--budget 5]

Prints p50 / p95 / p99 / max in seconds and exits 1 when p95 is not under --budget (the Phase 1
acceptance criterion is p95 < 5 s). Needs a running server with real AWS credentials, so every
call costs tokens: start with a small --n. `BENCH_API_KEY` can replace --api-key.
"""

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.stats import percentile  # noqa: E402

QUESTIONS = [
    "¿Cuánto cerró GGAL en NYSE?",
    "¿A cuánto está el riesgo país?",
    "¿Cuál es el dólar CCL hoy?",
    "¿Qué noticias hay sobre YPF?",
    "Mostrame mi cartera.",
]


def one_call(client: httpx.Client, question: str, cliente: str, headers: dict) -> tuple[float, int]:
    started = time.perf_counter()
    try:
        status = client.post(
            "/chat", json={"cliente_id": cliente, "message": question}, headers=headers
        ).status_code
    except httpx.HTTPError:
        status = 0
    return time.perf_counter() - started, status


def run_bench(
    client: httpx.Client,
    n: int,
    *,
    concurrency: int = 1,
    cliente: str = "demo",
    api_key: str | None = None,
) -> dict:
    headers = {"X-API-Key": api_key} if api_key else {}
    questions = itertools.islice(itertools.cycle(QUESTIONS), n)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as pool:
        results = list(pool.map(lambda q: one_call(client, q, cliente, headers), questions))
    wall = time.perf_counter() - started
    ok = [d for d, status in results if status == 200]
    return {
        "requests": len(results),
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "wall_seconds": round(wall, 2),
        **({f"p{p}": round(percentile(ok, p), 3) for p in (50, 95, 99)} if ok else {}),
        **({"max": round(max(ok), 3)} if ok else {}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY"))
    parser.add_argument("--cliente", default="demo")
    parser.add_argument("--budget", type=float, default=5.0, help="p95 budget in seconds")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    with httpx.Client(base_url=args.url, timeout=args.timeout) as client:
        report = run_bench(
            client,
            args.n,
            concurrency=args.concurrency,
            cliente=args.cliente,
            api_key=args.api_key,
        )
    for key, value in report.items():
        print(f"{key:>13}: {value}")
    if "p95" not in report:
        print("FAIL: no successful request")
        return 1
    passed = report["p95"] < args.budget and report["errors"] == 0
    print(f"{'PASS' if passed else 'FAIL'}: p95 {report['p95']}s vs budget {args.budget}s")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
