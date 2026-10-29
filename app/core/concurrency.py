"""Small helper to fan blocking calls out to threads without losing the request context."""

import contextvars
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_MAX_WORKERS = 8


def map_in_threads(
    func: Callable[[T], R], items: Iterable[T], max_workers: int = DEFAULT_MAX_WORKERS
) -> list[R]:
    """`[func(i) for i in items]` run concurrently; results keep the input order.

    Each call runs in a copy of the caller's contextvars, so the correlation id and the client
    bound by the request stay visible to audit logs. `func` is expected to handle its own
    errors (an exception propagates when its result is collected).
    """
    batch = list(items)
    if len(batch) <= 1:
        return [func(item) for item in batch]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as pool:
        futures = [pool.submit(contextvars.copy_context().run, func, item) for item in batch]
        return [future.result() for future in futures]
