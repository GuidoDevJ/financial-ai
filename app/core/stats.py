"""Small statistics helpers shared by the latency test and the benchmark script."""

import math


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (`pct` in 0..100) of a non-empty list."""
    if not values:
        raise ValueError("percentile of an empty list")
    ordered = sorted(values)
    rank = max(math.ceil(pct / 100 * len(ordered)), 1)
    return ordered[min(rank, len(ordered)) - 1]
