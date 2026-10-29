import time

from app.core.audit import request_context, request_id_var
from app.core.concurrency import map_in_threads


def test_results_keep_input_order_and_run_concurrently():
    def slow(n):
        time.sleep(0.2)
        return n * 2

    started = time.perf_counter()
    assert map_in_threads(slow, [1, 2, 3, 4, 5]) == [2, 4, 6, 8, 10]
    assert time.perf_counter() - started < 0.6  # sequential would take 1.0s


def test_request_context_is_visible_in_the_workers():
    with request_context(request_id="rid-12345678"):
        assert map_in_threads(lambda _: request_id_var.get(), [1, 2, 3]) == ["rid-12345678"] * 3


def test_empty_and_single_item_inputs():
    assert map_in_threads(lambda x: x, []) == []
    assert map_in_threads(lambda x: x + 1, [1]) == [2]
