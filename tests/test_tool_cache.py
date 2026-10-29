import fakeredis

from app.tools.cache import cached_call


def make():
    return fakeredis.FakeRedis(decode_responses=True)


def test_miss_then_hit_calls_compute_once():
    r, calls = make(), []

    def compute():
        calls.append(1)
        return {"a": 1}

    assert cached_call(r, "k", 60, compute) == {"a": 1}
    assert cached_call(r, "k", 60, compute) == {"a": 1}
    assert len(calls) == 1
    assert 0 < r.ttl("k") <= 60


def test_exceptions_are_not_cached():
    r = make()

    def boom():
        raise RuntimeError("x")

    try:
        cached_call(r, "k", 60, boom)
    except RuntimeError:
        pass
    assert r.get("k") is None


def test_redis_failure_degrades_to_compute():
    class Down:
        def get(self, *a):
            raise ConnectionError

        def setex(self, *a):
            raise ConnectionError

    assert cached_call(Down(), "k", 60, lambda: {"ok": True}) == {"ok": True}
