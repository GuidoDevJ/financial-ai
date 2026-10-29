import fakeredis

from app.core import redis_client
from app.core.config import Settings


def test_factory_uses_settings_url(monkeypatch):
    captured = {}

    def fake_from_url(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return fakeredis.FakeRedis(decode_responses=True)

    monkeypatch.setattr(redis_client.redis.Redis, "from_url", staticmethod(fake_from_url))
    client = redis_client.get_redis_client(Settings(_env_file=None, redis_url="redis://x:1/3"))
    assert captured["url"] == "redis://x:1/3"
    assert captured["kwargs"]["decode_responses"] is True
    assert captured["kwargs"]["socket_timeout"] == 5.0  # Settings default
    client.set("k", "v")
    assert client.get("k") == "v"
