import redis

from app.core.config import Settings, get_settings


def get_redis_client(settings: Settings | None = None) -> redis.Redis:
    """Build a Redis client from settings (lazy: no connection until first command)."""
    settings = settings or get_settings()
    return redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=settings.redis_socket_timeout,
    )
