"""Shared fixtures: keep tests independent of the developer's local .env file."""

import pytest

from app.core.config import get_settings

# Values the app reads through the cached get_settings(); pinned so a local .env cannot change
# the behaviour of endpoint tests (auth open in dev, default timeouts, valid prompt version).
_PINNED_ENV = {
    "APP_ENV": "dev",
    "AUTH_API_KEYS": "",
    "ADMIN_API_KEYS": "",
    "BEDROCK_PROMPT_VERSION": "DRAFT",
    "CHAT_TIMEOUT_SECONDS": "25",
    "CHAT_MAX_TOOL_ITERATIONS": "6",
    "CHAT_STREAM_TOKENS": "false",
    "WARMUP_ON_STARTUP": "false",  # lifespan tests must not wire real Bedrock/Redis clients
}


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    for key, value in _PINNED_ENV.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def set_env(monkeypatch):
    """Set env vars for the app settings and rebuild the cached settings."""

    def apply(**env: str):
        for key, value in env.items():
            monkeypatch.setenv(key.upper(), value)
        get_settings.cache_clear()
        return get_settings()

    return apply
