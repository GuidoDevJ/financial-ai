import pytest
from pydantic import ValidationError

from app.core.config import Settings


def make(monkeypatch, **env):
    for key in (
        "AWS_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "BEDROCK_MODEL_ID",
        "BEDROCK_MAX_TOKENS",
        "BEDROCK_TEMPERATURE",
        "APP_ENV",
        "IOL_USER",
        "IOL_PASSWORD",
        "NEWS_RSS_URLS",
        "REDIS_URL",
        "SMTP_HOST",
        "SMTP_USER",
        "SMTP_PASSWORD",
        "BEDROCK_PROMPT_CHAT_ID",
        "BEDROCK_PROMPT_BRIEF_ID",
        "BEDROCK_PROMPT_VERSION",
        "PROMPT_CACHE_TTL_SECONDS",
        "BANK_NAME",
        "AUTH_API_KEYS",
        "CHAT_TIMEOUT_SECONDS",
        "CHAT_MAX_TOOL_ITERATIONS",
        "PRICE_STALE_TTL_SECONDS",
        "MACRO_STALE_TTL_SECONDS",
        "TASA_BCRA_STALE_TTL_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_defaults(monkeypatch):
    s = make(monkeypatch)
    assert s.redis_url == "redis://localhost:6379/0"
    assert s.bedrock_model_id == "us.anthropic.claude-sonnet-5-5"
    assert s.aws_region == "us-east-1"
    assert s.aws_access_key_id == ""
    assert s.aws_secret_access_key is None
    assert s.aws_session_token is None
    assert not hasattr(s, "aws_profile")
    assert s.news_rss_urls == []
    assert not hasattr(s, "anthropic_api_key")


def test_news_rss_urls_comma_separated(monkeypatch):
    s = make(monkeypatch, NEWS_RSS_URLS="http://a.com/rss, http://b.com/rss ,")
    assert s.news_rss_urls == ["http://a.com/rss", "http://b.com/rss"]


def test_env_overrides(monkeypatch):
    s = make(monkeypatch, REDIS_URL="redis://r:1/2", BEDROCK_MODEL_ID="m", SMTP_HOST="smtp.x")
    assert s.redis_url == "redis://r:1/2"
    assert s.bedrock_model_id == "m"
    assert s.smtp_host == "smtp.x"


def test_bedrock_inference_defaults(monkeypatch):
    s = make(monkeypatch)
    assert s.bedrock_max_tokens == 1024
    assert s.bedrock_temperature == 0


def test_bedrock_inference_from_env(monkeypatch):
    s = make(
        monkeypatch,
        BEDROCK_MAX_TOKENS="2048",
        BEDROCK_TEMPERATURE="0.3",
    )
    assert (s.bedrock_max_tokens, s.bedrock_temperature) == (2048, 0.3)


def test_unknown_env_file_entries_are_ignored(monkeypatch, tmp_path):
    """An old .env that still has removed variables must not break startup."""
    env_file = tmp_path / ".env"
    removed = ["BEDROCK_SERVICE_TIER=flex", "BEDROCK_GUARDRAIL_ID=gr-1"]
    removed += ["BEDROCK_GUARDRAIL_VERSION=3", "BEDROCK_PROMPT_CACHING=true"]
    lines = [*removed, "BEDROCK_MAX_TOKENS=512"]
    env_file.write_text(chr(10).join(lines), encoding="utf-8")
    s = Settings(_env_file=env_file)
    assert s.bedrock_max_tokens == 512
    assert not hasattr(s, "bedrock_service_tier")


def test_aws_credentials_from_env_are_secret(monkeypatch):
    s = make(
        monkeypatch,
        AWS_ACCESS_KEY_ID="AKIAEXAMPLE",
        AWS_SECRET_ACCESS_KEY="s3cr3t-value",
        AWS_SESSION_TOKEN="tok-value",
    )
    assert s.aws_access_key_id == "AKIAEXAMPLE"
    assert s.aws_secret_access_key.get_secret_value() == "s3cr3t-value"
    assert s.aws_session_token.get_secret_value() == "tok-value"
    assert "s3cr3t-value" not in repr(s) and "tok-value" not in str(s)


def test_prompt_settings_defaults_and_overrides(monkeypatch):
    s = make(monkeypatch)
    assert s.bedrock_prompt_chat_id == "" and s.bedrock_prompt_brief_id == ""
    assert s.bedrock_prompt_version == "DRAFT"
    assert s.prompt_cache_ttl_seconds == 300
    # The old per-prompt names are gone.
    assert not hasattr(s, "chat_prompt_id") and not hasattr(s, "brief_prompt_version")
    s = make(
        monkeypatch,
        BEDROCK_PROMPT_CHAT_ID="ABCDEFGHIJ",
        BEDROCK_PROMPT_VERSION="3",
        BEDROCK_PROMPT_BRIEF_ID="KLMNOPQRST",
        PROMPT_CACHE_TTL_SECONDS="60",
    )
    assert (s.bedrock_prompt_chat_id, s.bedrock_prompt_version) == ("ABCDEFGHIJ", "3")
    assert s.bedrock_prompt_brief_id == "KLMNOPQRST" and s.prompt_cache_ttl_seconds == 60


@pytest.mark.parametrize("value", ["DRAFT", "1", "3", "99999", " 7 "])
def test_prompt_version_accepts_draft_or_a_number(monkeypatch, value):
    assert make(monkeypatch, BEDROCK_PROMPT_VERSION=value).bedrock_prompt_version == value.strip()


@pytest.mark.parametrize("value", ["DRIFT", "draft", "", "0", "100000", "-1", "1.5", "v2", "two"])
def test_prompt_version_typos_fail_fast_with_a_clear_message(monkeypatch, value):
    with pytest.raises(ValidationError) as err:
        make(monkeypatch, BEDROCK_PROMPT_VERSION=value)
    text = str(err.value)
    assert "BEDROCK_PROMPT_VERSION" in text and "DRAFT" in text and "1..99999" in text


def test_new_week4_settings_defaults(monkeypatch):
    s = make(monkeypatch)
    assert s.chat_timeout_seconds == 25
    assert s.chat_max_tool_iterations == 6
    assert s.auth_api_keys is None and s.api_key_map() == {}
    assert s.price_stale_ttl_seconds == 604800 and s.macro_stale_ttl_seconds == 604800
    assert s.tasa_bcra_stale_ttl_seconds == 2592000


def test_chat_limits_from_env(monkeypatch):
    s = make(monkeypatch, CHAT_TIMEOUT_SECONDS="3.5", CHAT_MAX_TOOL_ITERATIONS="4")
    assert (s.chat_timeout_seconds, s.chat_max_tool_iterations) == (3.5, 4)


def test_auth_api_keys_are_secret_and_parsed(monkeypatch):
    raw = '{"key-one": "client_001", "key-two": "client_002"}'
    s = make(monkeypatch, AUTH_API_KEYS=raw)
    assert s.api_key_map() == {"key-one": "client_001", "key-two": "client_002"}
    assert "key-one" not in repr(s) and "key-one" not in str(s)
    assert s.auth_api_keys.get_secret_value() == raw


@pytest.mark.parametrize("raw", ["", "   ", "{}"])
def test_auth_api_keys_blank_means_not_configured(monkeypatch, raw):
    assert make(monkeypatch, AUTH_API_KEYS=raw).api_key_map() == {}


@pytest.mark.parametrize(
    "raw",
    ["not json", '["a"]', '{"k": 1}', '{"": "c1"}', '{"k": "bad id!"}', '{"k": ""}'],
)
def test_auth_api_keys_invalid_value_fails_fast_without_echoing_it(monkeypatch, raw):
    with pytest.raises(ValidationError) as err:
        make(monkeypatch, AUTH_API_KEYS=raw)
    assert "AUTH_API_KEYS" in str(err.value)


def test_bank_name_defaults_to_banco_comafi_and_is_overridable(monkeypatch):
    assert make(monkeypatch).bank_name == "Banco Comafi"
    assert make(monkeypatch, BANK_NAME="Banco Demo").bank_name == "Banco Demo"
