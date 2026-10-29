import pytest
from botocore.exceptions import NoCredentialsError

from app.agents.llm import LLMConfigError, build_chat_model, ensure_llm_configured
from app.core.config import Settings


def settings(**kw) -> Settings:
    base = {"aws_region": "us-east-1", "bedrock_model_id": "us.anthropic.claude-test"}
    return Settings(_env_file=None, **{**base, **kw})


class FakeCreds:
    pass


@pytest.fixture
def creds(monkeypatch):
    monkeypatch.setattr("app.agents.llm._resolve_credentials", lambda: FakeCreds())


def test_missing_model_id_raises():
    with pytest.raises(LLMConfigError) as e:
        ensure_llm_configured(settings(bedrock_model_id=""))
    assert e.value.code == "BEDROCK_MODEL_ID_MISSING"


def test_missing_region_raises():
    with pytest.raises(LLMConfigError) as e:
        ensure_llm_configured(settings(aws_region=""))
    assert e.value.code == "AWS_REGION_MISSING"


SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
KEY_ID = "AKIAIOSFODNN7EXAMPLE"
TOKEN = "FwoGZXIvYXdzEXAMPLETOKEN"


def test_no_explicit_credentials_and_empty_chain_raises(monkeypatch):
    monkeypatch.setattr("app.agents.llm._resolve_credentials", lambda: None)
    with pytest.raises(LLMConfigError) as e:
        ensure_llm_configured(settings())
    assert e.value.code == "AWS_CREDENTIALS_MISSING"
    assert "AWS_ACCESS_KEY_ID" in e.value.message


def test_no_explicit_credentials_falls_back_to_default_chain(monkeypatch):
    monkeypatch.setattr("app.agents.llm._resolve_credentials", lambda: FakeCreds())
    ensure_llm_configured(settings())


def test_chain_lookup_error_is_credentials_missing(monkeypatch):
    def boom():
        raise RuntimeError("bad config")

    monkeypatch.setattr("app.agents.llm._resolve_credentials", boom)
    with pytest.raises(LLMConfigError) as e:
        ensure_llm_configured(settings())
    assert e.value.code == "AWS_CREDENTIALS_MISSING"


def test_explicit_credentials_pass_without_touching_the_chain(monkeypatch):
    def fail():
        raise AssertionError("chain must not be consulted")

    monkeypatch.setattr("app.agents.llm._resolve_credentials", fail)
    ensure_llm_configured(settings(aws_access_key_id=KEY_ID, aws_secret_access_key=SECRET))


@pytest.mark.parametrize(
    ("kw", "missing"),
    [
        ({"aws_access_key_id": KEY_ID}, "AWS_SECRET_ACCESS_KEY"),
        ({"aws_secret_access_key": SECRET}, "AWS_ACCESS_KEY_ID"),
    ],
)
def test_partial_credentials_raise_and_never_leak_values(monkeypatch, kw, missing):
    monkeypatch.setattr("app.agents.llm._resolve_credentials", lambda: FakeCreds())
    with pytest.raises(LLMConfigError) as e:
        ensure_llm_configured(settings(**kw))
    assert e.value.code == "AWS_CREDENTIALS_MISSING"
    assert missing in e.value.message
    assert KEY_ID not in e.value.message and SECRET not in e.value.message


def test_blank_values_count_as_unset(monkeypatch):
    monkeypatch.setattr("app.agents.llm._resolve_credentials", lambda: None)
    with pytest.raises(LLMConfigError):
        ensure_llm_configured(settings(aws_access_key_id="  ", aws_secret_access_key=""))


def test_configured_passes(creds):
    ensure_llm_configured(settings())


def test_build_chat_model_params(creds):
    model = build_chat_model(settings())
    assert model.model_id == "us.anthropic.claude-test"
    assert model.region_name == "us-east-1"
    assert model.temperature == 0
    assert model.max_tokens == 1024
    cfg = model.config
    assert cfg.read_timeout == 30
    assert cfg.retries["total_max_attempts"] == 4  # 3 retries + first try


def test_chat_profile_has_tighter_timeouts_and_fewer_retries(creds):
    model = build_chat_model(settings(), for_chat=True)
    assert model.config.read_timeout == 18
    assert model.config.retries["total_max_attempts"] == 2
    # worst case (read timeout x attempts) fits the 25s chat budget together with tool calls
    assert model.config.read_timeout * model.config.retries["total_max_attempts"] <= 36


def test_chat_profile_uses_the_optional_chat_model_id(creds):
    assert build_chat_model(settings(), for_chat=True).model_id == "us.anthropic.claude-test"
    fast = build_chat_model(
        settings(bedrock_chat_model_id="us.anthropic.claude-fast"), for_chat=True
    )
    assert fast.model_id == "us.anthropic.claude-fast"
    # the brief client never switches model
    brief = build_chat_model(settings(bedrock_chat_model_id="us.anthropic.claude-fast"))
    assert brief.model_id == "us.anthropic.claude-test"


def test_chat_model_receives_explicit_credentials():
    model = build_chat_model(
        settings(
            aws_access_key_id=KEY_ID,
            aws_secret_access_key=SECRET,
            aws_session_token=TOKEN,
        )
    )
    assert model.aws_access_key_id.get_secret_value() == KEY_ID
    assert model.aws_secret_access_key.get_secret_value() == SECRET
    assert model.aws_session_token.get_secret_value() == TOKEN
    assert model.region_name == "us-east-1"


def test_chat_model_without_session_token():
    model = build_chat_model(settings(aws_access_key_id=KEY_ID, aws_secret_access_key=SECRET))
    assert model.aws_session_token is None


def test_secrets_never_appear_in_repr_or_str():
    s = settings(aws_access_key_id=KEY_ID, aws_secret_access_key=SECRET, aws_session_token=TOKEN)
    text = f"{s!r} {s} {s.model_dump()} {s.model_dump_json()}"
    assert SECRET not in text and TOKEN not in text


def test_secrets_not_in_logs_when_config_fails(caplog):
    import logging

    with caplog.at_level(logging.DEBUG), pytest.raises(LLMConfigError):
        ensure_llm_configured(settings(aws_secret_access_key=SECRET))
    assert SECRET not in caplog.text


def test_no_credentials_error_type_is_importable():
    assert issubclass(NoCredentialsError, Exception)
