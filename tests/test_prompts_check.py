"""`python -m app.prompts.check`: fetches both prompts and prints a safe summary."""

from app.core.config import Settings
from app.prompts import reference
from app.prompts.check import main
from app.prompts.registry import PromptProvider
from tests.test_prompts_registry import FakeAgent, client_error, response

SETTINGS = Settings(
    _env_file=None,
    bedrock_prompt_chat_id="CHATCHATCH",
    bedrock_prompt_brief_id="BRIEFBRIEF",
    bedrock_prompt_version="2",
    aws_access_key_id="AKIASECRETKEYID",
    aws_secret_access_key="topsecret",
)


class ByIdAgent:
    """Fake agent answering per prompt id (or raising for it)."""

    def __init__(self, results):
        self.results = results

    def get_prompt(self, **kwargs):
        item = self.results[kwargs["promptIdentifier"]]
        if isinstance(item, Exception):
            raise item
        return item


def run(agent, settings=SETTINGS):
    return main(settings, PromptProvider(lambda: agent))


def ref_text(name):
    return reference.current_text(reference.load_reference()[name])


def test_prints_name_id_version_hash_length_drift_and_preview_and_exits_zero(capsys):
    long_text = "Linea uno\n" + "x" * 200
    agent = ByIdAgent({"CHATCHATCH": response(long_text, "2"), "BRIEFBRIEF": response("Brief")})
    assert run(agent) == 0
    out = capsys.readouterr().out
    chat_hash = reference.text_hash(long_text)
    assert f"OK   chat: id=CHATCHATCH version=2 hash={chat_hash} length={len(long_text)}" in out
    assert "OK   brief: id=BRIEFBRIEF version=DRAFT" in out
    chat_line = next(line for line in out.splitlines() if "CHATCHATCH" in line)
    assert "Linea uno xxxx" in chat_line and "x" * 100 not in chat_line  # 80-char preview


def test_reports_match_when_bedrock_text_equals_registry_yaml(capsys):
    agent = ByIdAgent(
        {
            "CHATCHATCH": response(ref_text("chat"), "2"),
            "BRIEFBRIEF": response(ref_text("brief"), "2"),
        }
    )
    assert run(agent) == 0
    out = capsys.readouterr().out
    assert out.count("drift=MATCH") == 2 and "WARN" not in out


def test_drift_prints_a_warning_but_exits_zero(capsys):
    agent = ByIdAgent(
        {
            "CHATCHATCH": response("edited in console", "2"),
            "BRIEFBRIEF": response(ref_text("brief")),
        }
    )
    assert run(agent) == 0
    out = capsys.readouterr().out
    assert "drift=DRIFT" in out and "WARN chat" in out and "drift=MATCH" in out


def test_never_prints_secrets(capsys):
    agent = ByIdAgent({"CHATCHATCH": response("a"), "BRIEFBRIEF": response("b")})
    run(agent)
    out = capsys.readouterr().out
    assert "AKIASECRETKEYID" not in out and "topsecret" not in out


def test_exits_non_zero_when_a_prompt_fails_but_still_checks_the_other(capsys):
    agent = ByIdAgent(
        {"CHATCHATCH": response("a"), "BRIEFBRIEF": client_error("ResourceNotFoundException")}
    )
    assert run(agent) == 1
    out = capsys.readouterr().out
    assert "OK   chat" in out and "FAIL brief: PROMPT_NOT_FOUND" in out


def test_missing_ids_exit_non_zero_and_name_the_variables(capsys):
    assert run(FakeAgent(response()), Settings(_env_file=None)) == 1
    out = capsys.readouterr().out
    assert "BEDROCK_PROMPT_CHAT_ID" in out and "BEDROCK_PROMPT_BRIEF_ID" in out


def test_other_aws_errors_exit_non_zero(capsys):
    agent = ByIdAgent(
        {"CHATCHATCH": client_error("ExpiredTokenException"), "BRIEFBRIEF": response("b")}
    )
    assert run(agent) == 1
    assert "FAIL chat: ClientError" in capsys.readouterr().out


def test_previous_registry_version_is_reported_as_drift(capsys):
    registry = reference.load_reference()
    v2 = reference.variant_text(registry["chat"]["versions"]["2"]["variants"][0])
    agent = ByIdAgent({"CHATCHATCH": response(v2, "2"), "BRIEFBRIEF": response(ref_text("brief"))})
    assert run(agent) == 0
    out = capsys.readouterr().out
    assert "WARN chat" in out and "version 4" in out
    assert out.count("drift=MATCH") == 1


def test_trailing_whitespace_in_bedrock_does_not_cause_drift(capsys):
    agent = ByIdAgent(
        {
            "CHATCHATCH": response(ref_text("chat") + "\n\n"),
            "BRIEFBRIEF": response("  " + ref_text("brief")),
        }
    )
    assert run(agent) == 0
    assert capsys.readouterr().out.count("drift=MATCH") == 2
