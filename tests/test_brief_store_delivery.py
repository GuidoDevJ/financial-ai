import json
import logging
import smtplib

import fakeredis

from app.brief.delivery import send_brief_email
from app.brief.store import BriefStore
from app.core.config import Settings
from app.tools.errors import ToolError
from app.tools.get_last_brief import make_get_last_brief


def redis_():
    return fakeredis.FakeRedis(decode_responses=True)


# ------------------------------------------------------------------- store


def test_store_legacy_payload_is_unchanged():
    r = redis_()
    store = BriefStore(r)
    store.save("c1", "2026-10-15", "texto", ["GGAL"])
    assert json.loads(r.get("brief:c1:2026-10-15")) == {
        "fecha": "2026-10-15",
        "texto_brief": "texto",
        "activos_mencionados": ["GGAL"],
    }


def test_store_extended_payload_is_backward_compatible_with_get_last_brief():
    store = BriefStore(redis_())
    store.save(
        "c1",
        "2026-10-15",
        "texto",
        ["GGAL"],
        html="<p>x</p>",
        degraded=True,
        missing_inputs=["price:YPFD.BA"],
        generated_at="2026-10-15T10:30:00+00:00",
    )
    stored = store.get("c1", "2026-10-15")
    assert stored["html"] == "<p>x</p>" and stored["degraded"] is True
    assert stored["missing_inputs"] == ["price:YPFD.BA"]
    assert stored["generated_at"] == "2026-10-15T10:30:00+00:00"
    assert store.exists("c1", "2026-10-15") and not store.exists("c1", "2026-10-16")
    tool = make_get_last_brief(store=store)
    assert tool(cliente_id="c1") == {
        "fecha": "2026-10-15",
        "texto_brief": "texto",
        "activos_mencionados": ["GGAL"],
    }


def test_get_last_brief_still_reports_missing_brief():
    tool = make_get_last_brief(store=BriefStore(redis_()))
    try:
        tool(cliente_id="c1")
    except ToolError as exc:
        assert exc.code == "NO_BRIEF_FOUND"
    else:
        raise AssertionError("expected ToolError")


# ---------------------------------------------------------------- delivery


class FakeSMTP:
    instances: list = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port = host, port
        self.calls = []
        self.message = None
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        self.calls.append("starttls")

    def login(self, user, password):
        self.calls.append(("login", user, password))

    def send_message(self, message):
        self.message = message
        self.calls.append("send")


def settings(**kw):
    base = dict(_env_file=None, smtp_host="smtp.test", smtp_from="brief@bank.test")
    return Settings(**{**base, "smtp_to": "qa@bank.test", **kw})


def test_no_smtp_host_skips_and_logs(caplog):
    FakeSMTP.instances.clear()
    with caplog.at_level(logging.INFO):
        result = send_brief_email(
            Settings(_env_file=None),
            fecha="2026-10-15",
            texto="t",
            html="<p>t</p>",
            smtp_factory=FakeSMTP,
        )
    assert result["status"] == "skipped"
    assert FakeSMTP.instances == []
    assert "email skipped" in caplog.text


def test_sends_multipart_message_with_starttls_and_login():
    FakeSMTP.instances.clear()
    s = settings(smtp_user="u", smtp_password="s3cret")
    result = send_brief_email(
        s, fecha="2026-10-15", texto="hola", html="<p>hola</p>", smtp_factory=FakeSMTP
    )
    assert result == {"status": "sent", "to": "qa@bank.test"}
    (smtp,) = FakeSMTP.instances
    assert (smtp.host, smtp.port) == ("smtp.test", 587)
    assert smtp.calls == ["starttls", ("login", "u", "s3cret"), "send"]
    msg = smtp.message
    assert msg["To"] == "qa@bank.test" and msg["From"] == "brief@bank.test"
    assert "2026-10-15" in msg["Subject"]
    assert msg.get_body(("plain",)).get_content().strip() == "hola"
    assert "<p>hola</p>" in msg.get_body(("html",)).get_content()


def test_smtp_failure_is_reported_not_raised_and_does_not_leak_password(caplog):
    class Boom(FakeSMTP):
        def send_message(self, message):
            raise smtplib.SMTPException("boom")

    with caplog.at_level(logging.INFO):
        result = send_brief_email(
            settings(smtp_user="u", smtp_password="s3cret"),
            fecha="2026-10-15",
            texto="t",
            html="h",
            smtp_factory=Boom,
        )
    assert result["status"] == "failed" and "SMTPException" in result["error"]
    assert "s3cret" not in caplog.text and "s3cret" not in json.dumps(result)


def test_missing_recipient_or_sender_is_skipped_with_reason():
    for kw in ({"smtp_to": ""}, {"smtp_from": ""}):
        result = send_brief_email(
            settings(**kw), fecha="2026-10-15", texto="t", html="h", smtp_factory=FakeSMTP
        )
        assert result["status"] == "skipped" and result["reason"]


def test_smtp_settings_defaults():
    s = Settings(_env_file=None)
    assert s.smtp_host == "" and s.smtp_port == 587 and s.smtp_starttls is True
    assert s.smtp_password is None
    assert "s3cret" not in repr(Settings(_env_file=None, smtp_password="s3cret"))


def test_store_saves_optional_prompt_identity_and_still_reads_old_payloads():
    r = redis_()
    store = BriefStore(r)
    meta = {"name": "brief", "version": "3", "hash": "abcdef0123456789"}
    store.save("c1", "2026-10-15", "texto", ["GGAL"], prompt=meta)
    assert store.get("c1", "2026-10-15")["prompt"] == meta
    store.save("c1", "2026-10-16", "texto", ["GGAL"])
    assert "prompt" not in store.get("c1", "2026-10-16")
