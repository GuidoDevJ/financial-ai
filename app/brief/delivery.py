"""Simulated brief delivery by email (SMTP to a single test mailbox).

Never raises: a delivery problem is returned as a status and must not fail the brief.
"""

import logging
import smtplib
from collections.abc import Callable
from email.message import EmailMessage

from app.core.config import Settings

logger = logging.getLogger(__name__)

SMTP_TIMEOUT_SECONDS = 15


def send_brief_email(
    settings: Settings,
    *,
    fecha: str,
    texto: str,
    html: str,
    smtp_factory: Callable[..., smtplib.SMTP] = smtplib.SMTP,
) -> dict:
    """Send the brief to SMTP_TO. Returns {"status": "sent"|"skipped"|"failed", ...}."""
    if not settings.smtp_host:
        logger.info("email skipped (simulated): SMTP_HOST is not set")
        return {"status": "skipped", "reason": "SMTP_HOST not set"}
    if not (settings.smtp_to and settings.smtp_from):
        logger.info("email skipped (simulated): SMTP_FROM / SMTP_TO not set")
        return {"status": "skipped", "reason": "SMTP_FROM or SMTP_TO not set"}

    message = EmailMessage()
    message["Subject"] = f"Brief diario {fecha}"
    message["From"] = settings.smtp_from
    message["To"] = settings.smtp_to
    message.set_content(texto)
    message.add_alternative(html, subtype="html")
    try:
        with smtp_factory(
            settings.smtp_host, settings.smtp_port, timeout=SMTP_TIMEOUT_SECONDS
        ) as smtp:
            if settings.smtp_starttls:
                smtp.starttls()
            if settings.smtp_user and settings.smtp_password:
                smtp.login(settings.smtp_user, settings.smtp_password.get_secret_value())
            smtp.send_message(message)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("brief email failed: %s", error)
        return {"status": "failed", "error": error}
    logger.info("brief email sent to %s", settings.smtp_to)
    return {"status": "sent", "to": settings.smtp_to}
