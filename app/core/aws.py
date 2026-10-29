"""Explicit AWS credentials from Settings, shared by the Bedrock chat and embeddings clients."""

from pydantic import SecretStr

from app.core.config import Settings


def _value(secret: SecretStr | str | None) -> str:
    raw = secret.get_secret_value() if isinstance(secret, SecretStr) else (secret or "")
    return raw.strip()


def explicit_credentials(settings: Settings) -> dict[str, str]:
    """Plain-string credential kwargs, or {} when none are configured.

    Only call after validating with `missing_credential_vars`: a lone key id or secret is
    never returned. {} means "use the default boto3 credential chain".
    """
    key_id = _value(settings.aws_access_key_id)
    secret = _value(settings.aws_secret_access_key)
    if not (key_id and secret):
        return {}
    creds = {"aws_access_key_id": key_id, "aws_secret_access_key": secret}
    if token := _value(settings.aws_session_token):
        creds["aws_session_token"] = token
    return creds


def missing_credential_vars(settings: Settings) -> list[str]:
    """Names (never values) of the missing half when exactly one of key id/secret is set."""
    has_key = bool(_value(settings.aws_access_key_id))
    has_secret = bool(_value(settings.aws_secret_access_key))
    if has_key == has_secret:
        return []
    return ["AWS_SECRET_ACCESS_KEY"] if has_key else ["AWS_ACCESS_KEY_ID"]
