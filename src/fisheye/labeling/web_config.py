"""Configuration helpers for web labeling serve and CLI commands."""

from __future__ import annotations

import os

from .web_auth import _utc_timestamp
from .web_responses import _is_loopback_host


LINK_SECRET_ENV_VAR = "PALETTE_LABELING_LINK_SECRET"


def _server_config_errors(config: object) -> list[str]:
    errors: list[str] = []
    if config.fixed_user and config.trust_auth_header:
        errors.append("choose either --user or --trust-auth-header, not both")
    if not config.fixed_user and not config.trust_auth_header:
        errors.append("serve requires --user for local development or --trust-auth-header behind a trusted proxy")
    if config.production and config.fixed_user:
        errors.append("--production requires proxy/header authentication; do not use --user")
    if config.production and not config.trust_auth_header:
        errors.append("--production requires --trust-auth-header behind a trusted proxy")
    if config.production and not config.admin_users:
        errors.append("--production requires at least one --admin-user")
    if config.require_operator_validation_for_start and config.validation_checklist_path is None:
        errors.append(
            "--require-operator-validation-for-start or --require-operator-validation-for-browser-work requires --validation-checklist"
        )
    if config.validation_checklist_path is not None and not config.validation_checklist_path.is_file():
        errors.append(
            f"--validation-checklist does not exist or is not a file: {config.validation_checklist_path}"
        )
    if config.trust_auth_header and not str(config.auth_header or "").strip():
        errors.append("--trust-auth-header requires a non-empty --auth-header")
    if config.link_not_before_utc:
        try:
            _utc_timestamp(config.link_not_before_utc)
        except Exception as exc:
            errors.append(f"--link-not-before-utc is invalid: {exc}")
    if not _is_loopback_host(config.host) and not config.allow_non_loopback:
        errors.append("--host is non-loopback; pass --allow-non-loopback only when network exposure is intentional")
    return errors


def _link_secret_from_arg(value: str | None) -> str:
    secret = str(value or os.environ.get(LINK_SECRET_ENV_VAR) or "").strip()
    if not secret:
        raise ValueError(f"Signed links require --link-secret or {LINK_SECRET_ENV_VAR}.")
    return secret
