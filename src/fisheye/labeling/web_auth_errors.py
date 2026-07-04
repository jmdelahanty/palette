"""Authentication error-message helpers for labeling web routes."""

from __future__ import annotations

from typing import Any


def _authentication_required_error_details(source: str, config: Any) -> str:
    """Return the browser/operator-facing reason authentication failed."""

    if source == "auth_header_not_trusted":
        return (
            "Header-based authentication is disabled. Start with --user for local development "
            "or --trust-auth-header behind a trusted proxy."
        )
    if source == "signed_invites_disabled":
        return "This invite link cannot be used because the server was not launched with --link-secret."
    if source == "invite_expected_user_mismatch":
        return "This invite link is for a different expected_user. Stop and ask the operator for a fresh invite."
    if source and (source.startswith("invite_error:") or source.startswith("signed_link_")):
        return "This invite link is invalid, expired, or revoked. Ask the operator for a fresh invite."
    return f"No user found from trusted auth header {config.auth_header}."


__all__ = ["_authentication_required_error_details"]
