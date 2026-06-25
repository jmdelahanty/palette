from __future__ import annotations

import json
from pathlib import Path

from fisheye.labeling import web as labeling_web


def _config(**overrides):
    values = {
        "store_path": Path(":memory:"),
        "host": "127.0.0.1",
        "port": 8795,
        "fixed_user": "alice",
        "auth_header": "X-Forwarded-User",
        "session_ttl_seconds": 600,
    }
    values.update(overrides)
    return labeling_web.ServerConfig(**values)


def test_local_development_config_allows_fixed_user_on_loopback():
    errors = labeling_web._server_config_errors(_config())
    assert errors == []


def test_header_auth_must_be_explicitly_trusted():
    errors = labeling_web._server_config_errors(_config(fixed_user=None, trust_auth_header=False))
    assert any("--user" in error and "--trust-auth-header" in error for error in errors)


def test_production_rejects_fixed_user_auth():
    errors = labeling_web._server_config_errors(_config(production=True, admin_users=("admin",)))
    assert any("--production requires proxy/header authentication" in error for error in errors)


def test_production_requires_trusted_header_and_admin_user():
    no_proxy_errors = labeling_web._server_config_errors(
        _config(fixed_user=None, trust_auth_header=False, production=True, admin_users=("admin",))
    )
    no_admin_errors = labeling_web._server_config_errors(
        _config(fixed_user=None, trust_auth_header=True, production=True, admin_users=())
    )

    assert any("--production requires --trust-auth-header" in error for error in no_proxy_errors)
    assert any("--production requires at least one --admin-user" in error for error in no_admin_errors)


def test_non_loopback_bind_requires_explicit_allow_flag():
    errors = labeling_web._server_config_errors(_config(host="0.0.0.0"))
    allowed_errors = labeling_web._server_config_errors(_config(host="0.0.0.0", allow_non_loopback=True))

    assert any("--allow-non-loopback" in error for error in errors)
    assert allowed_errors == []


def test_link_not_before_timestamp_is_validated():
    errors = labeling_web._server_config_errors(_config(link_not_before_utc="not-a-timestamp"))
    valid_errors = labeling_web._server_config_errors(_config(link_not_before_utc="2026-06-23T12:00:00Z"))

    assert any("--link-not-before-utc is invalid" in error for error in errors)
    assert valid_errors == []


def test_preflight_cli_emits_json_without_starting_server(tmp_path, capsys):
    result = labeling_web.main(
        [
            "--store",
            str(tmp_path / "labeling_work.sqlite"),
            "preflight",
            "--user",
            "alice",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["ok"] is True
    assert payload["error_count"] == 0
    assert payload["preflight"]["auth_mode"] == "fixed_user"


def test_preflight_cli_returns_nonzero_for_unsafe_auth_config(tmp_path, capsys):
    result = labeling_web.main(
        [
            "--store",
            str(tmp_path / "labeling_work.sqlite"),
            "preflight",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 2
    assert payload["ok"] is False
    assert payload["error_count"] >= 1
    assert any("--user" in error and "--trust-auth-header" in error for error in payload["errors"])
