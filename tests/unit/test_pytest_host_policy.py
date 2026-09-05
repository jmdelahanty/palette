from __future__ import annotations

import pytest

import tests.conftest as shared_fixtures


@pytest.mark.parametrize(
    "hostname",
    (
        "login1",
        "login2",
        "login1.hhmi.org",
        "LOGIN2.HHMI.ORG",
    ),
)
def test_pytest_host_policy_rejects_campus_login_nodes(
    monkeypatch: pytest.MonkeyPatch,
    hostname: str,
) -> None:
    monkeypatch.setattr(shared_fixtures.socket, "gethostname", lambda: hostname)

    with pytest.raises(pytest.UsageError, match="never on the campus login node"):
        shared_fixtures._require_palette_test_host()


def test_pytest_host_policy_allows_workstation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        shared_fixtures.socket,
        "gethostname",
        lambda: "delahantyj-ws1.hhmi.org",
    )

    shared_fixtures._require_palette_test_host()
