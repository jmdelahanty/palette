from __future__ import annotations

from datetime import date
from pathlib import Path

from scripts import check_contract_freshness as mod


def _write_contract(path: Path, metadata: str) -> None:
    path.write_text(
        f"# Contract\n\n<!-- contract-meta\n{metadata}\n-->\n",
        encoding="utf-8",
    )


def test_unmanaged_contract_is_outside_freshness_policy(tmp_path: Path) -> None:
    path = tmp_path / "unmanaged_contract.md"
    path.write_text("# Historical design contract\n", encoding="utf-8")

    assert mod._collect_issues(path, 90, today=date(2026, 8, 16)) == []


def test_only_active_contracts_expire(tmp_path: Path) -> None:
    active = tmp_path / "active_contract.md"
    draft = tmp_path / "draft_contract.md"
    _write_contract(
        active,
        "version: 1\nstatus: active\nimplementation: implemented\n"
        "last_verified: 2026-01-01",
    )
    _write_contract(
        draft,
        "version: 1\nstatus: draft\nimplementation: partial\n"
        "last_verified: 2026-01-01",
    )

    active_issues = mod._collect_issues(active, 90, today=date(2026, 8, 16))
    draft_issues = mod._collect_issues(draft, 90, today=date(2026, 8, 16))

    assert [issue.code for issue in active_issues] == ["stale"]
    assert draft_issues == []


def test_active_requires_verification_and_all_managed_require_implementation(
    tmp_path: Path,
) -> None:
    active = tmp_path / "active_contract.md"
    draft = tmp_path / "draft_contract.md"
    _write_contract(active, "version: 1\nstatus: active\nimplementation: implemented")
    _write_contract(draft, "version: 1\nstatus: draft")

    active_issues = mod._collect_issues(active, 90, today=date(2026, 8, 16))
    draft_issues = mod._collect_issues(draft, 90, today=date(2026, 8, 16))

    assert [(issue.code, issue.message) for issue in active_issues] == [
        ("missing_field", "active contract is missing required field: last_verified")
    ]
    assert [(issue.code, issue.message) for issue in draft_issues] == [
        ("missing_field", "missing required field: implementation")
    ]


def test_rejects_invalid_status_and_implementation(tmp_path: Path) -> None:
    path = tmp_path / "bad_contract.md"
    _write_contract(
        path,
        "version: 1\nstatus: implemented\nimplementation: complete",
    )

    issues = mod._collect_issues(path, 90, today=date(2026, 8, 16))

    assert {issue.code for issue in issues} == {
        "invalid_status",
        "invalid_implementation",
    }
