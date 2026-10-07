"""The intake runtime check catches a missing lazily imported dependency."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from fisheye.utils import check_intake_runtime as runtime

REPO = Path(__file__).resolve().parents[3]


def test_this_environment_can_run_intake(capsys) -> None:
    assert runtime.check() == []
    assert runtime.main([]) == 0
    assert "intake runtime check ok" in capsys.readouterr().out


def test_a_missing_lazy_dependency_fails_the_check(monkeypatch, capsys) -> None:
    # The cluster environment once lacked jsonschema; intake only imports it
    # when it first validates a document, so module imports alone pass.
    real = importlib.import_module

    def import_module(name, *args, **kwargs):
        if name == "jsonschema":
            raise ModuleNotFoundError("No module named 'jsonschema'")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(runtime.importlib, "import_module", import_module)
    assert runtime.check() == ["missing module jsonschema: No module named 'jsonschema'"]
    assert runtime.main([]) == 1
    assert "FAILED" in capsys.readouterr().err


def test_an_unbuildable_validator_fails_the_check(monkeypatch) -> None:
    def broken():
        raise ValueError("packaged_contract_drift:x.schema.json")

    monkeypatch.setattr(runtime, "_validators", lambda: [("transfer envelope", broken)])
    assert runtime.check() == [
        "cannot build transfer envelope validator: packaged_contract_drift:x.schema.json"
    ]


def test_every_pinned_intake_schema_is_checked() -> None:
    labels = {label for label, _ in runtime._validators()}
    assert {"transfer envelope", "frame identity proof", "citrus subject snapshot"} <= labels
    assert any(label.startswith("orange recording_output") for label in labels)
    assert any(label.startswith("orange subject reference") for label in labels)


def test_jsonschema_is_a_declared_dependency() -> None:
    assert '"jsonschema>=4.23,<5"' in (REPO / "pyproject.toml").read_text()
    assert '"jsonschema>=4.23,<5"' in (REPO / "environment.yml").read_text()


@pytest.mark.parametrize("needle", ["fisheye.utils.check_intake_runtime"])
def test_cluster_deploy_verification_runs_the_check(needle) -> None:
    script = (REPO / "scripts/deploy_palette_cluster_worktree.sh").read_text()
    remote = script[script.index('if [[ "$SKIP_HOST_VERIFY" -eq 0 ]]'):]
    assert needle in remote.split('run ssh -o BatchMode=yes "$VERIFY_HOST"')[0]
