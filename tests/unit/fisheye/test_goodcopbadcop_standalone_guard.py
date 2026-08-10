from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import pytest

from fisheye.analysis.goodcopbadcop_common import (
    STANDALONE_EXPLORATORY_STATUS_SCHEMA,
    STANDALONE_EXPLORATORY_STATUS_VERSION,
    exploratory_artifact_path,
    parse_standalone_exploratory_args,
    save_standalone_exploratory_figure,
    standalone_exploratory_status,
    write_standalone_exploratory_sidecar,
)


STANDALONE_SCRIPT_NAMES = (
    "analyze_goodcopbadcop_approach_avoidance.py",
    "analyze_goodcopbadcop_bout_kinematics_distance.py",
    "analyze_goodcopbadcop_bout_vigor_prepost.py",
    "analyze_goodcopbadcop_escape.py",
    "analyze_goodcopbadcop_habituation.py",
    "analyze_goodcopbadcop_immobility_artifact.py",
    "analyze_goodcopbadcop_lateral_gaze.py",
    "analyze_goodcopbadcop_learning_mixed_model.py",
    "analyze_goodcopbadcop_per_fish.py",
    "analyze_goodcopbadcop_radial_kinematics.py",
    "analyze_goodcopbadcop_radial_turn_direction.py",
    "analyze_goodcopbadcop_wall_mediator.py",
    "plot_goodcopbadcop_bout_rate.py",
    "plot_goodcopbadcop_freeze.py",
)
STANDALONE_MODULE_NAMES = tuple(
    f"fisheye.analysis.{Path(script_name).stem}"
    for script_name in STANDALONE_SCRIPT_NAMES
)


def test_standalone_status_is_explicitly_exploratory_and_ineligible() -> None:
    status = standalone_exploratory_status(analysis_id="example")

    assert status == {
        "schema": STANDALONE_EXPLORATORY_STATUS_SCHEMA,
        "version": STANDALONE_EXPLORATORY_STATUS_VERSION,
        "analysis_id": "example",
        "analysis_tier": "exploratory",
        "publication_eligibility": "ineligible",
        "confirmatory_use": False,
        "multiplicity_control": "none",
        "registered_group_statistics": False,
        "warning": status["warning"],
    }
    assert "Do not cite" in str(status["warning"])


def test_standalone_guard_requires_explicit_acknowledgement(capsys) -> None:
    parser = argparse.ArgumentParser(prog="standalone-test")

    with pytest.raises(SystemExit, match="2"):
        parse_standalone_exploratory_args(
            parser,
            analysis_id="standalone-test",
            argv=[],
        )

    assert "--exploratory-only" in capsys.readouterr().err


def test_standalone_guard_emits_machine_readable_status(capsys) -> None:
    parser = argparse.ArgumentParser(prog="standalone-test")

    args = parse_standalone_exploratory_args(
        parser,
        analysis_id="standalone-test",
        argv=["--exploratory-only"],
    )

    assert args.exploratory_only is True
    status_line = capsys.readouterr().err.splitlines()[0]
    prefix = "PALETTE_STANDALONE_ANALYSIS_STATUS="
    assert status_line.startswith(prefix)
    assert json.loads(status_line.removeprefix(prefix))["publication_eligibility"] == (
        "ineligible"
    )


def test_exploratory_sidecar_is_strict_json_and_does_not_replace_contract_fields(
    tmp_path: Path,
) -> None:
    artifact = exploratory_artifact_path(tmp_path / "result.png")
    artifact.write_bytes(b"figure")

    sidecar = write_standalone_exploratory_sidecar(
        artifact,
        analysis_id="example",
        extra={"artifact_kind": "figure"},
    )

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert artifact.name.endswith("_exploratory.png")
    assert payload["artifact_name"] == artifact.name
    assert payload["artifact_kind"] == "figure"
    assert payload["confirmatory_use"] is False
    with pytest.raises(ValueError, match="replace contract fields"):
        write_standalone_exploratory_sidecar(
            artifact,
            analysis_id="example",
            extra={"confirmatory_use": True},
        )


def test_exploratory_figure_is_watermarked_and_receipted(tmp_path: Path) -> None:
    class FakeFigure:
        def __init__(self) -> None:
            self.labels: list[str] = []

        def text(self, _x, _y, label, **_kwargs) -> None:
            self.labels.append(str(label))

        def savefig(self, path, **_kwargs) -> None:
            Path(path).write_bytes(b"figure")

    figure = FakeFigure()
    output, sidecar = save_standalone_exploratory_figure(
        figure,
        tmp_path / "result.png",
        analysis_id="example",
    )

    assert output.name == "result_exploratory.png"
    assert output.is_file()
    assert sidecar.is_file()
    assert any("EXPLORATORY ONLY" in label for label in figure.labels)


def test_all_inferential_goodcopbadcop_scripts_use_shared_guard() -> None:
    analysis_dir = Path(__file__).parents[3] / "src" / "fisheye" / "analysis"

    for script_name in STANDALONE_SCRIPT_NAMES:
        source = (analysis_dir / script_name).read_text(encoding="utf-8")
        assert "parse_standalone_exploratory_args(" in source, script_name
        assert ".parse_args(" not in source, script_name


@pytest.mark.parametrize("module_name", STANDALONE_MODULE_NAMES)
def test_every_standalone_cli_fails_before_work_without_acknowledgement(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", [module_name])

    with pytest.raises(SystemExit, match="2"):
        module.main()

    error = capsys.readouterr().err
    assert "--exploratory-only" in error
