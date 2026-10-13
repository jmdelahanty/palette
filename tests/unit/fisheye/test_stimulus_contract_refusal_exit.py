"""A stimulus protocol contract error is a deterministic refusal (exit 2 -> intake 65)."""

from __future__ import annotations

import pytest

from fisheye.analysis import import_stimulus_to_zarr as importer
from fisheye.intake.importing import ParentImportResult, _deterministic
from fisheye.shared.protocol_semantic_contract import ProtocolSemanticContractError

ARGS = ["in.h5", "out.zarr"]


def _failing(exc):
    def run(**_kwargs):
        raise exc

    return run


def test_a_protocol_contract_error_exits_2(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        importer, "import_stimulus_to_zarr",
        _failing(ProtocolSemanticContractError("semantic step 0 uses unknown stimulus_mode_id 19.")),
    )
    with pytest.raises(SystemExit) as exited:
        importer.main(ARGS)
    assert exited.value.code == importer.EXIT_CONTRACT_REFUSAL == 2
    assert "unknown stimulus_mode_id 19" in capsys.readouterr().err


def test_other_errors_still_propagate(monkeypatch) -> None:
    monkeypatch.setattr(importer, "import_stimulus_to_zarr", _failing(OSError("NFS hiccup")))
    with pytest.raises(OSError):
        importer.main(ARGS)


def test_intake_treats_the_stimulus_child_s_exit_2_as_a_refusal() -> None:
    def parent(returncode):
        return ParentImportResult(
            recording_id="r", camera_id="c", recording_dir="d", zarr_path="z",
            outcome="failed", failed_step="import_stimulus_to_zarr", returncode=returncode,
        )

    assert _deterministic(parent(2)) is True
    assert _deterministic(parent(1)) is False
