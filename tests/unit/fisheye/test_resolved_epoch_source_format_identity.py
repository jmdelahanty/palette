from __future__ import annotations

from types import SimpleNamespace

import pytest

from fisheye.analysis_workflows.resolved_epoch_selection import (
    ResolvedEpochSelectionError,
    _source_stimulus_format_identity,
)
from fisheye.shared.stimulus_coordinate_contract import (
    COORDINATE_CONTRACT_EPOCH,
    STIMULUS_IMPORT_VERSION,
)


def _source(**attrs: object) -> SimpleNamespace:
    return SimpleNamespace(attrs=attrs)


def test_source_format_identity_prefers_one_declared_schema() -> None:
    identity = _source_stimulus_format_identity(
        _source(schema_id="palette.stimulus.import.v1", schema_version=1)
    )

    assert identity == {
        "identity_kind": "declared_schema",
        "schema_id": "palette.stimulus.import.v1",
        "schema_version": 1,
    }


def test_source_format_identity_accepts_exact_maintained_legacy_import() -> None:
    identity = _source_stimulus_format_identity(
        _source(
            import_version=STIMULUS_IMPORT_VERSION,
            coordinate_contract_epoch=COORDINATE_CONTRACT_EPOCH,
            run_provenance={
                "command": "fisheye.analysis.import_stimulus_to_zarr"
            },
        )
    )

    assert identity == {
        "identity_kind": "maintained_legacy_import_contract",
        "import_version": STIMULUS_IMPORT_VERSION,
        "coordinate_contract_epoch": COORDINATE_CONTRACT_EPOCH,
        "writer_command": "fisheye.analysis.import_stimulus_to_zarr",
    }


@pytest.mark.parametrize(
    "attrs",
    (
        {"schema_id": "palette.stimulus.import.v1"},
        {"schema_version": 1},
        {
            "import_version": STIMULUS_IMPORT_VERSION,
            "coordinate_contract_epoch": COORDINATE_CONTRACT_EPOCH,
        },
        {
            "import_version": "1.0.0",
            "coordinate_contract_epoch": COORDINATE_CONTRACT_EPOCH,
            "run_provenance": {
                "command": "fisheye.analysis.import_stimulus_to_zarr"
            },
        },
    ),
)
def test_source_format_identity_rejects_partial_or_unknown_contracts(
    attrs: dict[str, object],
) -> None:
    with pytest.raises(ResolvedEpochSelectionError):
        _source_stimulus_format_identity(_source(**attrs))
