from __future__ import annotations

from dataclasses import replace

import pytest

from fisheye.analysis_workflows.chaser_input_provenance_proxy import (
    select_chaser_input_provenance_proxy,
)
from fisheye.analysis_workflows.chaser_input_provenance_proxy_storage import (
    ChaserInputProvenanceProxyStorageError,
    prepare_chaser_input_provenance_proxy,
    validate_prepared_chaser_input_provenance_proxy,
)
from fisheye.shared.zarr.chaser_input_provenance_proxy_schema import (
    ChaserInputProvenanceProxySchemaError,
)
from tests.unit.fisheye.test_chaser_input_provenance_proxy import _source


def _prepared():
    return prepare_chaser_input_provenance_proxy(
        select_chaser_input_provenance_proxy(_source())
    )


def test_prepare_produces_read_only_numeric_arrays_and_bound_manifest() -> None:
    prepared = _prepared()
    receipt = validate_prepared_chaser_input_provenance_proxy(prepared)

    assert receipt["dimensions"] == {
        "frame": 3,
        "candidate": 5,
        "chaser": 2,
        "frame_boundary": 4,
    }
    assert receipt["selector_eligible"] is False
    assert receipt["selection"] == "none"
    assert all(not value.flags.writeable for value in prepared.arrays.values())
    assert all(
        value.dtype.kind not in {"O", "U", "S"}
        for value in prepared.arrays.values()
    )
    assert len(prepared.payload_digest) == 64


def test_prepare_revalidates_source_before_accepting_result() -> None:
    source = _source()
    result = select_chaser_input_provenance_proxy(source)

    class _BrokenSource:
        def assert_verified(self) -> None:
            raise ValueError("source changed")

    with pytest.raises(ChaserInputProvenanceProxySchemaError, match="revalidation"):
        prepare_chaser_input_provenance_proxy(
            replace(result, source_handle=_BrokenSource())
        )


def test_prepared_array_tampering_is_rejected() -> None:
    prepared = _prepared()
    arrays = dict(prepared.arrays)
    changed = arrays["selected_stimulus_frame_num"].copy()
    changed[0] += 1
    changed.setflags(write=False)
    arrays["selected_stimulus_frame_num"] = changed

    with pytest.raises(ChaserInputProvenanceProxySchemaError, match="content digest"):
        validate_prepared_chaser_input_provenance_proxy(
            replace(prepared, arrays=arrays)
        )


def test_prepared_candidate_identity_tampering_is_rejected() -> None:
    prepared = _prepared()
    manifest = dict(prepared.manifest)
    manifest["prepared_candidate"] = {
        **manifest["prepared_candidate"],
        "candidate_state": "selected",
    }

    with pytest.raises(ChaserInputProvenanceProxyStorageError, match="identity"):
        validate_prepared_chaser_input_provenance_proxy(
            replace(prepared, manifest=manifest)
        )
