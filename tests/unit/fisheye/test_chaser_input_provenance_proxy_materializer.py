from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fisheye.analysis_workflows.chaser_input_provenance_proxy import (
    select_chaser_input_provenance_proxy,
)
from fisheye.analysis_workflows.chaser_input_provenance_proxy_storage import (
    prepare_chaser_input_provenance_proxy,
)
from fisheye.analysis_workflows.materializers.chaser_input_provenance_proxy import (
    ChaserInputProvenanceProxyMaterializationError,
    build_chaser_input_provenance_proxy_materialization_plan,
    materialize_chaser_input_provenance_proxy,
)
from fisheye.shared.zarr.chaser_input_provenance_proxy_schema import (
    CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from tests.unit.fisheye.test_chaser_input_provenance_proxy import _source


def _archive(tmp_path: Path, *, recording_id: str = "recording-1") -> Path:
    archive = tmp_path / "analysis.zarr"
    root = open_zarr_root(archive, mode="w-")
    root.attrs["recording_id"] = recording_id
    consolidate_metadata_capture_expected_warnings(archive)
    return archive


def _prepared():
    return prepare_chaser_input_provenance_proxy(
        select_chaser_input_provenance_proxy(_source())
    )


def test_plan_is_named_selector_ineligible_and_rejects_wrong_recording(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    prepared = _prepared()
    plan = build_chaser_input_provenance_proxy_materialization_plan(
        archive,
        scratch_root=tmp_path / "scratch",
        run_name="proxy_v1",
        prepared=prepared,
    )

    assert plan.run_path == (
        f"{CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH}/proxy_v1"
    )
    assert plan.to_json()["selector_eligible"] is False
    assert plan.to_json()["selection"] == "none"

    wrong = _archive(tmp_path / "wrong", recording_id="other-recording")
    with pytest.raises(
        ChaserInputProvenanceProxyMaterializationError,
        match="recording_id",
    ):
        build_chaser_input_provenance_proxy_materialization_plan(
            wrong,
            scratch_root=tmp_path / "wrong-scratch",
            run_name="proxy_v1",
            prepared=prepared,
        )


def test_materialize_publishes_direct_and_consolidated_without_selector(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    prepared = _prepared()

    receipt = materialize_chaser_input_provenance_proxy(
        archive,
        prepared=prepared,
        scratch_root=tmp_path / "scratch",
        run_name="proxy_v1",
    )

    assert receipt["status"] == "published_selector_ineligible"
    assert receipt["selector_eligible"] is False
    assert receipt["selection"] == "none"
    for consolidated in (False, True):
        root = open_zarr_root(
            archive,
            mode="r",
            use_consolidated=consolidated,
        )
        parent = root[CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH]
        assert "latest" not in parent.attrs
        assert "authoritative_run" not in parent.attrs
        run = parent["proxy_v1"]
        assert run.attrs["selector_eligible"] is False
        assert run.attrs["selection"] == "none"
        assert run["arrays/candidate_offsets"].shape == (4,)
        assert run["arrays/selected_chaser_position_xy"].shape == (3, 2, 2)
        assert np.asarray(run["arrays/selection_reason_code"]).dtype == np.uint8

    with pytest.raises(FileExistsError, match="existing target"):
        build_chaser_input_provenance_proxy_materialization_plan(
            archive,
            scratch_root=tmp_path / "second-scratch",
            run_name="proxy_v1",
            prepared=prepared,
        )


def test_plan_rejects_prepared_array_tampering(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    prepared = _prepared()
    arrays = dict(prepared.arrays)
    changed = arrays["candidate_offsets"].copy()
    changed[-1] -= 1
    changed.setflags(write=False)
    arrays["candidate_offsets"] = changed

    with pytest.raises(ValueError, match="content digest"):
        build_chaser_input_provenance_proxy_materialization_plan(
            archive,
            scratch_root=tmp_path / "scratch",
            run_name="proxy_v1",
            prepared=replace(prepared, arrays=arrays),
        )
