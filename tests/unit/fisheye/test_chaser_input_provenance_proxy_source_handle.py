from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fisheye.analysis_workflows.chaser_input_provenance_proxy_source_handle import (
    ChaserInputProvenanceProxySourceHandle,
    ChaserInputProvenanceProxySourceHandleError,
    load_chaser_input_provenance_proxy_source_handle,
)
from fisheye.analysis_workflows.materializers.chaser_input_provenance_proxy import (
    materialize_chaser_input_provenance_proxy,
)
from fisheye.shared.zarr.chaser_input_provenance_proxy_schema import (
    CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from tests.unit.fisheye.test_chaser_input_provenance_proxy_materializer import (
    _archive,
    _prepared,
)


def _published(tmp_path: Path) -> Path:
    archive = _archive(tmp_path)
    materialize_chaser_input_provenance_proxy(
        archive,
        prepared=_prepared(),
        scratch_root=tmp_path / "scratch",
        run_name="proxy_v1",
    )
    return archive


def test_strict_handle_reads_exact_consolidated_candidate(tmp_path: Path) -> None:
    archive = _published(tmp_path)
    handle = load_chaser_input_provenance_proxy_source_handle(
        archive,
        run_name="proxy_v1",
        expected_recording_id="recording-1",
    )

    assert handle.run_path == (
        f"{CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH}/proxy_v1"
    )
    assert handle.selector_eligible is False
    assert handle.dimensions.n_frames == 3
    assert handle.dimensions.n_candidates == 5
    assert handle.dimensions.n_chasers == 2
    np.testing.assert_array_equal(handle.acquisition_frame_index, [0, 2, 4])
    np.testing.assert_array_equal(handle.candidate_sample_count, [3, 1, 1])
    np.testing.assert_array_equal(handle.selected, [True, True, True])
    assert handle.selected_chaser_position_xy.shape == (3, 2, 2)
    assert handle.acquisition_frame_index.flags.writeable is False
    assert handle.acquisition_projection_record[
        "temporal_alignment_class"
    ] == "controller_input_provenance_proxy"
    assert handle.acquisition_projection_record[
        "physical_presentation_verified"
    ] is False
    binding = handle.publication_binding_record
    assert binding["run_path"] == handle.run_path
    assert binding["manifest_sha256"] == handle.manifest_sha256
    assert binding["n_candidates"] == 5
    assert binding["selector_eligible"] is False
    handle.assert_current()


@pytest.mark.parametrize(
    "run_name",
    [
        "latest",
        "authoritative_run",
        "../proxy_v1",
        "parent/proxy_v1",
        "/proxy_v1",
    ],
)
def test_loader_rejects_selectors_and_nonbare_paths(
    tmp_path: Path, run_name: str
) -> None:
    archive = _published(tmp_path)
    with pytest.raises(ChaserInputProvenanceProxySourceHandleError):
        load_chaser_input_provenance_proxy_source_handle(
            archive,
            run_name=run_name,
        )


def test_loader_rejects_wrong_recording_and_manifest_expectations(
    tmp_path: Path,
) -> None:
    archive = _published(tmp_path)
    with pytest.raises(ChaserInputProvenanceProxySourceHandleError, match="expectation"):
        load_chaser_input_provenance_proxy_source_handle(
            archive,
            run_name="proxy_v1",
            expected_recording_id="other-recording",
        )
    with pytest.raises(ChaserInputProvenanceProxySourceHandleError, match="expected"):
        load_chaser_input_provenance_proxy_source_handle(
            archive,
            run_name="proxy_v1",
            expected_manifest_sha256="0" * 64,
        )


def test_parent_selector_and_array_tampering_fail_closed(tmp_path: Path) -> None:
    archive = _published(tmp_path)
    root = open_zarr_root(archive, mode="a", use_consolidated=False)
    parent = root[CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH]
    parent.attrs["latest"] = "proxy_v1"
    consolidate_metadata_capture_expected_warnings(archive)
    with pytest.raises(ChaserInputProvenanceProxySourceHandleError, match="selector"):
        load_chaser_input_provenance_proxy_source_handle(
            archive,
            run_name="proxy_v1",
        )

    del parent.attrs["latest"]
    consolidate_metadata_capture_expected_warnings(archive)
    parent["proxy_v1/arrays/selected_stimulus_frame_num"][0] += 1
    with pytest.raises(
        ChaserInputProvenanceProxySourceHandleError,
        match="content digest",
    ):
        load_chaser_input_provenance_proxy_source_handle(
            archive,
            run_name="proxy_v1",
        )


def test_handle_rejects_post_seal_mutation(tmp_path: Path) -> None:
    archive = _published(tmp_path)
    handle = load_chaser_input_provenance_proxy_source_handle(
        archive,
        run_name="proxy_v1",
    )
    root = open_zarr_root(archive, mode="a", use_consolidated=False)
    root[f"{handle.run_path}/arrays/selected_stimulus_frame_num"][0] += 1

    with pytest.raises(ChaserInputProvenanceProxySourceHandleError):
        handle.assert_current()


def test_handle_cannot_be_minted_directly() -> None:
    with pytest.raises(ChaserInputProvenanceProxySourceHandleError, match="loader"):
        ChaserInputProvenanceProxySourceHandle()
