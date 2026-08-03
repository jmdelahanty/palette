from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.analysis import (
    chaser_bout_response,
    chaser_egocentric_bearing,
    chaser_epoch_behavior_summary,
    chaser_escape_events,
    chaser_escape_freeze_summary,
    chaser_gaze_tracking,
    chaser_near_field_occupancy,
    chaser_quadrant_occupancy,
    chaser_radial_occupancy,
    chaser_response_regimes,
)
from fisheye.analysis import chaser_component_writer as writer_mod
from fisheye.analysis.chaser_component_publication import (
    COMPONENT_MANIFEST_ATTR,
    COMPONENT_SELECTOR_ATTR,
    validate_chaser_component_manifest,
)
from fisheye.analysis.chaser_component_writer import (
    CHASER_COMPONENT_WRITER_RECEIPT_SCHEMA_ID,
    ChaserComponentWriterError,
    PublishedChaserComponentPath,
    require_chaser_component_staging_capability,
    sealed_chaser_component_writer,
)
from fisheye.analysis_workflows.materializers import chaser_component as materializer
from fisheye.shared.run_lineage_fingerprint import (
    build_run_lineage_payload,
    write_run_lineage_attrs,
)


_MAINTAINED_WRITERS = (
    (
        chaser_bout_response.write_chaser_bout_response_component,
        chaser_bout_response.COMPONENT_PARENT_NAME,
        chaser_bout_response.SCHEMA_ID,
        chaser_bout_response.SCHEMA_VERSION,
    ),
    (
        chaser_egocentric_bearing.write_chaser_egocentric_bearing_component,
        chaser_egocentric_bearing.COMPONENT_PARENT_NAME,
        chaser_egocentric_bearing.SCHEMA_ID,
        chaser_egocentric_bearing.SCHEMA_VERSION,
    ),
    (
        chaser_epoch_behavior_summary.write_chaser_epoch_behavior_summary_component,
        chaser_epoch_behavior_summary.COMPONENT_PARENT_NAME,
        chaser_epoch_behavior_summary.SCHEMA_ID,
        chaser_epoch_behavior_summary.SCHEMA_VERSION,
    ),
    (
        chaser_escape_events.write_chaser_escape_events_component,
        chaser_escape_events.COMPONENT_PARENT_NAME,
        chaser_escape_events.SCHEMA_ID,
        chaser_escape_events.SCHEMA_VERSION,
    ),
    (
        chaser_escape_freeze_summary.write_chaser_escape_freeze_summary_component,
        chaser_escape_freeze_summary.COMPONENT_PARENT_NAME,
        chaser_escape_freeze_summary.SCHEMA_ID,
        chaser_escape_freeze_summary.SCHEMA_VERSION,
    ),
    (
        chaser_gaze_tracking.write_chaser_gaze_tracking_component,
        chaser_gaze_tracking.COMPONENT_PARENT_NAME,
        chaser_gaze_tracking.SCHEMA_ID,
        chaser_gaze_tracking.SCHEMA_VERSION,
    ),
    (
        chaser_near_field_occupancy.write_chaser_near_field_occupancy_component,
        chaser_near_field_occupancy.COMPONENT_PARENT_NAME,
        chaser_near_field_occupancy.SCHEMA_ID,
        chaser_near_field_occupancy.SCHEMA_VERSION,
    ),
    (
        chaser_quadrant_occupancy.write_chaser_quadrant_occupancy_component,
        chaser_quadrant_occupancy.COMPONENT_PARENT_NAME,
        chaser_quadrant_occupancy.SCHEMA_ID,
        chaser_quadrant_occupancy.SCHEMA_VERSION,
    ),
    (
        chaser_radial_occupancy.write_chaser_radial_occupancy_component,
        chaser_radial_occupancy.COMPONENT_PARENT_NAME,
        chaser_radial_occupancy.SCHEMA_ID,
        chaser_radial_occupancy.SCHEMA_VERSION,
    ),
    (
        chaser_response_regimes.write_chaser_response_regimes_component,
        chaser_response_regimes.COMPONENT_PARENT_NAME,
        chaser_response_regimes.SCHEMA_ID,
        chaser_response_regimes.SCHEMA_VERSION,
    ),
)


def _snapshot():
    return SimpleNamespace(
        run_path="analysis/chaser_distance_runs/base",
        publication_seal_ref=(
            "/analysis/chaser_distance_runs/base@chaser_distance_publication_seal"
        ),
        publication_seal_sha256="a" * 64,
        surface_manifest_ref=(
            "/analysis/chaser_distance_runs/base@chaser_distance_surface_manifest"
        ),
        surface_manifest_sha256="b" * 64,
        row_identity_ref="/analysis/chaser_distance_runs/base@row_identity_contract",
        row_identity_sha256="c" * 64,
        authority_record=lambda: {
            "schema_id": "palette.chaser_distance_read_authority",
            "schema_version": 1,
            "run_ref": "/analysis/chaser_distance_runs/base",
        },
    )


def test_all_maintained_scientific_writers_use_exact_sealed_boundary() -> None:
    assert len(_MAINTAINED_WRITERS) == 10
    for writer, family, schema_id, schema_version in _MAINTAINED_WRITERS:
        assert writer.__chaser_component_sealed_writer__ is True
        assert writer.__chaser_component_family__ == family
        assert writer.__chaser_component_semantic_schema_id__ == schema_id
        assert writer.__chaser_component_semantic_schema_version__ == schema_version

        unsealed_builder = inspect.unwrap(writer)
        with pytest.raises(ChaserComponentWriterError, match="Unsealed"):
            unsealed_builder(
                Path("not-opened.zarr"),
                SimpleNamespace(),
            )


def test_sealed_writer_stages_publishes_and_returns_exact_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    root = zarr.open_group(
        str(source), mode="w", zarr_format=3, use_consolidated=False
    )
    run = root.require_group("analysis/chaser_distance_runs/base")
    run.attrs["recording_id"] = "recording"
    family = run.require_group("test_component")
    family.attrs["latest"] = "stale"
    family.attrs["latest_complete"] = "stale"

    monkeypatch.setattr(writer_mod, "load_chaser_distance_run", lambda *_a, **_k: _snapshot())
    monkeypatch.setattr(materializer, "load_chaser_distance_run", lambda *_a, **_k: _snapshot())

    @sealed_chaser_component_writer(
        component_family="test_component",
        semantic_schema_id="palette.test_chaser_component.v1",
        semantic_schema_version=1,
        method_id="test_chaser_component",
        method_version="1",
    )
    def write_component(
        zarr_path: Path,
        result: SimpleNamespace,
        *,
        overwrite: bool = False,
        _chaser_component_staging_capability: object | None = None,
    ) -> str:
        require_chaser_component_staging_capability(
            _chaser_component_staging_capability
        )
        local = zarr.open_group(
            str(zarr_path), mode="a", zarr_format=3, use_consolidated=False
        )
        component = local.require_group(
            f"{result.chaser_distance_run_path}/test_component/{result.component_name}"
        )
        component.create_array(
            "value",
            data=np.asarray([1.25, 3.5], dtype=np.float32),
            chunks=(2,),
        )
        component.attrs.update(
            {
                "schema_id": "palette.test_chaser_component.v1",
                "schema_version": 1,
                "method": "test_chaser_component",
                "method_version": "1",
                "status": "complete",
            }
        )
        write_run_lineage_attrs(
            component,
            build_run_lineage_payload(
                run_family=(
                    f"{result.chaser_distance_run_path}/test_component"
                ),
                analysis_schema={
                    "schema_id": "palette.test_chaser_component.v1",
                    "schema_version": 1,
                },
                method="test_chaser_component",
                method_version="1",
                source_refs={"source_run": result.chaser_distance_run_path},
                source_fingerprints={"source_run_sha256": "d" * 64},
                parameters={"threshold": 2.0},
            ),
            fingerprint_status="complete",
        )
        return (
            f"{result.chaser_distance_run_path}/test_component/"
            f"{result.component_name}"
        )

    result = SimpleNamespace(
        component_name="candidate_v1",
        chaser_distance_run_name="base",
        chaser_distance_run_path="analysis/chaser_distance_runs/base",
    )
    path = write_component(source, result, overwrite=True)

    assert isinstance(path, PublishedChaserComponentPath)
    assert str(path).endswith("test_component/candidate_v1")
    receipt = path.publication_receipt
    assert receipt["schema_id"] == CHASER_COMPONENT_WRITER_RECEIPT_SCHEMA_ID
    assert receipt["component_family"] == "test_component"
    assert receipt["component_manifest_sha256"]
    assert receipt["schema_version"] == 2
    assert receipt["dependency_handle"]["component_path"] == str(path)
    assert (
        receipt["dependency_handle"]["component_manifest_sha256"]
        == receipt["component_manifest_sha256"]
    )
    assert receipt["dependency_handle"]["record_sha256"]
    assert receipt["payload_array_count"] == 1
    assert receipt["selector_eligible"] is False
    assert receipt["validation"]["valid"] is True
    assert receipt["receipt_sha256"]

    check = zarr.open_group(
        str(source), mode="r", zarr_format=3, use_consolidated=False
    )
    parent = check["analysis/chaser_distance_runs/base/test_component"]
    component = parent["candidate_v1"]
    assert component.attrs["stage_selector_eligible"] is False
    assert component.attrs["palette_run_completion_status"] == "complete"
    assert COMPONENT_MANIFEST_ATTR in component.attrs
    assert parent.attrs["latest"] == "stale"
    assert parent.attrs["latest_complete"] == "stale"
    assert COMPONENT_SELECTOR_ATTR not in parent.attrs
    validate_chaser_component_manifest(
        component,
        snapshot=_snapshot(),
        expected_relative_path="test_component/candidate_v1",
        expected_manifest_sha256=receipt["component_manifest_sha256"],
    )

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        write_component(source, result, overwrite=True)


def test_missing_lineage_fails_before_any_destination_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    root = zarr.open_group(
        str(source), mode="w", zarr_format=3, use_consolidated=False
    )
    root.require_group("analysis/chaser_distance_runs/base")
    monkeypatch.setattr(writer_mod, "load_chaser_distance_run", lambda *_a, **_k: _snapshot())

    @sealed_chaser_component_writer(
        component_family="broken",
        semantic_schema_id="palette.broken.v1",
        semantic_schema_version=1,
        method_id="broken",
        method_version="1",
    )
    def write_broken(
        zarr_path: Path,
        result: SimpleNamespace,
        *,
        _chaser_component_staging_capability: object | None = None,
    ) -> str:
        require_chaser_component_staging_capability(
            _chaser_component_staging_capability
        )
        local = zarr.open_group(
            str(zarr_path), mode="a", zarr_format=3, use_consolidated=False
        )
        component = local.require_group(
            f"{result.chaser_distance_run_path}/broken/{result.component_name}"
        )
        component.create_array(
            "value", data=np.asarray([1], dtype=np.uint8), chunks=(1,)
        )
        component.attrs.update(
            {
                "schema_id": "palette.broken.v1",
                "schema_version": 1,
                "method": "broken",
                "method_version": "1",
            }
        )
        return f"{result.chaser_distance_run_path}/broken/{result.component_name}"

    result = SimpleNamespace(
        component_name="candidate",
        chaser_distance_run_name="base",
        chaser_distance_run_path="analysis/chaser_distance_runs/base",
    )
    with pytest.raises(ChaserComponentWriterError, match="run-lineage"):
        write_broken(source, result)

    check = zarr.open_group(
        str(source), mode="r", zarr_format=3, use_consolidated=False
    )
    assert "broken" not in check["analysis/chaser_distance_runs/base"]
