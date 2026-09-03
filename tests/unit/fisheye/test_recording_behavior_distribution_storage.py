from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fisheye.analysis_workflows.recording_behavior_distribution_storage import (
    PARENT_PATH,
    RecordingBehaviorDistributionStorageError,
    load_recording_behavior_distribution_source_handle,
    validate_recording_behavior_distribution_run,
    write_recording_behavior_distribution_run,
)
from fisheye.analysis_workflows.recording_behavior_distribution_publication import (
    build_recording_behavior_distribution_publication_plan,
    materialize_recording_behavior_distribution_locally,
    publish_recording_behavior_distribution_candidate,
)
from fisheye.analysis_workflows.recording_behavior_distribution_workflow import (
    PreparedRecordingBehaviorDistribution,
)
from fisheye.group_statistics.recording_behavior_distribution_specs import (
    DEFAULT_RECORDING_DISTRIBUTION_METRICS,
)
from fisheye.group_statistics.recording_behavior_distributions import (
    RecordingBehaviorDistributionConfig,
    RecordingDistributionMetricInput,
    compute_recording_behavior_distributions,
)
from fisheye.group_statistics.recording_distribution_scopes import (
    exact_source_membership_masks,
    frame_interval_scope,
    whole_session_scope,
)
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr_io import open_zarr_root


def _result(run_name: str):
    scopes = (
        whole_session_scope(),
        frame_interval_scope(
            scope_id="phase_a",
            scope_label="Phase A",
            scope_family="fixture",
            scope_provider_id="fixture.v1",
            order=1,
            start_frame=10,
            end_frame_exclusive=20,
            source_binding={"sha256": "1" * 64},
        ),
    )
    spec = next(
        item
        for item in DEFAULT_RECORDING_DISTRIBUTION_METRICS
        if item.metric_id == "bout.duration_s"
    )
    values = np.asarray([0.1, 0.2, np.nan])
    identity = {
        "source_run_path": np.asarray(["analysis/source/exact"] * 3, dtype=object),
        "source_manifest_sha256": np.asarray(["2" * 64] * 3, dtype=object),
    }
    fallback = {
        "source_run_path": "analysis/source/exact",
        "source_manifest_sha256": "2" * 64,
    }
    config = RecordingBehaviorDistributionConfig(
        distribution_run_id=run_name,
        recording_id="recording-1",
        scopes=scopes,
        source_record={"bundle_sha256": "3" * 64},
    )
    return compute_recording_behavior_distributions(
        config,
        (
            RecordingDistributionMetricInput(
                spec=spec,
                values=values,
                valid=np.asarray([True, True, True]),
                scope_projection=exact_source_membership_masks(
                    scopes, source_scope_id=["phase_a", None, "phase_a"]
                ),
                source_identity_arrays=identity,
                source_identity_fallback=fallback,
                valid_duration_s_by_scope={"whole_session": 10.0, "phase_a": 5.0},
            ),
        ),
    )


def test_recording_distribution_round_trip_is_object_free_and_receipt_bound(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "analysis.zarr"
    run_name = "recording-distributions-v1"
    result = _result(run_name)
    provenance = build_writer_run_provenance(
        command="recording-distribution-test",
        params={"run_name": run_name},
        input_run_ids={"source": "exact"},
        cwd=Path.cwd(),
    )

    write_recording_behavior_distribution_run(
        archive,
        run_name=run_name,
        result=result,
        run_provenance=provenance,
    )
    handle = load_recording_behavior_distribution_source_handle(
        archive, run_name=run_name, expected_recording_id="recording-1"
    )

    assert handle.result_record["record_sha256"] == result.record["record_sha256"]
    assert len(handle.tables["support"]) == 2
    root = open_zarr_root(archive, mode="r", use_consolidated=False)
    parent = root[PARENT_PATH]
    assert all(name not in parent.attrs for name in ("latest", "latest_complete"))
    run = parent[run_name]
    for table_name in handle.tables:
        assert run[f"{table_name}/row_utf8"].dtype == np.dtype(np.uint8)


def test_recording_distribution_validation_rejects_changed_payload(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "analysis.zarr"
    run_name = "recording-distributions-v1"
    result = _result(run_name)
    write_recording_behavior_distribution_run(
        archive,
        run_name=run_name,
        result=result,
        run_provenance=build_writer_run_provenance(
            command="recording-distribution-test",
            params={"run_name": run_name},
            cwd=Path.cwd(),
        ),
    )
    root = open_zarr_root(archive, mode="a", use_consolidated=False)
    run = root[f"{PARENT_PATH}/{run_name}"]
    payload = run["support/row_utf8"]
    changed = np.asarray(payload[...])
    changed[0] ^= np.uint8(1)
    payload[...] = changed

    with pytest.raises(
        RecordingBehaviorDistributionStorageError, match="changed"
    ):
        validate_recording_behavior_distribution_run(run)


def test_recording_distribution_publishes_atomically_without_selectors(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    open_zarr_root(source, mode="a", use_consolidated=False).attrs[
        "recording_id"
    ] = "recording-1"
    run_name = "recording-distributions-v1"
    result = _result(run_name)
    prepared = PreparedRecordingBehaviorDistribution(
        result=result,
        adapter_evidence={},
        omitted_metrics=(),
    )
    plan = build_recording_behavior_distribution_publication_plan(
        source,
        scratch_root=scratch,
        prepared=prepared,
    )

    local = materialize_recording_behavior_distribution_locally(
        plan, prepared=prepared
    )
    published = publish_recording_behavior_distribution_candidate(plan)

    assert local["valid"] is True
    assert published["final_validation"]["valid"] is True
    handle = load_recording_behavior_distribution_source_handle(
        source, run_name=run_name, expected_recording_id="recording-1"
    )
    assert handle.result_record["record_sha256"] == result.record["record_sha256"]
    direct = open_zarr_root(source, mode="r", use_consolidated=False)
    parent = direct[PARENT_PATH]
    assert all(
        name not in parent.attrs
        for name in ("latest", "latest_complete", "authoritative_run")
    )
