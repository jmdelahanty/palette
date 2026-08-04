from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time

import numpy as np
import pytest
import zarr

from fisheye.analysis import (
    bout_kinematics_schema,
    detection_occupancy_schema,
    swim_bout_schema,
)
from fisheye.analysis.exact_tabular_storage import (
    build_exact_tabular_storage_receipt,
)
from fisheye.analysis.swim_bout_frame_axis import (
    FRAME_AXIS_CONTRACT_ATTR,
    FRAME_AXIS_CONTRACT_SHA256_ATTR,
    build_frame_axis_contract,
)
from fisheye.analysis.track_kinematics_io import load_track_kinematics_track
from fisheye.analysis_workflows.analysis_candidate_execution import (
    PhysicalIOScope,
    build_candidate_execution_request,
    require_candidate_execution_receipt,
    require_candidate_execution_request,
)
from fisheye.analysis_workflows.materializers.runtime_telemetry import PhaseTelemetry
from fisheye.analysis_workflows.analysis_candidate_execution_catalog import (
    ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE,
)
from fisheye.analysis_workflows.analysis_candidate_invocation import (
    build_exact_tabular_invocation,
    build_occupancy_invocation,
)
from fisheye.analysis_workflows.materializers.exact_tabular_candidate import (
    compute_exact_tabular_logical_hashes,
)
from fisheye.analysis_workflows.occupancy_candidate_execution import (
    build_occupancy_source_identity,
    occupancy_source_identity_sha256,
)
from fisheye.diagnostics.analysis_candidate_execution import (
    ExactTabularCandidateExecutionFailed,
    exact_tabular_selector_snapshot_sha256,
    require_exact_tabular_execution_attempt,
    run_exact_tabular_candidate_fresh_process,
)
from fisheye.diagnostics import analysis_candidate_execution as runner_module
from fisheye.shared.zarr.analysis_benchmark_suite import (
    AnalysisBenchmarkScale,
    build_analysis_benchmark_suite,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1
from tests.unit.fisheye.test_benchmark_track_kinematics_v2_candidate import (
    SOURCE_RUN_NAME as TRACK_SOURCE_RUN_NAME,
    _build_canonical_sealed_source,
)
from tests.unit.fisheye.test_benchmark_exact_tabular_candidates import (
    _occupancy_archive,
)


@pytest.fixture(scope="module")
def canonical_track_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_canonical_sealed_source(
        tmp_path_factory.mktemp("analysis-candidate-canonical-track")
    )


def _copy_canonical_template(template: Path, workflow: Path) -> Path:
    source_recording = template.parents[1]
    target_recording = workflow / "recording"
    target_recording.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_recording, target_recording)
    return target_recording / "zarr" / template.name


def _create_array(
    group: zarr.Group,
    path: str,
    dtype: str | None,
    axes: tuple[str, ...],
    *,
    frame_count: int,
) -> None:
    parent = group
    parts = path.split("/")
    for name in parts[:-1]:
        parent = parent.require_group(name)
    resolved = np.dtype("S64" if dtype is None else dtype)
    first_extent = (
        2
        if axes[0] == "detector_signal"
        else (frame_count if axes[0] == "frame" else 3)
    )
    shape = (first_extent,) if len(axes) == 1 else (first_extent, frame_count)
    values = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    parent.create_array(parts[-1], data=values.astype(resolved))


def _set_columnar_attrs(
    group: zarr.Group,
    specs: dict[str, object],
    table_paths: tuple[str, ...],
) -> None:
    for table_path in table_paths:
        prefix = table_path + "/"
        fields = [
            (path[len(prefix) :], spec)
            for path, spec in specs.items()
            if path.startswith(prefix) and "/" not in path[len(prefix) :]
        ]
        if not fields:
            continue
        table = group[table_path]
        table.attrs["storage_layout"] = "columnar"
        table.attrs["field_names"] = [name for name, _spec in fields]
        table.attrs["field_dtypes"] = {
            name: spec.logical_dtype for name, spec in fields
        }


def _archive(path: Path, *, family: str) -> tuple[zarr.Group, zarr.Group]:
    root = zarr.open_group(str(path), mode="a", zarr_format=3, use_consolidated=False)
    analysis = root.require_group("analysis")
    track = load_track_kinematics_track(
        root,
        run_name=TRACK_SOURCE_RUN_NAME,
        scope="offline",
        track_id=7,
        required_speed_levels=(),
    )
    authority = track.authority_record()
    frames = np.asarray(track.source_acquisition_frame_index, dtype=np.int64)

    def create_source(stage: str, run_name: str) -> tuple[zarr.Group, zarr.Group]:
        if stage == "swim_bouts":
            schema = swim_bout_schema
            parent = analysis.require_group("swim_bout_runs")
            required = schema._required_specs()
            table_paths = schema._COLUMNAR_TABLE_PATHS
            attrs = {
                "schema_id": schema.SWIM_BOUT_RUN_SCHEMA_ID,
                "schema_version": schema.SWIM_BOUT_RUN_SCHEMA_VERSION,
                "layout": schema.SWIM_BOUT_LAYOUT,
            }
            writer = schema.write_swim_bout_array_manifest
        else:
            schema = bout_kinematics_schema
            parent = analysis.require_group("bout_kinematics_runs")
            required = schema._required_specs()
            table_paths = schema._COLUMNAR_TABLE_PATHS
            attrs = {
                "schema_id": schema.BOUT_KINEMATICS_RUN_SCHEMA_ID,
                "schema_version": schema.BOUT_KINEMATICS_RUN_SCHEMA_VERSION,
                "layout": schema.BOUT_KINEMATICS_LAYOUT,
            }
            writer = schema.write_bout_kinematics_array_manifest
        parent.attrs.update(
            {
                "latest": run_name,
                "latest_complete": run_name,
                "palette_completion_epoch": 1,
            }
        )
        run = parent.create_group(run_name)
        run.attrs.update(
            {
                **attrs,
                "palette_run_name": run_name,
                "palette_run_completion_status": "complete",
                "stage_selector_eligible": True,
                "provenance": {"stage": stage},
            }
        )
        for spec in required.values():
            _create_array(
                run,
                spec.path,
                spec.dtype,
                spec.axes,
                frame_count=int(frames.size),
            )
        _set_columnar_attrs(run, required, table_paths)
        run.require_group("report_tables").create_array(
            "diagnostic_only",
            data=np.asarray([1.0], dtype=np.float32),
        )
        if stage == "swim_bouts":
            contract = build_frame_axis_contract(
                frames,
                authoritative_path=str(
                    authority["source_acquisition_frame_index_ref"]
                ).lstrip("/"),
                source_track_kinematics_run=TRACK_SOURCE_RUN_NAME,
                track_id=7,
                source_track_motion_manifest_sha256=str(
                    authority["motion_manifest_sha256"]
                ),
            )
            run.attrs.update(
                {
                    "source_track_kinematics_run": TRACK_SOURCE_RUN_NAME,
                    "track_id": 7,
                    "source_track_motion_authority": authority,
                    "source_track_motion_manifest_sha256": authority[
                        "motion_manifest_sha256"
                    ],
                    FRAME_AXIS_CONTRACT_ATTR: contract,
                    FRAME_AXIS_CONTRACT_SHA256_ATTR: canonical_json_sha256(contract),
                }
            )
        else:
            run.attrs["source_refs"] = {
                "source_track_motion_authority": authority,
                "source_swim_bout_track_motion_authority": authority,
                "source_swim_bout_path": (
                    "analysis/swim_bout_runs/swim_source/levels/default"
                ),
            }
        writer(run)
        return parent, run

    if family == "bout_kinematics":
        _swim_parent, swim_run = create_source("swim_bouts", "swim_source")
        swim_run.require_group("levels").require_group("default")
    parent, _run = create_source(family, "source")
    return root, parent


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _request(
    archive: Path,
    *,
    family: str,
    scratch: Path,
) -> dict[str, object]:
    registry_probe = archive.parent / "registry.snapshot"
    profiles_probe = archive.parent / "production_profiles.snapshot"
    registry_probe.write_bytes(b"registry-v1")
    profiles_probe.write_bytes(b"production-profiles-v1")
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    parent_path = (
        "analysis/swim_bout_runs"
        if family == "swim_bouts"
        else "analysis/bout_kinematics_runs"
    )
    source = root[f"{parent_path}/source"]
    schema = swim_bout_schema if family == "swim_bouts" else bout_kinematics_schema
    source_declarations = (
        schema.build_swim_bout_array_declarations(source, byte_planner_adopted=False)
        if family == "swim_bouts"
        else schema.build_bout_kinematics_array_declarations(
            source, byte_planner_adopted=False
        )
    )
    candidate_declarations = (
        schema.build_swim_bout_array_declarations(source, byte_planner_adopted=True)
        if family == "swim_bouts"
        else schema.build_bout_kinematics_array_declarations(
            source, byte_planner_adopted=True
        )
    )
    source_hashes = compute_exact_tabular_logical_hashes(
        source,
        source_declarations,
    )
    storage = build_exact_tabular_storage_receipt(
        source,
        declarations=candidate_declarations,
        profile=PUBLISHED_HTTP_V1,
    )
    suite = build_analysis_benchmark_suite(
        family_id=family,
        scale=AnalysisBenchmarkScale(
            scale_id="test_fixture",
            dimensions=storage.dimensions,
            description="Exact-tabular fresh-process test fixture.",
        ),
        storage_receipt=storage,
        repetitions=1,
    )
    return build_candidate_execution_request(
        execution_id=f"{family}_fixture_rep0",
        adapter_manifest=ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE[
            family
        ].as_manifest(),
        invocation=build_exact_tabular_invocation(
            storage_profile_id="published_http_v1",
            copy_backend="python",
            keep_scratch=False,
        ),
        benchmark_suite=suite,
        archive_path=archive,
        source_run_path=f"{parent_path}/source",
        candidate_run_path=f"{parent_path}/candidate",
        scratch_root=scratch,
        source_identity_sha256=canonical_json_sha256(source_hashes),
        palette_commit=_git_commit(),
        repetition_index=0,
        candidate_order_index=0,
        candidate_order_count=1,
        cache_state="fresh_process_os_cache_uncontrolled",
        physical_io_scope=PhysicalIOScope.PROCESS_SELF_PROC_IO,
        selector_before_sha256=exact_tabular_selector_snapshot_sha256(
            archive,
            stage_id=family,
        ),
        registry_probe_path=registry_probe,
        production_profiles_probe_path=profiles_probe,
    )


def _occupancy_request(
    archive: Path,
    *,
    family: str,
    scratch: Path,
    candidate_name: str = "typed_candidate",
) -> dict[str, object]:
    registry_probe = archive.parent / f"{family}.registry.snapshot"
    profiles_probe = archive.parent / f"{family}.production_profiles.snapshot"
    registry_probe.write_bytes(b"registry-v1")
    profiles_probe.write_bytes(b"production-profiles-v1")
    parent_path = (
        "analysis/detection_occupancy_runs"
        if family == "detection_occupancy"
        else "analysis/session_occupancy_runs"
    )
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    source = root[f"{parent_path}/source"]
    session = family == "session_occupancy"
    source_declarations = detection_occupancy_schema.build_occupancy_array_declarations(
        source,
        session=session,
        byte_planner_adopted=False,
    )
    candidate_declarations = (
        detection_occupancy_schema.build_occupancy_array_declarations(
            source,
            session=session,
            byte_planner_adopted=True,
        )
    )
    source_hashes = compute_exact_tabular_logical_hashes(
        source,
        source_declarations,
    )
    source_identity = build_occupancy_source_identity(
        root,
        source,
        stage_id=family,
        logical_hashes=source_hashes,
    )
    spatiotemporal_identity_sha256 = occupancy_source_identity_sha256(source_identity)
    logical_identity_sha256 = canonical_json_sha256(source_hashes)
    storage = build_exact_tabular_storage_receipt(
        source,
        declarations=candidate_declarations,
        profile=PUBLISHED_HTTP_V1,
    )
    suite = build_analysis_benchmark_suite(
        family_id=family,
        scale=AnalysisBenchmarkScale(
            scale_id="test_fixture",
            dimensions=storage.dimensions,
            description="Spatiotemporally bound occupancy test fixture.",
        ),
        storage_receipt=storage,
        repetitions=1,
    )
    return build_candidate_execution_request(
        execution_id=f"{family}_fixture_rep0",
        adapter_manifest=ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE[
            family
        ].as_manifest(),
        invocation=build_occupancy_invocation(
            source_spatiotemporal_identity_sha256=spatiotemporal_identity_sha256,
            storage_profile_id="published_http_v1",
            copy_backend="python",
            keep_scratch=False,
        ),
        benchmark_suite=suite,
        archive_path=archive,
        source_run_path=f"{parent_path}/source",
        candidate_run_path=f"{parent_path}/{candidate_name}",
        scratch_root=scratch,
        source_identity_sha256=logical_identity_sha256,
        palette_commit=_git_commit(),
        repetition_index=0,
        candidate_order_index=0,
        candidate_order_count=1,
        cache_state="fresh_process_os_cache_uncontrolled",
        physical_io_scope=PhysicalIOScope.PROCESS_SELF_PROC_IO,
        selector_before_sha256=exact_tabular_selector_snapshot_sha256(
            archive,
            stage_id=family,
        ),
        registry_probe_path=registry_probe,
        production_profiles_probe_path=profiles_probe,
    )


def test_fresh_process_runner_has_no_unbound_execution_knobs() -> None:
    parameters = inspect.signature(run_exact_tabular_candidate_fresh_process).parameters
    assert "copy_backend" not in parameters
    assert "keep_scratch" not in parameters


@pytest.mark.parametrize("family", ["swim_bouts", "bout_kinematics"])
def test_exact_tabular_typed_runner_emits_complete_nonpromoting_receipt(
    tmp_path: Path,
    family: str,
    canonical_track_template: Path,
) -> None:
    workflow = tmp_path / ".palette_benchmarks" / family
    archive = _copy_canonical_template(canonical_track_template, workflow)
    _root, _parent = _archive(archive, family=family)
    request = _request(
        archive,
        family=family,
        scratch=tmp_path / f"scratch-{family}",
    )
    request_path = workflow / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    receipt_path = workflow / "evidence" / "receipt.json"
    attempt_path = workflow / "evidence" / "attempt.json"

    receipt = run_exact_tabular_candidate_fresh_process(
        request_path,
        receipt_path=receipt_path,
        attempt_path=attempt_path,
    )

    require_candidate_execution_receipt(
        receipt,
        expected_request_payload_digest=str(request["payload_digest"]),
    )
    assert receipt_path.is_file()
    assert not attempt_path.exists()
    assert receipt["payload"]["publication_gate_passed"] is False
    assert receipt["payload"]["physical_io"]["scope"] == "process_self_proc_io"
    assert receipt["payload"]["physical_io"]["physical_io_measured"] is False
    assert receipt["payload"]["coordinate_evidence"]["status"] == (
        "verified_bound_source"
    )
    assert len(receipt["payload"]["phases"]) == 12
    assert all(
        phase["outcome"] == "succeeded" for phase in receipt["payload"]["phases"]
    )
    reopened = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    parent_path = request["payload"]["adapter_manifest"]["payload"]["run_parent"]
    parent = reopened[parent_path]
    assert parent.attrs["latest"] == "source"
    assert parent.attrs["latest_complete"] == "source"
    assert parent["candidate"].attrs["stage_selector_eligible"] is False


@pytest.mark.parametrize(
    ("family", "expected_array_count", "expected_segment_role"),
    [
        ("detection_occupancy", 30, "stimulus_epoch_windows"),
        ("session_occupancy", 29, "recording_temporal_axis"),
    ],
)
def test_occupancy_runner_binds_spatiotemporal_authorities(
    tmp_path: Path,
    family: str,
    expected_array_count: int,
    expected_segment_role: str,
) -> None:
    workflow = tmp_path / ".palette_benchmarks" / family
    workflow.mkdir(parents=True)
    archive = _occupancy_archive(workflow / f"{family}.zarr", family=family)
    request = _occupancy_request(
        archive,
        family=family,
        scratch=tmp_path / f"scratch-{family}",
    )
    request_path = workflow / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    receipt_path = workflow / "evidence" / "receipt.json"
    attempt_path = workflow / "evidence" / "attempt.json"

    receipt = run_exact_tabular_candidate_fresh_process(
        request_path,
        receipt_path=receipt_path,
        attempt_path=attempt_path,
    )

    require_candidate_execution_receipt(
        receipt,
        expected_request_payload_digest=str(request["payload_digest"]),
    )
    assert receipt["payload"]["logical_equality"]["compared_array_count"] == (
        expected_array_count
    )
    coordinate = receipt["payload"]["coordinate_evidence"]
    assert coordinate["status"] == "verified_bound_source"
    assert [item["role"] for item in coordinate["source_authority_digests"]] == (
        sorted(["detection_geometry", expected_segment_role])
    )
    assert receipt["payload"]["publication_gate_passed"] is False
    assert receipt_path.is_file()
    assert not attempt_path.exists()
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    parent_path = (
        "analysis/detection_occupancy_runs"
        if family == "detection_occupancy"
        else "analysis/session_occupancy_runs"
    )
    parent = root[parent_path]
    assert parent.attrs["latest"] == "source"
    assert parent.attrs["latest_complete"] == "source"
    assert parent["typed_candidate"].attrs["stage_selector_eligible"] is False


@pytest.mark.parametrize(
    ("family", "tamper"),
    [
        ("detection_occupancy", "detection_geometry"),
        ("detection_occupancy", "stimulus_windows"),
        ("session_occupancy", "recording_time_axis"),
    ],
)
def test_occupancy_runner_rejects_authority_swaps_after_request(
    tmp_path: Path,
    family: str,
    tamper: str,
) -> None:
    workflow = tmp_path / ".palette_benchmarks" / f"{family}-{tamper}"
    workflow.mkdir(parents=True)
    archive = _occupancy_archive(workflow / f"{family}.zarr", family=family)
    request = _occupancy_request(
        archive,
        family=family,
        scratch=tmp_path / f"scratch-{family}-{tamper}",
    )
    root = zarr.open_group(str(archive), mode="r+", use_consolidated=False)
    if tamper == "detection_geometry":
        boxes = root["refined_detect_runs/refined_1/instances/bbox_norm_coords"]
        boxes[0, 0] = np.float32(float(boxes[0, 0]) + 1.0)
    elif tamper == "stimulus_windows":
        starts = root["analysis/stimulus_epoch_runs/epoch_source/windows/start_frame"]
        starts[0] = np.int64(1)
    else:
        root.attrs["total_frames"] = 9
    request_path = workflow / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    receipt_path = workflow / "evidence" / "receipt.json"
    attempt_path = workflow / "evidence" / "attempt.json"

    with pytest.raises(ExactTabularCandidateExecutionFailed) as caught:
        run_exact_tabular_candidate_fresh_process(
            request_path,
            receipt_path=receipt_path,
            attempt_path=attempt_path,
        )

    assert not receipt_path.exists()
    assert attempt_path.is_file()
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    require_exact_tabular_execution_attempt(
        attempt,
        expected_request_payload_digest=str(request["payload_digest"]),
    )
    assert caught.value.attempt == attempt
    assert attempt["payload"]["failure_phase"] == "runner_preflight"
    assert attempt["payload"]["target_state"]["exists"] is False
    assert attempt["payload"]["nonmutation_evidence"]["unchanged"] is True


def test_exact_tabular_typed_runner_writes_attempt_for_coordinate_tampering(
    tmp_path: Path,
    canonical_track_template: Path,
) -> None:
    workflow = tmp_path / ".palette_benchmarks" / "swim_bouts_failure"
    archive = _copy_canonical_template(canonical_track_template, workflow)
    root, _parent = _archive(archive, family="swim_bouts")
    request = _request(
        archive,
        family="swim_bouts",
        scratch=tmp_path / "scratch-failure",
    )
    authority = dict(
        root["analysis/swim_bout_runs/source"].attrs["source_track_motion_authority"]
    )
    authority["positions_px_coordinate_descriptor_sha256"] = "not-a-digest"
    root["analysis/swim_bout_runs/source"].attrs[
        "source_track_motion_authority"
    ] = authority
    request_path = workflow / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    receipt_path = workflow / "evidence" / "receipt.json"
    attempt_path = workflow / "evidence" / "attempt.json"

    with pytest.raises(ExactTabularCandidateExecutionFailed) as caught:
        run_exact_tabular_candidate_fresh_process(
            request_path,
            receipt_path=receipt_path,
            attempt_path=attempt_path,
        )

    assert not receipt_path.exists()
    assert attempt_path.is_file()
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    require_exact_tabular_execution_attempt(
        attempt,
        expected_request_payload_digest=str(request["payload_digest"]),
    )
    legacy_attempt = deepcopy(attempt)
    legacy_attempt["schema_version"] = 1
    with pytest.raises(ValueError, match="schema identity"):
        require_exact_tabular_execution_attempt(
            legacy_attempt,
            expected_request_payload_digest=str(request["payload_digest"]),
        )
    numeric_alias = deepcopy(attempt)
    numeric_alias["schema_version"] = 2.0
    with pytest.raises(ValueError, match="schema identity"):
        require_exact_tabular_execution_attempt(
            numeric_alias,
            expected_request_payload_digest=str(request["payload_digest"]),
        )
    assert caught.value.attempt == attempt
    assert attempt["payload"]["failure_phase"] == "runner_preflight"
    assert attempt["payload"]["target_state"]["exists"] is False
    assert attempt["payload"]["nonmutation_evidence"]["unchanged"] is True

    rebound = deepcopy(attempt)
    rebound_request = rebound["payload"]["request"]
    rebound_request["payload"]["execution_id"] = "rebound_execution"
    rebound_request["payload_digest"] = canonical_json_sha256(
        rebound_request["payload"]
    )
    rebound["payload"]["request_payload_digest"] = rebound_request["payload_digest"]
    rebound["payload_digest"] = canonical_json_sha256(rebound["payload"])
    with pytest.raises(ValueError, match="request binding"):
        require_exact_tabular_execution_attempt(
            rebound,
            expected_request_payload_digest=str(request["payload_digest"]),
        )

    extra_target_field = deepcopy(attempt)
    extra_target_field["payload"]["target_state"]["invented"] = True
    extra_target_field["payload_digest"] = canonical_json_sha256(
        extra_target_field["payload"]
    )
    with pytest.raises(ValueError, match="target-state field set"):
        require_exact_tabular_execution_attempt(
            extra_target_field,
            expected_request_payload_digest=str(request["payload_digest"]),
        )

    telemetry = PhaseTelemetry(materializer="exact_tabular_candidate")
    with telemetry.phase("plan"):
        pass
    injected_runtime = deepcopy(attempt)
    injected_runtime["payload"]["runtime_telemetry"] = telemetry.to_json()
    injected_runtime["payload_digest"] = canonical_json_sha256(
        injected_runtime["payload"]
    )
    require_exact_tabular_execution_attempt(
        injected_runtime,
        expected_request_payload_digest=str(request["payload_digest"]),
    )
    fabricated = deepcopy(injected_runtime)
    fabricated["payload"]["runtime_telemetry"]["phases"][0]["wall_seconds"] = -1.0
    fabricated["payload_digest"] = canonical_json_sha256(fabricated["payload"])
    with pytest.raises(ValueError, match="nonnegative"):
        require_exact_tabular_execution_attempt(
            fabricated,
            expected_request_payload_digest=str(request["payload_digest"]),
        )


def test_execution_request_rejects_descendant_run_alias(
    tmp_path: Path,
    canonical_track_template: Path,
) -> None:
    workflow = tmp_path / ".palette_benchmarks" / "descendant_alias"
    archive = _copy_canonical_template(canonical_track_template, workflow)
    _archive(archive, family="swim_bouts")
    request = _request(
        archive,
        family="swim_bouts",
        scratch=tmp_path / "scratch-descendant",
    )
    tampered = deepcopy(request)
    tampered["payload"][
        "candidate_run_path"
    ] = "analysis/swim_bout_runs/nested/candidate"
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    with pytest.raises(ValueError, match="immediate run child"):
        require_candidate_execution_request(tampered)


def test_driver_tombstones_candidate_when_protected_state_changes(
    tmp_path: Path,
    canonical_track_template: Path,
) -> None:
    workflow = tmp_path / ".palette_benchmarks" / "protected_state_change"
    archive = _copy_canonical_template(canonical_track_template, workflow)
    _archive(archive, family="swim_bouts")
    request = _request(
        archive,
        family="swim_bouts",
        scratch=tmp_path / "scratch-protected-state",
    )
    request_path = workflow / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    receipt_path = workflow / "evidence" / "receipt.json"
    attempt_path = workflow / "evidence" / "attempt.json"
    candidate_path = archive / "analysis" / "swim_bout_runs" / "candidate"
    registry_probe = Path(
        request["payload"]["protected_state_before"]["registry_probe_path"]
    )
    mutation_finished = threading.Event()

    def mutate_after_publication_starts() -> None:
        for _ in range(12_000):
            if candidate_path.exists():
                registry_probe.write_bytes(b"registry-v2-concurrent-change")
                mutation_finished.set()
                return
            time.sleep(0.005)

    mutator = threading.Thread(target=mutate_after_publication_starts, daemon=True)
    mutator.start()
    with pytest.raises(ExactTabularCandidateExecutionFailed) as caught:
        run_exact_tabular_candidate_fresh_process(
            request_path,
            receipt_path=receipt_path,
            attempt_path=attempt_path,
        )
    mutator.join(timeout=1.0)
    assert mutation_finished.is_set()
    assert not receipt_path.exists()
    assert attempt_path.is_file()
    assert caught.value.attempt["payload"]["failure_phase"] == (
        "driver_protected_poststate"
    )
    assert caught.value.attempt["payload"]["nonmutation_evidence"]["unchanged"] is False
    failed = zarr.open_group(str(archive), mode="r", use_consolidated=False)[
        "analysis/swim_bout_runs/candidate"
    ]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False
    assert "analysis_candidate_execution_tombstone" in failed.attrs


def test_receipt_write_failure_tombstones_published_candidate(
    tmp_path: Path,
    canonical_track_template: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = tmp_path / ".palette_benchmarks" / "receipt_failure"
    archive = _copy_canonical_template(canonical_track_template, workflow)
    _archive(archive, family="swim_bouts")
    request = _request(
        archive,
        family="swim_bouts",
        scratch=tmp_path / "scratch-receipt-failure",
    )
    request_path = workflow / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    receipt_path = workflow / "evidence" / "receipt.json"
    attempt_path = workflow / "evidence" / "attempt.json"
    original_write = runner_module._write_json_exclusive

    def fail_receipt(path: Path, value: dict[str, object]) -> None:
        if path == receipt_path.resolve():
            raise OSError("injected receipt publication failure")
        original_write(path, value)

    monkeypatch.setattr(runner_module, "_write_json_exclusive", fail_receipt)
    with pytest.raises(ExactTabularCandidateExecutionFailed) as caught:
        run_exact_tabular_candidate_fresh_process(
            request_path,
            receipt_path=receipt_path,
            attempt_path=attempt_path,
        )
    assert caught.value.attempt["payload"]["failure_phase"] == (
        "driver_receipt_publication"
    )
    assert not receipt_path.exists()
    assert attempt_path.is_file()
    failed = zarr.open_group(str(archive), mode="r", use_consolidated=False)[
        "analysis/swim_bout_runs/candidate"
    ]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False


def test_driver_rejects_child_self_certified_process_identity(
    tmp_path: Path,
    canonical_track_template: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = tmp_path / ".palette_benchmarks" / "child_identity_tampering"
    archive = _copy_canonical_template(canonical_track_template, workflow)
    _archive(archive, family="swim_bouts")
    request = _request(
        archive,
        family="swim_bouts",
        scratch=tmp_path / "scratch-child-identity",
    )
    request_path = workflow / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    receipt_path = workflow / "evidence" / "receipt.json"
    attempt_path = workflow / "evidence" / "attempt.json"
    real_popen = subprocess.Popen

    class TamperingPopen:
        def __init__(self, command, **kwargs):
            self._command = command
            self._process = real_popen(command, **kwargs)

        @property
        def pid(self):
            return self._process.pid

        @property
        def returncode(self):
            return self._process.returncode

        def communicate(self):
            result = self._process.communicate()
            hidden_receipt = Path(self._command[self._command.index("--receipt") + 1])
            value = json.loads(hidden_receipt.read_text(encoding="utf-8"))
            value["payload"]["fresh_process"]["child_pid"] += 1
            value["payload_digest"] = canonical_json_sha256(value["payload"])
            hidden_receipt.write_text(json.dumps(value), encoding="utf-8")
            return result

    monkeypatch.setattr(runner_module.subprocess, "Popen", TamperingPopen)
    with pytest.raises(ExactTabularCandidateExecutionFailed) as caught:
        run_exact_tabular_candidate_fresh_process(
            request_path,
            receipt_path=receipt_path,
            attempt_path=attempt_path,
        )

    assert caught.value.attempt["payload"]["failure_phase"] == ("driver_child_evidence")
    assert not receipt_path.exists()
    assert attempt_path.is_file()
    failed = zarr.open_group(str(archive), mode="r", use_consolidated=False)[
        "analysis/swim_bout_runs/candidate"
    ]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert "analysis_candidate_execution_tombstone" in failed.attrs


def test_malformed_child_attempt_still_uses_driver_failure_boundary(
    tmp_path: Path,
    canonical_track_template: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = tmp_path / ".palette_benchmarks" / "malformed_child_attempt"
    archive = _copy_canonical_template(canonical_track_template, workflow)
    root, _parent = _archive(archive, family="swim_bouts")
    request = _request(
        archive,
        family="swim_bouts",
        scratch=tmp_path / "scratch-malformed-attempt",
    )
    authority = dict(
        root["analysis/swim_bout_runs/source"].attrs["source_track_motion_authority"]
    )
    authority["positions_px_coordinate_descriptor_sha256"] = "invalid"
    root["analysis/swim_bout_runs/source"].attrs[
        "source_track_motion_authority"
    ] = authority
    request_path = workflow / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    receipt_path = workflow / "evidence" / "receipt.json"
    attempt_path = workflow / "evidence" / "attempt.json"
    real_popen = subprocess.Popen

    class TamperingPopen:
        def __init__(self, command, **kwargs):
            self._command = command
            self._process = real_popen(command, **kwargs)

        @property
        def pid(self):
            return self._process.pid

        @property
        def returncode(self):
            return self._process.returncode

        def communicate(self):
            result = self._process.communicate()
            hidden_attempt = Path(self._command[self._command.index("--attempt") + 1])
            value = json.loads(hidden_attempt.read_text(encoding="utf-8"))
            value["payload"]["invented"] = True
            hidden_attempt.write_text(json.dumps(value), encoding="utf-8")
            return result

    monkeypatch.setattr(runner_module.subprocess, "Popen", TamperingPopen)
    with pytest.raises(ExactTabularCandidateExecutionFailed) as caught:
        run_exact_tabular_candidate_fresh_process(
            request_path,
            receipt_path=receipt_path,
            attempt_path=attempt_path,
        )

    assert caught.value.attempt["payload"]["failure_phase"] == ("driver_child_evidence")
    assert not receipt_path.exists()
    assert attempt_path.is_file()
    require_exact_tabular_execution_attempt(
        caught.value.attempt,
        expected_request_payload_digest=str(request["payload_digest"]),
    )
