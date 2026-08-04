from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import uuid

import pytest
import zarr

from fisheye.analysis.stimulus_response_storage import (
    STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID,
)
from fisheye.analysis_workflows.analysis_candidate_execution import (
    PhysicalIOScope,
    build_candidate_execution_request,
)
from fisheye.analysis_workflows.analysis_candidate_execution_catalog import (
    ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE,
)
from fisheye.analysis_workflows.materializers import stimulus_response as materializer
from fisheye.analysis_workflows.stimulus_response_candidate_execution import (
    STIMULUS_RESPONSE_SOURCE_STAGING_MODE,
    build_stimulus_response_coordinate_evidence,
    build_stimulus_response_execution_suite,
    build_stimulus_response_source_identity,
    compute_stimulus_response_logical_hashes,
    require_stimulus_response_candidate_scientific_binding,
    require_stimulus_response_execution_suite,
    require_stimulus_response_invocation_parameters,
    stimulus_response_writer_arguments,
)
from fisheye.diagnostics.stimulus_response_candidate_execution import (
    StimulusResponseCandidateExecutionFailed,
    require_stimulus_response_execution_attempt,
    run_stimulus_response_candidate_fresh_process,
    stimulus_response_selector_snapshot_sha256,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.stimulus_response_schema import (
    STIMULUS_RESPONSE_LAYOUT,
)
from tests.unit.fisheye.test_stimulus_response_materializer import _fake_writer

SOURCE_PATH = "analysis/stimulus_response_runs/source"
CANDIDATE_PATH = "analysis/stimulus_response_runs/typed_candidate"
TRACK_PATH = "analysis/track_kinematics_runs/offline/track_1"
STIMULUS_PATH = "analysis/stimulus_runs/stim_1"
TRACK_DIGEST = "a" * 64
STIMULUS_RESPONSE_EXECUTION_PHASE_ORDER = (
    materializer.STIMULUS_RESPONSE_EXECUTION_PHASE_ORDER
)


def _scientific_parameters() -> dict[str, object]:
    return {
        "moving_threshold_mm_s": 2.0,
        "camera_to_projector_offset_deg": 0.0,
        "bin_size_s": 1.0,
        "follow_threshold": 0.5,
        "follow_window_s": 1.0,
        "omr_enabled": False,
        "omr_projection_deadzone": 0.0,
        "omr_projection_speed_deadzone_mm_s": 0.0,
        "omr_window_s": [10.0, 30.0, 60.0],
        "omr_early_window_s": [5.0, 10.0],
        "center_threshold_mm": 2.0,
        "concentric_radial_singularity_epsilon_mm": 0.5,
        "escape_speed_threshold_mm_s": 30.0,
        "escape_window_s": 5.0,
        "loom_pre_onset_s": 1.0,
        "loom_post_onset_s": 5.0,
        "loom_bin_size_s": 0.1,
    }


def _source_parameters() -> dict[str, object]:
    return {
        "layout": STIMULUS_RESPONSE_LAYOUT,
        "storage_profile_id": None,
        "fps": 30.0,
        "n_frames": 10,
        "omr_method_version": None,
        "concentric_radial_omr_method_version": None,
        **_scientific_parameters(),
    }


def _build_source(path: Path) -> zarr.Group:
    _fake_writer(
        [
            "unused-source.zarr",
            "--output-zarr-path",
            str(path),
            "--run-name",
            "source",
            "--layout",
            STIMULUS_RESPONSE_LAYOUT,
        ]
    )
    root = zarr.open_group(
        str(path),
        mode="a",
        zarr_format=3,
        use_consolidated=False,
    )
    track = (
        root.require_group("analysis")
        .require_group("track_kinematics_runs")
        .require_group("offline")
        .create_group("track_1")
    )
    track.attrs["track_motion_publication_manifest_sha256"] = TRACK_DIGEST
    stimulus = root["analysis"].require_group("stimulus_runs").create_group("stim_1")
    stimulus.attrs.update({"schema_id": "test.stimulus", "recording_id": "fixture"})

    run = root[SOURCE_PATH]
    refs = dict(run.attrs["source_refs"])
    refs["upstream_lineage"] = {
        "source_track_motion_run_ref": f"/{TRACK_PATH}",
        "source_track_motion_manifest_sha256": TRACK_DIGEST,
    }
    run.attrs.update(
        {
            "stage_selector_eligible": True,
            "parameters": _source_parameters(),
            "source_refs": refs,
            "source_track_kinematics_type": "offline",
            "source_track_kinematics_run": "track_1",
            "source_stimulus_run": "stim_1",
        }
    )
    parent = root["analysis/stimulus_response_runs"]
    parent.attrs.update({"latest": "source", "latest_complete": "source"})
    return root


def _invocation_parameters(root: zarr.Group) -> dict[str, object]:
    identity = build_stimulus_response_source_identity(
        root,
        source_run_path=SOURCE_PATH,
    )
    authorities = identity["source_authorities"]
    return {
        "source_track_kinematics_scope": "offline",
        "source_track_kinematics_run": "track_1",
        "source_track_motion_manifest_sha256": authorities[
            "track_motion_manifest_sha256"
        ],
        "source_stimulus_run": "stim_1",
        "source_stimulus_logical_tree_sha256": authorities[
            "stimulus_logical_tree_sha256"
        ],
        "source_stimulus_coordinate_lineage_sha256": authorities[
            "stimulus_coordinate_lineage_sha256"
        ],
        "source_bout_mode": "disabled",
        "source_swim_bout_run": None,
        "source_swim_bout_logical_tree_sha256": None,
        "scientific_parameters": _scientific_parameters(),
        "execution_backend": "dask_threads_per_step_v1",
        "source_staging_mode": STIMULUS_RESPONSE_SOURCE_STAGING_MODE,
        "storage_profile_id": STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID,
        "copy_backend": "python",
        "keep_scratch": False,
        "check_capacity": False,
    }


def _execution_writer(source: zarr.Group):
    source_refs = dict(source[SOURCE_PATH].attrs["source_refs"])

    def write(argv) -> None:
        _fake_writer(argv)
        output = Path(argv[argv.index("--output-zarr-path") + 1])
        run_name = argv[argv.index("--run-name") + 1]
        root = zarr.open_group(
            str(output),
            mode="a",
            zarr_format=3,
            use_consolidated=False,
        )
        run = root[f"analysis/stimulus_response_runs/{run_name}"]
        parameters = _source_parameters()
        parameters["storage_profile_id"] = STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID
        run.attrs.update(
            {
                "parameters": parameters,
                "source_refs": source_refs,
                "source_track_kinematics_type": "offline",
                "source_track_kinematics_run": "track_1",
                "source_stimulus_run": "stim_1",
            }
        )

    return write


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_invocation_is_closed_and_translates_every_scientific_knob(
    tmp_path: Path,
) -> None:
    root = _build_source(tmp_path / "source.zarr")
    parameters = _invocation_parameters(root)

    assert require_stimulus_response_invocation_parameters(parameters) == parameters
    arguments = stimulus_response_writer_arguments(parameters)
    assert arguments[:4] == (
        "--track-kinematics-type",
        "offline",
        "--track-kinematics-run",
        "track_1",
    )
    assert "--no-bouts" in arguments
    assert "--no-omr" in arguments
    assert arguments.count("--omr-window-s") == 3
    assert arguments.count("--omr-early-window-s") == 2

    changed = deepcopy(parameters)
    changed["unbound_knob"] = True
    with pytest.raises(ValueError, match="field set"):
        require_stimulus_response_invocation_parameters(changed)


def test_suite_replays_exact_byte_planned_v3_declarations_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    root = _build_source(tmp_path / "suite.zarr")
    suite = build_stimulus_response_execution_suite(
        root[SOURCE_PATH],
        repetitions=2,
    )

    require_stimulus_response_execution_suite("stimulus_response", suite)
    assert suite["payload"]["repetitions"] == 2

    changed = deepcopy(suite)
    receipt = changed["payload"]["storage_plan_receipt"]
    receipt["payload"]["arrays"][0]["declaration"]["fill_semantics"] = "tampered"
    receipt["payload_digest"] = canonical_json_sha256(receipt["payload"])
    changed["payload_digest"] = canonical_json_sha256(changed["payload"])
    with pytest.raises(ValueError, match="declaration|digest|storage plan"):
        require_stimulus_response_execution_suite("stimulus_response", changed)


def test_source_identity_binds_live_track_stimulus_and_coordinate_authorities(
    tmp_path: Path,
) -> None:
    root = _build_source(tmp_path / "identity.zarr")
    identity = build_stimulus_response_source_identity(
        root,
        source_run_path=SOURCE_PATH,
    )

    assert identity["compatibility_role"].endswith("_benchmark_authority_nonproduction")
    assert identity["source_authorities"]["track_motion_manifest_sha256"] == (
        TRACK_DIGEST
    )
    assert identity["source_authorities"]["stimulus_run_path"] == STIMULUS_PATH
    evidence = build_stimulus_response_coordinate_evidence(identity)
    roles = [record["role"] for record in evidence["source_authority_digests"]]
    assert roles == sorted(roles)
    root[STIMULUS_PATH].attrs["recording_id"] = "tampered"
    changed = build_stimulus_response_source_identity(
        root,
        source_run_path=SOURCE_PATH,
    )
    assert canonical_json_sha256(changed) != canonical_json_sha256(identity)


def test_execution_materializer_stages_publishes_validates_and_tombstones(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zarr"
    root = _build_source(archive)
    identity = build_stimulus_response_source_identity(
        root,
        source_run_path=SOURCE_PATH,
    )
    parameters = _invocation_parameters(root)
    source_hashes = compute_stimulus_response_logical_hashes(root[SOURCE_PATH])
    monkeypatch.setattr(
        materializer.response_writer,
        "main",
        _execution_writer(root),
    )
    binding = {
        "schema_id": "palette.analysis_candidate_execution_binding",
        "schema_version": 1,
        "execution_id": "stimulus-response-test",
    }

    with pytest.raises(ValueError, match="outside the source archive"):
        materializer.materialize_stimulus_response_execution_candidate(
            archive,
            source_run=SOURCE_PATH,
            run_name="unsafe_candidate",
            scratch_root=archive / "recursive-scratch",
            writer_arguments=stimulus_response_writer_arguments(parameters),
            copy_backend="python",
            keep_scratch=False,
            check_capacity=False,
            execution_binding=binding,
            expected_source_logical_hashes=source_hashes,
            expected_source_identity_sha256=canonical_json_sha256(identity),
        )

    with pytest.raises(ValueError, match="recognized node-local"):
        materializer.materialize_stimulus_response_execution_candidate(
            archive,
            source_run=SOURCE_PATH,
            run_name="nonlocal_candidate",
            scratch_root=Path("/groups/palette-stimulus-response-nonlocal"),
            writer_arguments=stimulus_response_writer_arguments(parameters),
            copy_backend="python",
            keep_scratch=False,
            check_capacity=False,
            execution_binding=binding,
            expected_source_logical_hashes=source_hashes,
            expected_source_identity_sha256=canonical_json_sha256(identity),
        )

    linked_archive = tmp_path / "linked-archive.zarr"
    linked_archive.symlink_to(archive, target_is_directory=True)
    with pytest.raises(ValueError, match="must not be a symlink"):
        materializer.materialize_stimulus_response_execution_candidate(
            linked_archive,
            source_run=SOURCE_PATH,
            run_name="linked_archive_candidate",
            scratch_root=tmp_path / "linked-archive-scratch",
            writer_arguments=stimulus_response_writer_arguments(parameters),
            copy_backend="python",
            keep_scratch=False,
            check_capacity=False,
            execution_binding=binding,
            expected_source_logical_hashes=source_hashes,
            expected_source_identity_sha256=canonical_json_sha256(identity),
        )

    external_payload = tmp_path / "network-like-payload"
    external_payload.write_bytes(b"must-not-be-followed")
    source_link = archive / "network-backed-symlink"
    source_link.symlink_to(external_payload)
    with pytest.raises(ValueError, match="self-contained and symlink-free"):
        materializer.materialize_stimulus_response_execution_candidate(
            archive,
            source_run=SOURCE_PATH,
            run_name="symlink_candidate",
            scratch_root=tmp_path / "symlink-scratch",
            writer_arguments=stimulus_response_writer_arguments(parameters),
            copy_backend="python",
            keep_scratch=False,
            check_capacity=False,
            execution_binding=binding,
            expected_source_logical_hashes=source_hashes,
            expected_source_identity_sha256=canonical_json_sha256(identity),
        )
    assert not (tmp_path / "symlink-scratch" / "staged-source.zarr").exists()
    assert not (
        archive / "analysis" / "stimulus_response_runs" / "symlink_candidate"
    ).exists()
    source_link.unlink()

    result = materializer.materialize_stimulus_response_execution_candidate(
        archive,
        source_run=SOURCE_PATH,
        run_name="typed_candidate",
        scratch_root=tmp_path / "scratch",
        writer_arguments=stimulus_response_writer_arguments(parameters),
        copy_backend="python",
        keep_scratch=False,
        check_capacity=False,
        execution_binding=binding,
        expected_source_logical_hashes=source_hashes,
        expected_source_identity_sha256=canonical_json_sha256(identity),
        publication_acceptance_validator=lambda _root, _parent, candidate: (
            require_stimulus_response_candidate_scientific_binding(
                candidate,
                source_identity=identity,
            )
            or {"scientific_binding": "exact"}
        ),
    )

    assert result["status"] == "complete"
    assert [phase["name"] for phase in result["runtime_telemetry"]["phases"]] == list(
        STIMULUS_RESPONSE_EXECUTION_PHASE_ORDER
    )
    assert result["source_logical_manifest_sha256"] == (
        result["published_logical_manifest_sha256"]
    )
    assert result["caller_acceptance"] == {"scientific_binding": "exact"}
    assert not (tmp_path / "scratch").exists()

    published = zarr.open_group(
        str(archive),
        mode="r",
        zarr_format=3,
        use_consolidated=False,
    )
    assert published["analysis/stimulus_response_runs"].attrs["latest"] == "source"
    assert published[CANDIDATE_PATH].attrs["stage_selector_eligible"] is False
    tombstone = materializer.tombstone_stimulus_response_execution_candidate(
        archive,
        run_name="typed_candidate",
        expected_execution_binding=binding,
        failure_phase="post_publication_gate",
        error_type="RuntimeError",
        error_message="synthetic failure",
    )
    assert tombstone["tombstoned"] is True
    fresh = zarr.open_group(
        str(archive),
        mode="r",
        zarr_format=3,
        use_consolidated=False,
    )
    assert fresh[CANDIDATE_PATH].attrs["palette_run_completion_status"] == "failed"
    assert fresh["analysis/stimulus_response_runs"].attrs["latest"] == "source"


def test_real_fresh_process_rejects_wrong_bound_stimulus_identity(
    tmp_path: Path,
) -> None:
    adapter = ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE["stimulus_response"]
    if adapter.runner_status.value != "implemented":
        pytest.skip("shared stimulus-response adapter activation is a separate commit")

    benchmark = tmp_path / ".palette_benchmarks" / "stimulus-response"
    benchmark.mkdir(parents=True)
    archive = benchmark / "fixture.zarr"
    root = _build_source(archive)
    parameters = _invocation_parameters(root)
    parameters["source_stimulus_logical_tree_sha256"] = "f" * 64
    invocation_payload = {
        "contract_id": "stimulus_response_v1",
        "parameters": parameters,
    }
    invocation = {
        "schema_id": "palette.analysis_candidate_invocation",
        "schema_version": 1,
        "payload": invocation_payload,
        "payload_digest": canonical_json_sha256(invocation_payload),
    }
    probe = benchmark / "protected.json"
    probe.write_text("{}\n", encoding="utf-8")
    request = build_candidate_execution_request(
        execution_id=f"stimulus_response_{uuid.uuid4().hex}",
        adapter_manifest=adapter.as_manifest(),
        invocation=invocation,
        benchmark_suite=build_stimulus_response_execution_suite(
            root[SOURCE_PATH],
            repetitions=1,
        ),
        archive_path=archive,
        source_run_path=SOURCE_PATH,
        candidate_run_path=CANDIDATE_PATH,
        scratch_root=Path("/tmp") / f"palette-stimulus-{uuid.uuid4().hex}",
        source_identity_sha256=canonical_json_sha256(
            compute_stimulus_response_logical_hashes(root[SOURCE_PATH])
        ),
        palette_commit=_git_commit(),
        repetition_index=0,
        candidate_order_index=0,
        candidate_order_count=1,
        cache_state="fresh_process_os_cache_uncontrolled",
        physical_io_scope=PhysicalIOScope.UNAVAILABLE,
        selector_before_sha256=stimulus_response_selector_snapshot_sha256(archive),
        registry_probe_path=probe,
        production_profiles_probe_path=probe,
    )
    workflow = benchmark / "fresh-process"
    workflow.mkdir()
    request_path = workflow / "request.json"
    receipt_path = workflow / "receipt.json"
    attempt_path = workflow / "attempt.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(
        StimulusResponseCandidateExecutionFailed,
        match="stimulus-response invocation",
    ) as raised:
        run_stimulus_response_candidate_fresh_process(
            request_path,
            receipt_path=receipt_path,
            attempt_path=attempt_path,
        )
    require_stimulus_response_execution_attempt(
        raised.value.attempt,
        expected_request_payload_digest=request["payload_digest"],
    )
    assert raised.value.attempt["payload"]["failure_phase"] == "runner_preflight"
    assert attempt_path.is_file()
    assert not receipt_path.exists()
    assert os.getpid() != raised.value.attempt["payload"]["child_pid"]
