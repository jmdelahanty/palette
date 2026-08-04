from __future__ import annotations

import copy
from pathlib import Path
import shutil

import pytest
import zarr

from fisheye.analysis import tail_posture_view_runs as tail_writer
from fisheye.analysis_workflows.analysis_candidate_invocation import (
    build_tail_posture_invocation,
)
from fisheye.analysis_workflows.materializers.tail_posture import (
    TAIL_POSTURE_EXECUTION_PHASE_ORDER,
    build_tail_posture_materialization_plan,
    materialize_tail_posture_candidate,
    snapshot_tail_posture_sources,
    tombstone_tail_posture_execution_candidate,
    write_local_tail_posture_candidate,
)
from fisheye.analysis_workflows.tail_posture_candidate_execution import (
    TAIL_POSTURE_EXECUTION_FAMILY_ID,
    build_tail_posture_coordinate_evidence,
    build_tail_posture_execution_suite,
    build_tail_posture_scientific_identity,
    compute_tail_posture_logical_hashes,
    require_tail_posture_execution_suite,
    require_tail_posture_invocation_parameters,
)
from fisheye.shared import tail_coordinate_publication
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1
from tests.unit.fisheye.test_subject_shape_coordinate_publication import (  # noqa: F401
    canonical_refined_template,
    canonical_subject_shape_profile_template,
)

SOURCE_RUN = "tail_posture_execution_source"
CANDIDATE_RUN = "tail_posture_execution_candidate"


def _patch_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tail_writer,
        "get_git_info",
        lambda repo_path=None: {  # noqa: ARG005
            "commit_hash": "d" * 40,
            "short_hash": "dddddddd",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "git@example.com:palette.git",
        },
    )
    monkeypatch.setattr(
        tail_writer,
        "get_environment_info",
        lambda **kwargs: {  # noqa: ARG005
            "environment": {"python": "3.11"},
            "platform": {
                "hostname": "tail-posture-execution-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.11.0",
                "machine": "x86_64",
            },
        },
    )


@pytest.fixture
def tail_posture_execution_archive(
    tmp_path: Path,
    canonical_subject_shape_profile_template: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    archive = tmp_path / ".palette_benchmarks" / "tail-posture-execution.zarr"
    archive.parent.mkdir(parents=True)
    shutil.copytree(canonical_subject_shape_profile_template, archive)
    _patch_provenance(monkeypatch)
    root = zarr.open_group(archive, mode="r+", use_consolidated=False)
    result = tail_writer.write_tail_posture_view_run_group(
        root,
        subject_shape_run="shape_profile_attack",
        run_name=SOURCE_RUN,
        stage_command="tail-posture execution source fixture",
    )
    assert result["status"] == "updated"
    zarr.consolidate_metadata(archive)
    return archive


def _source_facts(archive: Path) -> tuple[dict[str, object], object]:
    root = zarr.open_group(archive, mode="r", use_consolidated=False)
    source = root[f"analysis/tail_posture_view_runs/{SOURCE_RUN}"]
    publication = tail_coordinate_publication.load_tail_posture_coordinate_publication(
        root,
        f"analysis/tail_posture_view_runs/{SOURCE_RUN}",
    )
    return compute_tail_posture_logical_hashes(source), publication


def _invocation_parameters(source_publication: object) -> dict[str, object]:
    return {
        "source_schema_id": "analysis.tail_posture_view_runs",
        "source_schema_version": 3,
        "source_logical_schema_mode": "exact_tail_posture_v3_arrays_v1",
        "source_subject_shape_run": "shape_profile_attack",
        "source_tail_posture_manifest_sha256": (
            source_publication.manifest.record_sha256
        ),
        "source_subject_shape_manifest_sha256": (
            source_publication.source.manifest.record_sha256
        ),
        "source_tail_kinematics_run": None,
        "source_tail_kinematics_manifest_sha256": None,
        "view_family": "megabouts_compatible",
        "head_source": "head_endpoint_xy",
        "keypoint_count": 11,
        "execution_backend": "serial",
        "num_workers": 1,
        "source_staging_mode": "logical_array_snapshot_v1",
        "storage_profile_id": "published_http_v1",
        "copy_backend": "python",
        "keep_scratch": False,
        "check_capacity": False,
    }


def test_suite_and_invocation_parameter_grammar_are_closed(
    tail_posture_execution_archive: Path,
) -> None:
    root = zarr.open_group(
        tail_posture_execution_archive,
        mode="r",
        use_consolidated=False,
    )
    source = root[f"analysis/tail_posture_view_runs/{SOURCE_RUN}"]
    suite = build_tail_posture_execution_suite(source, repetitions=2)
    require_tail_posture_execution_suite(TAIL_POSTURE_EXECUTION_FAMILY_ID, suite)
    assert len(suite["payload"]["storage_plan_receipt"]["payload"]["arrays"]) == 10

    tampered = copy.deepcopy(suite)
    tampered["payload"]["storage_plan_receipt"]["payload"]["arrays"][0][
        "observed_facts"
    ]["dtype"] = "uint32"
    tampered["payload"]["storage_plan_receipt"]["payload_digest"] = (
        canonical_json_sha256(tampered["payload"]["storage_plan_receipt"]["payload"])
    )
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    with pytest.raises(ValueError):
        require_tail_posture_execution_suite(
            TAIL_POSTURE_EXECUTION_FAMILY_ID,
            tampered,
        )

    publication = tail_coordinate_publication.load_tail_posture_coordinate_publication(
        root,
        f"analysis/tail_posture_view_runs/{SOURCE_RUN}",
    )
    parameters = _invocation_parameters(publication)
    require_tail_posture_invocation_parameters(parameters)
    assert (
        build_tail_posture_invocation(**parameters)["payload"]["parameters"]
        == parameters
    )
    parameters["unexpected"] = True
    with pytest.raises(ValueError, match="field set"):
        require_tail_posture_invocation_parameters(parameters)


def test_materializer_rejects_non_node_local_scratch(
    tail_posture_execution_archive: Path,
) -> None:
    _source_hashes, publication = _source_facts(tail_posture_execution_archive)
    with pytest.raises(ValueError, match="recognized node-local"):
        build_tail_posture_materialization_plan(
            tail_posture_execution_archive,
            scratch_root=Path("/opt/palette-tail-posture-not-node-local"),
            source_run_name=SOURCE_RUN,
            run_name="invalid_scratch_candidate",
            subject_shape_run="shape_profile_attack",
            source_subject_shape_manifest_sha256=(
                publication.source.manifest.record_sha256
            ),
            source_tail_posture_manifest_sha256=(publication.manifest.record_sha256),
            source_tail_kinematics_run=None,
            source_tail_kinematics_manifest_sha256=None,
            view_family="megabouts_compatible",
            head_source="head_endpoint_xy",
            keypoint_count=11,
            storage_profile=PUBLISHED_HTTP_V1,
        )


def test_scientific_compute_uses_only_staged_subject_shape(
    tail_posture_execution_archive: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source_hashes, publication = _source_facts(tail_posture_execution_archive)
    plan = build_tail_posture_materialization_plan(
        tail_posture_execution_archive,
        scratch_root=tmp_path / "staged-source-proof",
        source_run_name=SOURCE_RUN,
        run_name="staged_source_candidate",
        subject_shape_run="shape_profile_attack",
        source_subject_shape_manifest_sha256=(
            publication.source.manifest.record_sha256
        ),
        source_tail_posture_manifest_sha256=publication.manifest.record_sha256,
        source_tail_kinematics_run=None,
        source_tail_kinematics_manifest_sha256=None,
        view_family="megabouts_compatible",
        head_source="head_endpoint_xy",
        keypoint_count=11,
        storage_profile=PUBLISHED_HTTP_V1,
    )
    snapshot = snapshot_tail_posture_sources(plan, check_capacity=False)
    assert Path(snapshot.staged_shape_group.store_path.store.root).resolve() == (
        plan.staged_source_zarr
    )
    original_prepare = tail_writer._prepare_run_group

    def require_staged_shape(*args, **kwargs):
        shape_group = kwargs["shape_group"]
        if Path(shape_group.store_path.store.root).resolve() != plan.staged_source_zarr:
            raise AssertionError("scientific compute reread live subject-shape state")
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(tail_writer, "_prepare_run_group", require_staged_shape)
    batch = tail_writer.compute_tail_posture_view_from_subject_shape_arrays(
        **snapshot.source_arrays,
        keypoint_count=plan.keypoint_count,
    )
    local, _receipt = write_local_tail_posture_candidate(
        plan,
        snapshot,
        batch=batch,
        storage_profile=PUBLISHED_HTTP_V1,
        execution_binding={"request_payload_digest": "b" * 64},
        stage_command="staged source proof",
    )
    assert local.attrs["source_refined_subject_masks_run"] == (
        snapshot.staged_shape_group.attrs["source_refined_subject_masks_run"]
    )


def test_atomic_candidate_recomputes_exact_arrays_without_selector_mutation(
    tail_posture_execution_archive: Path,
    tmp_path: Path,
) -> None:
    source_hashes, source_publication = _source_facts(tail_posture_execution_archive)
    root = zarr.open_group(
        tail_posture_execution_archive,
        mode="r",
        use_consolidated=False,
    )
    parent = root["analysis/tail_posture_view_runs"]
    selector_before = copy.deepcopy(dict(parent.attrs))
    binding = {
        "schema_id": "palette.test_tail_posture_execution_binding",
        "request_payload_digest": "a" * 64,
    }

    def accept(published_root, _parent, candidate):
        candidate_publication = tail_coordinate_publication._load_tail_coordinate_publication(  # noqa: SLF001
            published_root,
            f"analysis/tail_posture_view_runs/{CANDIDATE_RUN}",
            expected_selector_eligible=False,
            expected_kind="tail_posture_view",
            require_complete=True,
        )
        return {
            "coordinate_evidence": build_tail_posture_coordinate_evidence(
                source_publication=source_publication,
                candidate_publication=candidate_publication,
                source_tail_kinematics_manifest_sha256=None,
            )
        }

    result = materialize_tail_posture_candidate(
        tail_posture_execution_archive,
        scratch_root=tmp_path / "tail-posture-scratch",
        source_run_name=SOURCE_RUN,
        run_name=CANDIDATE_RUN,
        subject_shape_run="shape_profile_attack",
        source_subject_shape_manifest_sha256=(
            source_publication.source.manifest.record_sha256
        ),
        source_tail_posture_manifest_sha256=(source_publication.manifest.record_sha256),
        source_tail_kinematics_run=None,
        source_tail_kinematics_manifest_sha256=None,
        view_family="megabouts_compatible",
        head_source="head_endpoint_xy",
        keypoint_count=11,
        storage_profile=PUBLISHED_HTTP_V1,
        copy_backend="python",
        keep_scratch=False,
        check_capacity=False,
        execution_binding=binding,
        expected_source_logical_hashes=source_hashes,
        publication_acceptance_validator=accept,
        stage_command="tail-posture typed candidate fixture",
    )

    assert result["status"] == "complete"
    assert (
        result["source_logical_manifest_sha256"]
        == result["published_logical_manifest_sha256"]
    )
    assert result["local_direct_consolidated_array_count"] == 10
    assert result["published_direct_consolidated_array_count"] == 10
    assert [phase["name"] for phase in result["runtime_telemetry"]["phases"]] == list(
        TAIL_POSTURE_EXECUTION_PHASE_ORDER
    )
    assert not (tmp_path / "tail-posture-scratch").exists()

    fresh = zarr.open_group(
        tail_posture_execution_archive,
        mode="r",
        use_consolidated=False,
    )
    candidate = fresh[f"analysis/tail_posture_view_runs/{CANDIDATE_RUN}"]
    assert candidate.attrs["stage_selector_eligible"] is False
    assert candidate.attrs["storage_candidate_profile_promoted"] is False
    assert dict(fresh["analysis/tail_posture_view_runs"].attrs) == selector_before
    assert compute_tail_posture_logical_hashes(candidate) == source_hashes
    assert build_tail_posture_scientific_identity(candidate) == (
        build_tail_posture_scientific_identity(
            fresh[f"analysis/tail_posture_view_runs/{SOURCE_RUN}"]
        )
    )

    tombstone = tombstone_tail_posture_execution_candidate(
        tail_posture_execution_archive,
        run_name=CANDIDATE_RUN,
        expected_execution_binding=binding,
        failure_phase="test_postpublication_failure",
        error_type="RuntimeError",
        error_message="fixture failure",
    )
    assert tombstone["tombstoned"] is True
    failed = zarr.open_group(
        tail_posture_execution_archive,
        mode="r",
        use_consolidated=False,
    )[f"analysis/tail_posture_view_runs/{CANDIDATE_RUN}"]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False
    assert (
        dict(
            zarr.open_group(
                tail_posture_execution_archive,
                mode="r",
                use_consolidated=False,
            )["analysis/tail_posture_view_runs"].attrs
        )
        == selector_before
    )
