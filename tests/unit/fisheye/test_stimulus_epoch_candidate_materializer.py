from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest
import zarr

from fisheye.analysis.exact_tabular_storage import (
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
)
from fisheye.analysis.stimulus_epoch_schema import (
    STIMULUS_EPOCH_RUN_MANIFEST_ATTR,
    STIMULUS_EPOCH_RUN_SCHEMA_ID,
    validate_stimulus_epoch_array_manifest,
    validate_stimulus_epoch_candidate_lineage,
    validate_stimulus_epoch_run_manifest,
)
from fisheye.analysis_workflows.materializers import stimulus_epochs as materializer
from fisheye.analysis_workflows.materializers.stimulus_epochs import (
    build_stimulus_epoch_candidate_plan,
    materialize_stimulus_epoch_candidate,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .test_stimulus_epoch_schema import create_legacy_stimulus_epoch_archive


def test_candidate_dry_run_is_read_only(tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    root = create_legacy_stimulus_epoch_archive(archive)

    result = materialize_stimulus_epoch_candidate(
        archive,
        source_run="source",
        run_name="candidate",
        scratch_root=tmp_path / "scratch",
        apply=False,
    )

    assert result["status"] == "planned"
    assert result["mutates_archive"] is False
    assert "candidate" not in root["analysis/stimulus_epoch_runs"]
    assert not (tmp_path / "scratch").exists()


def test_candidate_is_exact_byte_planned_and_selector_ineligible(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    create_legacy_stimulus_epoch_archive(archive)

    result = materialize_stimulus_epoch_candidate(
        archive,
        source_run="source",
        run_name="candidate",
        scratch_root=tmp_path / "scratch",
        copy_backend="python",
        apply=True,
    )

    assert result["status"] == "complete"
    assert result["local_validation"]["array_count"] == 12
    assert result["local_direct_consolidated_array_count"] == 12
    assert result["archive_direct_consolidated_array_count"] == 12
    assert result["local_validation"]["logical_hashes"] == result["publication"][
        "source_logical_hashes"
    ]
    assert not (tmp_path / "scratch").exists()

    direct = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    consolidated = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    parent = direct["analysis/stimulus_epoch_runs"]
    candidate = parent["candidate"]
    assert parent.attrs["latest"] == "source"
    assert parent.attrs["latest_complete"] == "source"
    assert candidate.attrs["schema_id"] == STIMULUS_EPOCH_RUN_SCHEMA_ID
    assert candidate.attrs["stage_selector_eligible"] is False
    assert candidate.attrs["storage_candidate_profile_promoted"] is False
    assert candidate.attrs["palette_run_completion_status"] == "complete"
    assert validate_stimulus_epoch_array_manifest(
        candidate, byte_planner_adopted=True
    ) == ()
    assert validate_stimulus_epoch_candidate_lineage(candidate) == ()
    assert validate_stimulus_epoch_run_manifest(candidate) == ()
    source = parent["source"]
    source_lineage = json.loads(source.attrs["lineage_payload_json"])
    candidate_lineage = json.loads(candidate.attrs["lineage_payload_json"])
    assert candidate.attrs["lineage_hash"] != source.attrs["lineage_hash"]
    assert source_lineage["analysis_schema"]["schema_id"].endswith(".v1")
    assert candidate_lineage["analysis_schema"] == {
        "layout": "exact_columnar_v1",
        "row_axis": "epoch_windows",
        "schema_id": "palette.stimulus_epoch_windows.v2",
        "schema_version": 2,
    }
    assert candidate_lineage["source_refs"]["source_stimulus_epoch_run"] == "source"
    assert candidate_lineage["source_fingerprints"][
        "source_stimulus_epoch_lineage_hash"
    ] == source.attrs["lineage_hash"]
    receipt = candidate.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR]
    assert receipt["payload"]["storage_profile"]["profile_id"] == "published_http_v1"
    assert receipt["payload"]["storage_profile"]["codec_profile_id"] == "zstd_fast_v1"
    assert receipt["payload"]["object_estimate"]["payload_objects"] == 12
    assert len(receipt["payload"]["arrays"]) == 12
    assert all(array.metadata.zarr_format == 3 for _path, array in _walk_arrays(candidate))
    consolidated_candidate = consolidated["analysis/stimulus_epoch_runs/candidate"]
    assert dict(consolidated_candidate.attrs) == dict(candidate.attrs)


def _publish_candidate(tmp_path: Path, *, suffix: str) -> tuple[Path, zarr.Group]:
    archive = tmp_path / f"{suffix}_analysis.zarr"
    create_legacy_stimulus_epoch_archive(archive)
    result = materialize_stimulus_epoch_candidate(
        archive,
        source_run="source",
        run_name="candidate",
        scratch_root=tmp_path / f"scratch-{suffix}",
        copy_backend="python",
        apply=True,
    )
    assert result["status"] == "complete"
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    return archive, root["analysis/stimulus_epoch_runs/candidate"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("dimensions", "fps"), 99.0),
        (("source_stimulus", "fingerprint"), "f" * 64),
        (("source_epoch", "lineage_hash"), "e" * 64),
        (("protocol", "profile", "profile_id"), "tampered_profile"),
        (("candidate_lineage", "lineage_hash"), "d" * 64),
    ],
)
def test_rehashed_deep_run_manifest_tampering_fails_executable_binding(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    _archive, candidate = _publish_candidate(tmp_path, suffix=path[-1])
    manifest = copy.deepcopy(candidate.attrs[STIMULUS_EPOCH_RUN_MANIFEST_ATTR])
    target = manifest["payload"]
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    manifest["payload_digest"] = canonical_json_sha256(manifest["payload"])
    candidate.attrs[STIMULUS_EPOCH_RUN_MANIFEST_ATTR] = manifest

    errors = validate_stimulus_epoch_run_manifest(candidate)
    assert errors
    assert any("executable binding" in error for error in errors)


@pytest.mark.parametrize(
    ("attr_name", "value", "manifest_path"),
    [
        (
            "recording_id",
            "other_recording",
            ("run_identity", "recording_id"),
        ),
        (
            "source_stimulus_fingerprint",
            "f" * 64,
            ("source_stimulus", "fingerprint"),
        ),
    ],
)
def test_rehashed_attr_and_manifest_tampering_is_still_rejected_by_lineage(
    tmp_path: Path,
    attr_name: str,
    value: object,
    manifest_path: tuple[str, ...],
) -> None:
    _archive, candidate = _publish_candidate(tmp_path, suffix=attr_name)
    candidate.attrs[attr_name] = value
    manifest = copy.deepcopy(candidate.attrs[STIMULUS_EPOCH_RUN_MANIFEST_ATTR])
    target = manifest["payload"]
    for component in manifest_path[:-1]:
        target = target[component]
    target[manifest_path[-1]] = value
    manifest["payload_digest"] = canonical_json_sha256(manifest["payload"])
    candidate.attrs[STIMULUS_EPOCH_RUN_MANIFEST_ATTR] = manifest

    assert validate_stimulus_epoch_candidate_lineage(candidate)
    assert validate_stimulus_epoch_run_manifest(candidate)


def test_rehashed_profile_promotion_true_fails_exact_false_gate(tmp_path: Path) -> None:
    _archive, candidate = _publish_candidate(tmp_path, suffix="promoted")
    candidate.attrs["storage_candidate_profile_promoted"] = True
    manifest = copy.deepcopy(candidate.attrs[STIMULUS_EPOCH_RUN_MANIFEST_ATTR])
    manifest["payload"]["publication_state"][
        "storage_candidate_profile_promoted"
    ] = True
    manifest["payload_digest"] = canonical_json_sha256(manifest["payload"])
    candidate.attrs[STIMULUS_EPOCH_RUN_MANIFEST_ATTR] = manifest

    errors = materializer._validate_candidate_group(
        candidate,
        expected_hashes=materializer._logical_hashes(
            candidate,
            materializer.build_stimulus_epoch_array_declarations(
                candidate,
                byte_planner_adopted=True,
            ),
        ),
    )["errors"]
    assert any("exact false" in error for error in errors)


def test_direct_consolidated_check_includes_windows_group_declaration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, candidate = _publish_candidate(tmp_path, suffix="group-tree")
    candidate["windows"].attrs["field_names"] = ["window_id"]
    monkeypatch.setattr(
        materializer,
        "consolidate_metadata_capture_expected_warnings",
        lambda _path: None,
    )
    declarations = materializer.build_stimulus_epoch_array_declarations(
        candidate,
        byte_planner_adopted=True,
    )

    with pytest.raises(ValueError, match="declaration trees differ"):
        materializer._direct_consolidated_check(
            archive,
            run_path="analysis/stimulus_epoch_runs/candidate",
            declarations=declarations,
        )


def _walk_arrays(group: zarr.Group, prefix: str = ""):
    for name, array in group.arrays():
        yield (f"{prefix}/{name}" if prefix else name), array
    for name, child in group.groups():
        child_prefix = f"{prefix}/{name}" if prefix else name
        yield from _walk_arrays(child, child_prefix)


@pytest.mark.parametrize("name", ["latest", "latest_complete", "../candidate", "bad name"])
def test_plan_refuses_alias_or_unsafe_names(tmp_path: Path, name: str) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    create_legacy_stimulus_epoch_archive(archive)

    with pytest.raises(ValueError):
        build_stimulus_epoch_candidate_plan(
            archive,
            source_run="source",
            run_name=name,
            scratch_root=tmp_path / "scratch",
        )


def test_plan_requires_disjoint_scratch_and_rejects_source_symlink(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    create_legacy_stimulus_epoch_archive(archive)

    with pytest.raises(ValueError, match="disjoint"):
        build_stimulus_epoch_candidate_plan(
            archive,
            source_run="source",
            run_name="candidate",
            scratch_root=archive / "scratch",
        )
    with pytest.raises(ValueError, match="disjoint"):
        build_stimulus_epoch_candidate_plan(
            archive,
            source_run="source",
            run_name="candidate",
            scratch_root=tmp_path,
        )

    source_path = archive / "analysis" / "stimulus_epoch_runs" / "source"
    external = tmp_path / "external-source"
    shutil.move(source_path, external)
    source_path.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        build_stimulus_epoch_candidate_plan(
            archive,
            source_run="source",
            run_name="candidate",
            scratch_root=tmp_path / "separate-scratch",
        )


def test_post_consolidation_failure_repairs_failed_consolidated_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    create_legacy_stimulus_epoch_archive(archive)
    original = materializer._direct_consolidated_check
    calls = 0

    def fail_after_archive_consolidation(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            raise RuntimeError("injected failure after authoritative consolidation")
        return result

    monkeypatch.setattr(
        materializer,
        "_direct_consolidated_check",
        fail_after_archive_consolidation,
    )

    with pytest.raises(RuntimeError, match="injected failure"):
        materialize_stimulus_epoch_candidate(
            archive,
            source_run="source",
            run_name="failed_candidate",
            scratch_root=tmp_path / "scratch-failure",
            copy_backend="python",
            apply=True,
        )

    direct = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    consolidated = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    direct_run = direct["analysis/stimulus_epoch_runs/failed_candidate"]
    consolidated_run = consolidated[
        "analysis/stimulus_epoch_runs/failed_candidate"
    ]
    assert direct_run.attrs["palette_run_completion_status"] == "failed"
    assert direct_run.attrs["stage_selector_eligible"] is False
    assert "atomic_publication_tombstone" in direct_run.attrs
    assert dict(consolidated_run.attrs) == dict(direct_run.attrs)
    parent = direct["analysis/stimulus_epoch_runs"]
    assert parent.attrs["latest"] == "source"
    assert parent.attrs["latest_complete"] == "source"


def test_candidate_refuses_incomplete_or_explicitly_ineligible_source(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    root = create_legacy_stimulus_epoch_archive(archive)
    source = root["analysis/stimulus_epoch_runs/source"]
    source.attrs["palette_run_completion_status"] = "running"
    with pytest.raises(ValueError, match="not explicitly complete"):
        build_stimulus_epoch_candidate_plan(
            archive,
            source_run="source",
            run_name="candidate",
            scratch_root=tmp_path / "scratch-1",
        )

    source.attrs["palette_run_completion_status"] = "complete"
    source.attrs["stage_selector_eligible"] = False
    with pytest.raises(ValueError, match="explicitly selector-ineligible"):
        build_stimulus_epoch_candidate_plan(
            archive,
            source_run="source",
            run_name="candidate",
            scratch_root=tmp_path / "scratch-2",
        )
