from __future__ import annotations

from pathlib import Path
import shutil

import pytest
import zarr

from fisheye.analysis.exact_tabular_storage import (
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
)
from fisheye.analysis.stimulus_epoch_schema import (
    STIMULUS_EPOCH_RUN_SCHEMA_ID,
    validate_stimulus_epoch_array_manifest,
)
from fisheye.analysis_workflows.materializers import stimulus_epochs as materializer
from fisheye.analysis_workflows.materializers.stimulus_epochs import (
    build_stimulus_epoch_candidate_plan,
    materialize_stimulus_epoch_candidate,
)

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
    receipt = candidate.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR]
    assert receipt["payload"]["storage_profile"]["profile_id"] == "published_http_v1"
    assert receipt["payload"]["storage_profile"]["codec_profile_id"] == "zstd_fast_v1"
    assert receipt["payload"]["object_estimate"]["payload_objects"] == 12
    assert len(receipt["payload"]["arrays"]) == 12
    assert all(array.metadata.zarr_format == 3 for _path, array in _walk_arrays(candidate))
    consolidated_candidate = consolidated["analysis/stimulus_epoch_runs/candidate"]
    assert dict(consolidated_candidate.attrs) == dict(candidate.attrs)


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
