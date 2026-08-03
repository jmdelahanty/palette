from __future__ import annotations

import copy
from pathlib import Path
import shutil

import numpy as np
import pytest
import zarr

from fisheye.analysis.exact_tabular_storage import (
    ANALYSIS_STORAGE_PLAN_DIGEST_ATTR,
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
)
from fisheye.analysis.stimulus_epoch_consumer import (
    StimulusEpochCompatibilityPolicy,
    read_stimulus_epoch_snapshot,
)
from fisheye.analysis.stimulus_epoch_schema import (
    STIMULUS_EPOCH_RUN_MANIFEST_ATTR,
)
from fisheye.analysis_workflows.materializers.stimulus_epochs import (
    materialize_stimulus_epoch_candidate,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)

from .test_stimulus_epoch_schema import create_legacy_stimulus_epoch_archive


@pytest.fixture(scope="module")
def published_candidate(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("stimulus-epoch-consumer")
    archive = root / "source_analysis.zarr"
    create_legacy_stimulus_epoch_archive(archive)
    result = materialize_stimulus_epoch_candidate(
        archive,
        source_run="source",
        run_name="candidate",
        scratch_root=root / "scratch",
        copy_backend="python",
        apply=True,
    )
    assert result["status"] == "complete"
    return archive


def _copy_candidate(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


def _candidate_group(archive: Path) -> zarr.Group:
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    return root["analysis/stimulus_epoch_runs/candidate"]


def _reconsolidate(archive: Path) -> None:
    consolidate_metadata_capture_expected_warnings(archive)


def test_exact_consumer_validates_and_eagerly_reads_all_rows(
    published_candidate: Path,
) -> None:
    result = read_stimulus_epoch_snapshot(
        published_candidate,
        run_name="candidate",
    )

    assert result.run_path == "analysis/stimulus_epoch_runs/candidate"
    assert result.schema_id == "palette.stimulus_epoch_windows.v2"
    assert result.schema_version == 2
    assert not result.is_legacy_compatibility_read
    assert result.metadata_equivalence is not None
    assert result.metadata_equivalence.array_count == 12
    assert result.metadata_equivalence.group_count == 2
    assert [segment.segment_id for segment in result.segments] == [0, 1, 2]
    assert result.segments[1].source_start_event_frame == 10
    assert result.segments[1].source_end_event_frame == 20


def test_legacy_v1_requires_explicit_compatibility_policy(tmp_path: Path) -> None:
    archive = tmp_path / "legacy_analysis.zarr"
    create_legacy_stimulus_epoch_archive(archive)

    with pytest.raises(ValueError, match="ALLOW_EXPLICIT_V1"):
        read_stimulus_epoch_snapshot(archive, run_name="source")

    result = read_stimulus_epoch_snapshot(
        archive,
        run_name="source",
        compatibility_policy=StimulusEpochCompatibilityPolicy.ALLOW_EXPLICIT_V1,
    )
    assert result.is_legacy_compatibility_read
    assert result.metadata_equivalence is None
    assert len(result.segments) == 3


@pytest.mark.parametrize(
    "run_name", ["latest", "latest_complete", "../candidate", "bad name"]
)
def test_consumer_requires_one_explicit_safe_run_name(
    published_candidate: Path,
    run_name: str,
) -> None:
    with pytest.raises(ValueError, match="explicit immutable run name"):
        read_stimulus_epoch_snapshot(published_candidate, run_name=run_name)


@pytest.mark.parametrize(
    ("attr_name", "value", "expected"),
    [
        ("palette_run_completion_status", "running", "not complete"),
        ("palette_run_name", "other", "run-name binding"),
        ("stage_selector_eligible", True, "selector-ineligible"),
        ("storage_candidate_profile_promoted", True, "profile-promotion"),
    ],
)
def test_consumer_fails_closed_on_candidate_lifecycle_state(
    tmp_path: Path,
    published_candidate: Path,
    attr_name: str,
    value: object,
    expected: str,
) -> None:
    archive = _copy_candidate(
        published_candidate,
        tmp_path / f"{attr_name}_analysis.zarr",
    )
    _candidate_group(archive).attrs[attr_name] = value
    _reconsolidate(archive)

    with pytest.raises(ValueError, match=expected):
        read_stimulus_epoch_snapshot(archive, run_name="candidate")


def test_consumer_requires_persisted_direct_consolidated_equivalence(
    tmp_path: Path,
    published_candidate: Path,
) -> None:
    archive = _copy_candidate(
        published_candidate,
        tmp_path / "stale_metadata_analysis.zarr",
    )
    _candidate_group(archive)["windows"].attrs["field_names"] = ["window_id"]

    with pytest.raises(RuntimeError, match="Direct/consolidated declaration differs"):
        read_stimulus_epoch_snapshot(archive, run_name="candidate")


def test_consumer_rejects_symlinked_explicit_run(
    tmp_path: Path,
    published_candidate: Path,
) -> None:
    archive = _copy_candidate(
        published_candidate,
        tmp_path / "symlink_analysis.zarr",
    )
    parent = archive / "analysis" / "stimulus_epoch_runs"
    (parent / "alias").symlink_to(parent / "candidate", target_is_directory=True)

    with pytest.raises(ValueError, match="cannot be a symlink"):
        read_stimulus_epoch_snapshot(archive, run_name="alias")


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing", "persisted metadata inventory is not exact"),
        ("unexpected", "persisted metadata inventory is not exact"),
        ("wrong_dtype", "dtype mismatch"),
        ("wrong_rank", "rank mismatch"),
    ],
)
def test_consumer_rejects_adversarial_array_declarations(
    tmp_path: Path,
    published_candidate: Path,
    mutation: str,
    expected: str,
) -> None:
    archive = _copy_candidate(
        published_candidate,
        tmp_path / f"{mutation}_analysis.zarr",
    )
    windows = _candidate_group(archive)["windows"]
    if mutation == "unexpected":
        windows.create_array("frame_counts", data=np.ones(3, dtype=np.int32))
    else:
        original = np.asarray(windows["start_frame"][:])
        del windows["start_frame"]
        if mutation == "wrong_dtype":
            windows.create_array("start_frame", data=original.astype(np.int32))
        elif mutation == "wrong_rank":
            windows.create_array("start_frame", data=original.reshape(-1, 1))
    _reconsolidate(archive)

    with pytest.raises(ValueError, match=expected):
        read_stimulus_epoch_snapshot(archive, run_name="candidate")


def test_consumer_rejects_rehashed_run_manifest_tampering(
    tmp_path: Path,
    published_candidate: Path,
) -> None:
    archive = _copy_candidate(
        published_candidate,
        tmp_path / "rehashed_manifest_analysis.zarr",
    )
    candidate = _candidate_group(archive)
    manifest = copy.deepcopy(candidate.attrs[STIMULUS_EPOCH_RUN_MANIFEST_ATTR])
    manifest["payload"]["dimensions"]["fps"] = 99.0
    manifest["payload_digest"] = canonical_json_sha256(manifest["payload"])
    candidate.attrs[STIMULUS_EPOCH_RUN_MANIFEST_ATTR] = manifest
    _reconsolidate(archive)

    with pytest.raises(ValueError, match="executable binding"):
        read_stimulus_epoch_snapshot(archive, run_name="candidate")


def test_consumer_rejects_storage_receipt_digest_rebinding(
    tmp_path: Path,
    published_candidate: Path,
) -> None:
    archive = _copy_candidate(
        published_candidate,
        tmp_path / "receipt_analysis.zarr",
    )
    candidate = _candidate_group(archive)
    receipt = copy.deepcopy(candidate.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR])
    receipt["payload_digest"] = "f" * 64
    candidate.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR] = receipt
    candidate.attrs[ANALYSIS_STORAGE_PLAN_DIGEST_ATTR] = "f" * 64
    _reconsolidate(archive)

    with pytest.raises(ValueError, match="storage"):
        read_stimulus_epoch_snapshot(archive, run_name="candidate")


def test_invalid_explicit_v2_is_terminal_even_when_legacy_is_allowed(
    tmp_path: Path,
    published_candidate: Path,
) -> None:
    archive = _copy_candidate(
        published_candidate,
        tmp_path / "no_fallback_analysis.zarr",
    )
    candidate = _candidate_group(archive)
    candidate.attrs["palette_run_completion_status"] = "failed"
    _reconsolidate(archive)

    with pytest.raises(ValueError, match="not complete"):
        read_stimulus_epoch_snapshot(
            archive,
            run_name="candidate",
            compatibility_policy=StimulusEpochCompatibilityPolicy.ALLOW_EXPLICIT_V1,
        )
