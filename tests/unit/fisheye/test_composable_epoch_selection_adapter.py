from __future__ import annotations

from pathlib import Path
import shutil

import pytest
import zarr

from fisheye.analysis_workflows.composable_epoch_selection_adapter import (
    ComposableEpochSelectionAdapterError,
    EpochRoleBinding,
    TimelineAuthorityEvidence,
    compile_goodbatbadbat_selections,
)
from fisheye.analysis_workflows.resolved_epoch_selection import (
    resolve_exact_stimulus_epoch_selection,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from tests.unit.fisheye.test_provider_recording_timing_authority import (
    _install_clock_authority,
)

pytest_plugins = ("tests.unit.fisheye.test_stimulus_epoch_consumer",)


def _sealed_selection(
    tmp_path: Path,
    published_candidate: Path,
):
    archive = tmp_path / "goodbatbadbat_exact_selection.zarr"
    shutil.copytree(published_candidate, archive)
    _install_clock_authority(
        archive,
        tmp_path,
        frame_count=30,
        fps=10.0,
        recording_id="recording_1",
    )
    return archive, resolve_exact_stimulus_epoch_selection(
        archive,
        run_name="candidate",
    )


def _evidence(archive: Path, selection) -> TimelineAuthorityEvidence:
    root = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    video = dict(root.attrs["source_video_metadata"])
    clock = dict(selection.selection_record["recording_timing_authority"]["record"])
    source_metadata = {
        "recording_id": "recording_1",
        "source_timeline_digest": selection.source_timeline_digest,
        "source_epoch_run_path": selection.run_path,
        "source_epoch_run_manifest_sha256": selection.run_manifest_digest,
        "source_epoch_run_manifest_payload_sha256": (
            selection.run_manifest_payload_digest
        ),
        "source_epoch_logical_content_sha256": (
            selection.source_epoch_logical_content_digest
        ),
        "source_epoch_lineage_hash": selection.source_epoch_lineage_hash,
        "source_epoch_lineage_payload_sha256": (
            selection.source_epoch_lineage_payload_digest
        ),
        "timing_authority": selection.selection_record["timing"],
        "source_video_metadata_ref": "recording/source_video_metadata.v2",
        "source_video_metadata_sha256": canonical_json_sha256(video),
        "acquisition_clock_authority_ref": clock[
            "acquisition_frame_clock"
        ]["run_path"],
        "acquisition_clock_authority_sha256": canonical_json_sha256(clock),
        "acquisition_frame_domain": "camera_acquisition_frame_index",
        "frame_count": selection.native_frame_count,
        "fps": selection.fps,
    }
    return TimelineAuthorityEvidence(
        recording_id="recording_1",
        timeline_id=selection.source_timeline_digest,
        stimulus_authority_id=selection.run_path,
        acquisition_frame_domain="camera_acquisition_frame_index",
        source_video_metadata_ref="recording/source_video_metadata.v2",
        source_video_metadata_sha256=canonical_json_sha256(video),
        source_video_metadata=video,
        acquisition_clock_authority_ref=clock["acquisition_frame_clock"]["run_path"],
        acquisition_clock_authority_sha256=canonical_json_sha256(clock),
        acquisition_clock_authority=clock,
        source_metadata_sha256=canonical_json_sha256(source_metadata),
        source_metadata=source_metadata,
    )


def _bindings(selection):
    return {
        "black_before": EpochRoleBinding.by_window_id(selection.intervals[0].window_id),
        "chaser": EpochRoleBinding.by_window_id(selection.intervals[1].window_id),
        "black_after": EpochRoleBinding.by_window_id(selection.intervals[2].window_id),
    }


def test_three_window_goodbatbadbat_shape_is_explicit_and_source_bound(
    tmp_path: Path,
    published_candidate: Path,
) -> None:
    archive, selection = _sealed_selection(tmp_path, published_candidate)
    result = compile_goodbatbadbat_selections(
        selection,
        timeline_evidence=_evidence(archive, selection),
        role_bindings=_bindings(selection),
        include_all_black=True,
    )

    assert result.pre is result.black_before
    assert result.training is result.chaser
    assert result.post is result.black_after
    assert [result[name].selection_id for name in ("pre", "training", "post")] == [
        "black_before",
        "chaser",
        "black_after",
    ]
    assert result.timeline_authority.acquisition_frame_count == 30
    for role in ("black_before", "chaser", "black_after"):
        compiled = result[role]
        assert compiled.resolved_intervals[0].source_memberships[0].reference_kind == (
            "interval_annotation"
        )
        membership = compiled.resolved_intervals[0].source_memberships[0]
        metadata = membership.role.metadata
        assert metadata["source_interval_digest"] == (
            selection.intervals[{"black_before": 0, "chaser": 1, "black_after": 2}[role]].source_interval_digest
        )
        assert metadata["occurrence_identity"]["occurrence_id"]


def test_role_order_is_not_an_authority_and_digests_are_deterministic(
    tmp_path: Path,
    published_candidate: Path,
) -> None:
    archive, selection = _sealed_selection(tmp_path, published_candidate)
    evidence = _evidence(archive, selection)
    first = compile_goodbatbadbat_selections(
        selection,
        timeline_evidence=evidence,
        role_bindings=_bindings(selection),
        include_all_black=True,
    )
    reversed_bindings = {
        "black_after": _bindings(selection)["black_after"],
        "chaser": _bindings(selection)["chaser"],
        "black_before": _bindings(selection)["black_before"],
    }
    second = compile_goodbatbadbat_selections(
        selection,
        timeline_evidence=evidence,
        role_bindings=reversed_bindings,
        include_all_black=True,
    )
    assert first.named.keys() == second.named.keys()
    for name in ("black_before", "chaser", "black_after", "all_black"):
        assert first[name].to_dict() == second[name].to_dict()


def test_one_role_can_bind_by_exact_source_interval_digest(
    tmp_path: Path,
    published_candidate: Path,
) -> None:
    archive, selection = _sealed_selection(tmp_path, published_candidate)
    bindings = _bindings(selection)
    bindings["chaser"] = EpochRoleBinding.by_source_interval_digest(
        selection.intervals[1].source_interval_digest
    )
    result = compile_goodbatbadbat_selections(
        selection,
        timeline_evidence=_evidence(archive, selection),
        role_bindings=bindings,
    )
    membership = result.chaser.resolved_intervals[0].source_memberships[0]
    assert membership.reference_id == selection.intervals[1].source_interval_digest


def test_all_black_union_keeps_two_distinct_occurrences(
    tmp_path: Path,
    published_candidate: Path,
) -> None:
    archive, selection = _sealed_selection(tmp_path, published_candidate)
    result = compile_goodbatbadbat_selections(
        selection,
        timeline_evidence=_evidence(archive, selection),
        role_bindings=_bindings(selection),
        include_all_black=True,
    )
    assert result.all_black is not None
    assert result.all_black.aggregation_policy == "keep_occurrences"
    assert len(result.all_black.occurrences) == 2
    assert {
        occurrence.occurrence_id for occurrence in result.all_black.occurrences
    } == {
        selection.intervals[0].occurrence_identity["occurrence_id"],
        selection.intervals[2].occurrence_identity["occurrence_id"],
    }


@pytest.mark.parametrize(
    "bad_bindings",
    [
        {},
        {"black_before": EpochRoleBinding.by_window_id(0)},
        {
            "black_before": EpochRoleBinding.by_window_id(0),
            "chaser": EpochRoleBinding.by_window_id(1),
            "black_after": EpochRoleBinding.by_window_id(2),
            "extra": EpochRoleBinding.by_window_id(0),
        },
    ],
)
def test_missing_or_extra_role_bindings_fail_closed(
    tmp_path: Path,
    published_candidate: Path,
    bad_bindings,
) -> None:
    archive, selection = _sealed_selection(tmp_path, published_candidate)
    with pytest.raises(ComposableEpochSelectionAdapterError, match="exactly"):
        compile_goodbatbadbat_selections(
            selection,
            timeline_evidence=_evidence(archive, selection),
            role_bindings=bad_bindings,
        )


def test_labels_are_not_accepted_as_implicit_role_authority(
    tmp_path: Path,
    published_candidate: Path,
) -> None:
    archive, selection = _sealed_selection(tmp_path, published_candidate)
    with pytest.raises(ComposableEpochSelectionAdapterError, match="explicit"):
        compile_goodbatbadbat_selections(
            selection,
            timeline_evidence=_evidence(archive, selection),
            role_bindings={
                "black_before": "pre_event",
                "chaser": "training_event",
                "black_after": "post_event",
            },
        )


def test_wrong_recording_or_frame_evidence_fails_closed(
    tmp_path: Path,
    published_candidate: Path,
) -> None:
    archive, selection = _sealed_selection(tmp_path, published_candidate)
    evidence = _evidence(archive, selection)
    with pytest.raises(ComposableEpochSelectionAdapterError, match="recording_id"):
        compile_goodbatbadbat_selections(
            selection,
            timeline_evidence=TimelineAuthorityEvidence(
                recording_id="another-recording",
                timeline_id=evidence.timeline_id,
                stimulus_authority_id=evidence.stimulus_authority_id,
                acquisition_frame_domain=evidence.acquisition_frame_domain,
                source_video_metadata_ref=evidence.source_video_metadata_ref,
                source_video_metadata_sha256=evidence.source_video_metadata_sha256,
                source_video_metadata=evidence.source_video_metadata,
                acquisition_clock_authority_ref=evidence.acquisition_clock_authority_ref,
                acquisition_clock_authority_sha256=evidence.acquisition_clock_authority_sha256,
                acquisition_clock_authority=evidence.acquisition_clock_authority,
                source_metadata_sha256=evidence.source_metadata_sha256,
                source_metadata=evidence.source_metadata,
            ),
            role_bindings=_bindings(selection),
        )


def test_inconsistent_frame_count_evidence_fails_closed(
    tmp_path: Path,
    published_candidate: Path,
) -> None:
    archive, selection = _sealed_selection(tmp_path, published_candidate)
    evidence = _evidence(archive, selection)
    root = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    video = dict(root.attrs["source_video_metadata"])
    video["total_frames"] = 29
    with pytest.raises(ComposableEpochSelectionAdapterError, match="frame count"):
        compile_goodbatbadbat_selections(
            selection,
            timeline_evidence=TimelineAuthorityEvidence(
                recording_id=evidence.recording_id,
                timeline_id=evidence.timeline_id,
                stimulus_authority_id=evidence.stimulus_authority_id,
                acquisition_frame_domain=evidence.acquisition_frame_domain,
                source_video_metadata_ref=evidence.source_video_metadata_ref,
                source_video_metadata_sha256=canonical_json_sha256(video),
                source_video_metadata=video,
                acquisition_clock_authority_ref=evidence.acquisition_clock_authority_ref,
                acquisition_clock_authority_sha256=evidence.acquisition_clock_authority_sha256,
                acquisition_clock_authority=evidence.acquisition_clock_authority,
                source_metadata_sha256=evidence.source_metadata_sha256,
                source_metadata=evidence.source_metadata,
            ),
            role_bindings=_bindings(selection),
        )


def test_unsealed_selection_fails_closed(
    tmp_path: Path,
    published_candidate: Path,
) -> None:
    archive, selection = _sealed_selection(tmp_path, published_candidate)
    object.__setattr__(selection, "_seal", object())
    with pytest.raises(ComposableEpochSelectionAdapterError, match="verified"):
        compile_goodbatbadbat_selections(
            selection,
            timeline_evidence=_evidence(archive, selection),
            role_bindings=_bindings(selection),
        )


def test_selector_alias_selection_name_fails_closed(
    tmp_path: Path,
    published_candidate: Path,
) -> None:
    archive, selection = _sealed_selection(tmp_path, published_candidate)
    evidence = _evidence(archive, selection)
    object.__setattr__(selection, "run_name", "latest")
    with pytest.raises(ComposableEpochSelectionAdapterError, match="non-selector"):
        compile_goodbatbadbat_selections(
            selection,
            timeline_evidence=evidence,
            role_bindings=_bindings(selection),
        )
