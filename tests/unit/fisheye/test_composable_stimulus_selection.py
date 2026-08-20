from __future__ import annotations

from dataclasses import replace

import pytest

from fisheye.analysis_workflows.composable_stimulus_selection import (
    AuthorityMismatchError,
    RoleMetadata,
    SelectionSpec,
    StimulusSelectionError,
    TimelineAuthority,
    TrimSpec,
    UnsupportedExpressionError,
    canonical_json,
    compile_selection,
    difference,
    intersection,
    interval_annotation_reference,
    member,
    stimulus_step_reference,
    union,
)


def _authority() -> TimelineAuthority:
    return TimelineAuthority(
        recording_id="recording-001",
        timeline_id="acquisition-timeline-v1",
        stimulus_authority_id="stimulus-run-001",
        stimulus_authority_sha256="a" * 64,
        acquisition_frame_domain="camera_acquisition_frame_index",
        acquisition_frame_count=120,
        source_video_metadata_ref="recording/video/metadata.json",
        source_video_metadata_sha256="b" * 64,
        acquisition_clock_authority_ref="recording/acquisition_clock.json",
        acquisition_clock_authority_sha256="c" * 64,
        source_metadata_sha256="d" * 64,
    )


def _step(
    authority: TimelineAuthority,
    reference_id: str,
    label: str,
    start: int,
    end: int,
    *,
    occurrence_id: str | None = None,
):
    return stimulus_step_reference(
        reference_id=reference_id,
        label=label,
        start_frame=start,
        end_frame=end,
        authority=authority,
        occurrence_id=occurrence_id,
    )


def _compile(expression, *, policy="keep_occurrences"):
    return compile_selection(
        SelectionSpec(
            selection_id="selection-v1",
            expression=expression,
            aggregation_policy=policy,
        )
    )


def test_solid_black_chaser_solid_black_keeps_distinct_roles_and_occurrences():
    authority = _authority()
    before = member(
        _step(authority, "step-before", "SOLID_BLACK", 0, 10),
        role=RoleMetadata("baseline", "black_before"),
    )
    chaser = member(
        _step(authority, "step-chaser", "CHASER_PRESENTATION", 12, 20),
        role=RoleMetadata("treatment", "chaser"),
    )
    after = member(
        _step(authority, "step-after", "SOLID_BLACK", 22, 30),
        role=RoleMetadata("control", "black_after"),
    )

    compiled = _compile(union(before, chaser, after))

    assert compiled.pooled_intervals == ((0, 10), (12, 20), (22, 30))
    assert [occurrence.role.role for occurrence in compiled.occurrences] == [
        "baseline",
        "treatment",
        "control",
    ]
    assert [occurrence.role.label for occurrence in compiled.occurrences] == [
        "black_before",
        "chaser",
        "black_after",
    ]
    assert [occurrence.label for occurrence in compiled.occurrences] == [
        "SOLID_BLACK",
        "CHASER_PRESENTATION",
        "SOLID_BLACK",
    ]
    assert compiled.occurrences[0].intervals == ((0, 10),)
    assert compiled.occurrences[2].intervals == ((22, 30),)


def test_gaps_are_preserved_and_empty_difference_is_valid():
    authority = _authority()
    left = member(_step(authority, "left", "A", 2, 5))
    right = member(_step(authority, "right", "B", 5, 8))
    assert _compile(union(left, right)).pooled_intervals == ((2, 8),)

    same_span_a = member(_step(authority, "same-a", "A", 10, 20))
    same_span_b = member(_step(authority, "same-b", "B", 10, 20))
    empty = _compile(difference(same_span_a, same_span_b))
    assert empty.empty
    assert empty.resolved_intervals == ()
    assert empty.pooled_intervals == ()
    assert empty.occurrences[0].intervals == ()
    assert empty.occurrences[1].intervals == ()


def test_overlap_deduplicates_frames_but_retains_both_source_memberships():
    authority = _authority()
    first = member(_step(authority, "first", "A", 0, 6))
    second = member(_step(authority, "second", "B", 3, 9))

    compiled = _compile(union(first, second), policy="pool_intervals")

    assert compiled.pooled_intervals == ((0, 9),)
    assert compiled.selected_frame_count == 9
    overlap = compiled.resolved_intervals[1]
    assert (overlap.start_frame, overlap.end_frame) == (3, 6)
    assert {item.reference_id for item in overlap.source_memberships} == {
        "first",
        "second",
    }
    assert sum(occurrence.frame_count for occurrence in compiled.occurrences) == 12


def test_trim_uses_ceil_seconds_times_fps_and_records_policy():
    authority = _authority()
    reference = _step(authority, "trimmed", "A", 10, 30)
    compiled = _compile(
        member(
            reference,
            trim=TrimSpec(leading_seconds=0.021, trailing_seconds=0.041, fps=50),
        )
    )

    assert compiled.pooled_intervals == ((12, 27),)
    membership = compiled.resolved_intervals[0].source_memberships[0]
    assert membership.selected_start_frame == 12
    assert membership.selected_end_frame == 27
    assert membership.trim is not None
    assert membership.trim.leading_frames == 2
    assert membership.trim.trailing_frames == 3
    assert membership.trim.to_dict()["rounding_policy"] == "ceil_seconds_times_fps_v1"


def test_zero_length_after_exact_trim_is_empty_but_inverting_trim_fails():
    authority = _authority()
    reference = _step(authority, "trimmed", "A", 10, 20)
    assert _compile(member(reference, trim=TrimSpec(0.1, 0.1, 50))).empty
    with pytest.raises(StimulusSelectionError, match="invert"):
        _compile(member(reference, trim=TrimSpec(0.11, 0.1, 50)))
    with pytest.raises(StimulusSelectionError, match="non-negative"):
        TrimSpec(-0.01, 0, 50)


def test_annotation_and_step_are_exact_typed_members():
    authority = _authority()
    annotation = interval_annotation_reference(
        reference_id="annotation-1",
        label="reviewed_chaser_window",
        start_frame=40,
        end_frame=50,
        authority=authority,
    )
    compiled = _compile(member(annotation))
    assert compiled.resolved_intervals[0].source_memberships[0].reference_kind == (
        "interval_annotation"
    )


def test_canonical_digest_is_stable_under_union_mapping_and_child_order():
    authority = _authority()
    a = member(
        _step(authority, "a", "A", 0, 4),
        role=RoleMetadata("baseline", metadata={"z": 2, "a": 1}),
    )
    b = member(_step(authority, "b", "B", 8, 12))
    left = _compile(union(a, b))
    right = _compile(union(b, a))
    assert left.request_digest == right.request_digest
    assert left.resolved_digest == right.resolved_digest
    assert canonical_json(left.requested) == canonical_json(right.requested)


def test_authority_mismatch_and_stale_expected_authority_fail_closed():
    authority = _authority()
    other = replace(authority, timeline_id="other-timeline")
    expression = union(
        member(_step(authority, "a", "A", 0, 4)),
        member(_step(other, "b", "B", 4, 8)),
    )
    with pytest.raises(AuthorityMismatchError, match="one recording"):
        _compile(expression)
    with pytest.raises(AuthorityMismatchError, match="expected authority"):
        compile_selection(
            SelectionSpec(
                "selection-v1",
                member(_step(authority, "a", "A", 0, 4)),
                "pool_intervals",
            ),
            expected_authority=replace(authority, source_metadata_sha256="e" * 64),
        )


def test_duplicate_atomic_reference_and_unsupported_operation_fail_closed():
    authority = _authority()
    reference = _step(authority, "duplicate", "A", 0, 4)
    expression = union(member(reference), member(reference))
    with pytest.raises(StimulusSelectionError, match="duplicate"):
        _compile(expression)

    class Unsupported:
        def to_dict(self):
            return {"operation": "xor"}

    with pytest.raises(UnsupportedExpressionError, match="unsupported expression"):
        _compile(Unsupported())


def test_out_of_domain_and_invalid_authority_fields_fail_closed():
    authority = _authority()
    with pytest.raises(StimulusSelectionError, match="SHA-256"):
        replace(authority, source_metadata_sha256="missing")
    with pytest.raises(StimulusSelectionError, match="frame domain"):
        _step(authority, "outside", "A", 110, 121)
    with pytest.raises(StimulusSelectionError, match="non-empty half-open"):
        _step(authority, "inverting", "A", 5, 5)


def test_set_algebra_intersection_and_difference_preserve_surviving_membership():
    authority = _authority()
    first = member(_step(authority, "first", "A", 0, 10))
    second = member(_step(authority, "second", "B", 5, 15))
    overlap = _compile(intersection(first, second))
    assert overlap.pooled_intervals == ((5, 10),)
    assert {
        item.reference_id for item in overlap.resolved_intervals[0].source_memberships
    } == {
        "first",
        "second",
    }

    excluded = member(_step(authority, "excluded", "C", 7, 8))
    remaining = _compile(difference(union(first, second), excluded))
    assert remaining.pooled_intervals == ((0, 7), (8, 15))
    assert all(
        item.reference_id != "excluded"
        for interval in remaining.resolved_intervals
        for item in interval.source_memberships
    )


def test_selection_payload_keeps_aggregation_policy_explicit():
    authority = _authority()
    expression = union(
        member(_step(authority, "a", "A", 0, 3)),
        member(_step(authority, "b", "B", 6, 9)),
    )
    keep = _compile(expression, policy="keep_occurrences")
    pool = _compile(expression, policy="pool_intervals")
    assert keep.aggregation_policy == "keep_occurrences"
    assert pool.aggregation_policy == "pool_intervals"
    assert keep.request_digest != pool.request_digest


def test_nested_metadata_and_compiled_request_are_immutable_after_digest() -> None:
    authority = _authority()
    nested = {"labels": ["first"]}
    role = RoleMetadata("baseline", metadata=nested)
    spec = SelectionSpec(
        selection_id="immutable-selection.v1",
        expression=member(_step(authority, "a", "A", 0, 3), role=role),
        aggregation_policy="keep_occurrences",
        metadata=nested,
    )
    compiled = compile_selection(spec)
    digest = compiled.request_digest

    nested["labels"].append("mutated")
    assert role.to_dict()["metadata"] == {"labels": ["first"]}
    assert spec.to_dict(authority=authority)["metadata"] == {"labels": ["first"]}
    assert compiled.request_digest == digest
    with pytest.raises(TypeError):
        compiled.requested["selection_id"] = "changed"
