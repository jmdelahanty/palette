from __future__ import annotations

from fisheye.registry.maintenance import (
    _extract_tracking_summary_details,
    _tracking_source_freshness,
)


class _Group:
    def __init__(self, attrs: dict[str, object]) -> None:
        self.attrs = attrs


def test_registry_projects_tracking_identity_and_rowset_fingerprint() -> None:
    group = _Group(
        {
            "tracking_identity_mode": "instance_key",
            "source_rowset_path": "crop_runs/crop_a",
            "source_rowset_fingerprint": "abc123",
            "source_rowset_fingerprint_status": "complete",
            "source_rowset_row_count": 12,
            "source_rowset_edit_revision": 4,
            "source_arena_assignment_run": "arena_a",
        }
    )

    details = _extract_tracking_summary_details(group)

    assert details["tracking_identity_mode"] == "instance_key"
    assert details["source_rowset_fingerprint"] == "abc123"
    assert details["source_rowset_row_count"] == 12
    assert details["source_rowset_edit_revision"] == 4


def test_registry_marks_tracking_stale_when_arena_or_rowset_changes() -> None:
    tracks = _Group(
        {
            "source_arena_assignment_run": "arena_a",
            "source_rowset_fingerprint": "fingerprint_a",
        }
    )
    matching_arena = _Group({"source_rowset_fingerprint": "fingerprint_a"})
    changed_arena = _Group({"source_rowset_fingerprint": "fingerprint_b"})

    assert _tracking_source_freshness(
        tracks_group=tracks,
        arena_assignment_group=matching_arena,
        selected_arena_assignment_run="arena_a",
    ) == ("ok", "present")
    assert _tracking_source_freshness(
        tracks_group=tracks,
        arena_assignment_group=matching_arena,
        selected_arena_assignment_run="arena_b",
    ) == ("stale", "stale_vs_selected_arena_assignment")
    assert _tracking_source_freshness(
        tracks_group=tracks,
        arena_assignment_group=changed_arena,
        selected_arena_assignment_run="arena_a",
    ) == ("stale", "source_rowset_fingerprint_mismatch")
