from __future__ import annotations

from fisheye.shared.subject_mask_component_provenance import write_subject_mask_component_provenance


class _FakeGroup(dict):
    def __init__(self) -> None:
        super().__init__()
        self.attrs: dict[str, object] = {}

    def require_group(self, name: str):
        value = self.get(name)
        if isinstance(value, _FakeGroup):
            return value
        value = _FakeGroup()
        self[name] = value
        return value


def test_write_subject_mask_component_provenance_records_crop_snapshot_fields() -> None:
    run_group = _FakeGroup()

    provenance_group = write_subject_mask_component_provenance(
        run_group,
        component_name="subject_body",
        source_stage="subject_mask_runs",
        source_run="subject_masks_body_001",
        source_method="sam_subject_body_v1",
        source_channels=["subject_body"],
        source_label_schema_id="subject_v1_union",
        source_created_at_utc="2026-04-15T12:00:00Z",
        source_crop_run="crop_001",
        source_crop_snapshot={
            "source_crop_storage_mode": "geometry_only",
            "source_crop_signature": {"signature_version": 2, "crop_revision": 7},
            "source_crop_revision": 7,
            "source_detect_review_status_ref": "refined_detect_runs/refined_detect_001/review_status",
        },
    )

    assert provenance_group.attrs["source_stage"] == "subject_mask_runs"
    assert provenance_group.attrs["source_run"] == "subject_masks_body_001"
    assert provenance_group.attrs["source_method"] == "sam_subject_body_v1"
    assert provenance_group.attrs["source_channels"] == ["subject_body"]
    assert provenance_group.attrs["source_crop_run"] == "crop_001"
    assert provenance_group.attrs["source_crop_storage_mode"] == "geometry_only"
    assert provenance_group.attrs["source_crop_signature"] == "{'signature_version': 2, 'crop_revision': 7}"
    assert provenance_group.attrs["source_crop_revision"] == 7
    assert (
        provenance_group.attrs["source_detect_review_status_ref"]
        == "refined_detect_runs/refined_detect_001/review_status"
    )
