from fisheye.utils.backfill_keypoint_heading_computation import _backfill_run_group


class _FakeGroup(dict):
    def __init__(self, *args, attrs: dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}


def test_backfill_run_group_populates_pose_schema_heading_computation() -> None:
    run = _FakeGroup(
        attrs={
            "pose_schema": {
                "name": "traditional_v1",
                "nodes": [
                    {"id": 0, "name": "swim_bladder"},
                    {"id": 1, "name": "eye_left"},
                    {"id": 2, "name": "eye_right"},
                ],
                "edges": [[0, 1], [0, 2], [1, 2]],
                "metadata": {
                    "version": 1,
                    "coordinate_system": "roi",
                },
            }
        }
    )

    result = _backfill_run_group(run, apply=True)

    assert result.status == "ok"
    heading = run.attrs["pose_schema"]["metadata"]["heading_computation"]
    assert heading["origin"] == {"op": "midpoint", "labels": ["eye_left", "eye_right"]}
    assert heading["direction_from"] == {"op": "keypoint", "label": "swim_bladder"}
    assert heading["dependent_keypoints"] == ["swim_bladder", "eye_left", "eye_right"]


def test_backfill_run_group_uses_run_consistent_legacy_bladder_label() -> None:
    run = _FakeGroup(
        attrs={
            "keypoint_labels": ["bladder", "eye_left", "eye_right"],
            "pose_schema": {
                "name": "traditional_v1",
                "nodes": [
                    {"id": 0, "name": "bladder"},
                    {"id": 1, "name": "eye_left"},
                    {"id": 2, "name": "eye_right"},
                ],
                "metadata": {},
            },
        }
    )

    result = _backfill_run_group(run, apply=True)

    assert result.status == "ok"
    heading = run.attrs["pose_schema"]["metadata"]["heading_computation"]
    assert heading["direction_from"] == {"op": "keypoint", "label": "bladder"}
    assert heading["dependent_keypoints"] == ["bladder", "eye_left", "eye_right"]


def test_backfill_run_group_skips_when_heading_computation_already_present() -> None:
    run = _FakeGroup(
        attrs={
            "pose_schema": {
                "name": "traditional_v2",
                "nodes": [
                    {"id": 0, "name": "swim_bladder"},
                    {"id": 1, "name": "eye_left"},
                    {"id": 2, "name": "eye_right"},
                    {"id": 3, "name": "snout_tip"},
                    {"id": 4, "name": "tail_tip"},
                ],
                "metadata": {
                    "heading_computation": {
                        "version": 1,
                        "enabled": True,
                        "origin": {"op": "midpoint", "labels": ["eye_left", "eye_right"]},
                        "direction_from": {"op": "keypoint", "label": "swim_bladder"},
                        "direction_to": {"op": "midpoint", "labels": ["eye_left", "eye_right"]},
                        "dependent_keypoints": ["swim_bladder", "eye_left", "eye_right"],
                    }
                },
            }
        }
    )

    result = _backfill_run_group(run, apply=True)

    assert result.status == "skipped_existing"


def test_backfill_run_group_reports_unsupported_when_heading_labels_missing() -> None:
    run = _FakeGroup(
        attrs={
            "pose_schema": {
                "name": "weird_schema",
                "nodes": [
                    {"id": 0, "name": "dorsal_fin"},
                    {"id": 1, "name": "tail_tip"},
                ],
                "metadata": {},
            }
        }
    )

    result = _backfill_run_group(run, apply=True)

    assert result.status == "unsupported_labels"
