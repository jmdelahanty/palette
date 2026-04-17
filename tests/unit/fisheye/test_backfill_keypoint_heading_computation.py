from pathlib import Path

from fisheye.utils import backfill_keypoint_heading_computation as mod
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


def test_backfill_run_group_prefers_run_level_labels_over_stale_schema_nodes() -> None:
    run = _FakeGroup(
        attrs={
            "keypoint_labels": ["swim_bladder", "eye_left", "eye_right"],
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
    assert heading["direction_from"] == {"op": "keypoint", "label": "swim_bladder"}
    assert heading["dependent_keypoints"] == ["swim_bladder", "eye_left", "eye_right"]


def test_backfill_run_group_canonicalizes_legacy_bladder_label() -> None:
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
    assert heading["direction_from"] == {"op": "keypoint", "label": "swim_bladder"}
    assert heading["dependent_keypoints"] == ["swim_bladder", "eye_left", "eye_right"]


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


def test_select_runs_includes_direct_fs_run_names(monkeypatch) -> None:
    root = _FakeGroup(
        attrs={},
        keypoints_runs=_FakeGroup(
            attrs={"latest": "keypoints_001"},
            keypoints_001=_FakeGroup(attrs={"name": "embedded"}),
        ),
    )
    direct_groups = {
        "keypoints_001": _FakeGroup(attrs={"name": "direct-001"}),
        "keypoints_002": _FakeGroup(attrs={"name": "direct-002"}),
    }
    zarr_path = Path("/tmp/fake_training.zarr")
    seen_modes: list[str] = []

    monkeypatch.setattr(mod, "_direct_group_names", lambda path: ["keypoints_001", "keypoints_002"])
    monkeypatch.setattr(
        mod,
        "_open_group_direct",
        lambda path, mode: seen_modes.append(mode) or direct_groups[Path(path).name],
    )

    groups = mod._select_runs(root, all_runs=True, zarr_path=zarr_path, open_mode="a")

    assert len(groups) == 2
    assert groups[0] is direct_groups["keypoints_001"]
    assert groups[1] is direct_groups["keypoints_002"]
    assert seen_modes == ["a", "a"]
