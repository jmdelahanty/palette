"""Tests for syncing eye-mask profile summaries into registry rows."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils import sync_eye_mask_profile_registry as sync_module


class _FakeGroup:
    def __init__(
        self,
        *,
        children: dict[str, "_FakeGroup"] | None = None,
        attrs: dict[str, object] | None = None,
        get_misses: set[str] | None = None,
        item_misses: set[str] | None = None,
        store: object | None = None,
        path: str = "",
    ) -> None:
        self._children = dict(children or {})
        self.attrs = dict(attrs or {})
        self._get_misses = set(get_misses or set())
        self._item_misses = set(item_misses or set())
        self.store = object() if store is None else store
        self.path = path

    def get(self, name: str):
        if name in self._get_misses:
            return None
        return self._children.get(name)

    def __getitem__(self, name: str):
        if name in self._item_misses:
            raise KeyError(name)
        return self._children[name]

    def __contains__(self, name: str) -> bool:
        return name in self._children

    def keys(self):
        return self._children.keys()

    def group_keys(self):
        return self._children.keys()


def _seed_dataset_row(
    registry_path: Path,
    *,
    dataset_id: str,
    zarr_path: Path,
    zarr_use: str,
    genotype: str | None = None,
    dpf_at_acquisition: int | None = None,
) -> None:
    registry = Registry(registry_path)
    registry.upsert_dataset(
        dataset_id,
        session_uuid=f"{dataset_id}_session",
        zarr_path=zarr_path,
        recording_id=f"{dataset_id}_recording",
        artifact_kind="source_recording",
        zarr_use=zarr_use,
    )
    if genotype is not None or dpf_at_acquisition is not None:
        registry.upsert_provenance(
            dataset_id,
            provenance={
                "genotype": genotype,
                "dpf_at_acquisition": dpf_at_acquisition,
            },
            context={},
            protocol_name=None,
            protocol_hash=None,
            acquisition={},
        )
    registry.close()


def _summary_payload(*, dataset_id: str, recording_id: str, zarr_use: str) -> dict[str, object]:
    return {
        "schema_name": "eye_mask_dataset_profile",
        "schema_version": "v1",
        "created_at_utc": "2026-02-25T03:00:00+00:00",
        "dataset": {
            "dataset_id": dataset_id,
            "recording_id": recording_id,
            "zarr_use": zarr_use,
        },
        "source": {
            "stage_group": "refined_eye_masks_runs",
            "eye_mask_path": "refined_eye_masks_runs/refined_eye_masks_2026-02-25_03-00-00",
            "eye_mask_run": "refined_eye_masks_2026-02-25_03-00-00",
            "eye_mask_method": "traditional",
            "source_keypoint_path": "refined_keypoints_runs/refined_keypoints_2026-02-25_02-30-00",
            "source_keypoint_run": "refined_keypoints_2026-02-25_02-30-00",
            "source_crop_run": "crop_2026-02-25_02-20-00",
            "review_state": "approved",
            "review_method": "manual",
            "review_intended_use": "training",
            "review_timestamp_utc": "2026-02-25T02:59:00+00:00",
            "source_keypoint_stale": {
                "state": "fresh",
                "reason": None,
                "timestamp_utc": "2026-02-25T02:58:30+00:00",
            },
        },
        "quality": {
            "rows_total": 100,
            "rows_usable": 95,
            "usable_rate": 0.95,
            "reviewed_rate": 1.0,
            "excluded_rate": 0.05,
            "exclusion_reasons": {"ellipse_fit_failed": 5},
            "ellipse_success_rate": 0.98,
            "pair_success_rate": 0.95,
        },
        "geometry": {
            "area": {"stats": {"p10": 300.0, "p50": 400.0, "p90": 500.0}},
            "left_area": {"stats": {"p10": 140.0, "p50": 190.0, "p90": 240.0}},
            "right_area": {"stats": {"p10": 150.0, "p50": 200.0, "p90": 250.0}},
            "union_area": {"stats": {"p10": 300.0, "p50": 390.0, "p90": 480.0}},
            "area_lr_ratio": {"stats": {"p10": 0.92, "p50": 0.97, "p90": 1.03}},
            "major_axis": {"stats": {"p10": 10.0, "p50": 12.0, "p90": 14.0}},
            "minor_axis": {"stats": {"p10": 5.0, "p50": 6.0, "p90": 7.0}},
            "aspect_ratio": {"stats": {"p10": 1.5, "p50": 1.8, "p90": 2.1}},
            "eye_separation": {"stats": {"p10": 20.0, "p50": 25.0, "p90": 30.0}},
        },
        "spatial": {
            "edge_proximity_rate": 0.03,
        },
        "composition": {
            "rig_id": "rig_a",
            "camera_id": "cam_1",
            "arena_id": "arena_x",
            "dish_design": "cedar",
            "canvas_name": "canvas_a",
            "protocol_name": "DefaultScreen",
            "genotype": "Tg(elavl3:gcamp7f)",
            "dpf_at_acquisition": 7,
        },
    }


def test_sync_eye_mask_profile_registry_apply_upserts_latest_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "dataset_a_training.zarr"
    zarr_path.mkdir(parents=True)
    _seed_dataset_row(registry_path, dataset_id="dataset_a", zarr_path=zarr_path, zarr_use="training")

    summary = _summary_payload(dataset_id="dataset_a", recording_id="recording_a", zarr_use="training")
    monkeypatch.setattr(sync_module, "_open_root", lambda _: object())
    monkeypatch.setattr(
        sync_module,
        "_latest_profile_summary",
        lambda _root: ("eye_mask_profile_2026-02-25_03-00-00", summary, None),
    )

    rc = sync_module.main(
        [
            "--registry",
            str(registry_path),
            "--dataset-id",
            "dataset_a",
            "--apply",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "inserted=1" in out
    assert "updated=0" in out
    assert "unchanged=0" in out

    registry = Registry(registry_path)
    try:
        rows = registry.query_eye_mask_data_profile_latest(dataset_ids=["dataset_a"])
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["dataset_id"] == "dataset_a"
        assert row["profile_run"] == "eye_mask_profile_2026-02-25_03-00-00"
        assert row["recording_id"] == "recording_a"
        assert row["zarr_use"] == "training"
        assert row["stage_group"] == "refined_eye_masks_runs"
        assert row["eye_mask_method"] == "traditional"
        assert row["usable_rate"] == pytest.approx(0.95)
        assert row["area_p50"] == pytest.approx(400.0)
        assert row["left_area_p50"] == pytest.approx(190.0)
        assert row["right_area_p50"] == pytest.approx(200.0)
        assert row["union_area_p50"] == pytest.approx(390.0)
        assert row["area_lr_ratio_p50"] == pytest.approx(0.97)
        assert row["source_keypoint_stale_state"] == "fresh"
        assert row["genotype"] == "Tg(elavl3:gcamp7f)"
        assert int(row["dpf_at_acquisition"]) == 7
    finally:
        registry.close()


def test_sync_eye_mask_profile_registry_apply_detects_unchanged_then_updated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "dataset_b_training.zarr"
    zarr_path.mkdir(parents=True)
    _seed_dataset_row(registry_path, dataset_id="dataset_b", zarr_path=zarr_path, zarr_use="training")

    summary_initial = _summary_payload(dataset_id="dataset_b", recording_id="recording_b", zarr_use="training")
    monkeypatch.setattr(sync_module, "_open_root", lambda _: object())
    monkeypatch.setattr(
        sync_module,
        "_latest_profile_summary",
        lambda _root: ("eye_mask_profile_2026-02-25_04-00-00", summary_initial, None),
    )

    first_rc = sync_module.main(
        [
            "--registry",
            str(registry_path),
            "--dataset-id",
            "dataset_b",
            "--apply",
        ]
    )
    assert first_rc == 0
    _ = capsys.readouterr()

    second_rc = sync_module.main(
        [
            "--registry",
            str(registry_path),
            "--dataset-id",
            "dataset_b",
            "--apply",
        ]
    )
    assert second_rc == 0
    second_out = capsys.readouterr().out
    assert "inserted=0" in second_out
    assert "updated=0" in second_out
    assert "unchanged=1" in second_out

    summary_updated = _summary_payload(dataset_id="dataset_b", recording_id="recording_b", zarr_use="training")
    quality = dict(summary_updated["quality"])  # type: ignore[index]
    quality["usable_rate"] = 0.88
    summary_updated["quality"] = quality
    monkeypatch.setattr(
        sync_module,
        "_latest_profile_summary",
        lambda _root: ("eye_mask_profile_2026-02-25_04-00-00", summary_updated, None),
    )

    third_rc = sync_module.main(
        [
            "--registry",
            str(registry_path),
            "--dataset-id",
            "dataset_b",
            "--apply",
        ]
    )
    assert third_rc == 0
    third_out = capsys.readouterr().out
    assert "inserted=0" in third_out
    assert "updated=1" in third_out
    assert "unchanged=0" in third_out

    registry = Registry(registry_path)
    try:
        rows = registry.query_eye_mask_data_profile_latest(dataset_ids=["dataset_b"])
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["usable_rate"] == pytest.approx(0.88)
    finally:
        registry.close()


def test_sync_eye_mask_profile_registry_dry_run_does_not_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "dataset_c_analysis.zarr"
    zarr_path.mkdir(parents=True)
    _seed_dataset_row(registry_path, dataset_id="dataset_c", zarr_path=zarr_path, zarr_use="analysis")

    summary = _summary_payload(dataset_id="dataset_c", recording_id="recording_c", zarr_use="analysis")
    monkeypatch.setattr(sync_module, "_open_root", lambda _: object())
    monkeypatch.setattr(
        sync_module,
        "_latest_profile_summary",
        lambda _root: ("eye_mask_profile_2026-02-25_05-00-00", summary, None),
    )

    rc = sync_module.main(
        [
            "--registry",
            str(registry_path),
            "--dataset-id",
            "dataset_c",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "would_insert=1" in out
    assert "would_update=0" in out
    assert "would_unchanged=0" in out

    registry = Registry(registry_path)
    try:
        count = registry.conn.execute("SELECT COUNT(*) AS n FROM eye_mask_data_profile;").fetchone()
        assert count is not None
        assert int(count["n"]) == 0
    finally:
        registry.close()


def test_sync_eye_mask_profile_registry_missing_profile_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "dataset_d_training.zarr"
    zarr_path.mkdir(parents=True)
    _seed_dataset_row(registry_path, dataset_id="dataset_d", zarr_path=zarr_path, zarr_use="training")

    monkeypatch.setattr(sync_module, "_open_root", lambda _: object())
    monkeypatch.setattr(sync_module, "_latest_profile_summary", lambda _root: (None, None, "missing summary"))

    rc = sync_module.main(
        [
            "--registry",
            str(registry_path),
            "--dataset-id",
            "dataset_d",
            "--apply",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "missing_profile" in out
    assert "missing_profile=1" in out

    registry = Registry(registry_path)
    try:
        count = registry.conn.execute("SELECT COUNT(*) AS n FROM eye_mask_data_profile;").fetchone()
        assert count is not None
        assert int(count["n"]) == 0
    finally:
        registry.close()


def test_sync_eye_mask_profile_registry_falls_back_to_provenance_subject_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "dataset_e_training.zarr"
    zarr_path.mkdir(parents=True)
    _seed_dataset_row(
        registry_path,
        dataset_id="dataset_e",
        zarr_path=zarr_path,
        zarr_use="training",
        genotype="Tg(test:line)",
        dpf_at_acquisition=9,
    )

    summary = _summary_payload(dataset_id="dataset_e", recording_id="recording_e", zarr_use="training")
    composition = dict(summary["composition"])  # type: ignore[index]
    composition.pop("genotype", None)
    composition.pop("dpf_at_acquisition", None)
    summary["composition"] = composition
    monkeypatch.setattr(sync_module, "_open_root", lambda _: object())
    monkeypatch.setattr(
        sync_module,
        "_latest_profile_summary",
        lambda _root: ("eye_mask_profile_2026-02-25_06-00-00", summary, None),
    )

    rc = sync_module.main(
        [
            "--registry",
            str(registry_path),
            "--dataset-id",
            "dataset_e",
            "--apply",
        ]
    )
    assert rc == 0

    registry = Registry(registry_path)
    try:
        rows = registry.query_eye_mask_data_profile_latest(dataset_ids=["dataset_e"])
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["genotype"] == "Tg(test:line)"
        assert int(row["dpf_at_acquisition"]) == 9
    finally:
        registry.close()


def test_latest_profile_summary_falls_back_to_keyed_group_lookup() -> None:
    summary_payload = {
        "schema_name": "eye_mask_dataset_profile",
        "schema_version": "v1",
    }
    run_group = _FakeGroup(attrs={"profile_summary": summary_payload})
    runs_parent = _FakeGroup(
        children={"run_1": run_group},
        attrs={"latest": "run_1"},
        get_misses={"run_1"},
    )
    analysis = _FakeGroup(
        children={"eye_mask_profile_runs": runs_parent},
        get_misses={"eye_mask_profile_runs"},
    )
    root = _FakeGroup(
        children={"analysis": analysis},
        get_misses={"analysis"},
    )

    run_name, summary, error = sync_module._latest_profile_summary(root)  # type: ignore[arg-type]
    assert error is None
    assert run_name == "run_1"
    assert summary is not None
    assert summary["schema_name"] == "eye_mask_dataset_profile"
