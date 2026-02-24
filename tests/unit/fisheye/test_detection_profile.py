from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fisheye.registry.db import Registry
from fisheye.utils import backfill_detection_profiles as backfill_mod
from fisheye.utils import detection_profile as detection_profile_mod
from fisheye.utils.detection_profile import build_detection_profile_summary, write_detection_profile


class _FakeArray:
    def __init__(self, data: Any, *, chunks: tuple[int, ...] | None = None) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = self._data.dtype
        self.chunks = chunks

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, item, value) -> None:
        self._data[item] = value


class _FakeGroup:
    def __init__(self, *, store: object | None = None, path: str = "") -> None:
        self._children: dict[str, _FakeGroup | _FakeArray] = {}
        self.attrs: dict[str, Any] = {}
        self.store = object() if store is None else store
        self.path = path

    def create_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).create_group(tail)
        if name in self._children:
            raise ValueError(f"{name} already exists")
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(store=self.store, path=child_path)
        self._children[name] = child
        return child

    def require_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).require_group(tail)
        existing = self._children.get(name)
        if isinstance(existing, _FakeGroup):
            return existing
        if existing is not None:
            raise TypeError(f"{name} already exists and is not a group")
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(store=self.store, path=child_path)
        self._children[name] = child
        return child

    def create_array(
        self,
        name: str,
        *,
        data: Any,
        chunks: tuple[int, ...] | None = None,
        overwrite: bool = False,
    ) -> _FakeArray:
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).create_array(tail, data=data, chunks=chunks, overwrite=overwrite)
        if not overwrite and name in self._children:
            raise ValueError(f"{name} already exists")
        arr = _FakeArray(data, chunks=chunks)
        self._children[name] = arr
        return arr

    def get(self, name: str):
        try:
            return self[name]
        except Exception:
            return None

    def group_keys(self):
        return [key for key, value in self._children.items() if isinstance(value, _FakeGroup)]

    def keys(self):
        return list(self._children.keys())

    def __contains__(self, key: str) -> bool:
        try:
            _ = self[key]
            return True
        except Exception:
            return False

    def __getitem__(self, key: str):
        if "/" in key:
            current: _FakeGroup | _FakeArray = self
            for token in key.split("/"):
                if not isinstance(current, _FakeGroup):
                    raise KeyError(key)
                current = current._children[token]
            return current
        return self._children[key]

    def __delitem__(self, key: str) -> None:
        if "/" in key:
            head, tail = key.split("/", 1)
            child = self._children[head]
            if not isinstance(child, _FakeGroup):
                raise KeyError(key)
            del child[tail]
            return
        del self._children[key]


def _make_detect_root() -> _FakeGroup:
    root = _FakeGroup()
    root.attrs["recording_id"] = "rec_001"
    root.attrs["zarr_purpose"] = "analysis"
    root.attrs["rig_id"] = "omnifin0"
    root.attrs["dish_design"] = "cedar"

    analysis_meta = root.create_group("analysis_metadata")
    analysis_meta.attrs["session_context"] = json.dumps(
        {
            "camera_id": "2010094",
            "arena_id": "arena_2",
            "canvas_name": "shadow",
            "protocol_name_from_definition": "DefaultScreen",
        }
    )

    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_001"
    detect_run = detect_parent.create_group("detect_001")
    detect_run.create_array("frame_indices", data=np.asarray([0, 0, 1, 3, 4], dtype=np.int32))
    detect_run.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [
                [0.20, 0.25, 0.10, 0.12],
                [0.65, 0.72, 0.20, 0.18],
                [0.52, 0.30, 0.14, 0.10],
                [0.08, 0.12, 0.08, 0.09],
                [0.88, 0.90, 0.10, 0.11],
            ],
            dtype=np.float64,
        ),
    )
    detect_run.create_array("frame_counts", data=np.asarray([2, 1, 0, 1, 1, 0], dtype=np.int32))
    return root


class _FlakyLookupGroup(_FakeGroup):
    def __init__(
        self,
        *,
        children: dict[str, _FakeGroup | _FakeArray],
        attrs: dict[str, Any],
        store: object,
        path: str,
        miss_key: str,
    ) -> None:
        super().__init__(store=store, path=path)
        self._children = children
        self.attrs = attrs
        self._miss_key = miss_key

    def get(self, name: str):
        if name == self._miss_key:
            return None
        return super().get(name)

    def __getitem__(self, key: str):
        if key == self._miss_key:
            raise KeyError(key)
        return super().__getitem__(key)


def _is_monotonic(values: list[float | None]) -> bool:
    filtered = [float(v) for v in values if v is not None]
    return all(a <= b for a, b in zip(filtered, filtered[1:]))


def test_build_detection_profile_summary_invariants() -> None:
    root = _make_detect_root()

    summary = build_detection_profile_summary(
        root,
        zarr_path=Path("/tmp/rec_001_analysis.zarr"),
        created_at_utc="2026-02-24T10:00:00+00:00",
    )

    coverage = summary["coverage"]
    counts = summary["counts"]
    detections_total = int(counts["detections_total"])
    frames_total = int(coverage["frames_total"])
    frames_with_detections = int(coverage["frames_with_detections"])
    expected_coverage = 100.0 * frames_with_detections / frames_total

    assert np.isclose(coverage["coverage_percent"], expected_coverage)
    assert frames_total >= frames_with_detections
    assert detections_total >= frames_with_detections

    histograms = summary["histograms"]
    assert int(np.sum(histograms["w_norm"]["counts"])) == detections_total
    assert int(np.sum(histograms["h_norm"]["counts"])) == detections_total
    assert int(np.sum(histograms["area_norm"]["counts"])) == detections_total
    assert int(np.sum(histograms["aspect_ratio"]["counts"])) == detections_total

    geometry = summary["geometry_norm"]
    for metric_name in ("cx", "cy", "w", "h", "area", "aspect_ratio"):
        metric = geometry[metric_name]
        assert _is_monotonic(
            [
                metric["p01"],
                metric["p05"],
                metric["p10"],
                metric["p25"],
                metric["p50"],
                metric["p75"],
                metric["p90"],
                metric["p95"],
                metric["p99"],
            ]
        )

    dpf = counts["detections_per_frame"]
    assert dpf["p10"] <= dpf["p50"] <= dpf["p90"]

    composition = summary["composition"]
    assert composition["rig_id"] == "omnifin0"
    assert composition["camera_id"] == "2010094"
    assert composition["arena_id"] == "arena_2"
    assert composition["canvas_name"] == "shadow"
    assert composition["protocol_name"] == "DefaultScreen"
    assert composition["dish_design"] == "cedar"


def test_write_detection_profile_writes_run_attrs_and_latest_pointer() -> None:
    root = _make_detect_root()
    result = write_detection_profile(
        root,
        zarr_path=Path("/tmp/rec_001_analysis.zarr"),
        run_name="detection_profile_2026-02-24_10-10-10",
        created_at_utc="2026-02-24T10:10:10+00:00",
        source_detection_path="detect_runs/detect_001",
    )

    assert result.run_name == "detection_profile_2026-02-24_10-10-10"
    assert result.source_detection_path == "detect_runs/detect_001"
    assert result.source_detection_type == "detect"

    parent = root["analysis/detection_profile_runs"]
    assert parent.attrs["latest"] == "detection_profile_2026-02-24_10-10-10"

    run_group = parent["detection_profile_2026-02-24_10-10-10"]
    assert run_group.attrs["schema_name"] == "detection_dataset_profile"
    assert run_group.attrs["schema_version"] == "v1"
    assert run_group.attrs["source_detection_path"] == "detect_runs/detect_001"
    assert run_group.attrs["source_detection_type"] == "detect"
    assert run_group.attrs["source_resolution"] == "full"
    assert run_group.attrs["source_frame_count"] == 6
    assert run_group.attrs["source_frame_count_full"] is None
    assert isinstance(run_group.attrs["profile_summary"], dict)


def test_write_detection_profile_reuses_existing_runs_parent() -> None:
    root = _make_detect_root()
    analysis = root.create_group("analysis")
    runs_parent = analysis.create_group("detection_profile_runs")
    runs_parent.attrs["latest"] = "existing_run"
    runs_parent.create_group("existing_run")

    result = write_detection_profile(
        root,
        zarr_path=Path("/tmp/rec_001_analysis.zarr"),
        run_name="detection_profile_2026-02-24_10-20-20",
        created_at_utc="2026-02-24T10:20:20+00:00",
        source_detection_path="detect_runs/detect_001",
    )

    assert result.run_name == "detection_profile_2026-02-24_10-20-20"
    assert runs_parent.attrs["latest"] == "detection_profile_2026-02-24_10-20-20"
    assert "existing_run" in runs_parent
    assert "detection_profile_2026-02-24_10-20-20" in runs_parent


def test_write_detection_profile_handles_flaky_group_lookup() -> None:
    root = _make_detect_root()
    analysis = root.create_group("analysis")
    runs_parent = analysis.create_group("detection_profile_runs")
    runs_parent.attrs["latest"] = "existing_run"
    runs_parent.create_group("existing_run")

    # Simulate store behavior where get("detection_profile_runs") unexpectedly
    # returns None even though the group exists.
    analysis.get = (
        lambda name: None
        if name == "detection_profile_runs"
        else _FakeGroup.get(analysis, name)
    )

    result = write_detection_profile(
        root,
        zarr_path=Path("/tmp/rec_001_analysis.zarr"),
        run_name="detection_profile_2026-02-24_10-25-25",
        created_at_utc="2026-02-24T10:25:25+00:00",
        source_detection_path="detect_runs/detect_001",
    )

    assert result.run_name == "detection_profile_2026-02-24_10-25-25"
    assert runs_parent.attrs["latest"] == "detection_profile_2026-02-24_10-25-25"
    assert "existing_run" in runs_parent
    assert "detection_profile_2026-02-24_10-25-25" in runs_parent


def test_write_detection_profile_falls_back_to_open_group_lookup(monkeypatch) -> None:
    root = _make_detect_root()
    analysis = root.create_group("analysis")
    runs_parent = analysis.create_group("detection_profile_runs")
    runs_parent.attrs["latest"] = "existing_run"
    runs_parent.create_group("existing_run")
    root._children["analysis"] = _FlakyLookupGroup(
        children=analysis._children,
        attrs=analysis.attrs,
        store=analysis.store,
        path=analysis.path,
        miss_key="detection_profile_runs",
    )

    opened_paths: list[str] = []

    def _fake_open_group(*_args, **kwargs):
        path = kwargs.get("path")
        if isinstance(path, str):
            opened_paths.append(path)
        if path == "analysis/detection_profile_runs":
            return runs_parent
        raise KeyError(path)

    monkeypatch.setattr(detection_profile_mod.zarr, "open_group", _fake_open_group)

    result = write_detection_profile(
        root,
        zarr_path=Path("/tmp/rec_001_analysis.zarr"),
        run_name="detection_profile_2026-02-24_10-30-30",
        created_at_utc="2026-02-24T10:30:30+00:00",
        source_detection_path="detect_runs/detect_001",
    )

    assert result.run_name == "detection_profile_2026-02-24_10-30-30"
    assert "analysis/detection_profile_runs" in opened_paths
    assert runs_parent.attrs["latest"] == "detection_profile_2026-02-24_10-30-30"
    assert "existing_run" in runs_parent
    assert "detection_profile_2026-02-24_10-30-30" in runs_parent


def test_build_detection_profile_summary_prefers_manual_refined_group() -> None:
    root = _make_detect_root()
    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"

    refined = refined_parent.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    refined.attrs["manual_review_latest"] = "manual_review_001"
    refined.attrs["detect_review_status"] = {
        "state": "approved",
        "method": "manual",
        "intended_use": "training",
        "timestamp": "2026-02-24T10:15:00+00:00",
    }
    manual = refined.create_group("manual_review_001")
    manual.create_array("frame_indices", data=np.asarray([0, 1, 2], dtype=np.int32))
    manual.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [
                [0.3, 0.3, 0.1, 0.1],
                [0.4, 0.4, 0.1, 0.1],
                [0.5, 0.5, 0.1, 0.1],
            ],
            dtype=np.float64,
        ),
    )
    manual.create_array("frame_counts", data=np.asarray([1, 1, 1], dtype=np.int32))

    summary = build_detection_profile_summary(
        root,
        zarr_path=Path("/tmp/rec_001_analysis.zarr"),
        created_at_utc="2026-02-24T10:20:00+00:00",
    )
    source = summary["source"]

    assert source["detection_path"] == "refined_detect_runs/refined_detect_001/manual_review_001"
    assert source["detection_type"] == "manual"
    assert source["manual_group"] == "manual_review_001"
    assert source["review_state"] == "approved"
    assert source["review_method"] == "manual"
    assert source["review_intended_use"] == "training"
    assert source["review_timestamp_utc"] == "2026-02-24T10:15:00+00:00"


def test_backfill_detection_profiles_cli_dry_run_and_apply(capsys, monkeypatch, tmp_path: Path) -> None:
    zarr_ok = tmp_path / "ok_analysis.zarr"
    zarr_missing = tmp_path / "missing_analysis.zarr"

    root_ok = _make_detect_root()
    root_missing = _FakeGroup()
    root_missing.attrs["zarr_purpose"] = "analysis"

    mapping = {
        zarr_ok: root_ok,
        zarr_missing: root_missing,
    }

    monkeypatch.setattr(backfill_mod, "_iter_zarr", lambda *_args, **_kwargs: iter(mapping.keys()))
    monkeypatch.setattr(
        backfill_mod.zarr,
        "open_group",
        lambda path, mode="r": mapping[Path(path)],  # noqa: ARG005
    )

    dry_rc = backfill_mod.main([str(tmp_path)])
    assert dry_rc == 0
    dry_out = capsys.readouterr().out
    assert "would_write" in dry_out
    assert "missing_source" in dry_out
    assert "would_write=1" in dry_out
    assert "errors=0" in dry_out
    assert root_ok.get("analysis/detection_profile_runs") is None

    apply_rc = backfill_mod.main(
        [
            str(tmp_path),
            "--apply",
            "--run-name",
            "detection_profile_fixed",
        ]
    )
    assert apply_rc == 0
    apply_out = capsys.readouterr().out
    assert "updated" in apply_out
    assert "missing_source=1" in apply_out
    assert "updated=1" in apply_out
    assert root_ok["analysis/detection_profile_runs"].attrs["latest"] == "detection_profile_fixed"
    assert "profile_summary" in root_ok["analysis/detection_profile_runs/detection_profile_fixed"].attrs


def test_backfill_detection_profiles_populates_dataset_identity_from_registry(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "identity_training.zarr"
    root = _make_detect_root()
    root.attrs.pop("recording_id", None)
    root.attrs["zarr_purpose"] = "training"
    mapping = {zarr_path: root}

    monkeypatch.setattr(backfill_mod, "_iter_zarr", lambda *_args, **_kwargs: iter(mapping.keys()))
    monkeypatch.setattr(
        backfill_mod.zarr,
        "open_group",
        lambda path, mode="r": mapping[Path(path)],  # noqa: ARG005
    )

    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    registry.upsert_dataset(
        "dataset_identity",
        session_uuid="session_identity",
        zarr_path=zarr_path,
        recording_id="recording_identity",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.close()

    rc = backfill_mod.main(
        [
            str(zarr_path),
            "--apply",
            "--registry",
            str(registry_path),
            "--run-name",
            "detection_profile_identity",
            "--zarr-use",
            "any",
        ]
    )
    assert rc == 0

    run_group = root["analysis/detection_profile_runs/detection_profile_identity"]
    profile_summary = run_group.attrs["profile_summary"]
    assert profile_summary["dataset"]["dataset_id"] == "dataset_identity"
    assert profile_summary["dataset"]["recording_id"] == "recording_identity"
    assert profile_summary["dataset"]["zarr_use"] == "training"
