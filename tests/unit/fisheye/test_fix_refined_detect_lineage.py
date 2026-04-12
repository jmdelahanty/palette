from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fisheye.utils import fix_refined_detect_lineage as mod


class _FakeArray:
    def __init__(self, shape: tuple[int, ...] = (1,)) -> None:
        self.shape = shape


class _FakeGroup:
    def __init__(self, children: dict[str, "_FakeGroup | _FakeArray"] | None = None) -> None:
        self._children: dict[str, _FakeGroup | _FakeArray] = children or {}
        self.attrs: dict[str, Any] = {}

    def create_group(self, name: str) -> "_FakeGroup":
        child = _FakeGroup()
        self._children[name] = child
        return child

    def create_array(self, name: str, shape: tuple[int, ...] = (1,)) -> _FakeArray:
        arr = _FakeArray(shape=shape)
        self._children[name] = arr
        return arr

    def get(self, name: str) -> "_FakeGroup | _FakeArray | None":
        return self._children.get(name)

    def group_keys(self):  # pragma: no cover - exercised by production helpers
        return [key for key, value in self._children.items() if isinstance(value, _FakeGroup)]

    def keys(self):  # pragma: no cover - exercised by production helpers
        return self._children.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._children

    def __getitem__(self, key: str) -> "_FakeGroup | _FakeArray":
        return self._children[key]


def _patch_scan(monkeypatch, mapping: dict[Path, _FakeGroup]) -> None:
    ordered_paths = list(mapping.keys())
    monkeypatch.setattr(mod, "_iter_zarr", lambda *_args, **_kwargs: iter(ordered_paths))
    monkeypatch.setattr(mod.zarr, "open_group", lambda path, mode="r": mapping[Path(path)])  # noqa: ARG005


def _add_refined_run(
    root: _FakeGroup,
    run_name: str,
    *,
    valid: bool,
) -> None:
    parent = root.get("refined_detect_runs")
    if parent is None:
        parent = root.create_group("refined_detect_runs")
    assert isinstance(parent, _FakeGroup)
    run = parent.create_group(run_name)
    manual = run.create_group("manual")
    manual.create_array("frame_indices", shape=(1,))
    if valid:
        manual.create_array("bbox_norm_coords", shape=(1, 4))


def _add_sparse_refined_run(root: _FakeGroup, run_name: str) -> None:
    parent = root.get("refined_detect_runs")
    if parent is None:
        parent = root.create_group("refined_detect_runs")
    assert isinstance(parent, _FakeGroup)
    run = parent.create_group(run_name)
    instances = run.create_group("instances")
    for name, shape in (
        ("refined_row_ids", (1,)),
        ("frame_indices", (1,)),
        ("frame_offsets", (2,)),
        ("bbox_img_xyxy", (1, 4)),
        ("bbox_norm_coords", (1, 4)),
        ("source_kind_codes", (1,)),
        ("manual_edit_flags", (1,)),
        ("source_detect_row_index", (1,)),
        ("frame_counts", (1,)),
    ):
        instances.create_array(name, shape=shape)


def _set_refined_latest(root: _FakeGroup, run_name: str) -> None:
    parent = root["refined_detect_runs"]
    assert isinstance(parent, _FakeGroup)
    parent.attrs["latest"] = run_name


def _add_crop_run(
    root: _FakeGroup,
    run_name: str,
    *,
    detection_source_path: str | None,
    source_refined_run: str | None,
    detection_source_type: str | None,
    source_coords_path: str | None = None,
    crop_signature: dict[str, Any] | None = None,
) -> None:
    parent = root.get("crop_runs")
    if parent is None:
        parent = root.create_group("crop_runs")
    assert isinstance(parent, _FakeGroup)
    run = parent.create_group(run_name)
    if detection_source_path is not None:
        run.attrs["detection_source_path"] = detection_source_path
    if source_refined_run is not None:
        run.attrs["source_refined_run"] = source_refined_run
    if detection_source_type is not None:
        run.attrs["detection_source_type"] = detection_source_type
    if source_coords_path is not None:
        run.attrs["source_coords_path"] = source_coords_path
    if crop_signature is not None:
        run.attrs["crop_signature"] = crop_signature
    parent.attrs["latest"] = run_name


def test_main_dry_run_prefers_crop_referenced_valid_run_for_stale_latest(monkeypatch, capsys, tmp_path: Path) -> None:
    zarr_path = tmp_path / "rec_analysis.zarr"
    root = _FakeGroup()
    root.attrs["zarr_use"] = "analysis"
    _add_refined_run(root, "refined_detect_old", valid=True)
    _add_refined_run(root, "refined_detect_new", valid=False)
    _set_refined_latest(root, "refined_detect_new")
    _add_crop_run(
        root,
        "crop_001",
        detection_source_path="refined_detect_runs/refined_detect_old/manual",
        source_refined_run="refined_detect_old",
        detection_source_type="manual",
    )
    _patch_scan(monkeypatch, {zarr_path: root})

    rc = mod.main([str(zarr_path), "--dry-run"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "plan:" in out
    assert "refined_detect_runs latest 'refined_detect_new' -> 'refined_detect_old'" in out

    refined_parent = root["refined_detect_runs"]
    assert isinstance(refined_parent, _FakeGroup)
    assert refined_parent.attrs["latest"] == "refined_detect_new"


def test_main_apply_aligns_crop_source_with_canonical_latest(monkeypatch, capsys, tmp_path: Path) -> None:
    zarr_path = tmp_path / "rec_analysis.zarr"
    root = _FakeGroup()
    root.attrs["zarr_use"] = "analysis"
    _add_refined_run(root, "refined_detect_old", valid=True)
    _add_refined_run(root, "refined_detect_new", valid=True)
    _set_refined_latest(root, "refined_detect_new")
    _add_crop_run(
        root,
        "crop_001",
        detection_source_path="refined_detect_runs/refined_detect_old/manual",
        source_refined_run="refined_detect_old",
        detection_source_type="manual",
    )
    _patch_scan(monkeypatch, {zarr_path: root})

    rc = mod.main([str(zarr_path), "--apply"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "apply:" in out
    assert "applied_repairs: 3" in out

    crop_parent = root["crop_runs"]
    assert isinstance(crop_parent, _FakeGroup)
    crop = crop_parent["crop_001"]
    assert isinstance(crop, _FakeGroup)
    assert crop.attrs["detection_source_path"] == "refined_detect_runs/refined_detect_new/manual"
    assert crop.attrs["source_coords_path"] == "refined_detect_runs/refined_detect_new/manual"
    assert crop.attrs["source_refined_run"] == "refined_detect_new"
    assert crop.attrs["detection_source_type"] == "manual"


def test_main_zarr_use_filter_only_modifies_selected_archives(monkeypatch, capsys, tmp_path: Path) -> None:
    analysis_path = tmp_path / "analysis.zarr"
    training_path = tmp_path / "training.zarr"

    analysis_root = _FakeGroup()
    analysis_root.attrs["zarr_use"] = "analysis"
    _add_refined_run(analysis_root, "refined_detect_old", valid=True)
    _add_refined_run(analysis_root, "refined_detect_new", valid=True)
    _set_refined_latest(analysis_root, "refined_detect_new")
    _add_crop_run(
        analysis_root,
        "crop_001",
        detection_source_path="refined_detect_runs/refined_detect_old/manual",
        source_refined_run="refined_detect_old",
        detection_source_type="manual",
    )

    training_root = _FakeGroup()
    training_root.attrs["zarr_use"] = "training"
    _add_refined_run(training_root, "refined_detect_old", valid=True)
    _add_refined_run(training_root, "refined_detect_new", valid=True)
    _set_refined_latest(training_root, "refined_detect_new")
    _add_crop_run(
        training_root,
        "crop_001",
        detection_source_path="refined_detect_runs/refined_detect_old/manual",
        source_refined_run="refined_detect_old",
        detection_source_type="manual",
    )

    _patch_scan(monkeypatch, {analysis_path: analysis_root, training_path: training_root})

    rc = mod.main([str(tmp_path), "--recursive", "--zarr-use", "analysis", "--apply"])
    assert rc == 0
    capsys.readouterr()

    analysis_crop = analysis_root["crop_runs"]["crop_001"]
    assert isinstance(analysis_crop, _FakeGroup)
    assert analysis_crop.attrs["source_refined_run"] == "refined_detect_new"

    training_crop = training_root["crop_runs"]["crop_001"]
    assert isinstance(training_crop, _FakeGroup)
    assert training_crop.attrs["source_refined_run"] == "refined_detect_old"


def test_main_writes_json_report_in_dry_run_mode(monkeypatch, capsys, tmp_path: Path) -> None:
    zarr_path = tmp_path / "rec_analysis.zarr"
    report_path = tmp_path / "reports" / "lineage_dry_run.json"
    root = _FakeGroup()
    root.attrs["zarr_use"] = "analysis"
    _add_refined_run(root, "refined_detect_old", valid=True)
    _add_refined_run(root, "refined_detect_new", valid=False)
    _set_refined_latest(root, "refined_detect_new")
    _add_crop_run(
        root,
        "crop_001",
        detection_source_path="refined_detect_runs/refined_detect_old/manual",
        source_refined_run="refined_detect_old",
        detection_source_type="manual",
    )
    _patch_scan(monkeypatch, {zarr_path: root})

    rc = mod.main([str(zarr_path), "--dry-run", "--json-report", str(report_path)])
    assert rc == 0
    capsys.readouterr()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "dry-run"
    assert payload["summary"]["zarr_with_changes"] == 1
    assert payload["summary"]["planned_repairs"] == 2
    assert payload["summary"]["applied_repairs"] == 0
    assert len(payload["repairs"]) == 2
    fields = {repair["field"]: repair for repair in payload["repairs"]}
    assert set(fields.keys()) == {"latest", "source_coords_path"}
    assert fields["latest"]["zarr_path"] == str(zarr_path)
    assert fields["latest"]["target_path"] == "refined_detect_runs"
    assert fields["latest"]["old_value"] == "refined_detect_new"
    assert fields["latest"]["new_value"] == "refined_detect_old"
    assert fields["latest"]["applied"] is False
    assert fields["source_coords_path"]["target_path"] == "crop_runs/crop_001"
    assert fields["source_coords_path"]["new_value"] == "refined_detect_runs/refined_detect_old/manual"
    assert fields["source_coords_path"]["applied"] is False


def test_main_writes_json_report_with_applied_flags(monkeypatch, capsys, tmp_path: Path) -> None:
    zarr_path = tmp_path / "rec_analysis.zarr"
    report_path = tmp_path / "reports" / "lineage_apply.json"
    root = _FakeGroup()
    root.attrs["zarr_use"] = "analysis"
    _add_refined_run(root, "refined_detect_old", valid=True)
    _add_refined_run(root, "refined_detect_new", valid=True)
    _set_refined_latest(root, "refined_detect_new")
    _add_crop_run(
        root,
        "crop_001",
        detection_source_path="refined_detect_runs/refined_detect_old/manual",
        source_refined_run="refined_detect_old",
        detection_source_type="manual",
    )
    _patch_scan(monkeypatch, {zarr_path: root})

    rc = mod.main([str(zarr_path), "--apply", "--json-report", str(report_path)])
    assert rc == 0
    capsys.readouterr()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "apply"
    assert payload["summary"]["planned_repairs"] == 3
    assert payload["summary"]["applied_repairs"] == 3
    fields = {repair["field"]: repair for repair in payload["repairs"]}
    assert set(fields.keys()) == {"detection_source_path", "source_coords_path", "source_refined_run"}
    assert fields["detection_source_path"]["applied"] is True
    assert fields["source_coords_path"]["applied"] is True
    assert fields["source_refined_run"]["applied"] is True


def test_main_apply_normalizes_sparse_refined_root_alias_paths(monkeypatch, capsys, tmp_path: Path) -> None:
    zarr_path = tmp_path / "rec_analysis.zarr"
    root = _FakeGroup()
    root.attrs["zarr_use"] = "analysis"
    _add_sparse_refined_run(root, "refined_detect_new")
    _set_refined_latest(root, "refined_detect_new")
    _add_crop_run(
        root,
        "crop_001",
        detection_source_path="refined_detect_runs/refined_detect_new",
        source_coords_path="refined_detect_runs/refined_detect_new",
        source_refined_run="refined_detect_new",
        detection_source_type="refined",
        crop_signature={
            "signature_version": 2,
            "detection_source_path": "refined_detect_runs/refined_detect_new",
            "source_coords_path": "refined_detect_runs/refined_detect_new",
            "detection_source_type": "refined",
            "source_refined_run": "refined_detect_new",
        },
    )
    _patch_scan(monkeypatch, {zarr_path: root})

    rc = mod.main([str(zarr_path), "--apply"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "applied_repairs: 3" in out

    crop_parent = root["crop_runs"]
    assert isinstance(crop_parent, _FakeGroup)
    crop = crop_parent["crop_001"]
    assert isinstance(crop, _FakeGroup)
    assert crop.attrs["detection_source_path"] == "refined_detect_runs/refined_detect_new/instances"
    assert crop.attrs["source_coords_path"] == "refined_detect_runs/refined_detect_new/instances"
    assert crop.attrs["crop_signature"]["detection_source_path"] == (
        "refined_detect_runs/refined_detect_new/instances"
    )
    assert crop.attrs["crop_signature"]["source_coords_path"] == (
        "refined_detect_runs/refined_detect_new/instances"
    )
