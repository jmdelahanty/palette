from __future__ import annotations

import multiprocessing
from pathlib import Path
import threading
from typing import Any
import warnings

import numpy as np
import pytest
import zarr

from fisheye.shared import zarr_helpers as zarr_helpers_module
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    archive_metadata_publication_lock_path,
    consolidate_metadata_capture_expected_warnings,
    first_array_length,
    first_array_length_in_group,
    infer_zarr_use,
    normalize_zarr_path,
    read_zarr_array_mapping,
    reconsolidate_zarr_metadata,
    resolve_zarr_run,
    safe_int,
    zarr_array_names,
    zarr_attrs_dict,
    zarr_child_group,
    zarr_group_keys,
)
from fisheye.shared.zarr_run_completion import set_authoritative_run


class _FakeArray:
    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values)
        self.shape = self._values.shape

    def __getitem__(self, key: object) -> np.ndarray:
        return self._values[key]


def _acquire_archive_lock_in_fork(
    archive: str,
    sender: Any,
) -> None:
    with archive_metadata_publication_lock(archive):
        sender.send("acquired")
    sender.close()


def _signal_fork_completed(sender: Any) -> None:
    sender.send("forked")
    sender.close()


class _FakeGroup:
    def __init__(self, *, path: str = "") -> None:
        self._children: dict[str, Any] = {}
        self.attrs: dict[str, Any] = {}
        self.path = path

    def create_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).create_group(tail)
        if name in self._children:
            raise ValueError(f"{name} already exists")
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def create_array(self, name: str, values: np.ndarray) -> _FakeArray:
        array = _FakeArray(values)
        self._children[name] = array
        return array

    def require_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).require_group(tail)
        existing = self._children.get(name)
        if existing is not None:
            return existing
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def get(self, name: str):
        try:
            return self[name]
        except Exception:
            return None

    def group_keys(self):
        return [name for name, value in self._children.items() if isinstance(value, _FakeGroup)]

    def keys(self):
        return list(self._children.keys())

    def __contains__(self, key: str) -> bool:
        try:
            _ = self[key]
            return True
        except Exception:
            return False

    def __getitem__(self, key: str) -> Any:
        if "/" in key:
            current: Any = self
            for token in key.split("/"):
                current = current._children[token]
            return current
        return self._children[key]


def _build_root() -> _FakeGroup:
    root = _FakeGroup()
    runs = root.require_group("analysis/stimulus_runs")
    runs.create_group("stimulus_001")
    runs.create_group("stimulus_002")
    runs.create_group("stimulus_003")
    return root


def test_zarr_reader_helpers_normalize_attrs_groups_and_arrays() -> None:
    root = _FakeGroup()
    group = root.require_group("analysis/example/run_1")
    group.attrs[1] = "one"
    group.create_array("b_array", np.asarray([1, 2, 3], dtype=np.int32))
    group.create_array("a_array", np.asarray([[1.0], [2.0]], dtype=np.float32))
    group.create_group("child")

    assert normalize_zarr_path("/analysis//example/run_1/") == "analysis/example/run_1"
    assert zarr_attrs_dict(group) == {"1": "one"}
    assert zarr_group_keys(group) == ["child"]
    assert zarr_child_group(root, "analysis/example/run_1/child") is group["child"]
    assert zarr_child_group(root, "analysis/example/missing") is None
    assert zarr_array_names(group) == ["a_array", "b_array"]
    assert safe_int("7") == 7
    assert safe_int("not-int") is None


def test_infer_zarr_use_prefers_canonical_store_purpose_attr() -> None:
    root = _FakeGroup()
    root.attrs["zarr_use"] = "training"
    root.attrs["zarr_purpose"] = "analysis"

    assert infer_zarr_use(root, Path("recording_training.zarr")) == "analysis"


def test_infer_zarr_use_accepts_legacy_purpose_and_filename_fallback() -> None:
    root = _FakeGroup()
    root.attrs["zarr_purpose"] = "analysis"

    assert infer_zarr_use(root, Path("unknown.zarr")) == "analysis"
    assert infer_zarr_use({}, Path("recording_training.zarr")) == "training"
    assert infer_zarr_use({}, Path("recording.zarr"), default="unknown") == "unknown"


def test_infer_zarr_use_ignores_invalid_attrs_before_suffix_fallback() -> None:
    root = _FakeGroup()
    root.attrs["zarr_use"] = "not-a-use"
    root.attrs["zarr_purpose"] = "also-not-a-use"

    assert infer_zarr_use(root, Path("recording_analysis.zarr")) == "analysis"


def test_infer_zarr_use_can_accept_extended_or_arbitrary_vocab() -> None:
    root = _FakeGroup()
    root.attrs["zarr_use"] = "custom_review"

    assert infer_zarr_use(root, Path("recording.zarr")) is None
    assert infer_zarr_use(root, Path("recording.zarr"), valid_uses=None) == "custom_review"


def test_read_zarr_array_mapping_records_logical_source_paths() -> None:
    root = _FakeGroup()
    group = root.require_group("analysis/example/run_1")
    group.create_array("kept", np.asarray([1, 2, 3], dtype=np.int64))
    group.create_array("skipped", np.asarray([4, 5, 6], dtype=np.int64))
    source_paths: dict[str, str] = {}

    arrays = read_zarr_array_mapping(
        group,
        physical_prefix="analysis/example/run_1",
        logical_prefix="logical/run",
        source_paths=source_paths,
        array_names=("kept", "missing"),
    )

    assert list(arrays) == ["kept"]
    np.testing.assert_array_equal(arrays["kept"], [1, 2, 3])
    assert source_paths == {"logical/run/kept": "analysis/example/run_1/kept"}
    assert first_array_length(arrays, ("missing", "kept")) == 3
    assert first_array_length_in_group(group, ("missing", "skipped")) == 3


def test_resolve_zarr_run_uses_explicit_run_name() -> None:
    root = _build_root()

    run_group, run_name = resolve_zarr_run(
        root,
        "analysis/stimulus_runs",
        "stimulus_002",
        run_label="Stimulus run",
    )

    assert run_name == "stimulus_002"
    assert run_group.path == "analysis/stimulus_runs/stimulus_002"


def test_resolve_zarr_run_uses_latest_attr_and_latest_alias() -> None:
    root = _build_root()
    root["analysis/stimulus_runs"].attrs["latest"] = b"stimulus_003"

    run_group, run_name = resolve_zarr_run(
        root,
        ("analysis", "stimulus_runs"),
        "latest",
        latest_aliases=("latest",),
        run_label="Stimulus run",
    )

    assert run_name == "stimulus_003"
    assert run_group.path == "analysis/stimulus_runs/stimulus_003"


def test_resolve_zarr_run_prefers_authoritative_run_over_later_latest() -> None:
    root = _build_root()
    parent = root["analysis/stimulus_runs"]
    parent.attrs["latest"] = "stimulus_003"
    set_authoritative_run(parent, "stimulus_002", approved_by="jeremy")

    run_group, run_name = resolve_zarr_run(
        root,
        ("analysis", "stimulus_runs"),
        None,
        run_label="Stimulus run",
    )

    assert run_name == "stimulus_002"
    assert run_group.path == "analysis/stimulus_runs/stimulus_002"


def test_resolve_zarr_run_falls_back_to_sorted_last() -> None:
    root = _build_root()

    run_group, run_name = resolve_zarr_run(
        root,
        "analysis/stimulus_runs",
        None,
        fallback_to_sorted="last",
        run_label="Stimulus run",
    )

    assert run_name == "stimulus_003"
    assert run_group.path == "analysis/stimulus_runs/stimulus_003"


def test_resolve_zarr_run_sorted_fallback_skips_explicit_nonselector() -> None:
    root = _build_root()
    parent = root["analysis/stimulus_runs"]
    parent["stimulus_003"].attrs["stage_selector_eligible"] = False

    run_group, run_name = resolve_zarr_run(
        root,
        "analysis/stimulus_runs",
        None,
        fallback_to_sorted="last",
        run_label="Stimulus run",
    )

    assert run_name == "stimulus_002"
    assert run_group.path == "analysis/stimulus_runs/stimulus_002"


def test_resolve_zarr_run_falls_back_to_sorted_first() -> None:
    root = _build_root()

    run_group, run_name = resolve_zarr_run(
        root,
        "analysis/stimulus_runs",
        None,
        fallback_to_sorted="first",
        run_label="Stimulus run",
    )

    assert run_name == "stimulus_001"
    assert run_group.path == "analysis/stimulus_runs/stimulus_001"


def test_resolve_zarr_run_reports_missing_run_with_available_names() -> None:
    root = _build_root()

    with pytest.raises(ValueError, match="Stimulus run 'stimulus_999' not found under analysis/stimulus_runs"):
        resolve_zarr_run(
            root,
            "analysis/stimulus_runs",
            "stimulus_999",
            run_label="Stimulus run",
        )


def test_resolve_zarr_run_reports_missing_parent() -> None:
    root = _FakeGroup()

    with pytest.raises(ValueError, match="analysis/stimulus_runs not found in store"):
        resolve_zarr_run(
            root,
            "analysis/stimulus_runs",
            None,
            run_label="Stimulus run",
        )


def test_resolve_zarr_run_uses_direct_filesystem_group_for_explicit_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _FakeGroup()
    root.require_group("analysis/stimulus_runs")
    root._palette_fs_path = str(tmp_path / "archive.zarr")  # type: ignore[attr-defined]
    root._palette_open_mode = "r"  # type: ignore[attr-defined]
    direct_group_path = tmp_path / "archive.zarr" / "analysis" / "stimulus_runs" / "stimulus_777"
    direct_group_path.mkdir(parents=True)
    (direct_group_path / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group", "attributes": {}}')
    sentinel = _FakeGroup(path="analysis/stimulus_runs/stimulus_777")

    def _fake_open_group(path: str, *, mode: str, use_consolidated: bool) -> _FakeGroup:
        assert Path(path) == direct_group_path
        assert mode == "r"
        assert use_consolidated is False
        return sentinel

    monkeypatch.setattr(zarr, "open_group", _fake_open_group)

    run_group, run_name = resolve_zarr_run(
        root,
        "analysis/stimulus_runs",
        "stimulus_777",
        run_label="Stimulus run",
    )

    assert run_name == "stimulus_777"
    assert run_group is sentinel


def test_resolve_zarr_run_uses_direct_filesystem_group_for_latest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _FakeGroup()
    parent = root.require_group("analysis/stimulus_runs")
    parent.attrs["latest"] = "stimulus_888"
    root._palette_fs_path = str(tmp_path / "archive.zarr")  # type: ignore[attr-defined]
    root._palette_open_mode = "r"  # type: ignore[attr-defined]
    direct_group_path = tmp_path / "archive.zarr" / "analysis" / "stimulus_runs" / "stimulus_888"
    direct_group_path.mkdir(parents=True)
    (direct_group_path / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group", "attributes": {}}')
    sentinel = _FakeGroup(path="analysis/stimulus_runs/stimulus_888")

    def _fake_open_group(path: str, *, mode: str, use_consolidated: bool) -> _FakeGroup:
        assert Path(path) == direct_group_path
        assert mode == "r"
        assert use_consolidated is False
        return sentinel

    monkeypatch.setattr(zarr, "open_group", _fake_open_group)

    run_group, run_name = resolve_zarr_run(
        root,
        "analysis/stimulus_runs",
        None,
        run_label="Stimulus run",
    )

    assert run_name == "stimulus_888"
    assert run_group is sentinel


def test_resolve_zarr_run_uses_store_path_file_uri_when_palette_annotation_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _FakeGroup()
    root.require_group("analysis/stimulus_runs")
    root.store_path = f"file://{tmp_path / 'archive.zarr'}"  # type: ignore[attr-defined]
    direct_group_path = tmp_path / "archive.zarr" / "analysis" / "stimulus_runs" / "stimulus_999"
    direct_group_path.mkdir(parents=True)
    (direct_group_path / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group", "attributes": {}}')
    sentinel = _FakeGroup(path="analysis/stimulus_runs/stimulus_999")

    def _fake_open_group(path: str, *, mode: str, use_consolidated: bool) -> _FakeGroup:
        assert Path(path) == direct_group_path
        assert mode == "r"
        assert use_consolidated is False
        return sentinel

    monkeypatch.setattr(zarr, "open_group", _fake_open_group)

    run_group, run_name = resolve_zarr_run(
        root,
        "analysis/stimulus_runs",
        "stimulus_999",
        run_label="Stimulus run",
    )

    assert run_name == "stimulus_999"
    assert run_group is sentinel


def test_reconsolidate_zarr_metadata_records_attrs_and_calls_zarr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = _FakeGroup(path="detect_runs/detect_001/quality_reports")
    opened: list[tuple[Path, str]] = []
    consolidated: list[tuple[str, str | None]] = []

    def _fake_open(path: str | Path, *, mode: str) -> _FakeGroup:
        opened.append((Path(path), mode))
        return target

    def _fake_consolidate(store: str, *, path: str | None = None):
        consolidated.append((store, path))
        return target

    monkeypatch.setattr("fisheye.shared.zarr_helpers.open_zarr_group_direct", _fake_open)
    monkeypatch.setattr(zarr, "consolidate_metadata", _fake_consolidate)

    zarr_path = tmp_path / "archive.zarr"
    report = reconsolidate_zarr_metadata(
        zarr_path,
        group_path="detect_runs/detect_001/quality_reports",
        policy="unit_test",
    )

    assert report["status"] == "ok"
    assert report["zarr_path"] == str(zarr_path.resolve())
    assert report["group_path"] == "detect_runs/detect_001/quality_reports"
    assert opened == [
        (
            zarr_path.resolve() / "detect_runs" / "detect_001" / "quality_reports",
            "r+",
        )
    ]
    assert consolidated == [
        (
            str(zarr_path.resolve()),
            "detect_runs/detect_001/quality_reports",
        )
    ]
    assert target.attrs["metadata_consolidation_policy"] == "unit_test"
    assert target.attrs["metadata_consolidation_status"] == "ok"
    assert target.attrs["metadata_consolidation_group_path"] == "detect_runs/detect_001/quality_reports"
    assert report["suppressed_expected_warning_count"] == 0
    assert report["unexpected_warning_count"] == 0


def test_consolidate_metadata_capture_expected_warnings_suppresses_sidecars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str | None]] = []

    def _fake_consolidate(store: str, *, path: str | None = None) -> None:
        calls.append((store, path))
        warnings.warn(
            "Object at .failed is not recognized as a component of a Zarr hierarchy.",
            UserWarning,
            stacklevel=2,
        )
        warnings.warn(
            "Object at .incoming is not recognized as a component of a Zarr hierarchy.",
            UserWarning,
            stacklevel=2,
        )

    monkeypatch.setattr(zarr, "consolidate_metadata", _fake_consolidate)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = consolidate_metadata_capture_expected_warnings(
            tmp_path / "archive.zarr",
            path="analysis",
        )

    assert caught == []
    assert calls == [(str(tmp_path / "archive.zarr"), "analysis")]
    assert report["suppressed_expected_warning_count"] == 2
    assert report["unexpected_warning_count"] == 0
    assert report["suppressed_expected_warning_messages"] == [
        "Object at .failed is not recognized as a component of a Zarr hierarchy.",
        "Object at .incoming is not recognized as a component of a Zarr hierarchy.",
    ]


def test_consolidate_metadata_capture_expected_warnings_reemits_unexpected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_consolidate(store: str, *, path: str | None = None) -> None:
        warnings.warn(
            "unexpected metadata warning",
            UserWarning,
            stacklevel=2,
        )

    monkeypatch.setattr(zarr, "consolidate_metadata", _fake_consolidate)

    with pytest.warns(UserWarning, match="unexpected metadata warning"):
        report = consolidate_metadata_capture_expected_warnings(tmp_path / "archive.zarr")

    assert report["suppressed_expected_warning_count"] == 0
    assert report["unexpected_warning_count"] == 1


def test_consolidate_metadata_walks_direct_tree_below_stale_nested_cache(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zarr"
    root = zarr.open_group(str(archive), mode="w")
    root.require_group("analysis").require_group("streams").require_group("crop")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zarr.consolidate_metadata(str(archive), path="analysis/streams/crop")
        zarr.consolidate_metadata(str(archive), path="analysis")
        zarr.consolidate_metadata(str(archive))

    direct = zarr.open_group(str(archive), mode="r+", use_consolidated=False)
    direct_crop = direct["analysis/streams/crop"]
    direct_crop.require_group("ledger_runs").create_group("crop_ledger_new")
    stale = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    assert stale["analysis/streams/crop"].get("ledger_runs") is None

    report = consolidate_metadata_capture_expected_warnings(archive)

    consolidated = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    assert "crop_ledger_new" in consolidated["analysis/streams/crop/ledger_runs"]
    assert report["suppressed_expected_warning_count"] == 0
    assert report["unexpected_warning_count"] == 0


def test_archive_lock_resets_inherited_reentrancy_state_after_fork(
    tmp_path: Path,
) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork start method is unavailable")
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    archive = tmp_path / "archive.zarr"
    process = context.Process(
        target=_acquire_archive_lock_in_fork,
        args=(str(archive), sender),
    )
    try:
        with archive_metadata_publication_lock(archive):
            process.start()
            sender.close()
            assert not receiver.poll(0.25)
        assert receiver.poll(5)
        assert receiver.recv() == "acquired"
        process.join(timeout=5)
        assert process.exitcode == 0
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def test_archive_lock_resolves_symlink_aliases_to_one_lock(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zarr"
    archive.mkdir()
    alias = tmp_path / "archive-alias.zarr"
    alias.symlink_to(archive, target_is_directory=True)

    assert archive_metadata_publication_lock_path(alias) == (
        archive_metadata_publication_lock_path(archive)
    )
    with archive_metadata_publication_lock(archive) as direct_lock:
        with archive_metadata_publication_lock(alias) as alias_lock:
            assert alias_lock == direct_lock


def test_archive_lock_drops_handle_inherited_from_other_thread(
    tmp_path: Path,
) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork start method is unavailable")
    archive = tmp_path / "archive.zarr"
    held = threading.Event()
    release = threading.Event()

    def hold_parent_lock() -> None:
        with archive_metadata_publication_lock(archive):
            held.set()
            assert release.wait(5)

    thread = threading.Thread(target=hold_parent_lock)
    thread.start()
    assert held.wait(5)
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_acquire_archive_lock_in_fork,
        args=(str(archive), sender),
    )
    try:
        process.start()
        sender.close()
        assert not receiver.poll(0.25)
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert receiver.poll(5)
        assert receiver.recv() == "acquired"
        process.join(timeout=5)
        assert process.exitcode == 0
    finally:
        release.set()
        thread.join(timeout=5)
        receiver.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def test_archive_lock_open_and_registration_are_atomic_with_fork(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork start method is unavailable")
    archive = tmp_path / "archive.zarr"
    lock_path = archive_metadata_publication_lock_path(archive)
    opened = threading.Event()
    allow_registration = threading.Event()
    parent_held = threading.Event()
    release_parent = threading.Event()
    original_open = Path.open

    def delay_lock_open(path: Path, *args: Any, **kwargs: Any):
        handle = original_open(path, *args, **kwargs)
        if path == lock_path:
            opened.set()
            assert allow_registration.wait(5)
        return handle

    monkeypatch.setattr(Path, "open", delay_lock_open)

    def hold_parent_lock() -> None:
        with archive_metadata_publication_lock(archive):
            parent_held.set()
            assert release_parent.wait(5)

    lock_thread = threading.Thread(target=hold_parent_lock)
    lock_thread.start()
    assert opened.wait(5)

    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_signal_fork_completed, args=(sender,))
    fork_returned = threading.Event()

    def start_child() -> None:
        process.start()
        sender.close()
        fork_returned.set()

    fork_thread = threading.Thread(target=start_child)
    fork_thread.start()
    try:
        assert not fork_returned.wait(0.25)
        allow_registration.set()
        assert parent_held.wait(5)
        assert fork_returned.wait(5)
        assert receiver.poll(5)
        assert receiver.recv() == "forked"
        process.join(timeout=5)
        assert process.exitcode == 0
    finally:
        allow_registration.set()
        release_parent.set()
        lock_thread.join(timeout=5)
        fork_thread.join(timeout=5)
        receiver.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert not lock_thread.is_alive()
    assert not fork_thread.is_alive()
    assert not zarr_helpers_module._ARCHIVE_LIVE_HANDLES


def test_reconsolidate_zarr_metadata_returns_error_without_raising(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = _FakeGroup()

    def _fake_open(path: str | Path, *, mode: str) -> _FakeGroup:
        return target

    def _fake_consolidate(store: str, *, path: str | None = None):
        raise TypeError("store does not support consolidated metadata")

    monkeypatch.setattr("fisheye.shared.zarr_helpers.open_zarr_group_direct", _fake_open)
    monkeypatch.setattr(zarr, "consolidate_metadata", _fake_consolidate)

    report = reconsolidate_zarr_metadata(tmp_path / "archive.zarr", policy="unit_test")

    assert report["status"] == "error"
    assert "does not support" in report["error"]
    assert target.attrs["metadata_consolidation_status"] == "error"
    assert "does not support" in target.attrs["metadata_consolidation_error"]
