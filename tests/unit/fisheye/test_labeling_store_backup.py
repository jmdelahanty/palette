from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import pytest
import zstandard

from fisheye.labeling import store_backup
from fisheye.labeling.assignment_store import LabelingStore
from fisheye.labeling.store_backup import (
    LabelingStoreBackupError,
    backup_labeling_store,
    prune_snapshots,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _labeling_store(path: Path, *, tasks: int = 1) -> Path:
    store = LabelingStore(path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        for index in range(tasks):
            store.upsert_task(
                task_id=f"task-{index}", recording_id="rec-a", workflow_kind="keypoints"
            )
    finally:
        store.close()
    return path


def _restore(backup: Path, destination: Path) -> Path:
    with backup.open("rb") as reader, destination.open("wb") as writer:
        zstandard.ZstdDecompressor().copy_stream(reader, writer)
    return destination


def test_backup_publishes_validated_restorable_snapshot_without_touching_source(
    tmp_path: Path,
) -> None:
    source = _labeling_store(tmp_path / "live" / "labeling_work.sqlite", tasks=3)
    before = (_sha256(source), source.stat().st_mtime_ns)

    receipt = backup_labeling_store(source, tmp_path / "backups")

    assert (_sha256(source), source.stat().st_mtime_ns) == before
    assert not list(source.parent.glob("*-journal"))
    assert receipt["status"] == "complete"
    assert receipt["validation"]["integrity_check"] == "ok"
    assert receipt["validation"]["row_counts"]["labeling_tasks"] == 3
    backup = Path(str(receipt["backup_path"]))
    assert backup.parent == tmp_path / "backups" / "labeling_work"
    assert json.loads(Path(f"{backup}.receipt.json").read_text()) == {
        key: value for key, value in receipt.items() if key != "deleted_old_backups"
    }
    assert json.loads((backup.parent / "latest.json").read_text())["backup_path"] == str(
        backup
    )

    restored = _restore(backup, tmp_path / "restored.sqlite")
    assert _sha256(restored) == receipt["snapshot_sha256"]
    reopened = LabelingStore(restored)
    try:
        assert reopened.get_task("task-2") is not None
    finally:
        reopened.close()


def test_backup_skips_unchanged_store_and_captures_later_changes(tmp_path: Path) -> None:
    source = _labeling_store(tmp_path / "labeling_work.sqlite")
    backups = tmp_path / "backups"
    first = backup_labeling_store(source, backups, label="main")

    unchanged = backup_labeling_store(source, backups, label="main")
    assert unchanged["status"] == "unchanged"
    assert unchanged["backup_path"] == first["backup_path"]
    assert len(list((backups / "main").glob("*.sqlite.zst"))) == 1

    store = LabelingStore(source)
    try:
        store.upsert_task(task_id="task-new", recording_id="rec-a", workflow_kind="keypoints")
    finally:
        store.close()
    # Snapshot names carry one-second timestamps; keep the second name distinct.
    renamed = Path(str(first["backup_path"]))
    renamed.rename(renamed.with_name("main_20000101_000000.sqlite.zst"))
    latest = json.loads((backups / "main" / "latest.json").read_text())
    latest["backup_path"] = str(renamed.with_name("main_20000101_000000.sqlite.zst"))
    (backups / "main" / "latest.json").write_text(json.dumps(latest))

    changed = backup_labeling_store(source, backups, label="main")
    assert changed["status"] == "complete"
    assert changed["snapshot_sha256"] != first["snapshot_sha256"]
    assert changed["validation"]["row_counts"]["labeling_tasks"] == 2


def test_backup_refuses_non_labeling_and_inconsistent_stores(tmp_path: Path) -> None:
    other = tmp_path / "other.sqlite"
    with sqlite3.connect(other) as conn:
        conn.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY);")
    with pytest.raises(LabelingStoreBackupError, match="not a labeling store"):
        backup_labeling_store(other, tmp_path / "backups")

    broken = _labeling_store(tmp_path / "broken.sqlite")
    with sqlite3.connect(broken) as conn:
        conn.execute("PRAGMA foreign_keys = OFF;")
        conn.execute(
            "INSERT INTO labeling_task_events (event_id, task_id, recording_id, user, "
            "event_type, created_at_utc) "
            "VALUES ('orphan', 'missing-task', 'rec-a', 'alice', 'probe', 'now');"
        )
    with pytest.raises(LabelingStoreBackupError, match="foreign_key_check"):
        backup_labeling_store(broken, tmp_path / "backups")
    assert not (tmp_path / "backups" / "broken").exists()

    with pytest.raises(LabelingStoreBackupError, match="does not exist"):
        backup_labeling_store(tmp_path / "missing.sqlite", tmp_path / "backups")


def test_prune_keeps_recent_and_newest_per_day(tmp_path: Path) -> None:
    names = [
        "s_20260920_010000",
        "s_20260920_230000",
        "s_20260921_120000",
        "s_20260922_080000",
        "s_20260922_090000",
        "s_20260922_100000",
    ]
    for name in names:
        (tmp_path / f"{name}.sqlite.zst").write_bytes(b"x")
        (tmp_path / f"{name}.sqlite.zst.receipt.json").write_text("{}")

    deleted = prune_snapshots(tmp_path, "s", keep_recent=2, keep_daily=2)

    remaining = sorted(path.name for path in tmp_path.glob("*.sqlite.zst"))
    # Two most recent, plus the newest of each of the two newest days.
    assert remaining == [
        "s_20260921_120000.sqlite.zst",
        "s_20260922_090000.sqlite.zst",
        "s_20260922_100000.sqlite.zst",
    ]
    assert len(deleted) == 3
    assert not (tmp_path / "s_20260920_230000.sqlite.zst.receipt.json").exists()


def test_cli_backs_up_every_store_and_fails_if_any_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = _labeling_store(tmp_path / "good.sqlite")

    code = store_backup.main(
        [
            "--store",
            f"{good}=main",
            "--store",
            str(tmp_path / "missing.sqlite"),
            "--backup-dir",
            str(tmp_path / "backups"),
        ]
    )

    results = json.loads(capsys.readouterr().out)
    assert code == 1
    assert [result["status"] for result in results] == ["complete", "failed"]
    assert list((tmp_path / "backups" / "main").glob("main_*.sqlite.zst"))


def test_backup_store_command_writes_validated_copy_without_initializing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from fisheye.labeling import web

    source = _labeling_store(tmp_path / "labeling_work.sqlite", tasks=2)
    before = _sha256(source)

    def _no_initialize(self: LabelingStore) -> None:
        raise AssertionError("backup-store must not initialize the live store")

    monkeypatch.setattr(LabelingStore, "initialize", _no_initialize)
    output = tmp_path / "copy" / "labeling_work.sqlite"
    assert web.main(["--store", str(source), "backup-store", "--output", str(output)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["backup_path"] == str(output.resolve())
    assert payload["validation"]["row_counts"]["labeling_tasks"] == 2
    assert _sha256(source) == before
    with pytest.raises(FileExistsError):
        web.main(["--store", str(source), "backup-store", "--output", str(output)])
