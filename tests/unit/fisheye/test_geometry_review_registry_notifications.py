from __future__ import annotations

import json
import sqlite3
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pytest

from fisheye.labeling.notifications import LabelingNotificationConfig
from fisheye.registry.geometry_review import (
    GeometryReviewRegistryError,
    actionable_geometry_transitions,
    load_geometry_review_queue,
)
from fisheye.registry.geometry_review_notifications import (
    scan_geometry_review_notifications,
)


def _registry(path: Path, *, checked_status: bool = True) -> Path:
    status_column = (
        "TEXT NOT NULL CHECK (status IN ('ok','missing','absent','na','error'))"
        if checked_status
        else "TEXT NOT NULL"
    )
    with sqlite3.connect(path) as conn:
        conn.executescript(f"""
            CREATE TABLE datasets (
                dataset_id TEXT PRIMARY KEY,
                recording_id TEXT,
                zarr_path TEXT NOT NULL,
                zarr_use TEXT,
                status TEXT
            );
            CREATE TABLE recordings (
                recording_id TEXT PRIMARY KEY,
                camera_id TEXT,
                arena_id TEXT
            );
            CREATE TABLE recording_step_status (
                dataset_id TEXT NOT NULL,
                recording_id TEXT,
                step_name TEXT NOT NULL,
                status {status_column},
                run_name TEXT,
                review_status_json TEXT,
                details_json TEXT,
                source TEXT,
                updated_utc TEXT,
                PRIMARY KEY (dataset_id, step_name)
            );
            """)
    return path


def _insert_dataset(
    registry: Path,
    *,
    index: int,
    step: str = "arena_geometry_offline_fit",
    status: str = "ok",
    review: dict[str, object] | None = None,
    details: dict[str, object] | None = None,
) -> None:
    dataset_id = f"dataset-{index}"
    recording_id = f"recording-{index}"
    run_id = f"arena-geometry-fit-review-{index}"
    with sqlite3.connect(registry) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO datasets VALUES (?, ?, ?, 'analysis', 'ok')",
            (dataset_id, recording_id, f"/recordings/{recording_id}_analysis.zarr"),
        )
        conn.execute(
            "INSERT OR IGNORE INTO recordings VALUES (?, ?, ?)",
            (recording_id, f"camera-{index}", f"arena-{index}"),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO recording_step_status VALUES (
                ?, ?, ?, ?, ?, ?, ?, 'test', '2026-08-14T12:00:00Z'
            )
            """,
            (
                dataset_id,
                recording_id,
                step,
                status,
                run_id,
                json.dumps(review) if review is not None else None,
                json.dumps(details) if details is not None else None,
            ),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO recording_step_status VALUES (
                ?, ?, 'detect', 'ok', ?, NULL, ?, 'test',
                '2026-08-14T12:00:00Z'
            )
            """,
            (
                dataset_id,
                recording_id,
                f"detect-canonical-{index}",
                json.dumps(
                    {
                        "canonical_detection_authority_errors": [],
                        "source_detect_identity_kind": "run",
                        "canonical_detection_manifest_digest": (
                            format(index % 16, "x") * 64
                        ),
                    }
                ),
            ),
        )


def _pending(index: int) -> dict[str, object]:
    return {
        "state": "evidence_complete_review_pending",
        "runs": [f"arena-geometry-fit-review-{index}"],
        "review_record_sha256": str(index) * 64,
    }


def _outbox_config(path: Path, *, mode: str = "outbox") -> LabelingNotificationConfig:
    return LabelingNotificationConfig(
        mode=mode,
        sender="Palette Geometry <palette@localhost>",
        base_url="http://127.0.0.1:8772",
        outbox_dir=path,
    )


def test_registry_queue_uses_valid_status_plus_review_json(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.sqlite")
    _insert_dataset(registry, index=1, review=_pending(1))
    _insert_dataset(
        registry,
        index=2,
        status="missing",
        details={"reason": "upstream_job_running"},
    )
    registry_before = registry.read_bytes()

    queue = load_geometry_review_queue(registry)

    assert registry.read_bytes() == registry_before
    assert [item.dataset_id for item in queue] == ["dataset-1"]
    assert queue[0].geometry_state == "fit_evidence_awaiting_review"
    assert queue[0].camera_serial == "camera-1"
    assert queue[0].arena_id == "arena-1"
    assert queue[0].detection_run == "detect-canonical-1"
    assert queue[0].detection_manifest_digest == "1" * 64
    transitions = actionable_geometry_transitions(queue)
    assert len(transitions) == 1
    assert transitions[0].run_id == "arena-geometry-fit-review-1"
    assert transitions[0].digest == "1" * 64


def test_registry_queue_hides_geometry_without_eligible_canonical_detection(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path / "registry.sqlite")
    _insert_dataset(registry, index=1, review=_pending(1))
    with sqlite3.connect(registry) as conn:
        conn.execute(
            """
            UPDATE recording_step_status
            SET details_json = ?
            WHERE dataset_id = 'dataset-1' AND step_name = 'detect'
            """,
            (
                json.dumps(
                    {
                        "canonical_detection_authority_errors": ["legacy_flat_layout"],
                        "source_detect_identity_kind": "run",
                        "canonical_detection_manifest_digest": "1" * 64,
                    }
                ),
            ),
        )

    assert load_geometry_review_queue(registry) == []


def test_registry_queue_rejects_invalid_status_review(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.sqlite", checked_status=False)
    _insert_dataset(
        registry,
        index=1,
        status="review",
        review={"state": "review_required", "runs": ["comparison-1"]},
    )

    with pytest.raises(GeometryReviewRegistryError, match="human review belongs"):
        load_geometry_review_queue(registry)


def test_registry_queue_rejects_dataset_stage_recording_mismatch(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path / "registry.sqlite")
    _insert_dataset(registry, index=1, review=_pending(1))
    with sqlite3.connect(registry) as conn:
        conn.execute(
            "UPDATE recording_step_status SET recording_id = 'wrong-recording' "
            "WHERE dataset_id = 'dataset-1'"
        )

    with pytest.raises(GeometryReviewRegistryError, match="binding mismatch"):
        load_geometry_review_queue(registry)


def test_errors_are_actionable_but_missing_and_running_are_not(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.sqlite")
    _insert_dataset(
        registry,
        index=1,
        status="error",
        details={"reason": "offline_fit_failed"},
    )
    _insert_dataset(
        registry,
        index=2,
        status="missing",
        details={"reason": "still_running"},
    )

    queue = load_geometry_review_queue(registry)

    assert [item.dataset_id for item in queue] == ["dataset-1"]
    assert queue[0].geometry_state == "fit_failure"
    assert actionable_geometry_transitions(queue)[0].semantic_state == (
        "offline_fit_failed"
    )


def test_outbox_digest_is_batched_and_durable_across_scans(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.sqlite")
    _insert_dataset(registry, index=1, review=_pending(1))
    _insert_dataset(registry, index=2, review=_pending(2))
    state = tmp_path / "state" / "geometry.sqlite"
    outbox = tmp_path / "outbox"

    first = scan_geometry_review_notifications(
        registry_path=registry,
        state_db=state,
        recipients="operator@example.org",
        config=_outbox_config(outbox),
    )
    second = scan_geometry_review_notifications(
        registry_path=registry,
        state_db=state,
        recipients="operator@example.org",
        config=_outbox_config(outbox),
    )

    assert len(first.new) == 2
    assert first.delivery["status"] == "queued"
    assert second.new == ()
    assert second.delivery["status"] == "no_new_events"
    assert len(list(outbox.glob("*.eml"))) == 1
    message = BytesParser(policy=policy.default).parsebytes(
        next(outbox.glob("*.eml")).read_bytes()
    )
    body = message.get_content()
    assert "recording-1" in body
    assert "recording-2" in body
    assert "dataset_id=dataset-1" in body
    with sqlite3.connect(state) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM geometry_review_notification_events "
                "WHERE delivered_utc IS NOT NULL"
            ).fetchone()[0]
            == 2
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM geometry_review_notification_scans"
            ).fetchone()[0]
            == 2
        )


@pytest.mark.parametrize("mode,dry_run", [("disabled", False), ("outbox", True)])
def test_disabled_and_dry_run_do_not_consume_dedup_state(
    tmp_path: Path, mode: str, dry_run: bool
) -> None:
    registry = _registry(tmp_path / "registry.sqlite")
    _insert_dataset(registry, index=1, review=_pending(1))
    state = tmp_path / "state.sqlite"
    outbox = tmp_path / "outbox"

    skipped = scan_geometry_review_notifications(
        registry_path=registry,
        state_db=state,
        recipients="operator@example.org",
        config=_outbox_config(outbox, mode=mode),
        dry_run=dry_run,
    )
    delivered = scan_geometry_review_notifications(
        registry_path=registry,
        state_db=state,
        recipients="operator@example.org",
        config=_outbox_config(outbox),
    )

    assert len(skipped.new) == 1
    assert skipped.delivery["status"] == ("dry_run" if dry_run else "skipped")
    assert len(delivered.new) == 1
    assert delivered.delivery["status"] == "queued"


def test_new_semantic_transition_notifies_after_prior_delivery(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.sqlite")
    _insert_dataset(registry, index=1, review=_pending(1))
    state = tmp_path / "state.sqlite"
    config = _outbox_config(tmp_path / "outbox")
    first = scan_geometry_review_notifications(
        registry_path=registry,
        state_db=state,
        recipients="operator@example.org",
        config=config,
    )
    _insert_dataset(
        registry,
        index=1,
        step="arena_geometry_offline_fit",
        review={
            "state": "review_required",
            "reason": "semantic_feature_incompatible",
            "runs": ["arena-geometry-fit-review-1"],
            "review_record_sha256": "1" * 64,
        },
    )
    second = scan_geometry_review_notifications(
        registry_path=registry,
        state_db=state,
        recipients="operator@example.org",
        config=config,
    )

    assert len(first.new) == 1
    assert len(second.new) == 1
    assert first.new[0].event_key != second.new[0].event_key
    assert second.new[0].semantic_state == "semantic_feature_incompatible"


def test_smtp_transport_is_reused_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path / "registry.sqlite")
    _insert_dataset(registry, index=1, review=_pending(1))
    sent = []

    class _SMTP:
        def __init__(self, host: str, port: int, timeout: int) -> None:
            assert (host, port, timeout) == ("smtp.example.org", 2525, 30)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def send_message(self, message) -> None:
            sent.append(message)

    monkeypatch.setattr("fisheye.labeling.notifications.smtplib.SMTP", _SMTP)
    config = LabelingNotificationConfig(
        mode="smtp",
        sender="palette@example.org",
        smtp_host="smtp.example.org",
        smtp_port=2525,
        smtp_starttls=False,
    )

    result = scan_geometry_review_notifications(
        registry_path=registry,
        state_db=tmp_path / "state.sqlite",
        recipients="operator@example.org",
        config=config,
    )

    assert result.delivery["status"] == "sent"
    assert len(sent) == 1
    assert sent[0]["X-Palette-Labeling-Notification-Kind"] == ("geometry_review_digest")


def test_notification_state_cannot_be_registry_or_zarr_data(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.sqlite")
    _insert_dataset(registry, index=1, review=_pending(1))
    config = _outbox_config(tmp_path / "outbox")

    with pytest.raises(ValueError, match="canonical registry"):
        scan_geometry_review_notifications(
            registry_path=registry,
            state_db=registry,
            recipients="operator@example.org",
            config=config,
        )
    with pytest.raises(ValueError, match="outside canonical analysis Zarrs"):
        scan_geometry_review_notifications(
            registry_path=registry,
            state_db=tmp_path / "recording.zarr" / "state.sqlite",
            recipients="operator@example.org",
            config=config,
        )
