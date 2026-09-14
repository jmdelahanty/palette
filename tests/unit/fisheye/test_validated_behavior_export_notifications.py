from __future__ import annotations

import json
from email import policy
from email.parser import BytesParser
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisheye.analytics_exports.validated_behavior_notifications import (
    deliver_validated_behavior_export_announcement,
    prepare_validated_behavior_export_announcement,
)
from fisheye.labeling.notifications import LabelingNotificationConfig
from fisheye.utils.notify_validated_behavior_export import main

RUN_ID = "sleepyfish-bout-v1"


def _reader_stub(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> list[tuple[object, ...]]:
    calls: list[tuple[object, ...]] = []

    def open_dataset(
        requested_root: Path,
        requested_run: str,
        *,
        validate: bool,
        full_part_hashes: bool,
    ) -> SimpleNamespace:
        calls.append((requested_root, requested_run, validate, full_part_hashes))
        return SimpleNamespace(
            root=root,
            export_run_id=RUN_ID,
            validation_mode="receipt",
            table_names=("canonical_swim_bouts", "bout_movement_metrics"),
            manifest={
                "export_profile": {
                    "profile_id": "validated_core_behavior_bout_kinematics_v1"
                },
                "record_sha256": "a" * 64,
                "status": "complete_selector_ineligible",
            },
        )

    monkeypatch.setattr(
        "fisheye.analytics_exports.validated_behavior_notifications."
        "ValidatedBehaviorExportDataset.open",
        open_dataset,
    )
    return calls


def test_preview_binds_exact_validated_export_without_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _reader_stub(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "fisheye.utils.notify_validated_behavior_export."
        "deliver_validated_behavior_export_announcement",
        lambda *_args, **_kwargs: pytest.fail("preview must not deliver"),
    )
    code = main(
        [
            "--publication-root",
            str(tmp_path),
            "--export-run-id",
            RUN_ID,
            "--to",
            "reader@example.org",
            "--audience",
            "Johnson Lab collaborators",
            "--location",
            "/groups/shared/sleepyfish",
            "--handoff",
            "https://example.org/sleepyfish-guide",
            "--access-note",
            "Ask the owner for group access.",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert calls == [(tmp_path, RUN_ID, True, False)]
    assert "reader@example.org" in output
    assert "DATASET\n" in output
    assert "ACCESS\n" in output
    assert "TABLES (2)\n- canonical_swim_bouts" in output
    assert "HOW TO READ IT\n" in output
    assert "PROVENANCE\n" in output
    assert "Manifest record SHA-256: " + "a" * 64 in output
    assert "Validation mode: receipt" in output
    assert "Intended audience: Johnson Lab collaborators" in output
    assert "Location (provided by sender): /groups/shared/sleepyfish" in output
    assert "Reading guide: https://example.org/sleepyfish-guide" in output
    assert "does not grant filesystem access" in output


def test_invalid_recipient_or_unvalidated_publication_cannot_deliver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _reader_stub(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="plain email address"):
        prepare_validated_behavior_export_announcement(
            publication_root=tmp_path,
            export_run_id=RUN_ID,
            to=["Name <reader@example.org>"],
        )
    assert calls == []

    def reject(*_args: object, **_kwargs: object) -> None:
        raise ValueError("stale validation receipt")

    monkeypatch.setattr(
        "fisheye.analytics_exports.validated_behavior_notifications."
        "ValidatedBehaviorExportDataset.open",
        reject,
    )
    with pytest.raises(ValueError, match="stale validation receipt"):
        prepare_validated_behavior_export_announcement(
            publication_root=tmp_path, export_run_id=RUN_ID, to=["reader@example.org"]
        )


def test_header_lines_cannot_be_injected_into_sender_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reader_stub(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="note must be one line"):
        prepare_validated_behavior_export_announcement(
            publication_root=tmp_path,
            export_run_id=RUN_ID,
            to=["reader@example.org"],
            note="Ready\nBcc: someone@example.org",
        )


def test_explicit_delivery_uses_existing_outbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reader_stub(monkeypatch, tmp_path)
    announcement = prepare_validated_behavior_export_announcement(
        publication_root=tmp_path,
        export_run_id=RUN_ID,
        to=["reader@example.org", "colleague@example.org"],
        handoff="https://example.org/guide?a=1&b=2",
        note="Review <this> & reply.",
    )
    outbox = tmp_path / "outbox"
    result = deliver_validated_behavior_export_announcement(
        announcement,
        config=LabelingNotificationConfig(
            mode="outbox",
            sender="Palette Exports <palette@example.org>",
            outbox_dir=outbox,
        ),
    )
    assert result["status"] == "queued"
    assert result["sent"] is False
    assert result["context"]["manifest_record_sha256"] == "a" * 64
    message = BytesParser(policy=policy.default).parsebytes(
        Path(result["outbox_eml_path"]).read_bytes()
    )
    assert message["To"] == "reader@example.org, colleague@example.org"
    assert message["X-Palette-Labeling-Notification-Kind"] == (
        "validated_behavior_export_available"
    )
    assert message.get_content_type() == "multipart/alternative"
    plain = message.get_body(preferencelist=("plain",)).get_content()
    html = message.get_body(preferencelist=("html",)).get_content()
    assert "TABLES (2)\n- canonical_swim_bouts" in plain
    assert "SENDER NOTE\nReview <this> & reply." in plain
    assert "<h2" in html and ">Access</h2>" in html
    assert ">Tables</h2>" in html and ">Provenance</h2>" in html
    assert '<a href="https://example.org/guide?a=1&amp;b=2">' in html
    assert "Review &lt;this&gt; &amp; reply." in html
    assert "<this>" not in html
    assert (
        json.loads(Path(result["outbox_json_path"]).read_text())["status"] == "queued"
    )


def test_cli_delivery_defaults_to_outbox_even_if_labeling_uses_smtp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _reader_stub(monkeypatch, tmp_path)
    monkeypatch.setenv("PALETTE_LABELING_NOTIFICATION_MODE", "smtp")
    monkeypatch.setenv("PALETTE_LABELING_SMTP_HOST", "smtp.example.org")
    monkeypatch.setenv("PALETTE_EXPORT_NOTIFICATION_OUTBOX", str(tmp_path / "outbox"))
    monkeypatch.setattr(
        "fisheye.labeling.notifications.smtplib.SMTP",
        lambda *_args, **_kwargs: pytest.fail("SMTP must require --mode smtp"),
    )
    code = main(
        [
            "--publication-root",
            str(tmp_path),
            "--export-run-id",
            RUN_ID,
            "--to",
            "reader@example.org",
            "--deliver",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["status"] == "queued"
    assert Path(result["outbox_eml_path"]).exists()


def test_explicit_smtp_delivery_uses_existing_relay_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reader_stub(monkeypatch, tmp_path)
    sent = []

    class SMTP:
        def __init__(self, host: str, port: int, timeout: int) -> None:
            assert (host, port, timeout) == ("smtp.example.org", 2525, 30)

        def __enter__(self) -> SMTP:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def send_message(self, message: object) -> None:
            sent.append(message)

    monkeypatch.setattr("fisheye.labeling.notifications.smtplib.SMTP", SMTP)
    announcement = prepare_validated_behavior_export_announcement(
        publication_root=tmp_path, export_run_id=RUN_ID, to=["reader@example.org"]
    )
    result = deliver_validated_behavior_export_announcement(
        announcement,
        config=LabelingNotificationConfig(
            mode="smtp",
            sender="palette@example.org",
            smtp_host="smtp.example.org",
            smtp_port=2525,
            smtp_starttls=False,
        ),
    )
    assert result["status"] == "sent"
    assert len(sent) == 1
    assert sent[0]["Subject"] == f"Palette dataset available: {RUN_ID}"
