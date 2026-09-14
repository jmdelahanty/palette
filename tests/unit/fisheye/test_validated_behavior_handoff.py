from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisheye.analytics_exports.validated_behavior_handoff import (
    publish_validated_behavior_handoff,
    read_validated_behavior_handoff,
)
from fisheye.analytics_exports.validated_behavior_notifications import (
    prepare_validated_behavior_export_announcement,
)

RUN_ID = "sleepyfish-bout-v1"
DIGEST = "a" * 64


def _dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    dataset = SimpleNamespace(
        root=tmp_path,
        export_run_id=RUN_ID,
        validation_mode="receipt",
        table_names=("canonical_swim_bouts",),
        manifest={
            "record_sha256": DIGEST,
            "status": "complete_selector_ineligible",
            "export_profile": {
                "profile_id": "validated_core_behavior_bout_kinematics_v1"
            },
        },
    )

    def open_dataset(
        root: Path, run_id: str, *, validate: bool, full_part_hashes: bool
    ) -> SimpleNamespace:
        assert (root, run_id, validate, full_part_hashes) == (
            tmp_path,
            RUN_ID,
            True,
            False,
        )
        return dataset

    monkeypatch.setattr(
        "fisheye.analytics_exports.validated_behavior_handoff."
        "ValidatedBehaviorExportDataset.open",
        open_dataset,
    )
    return dataset


def _source(tmp_path: Path, *, text: str = "Reviewed guide") -> Path:
    path = tmp_path / "reviewed.md"
    path.write_text(f"# {text}\n\nExport run: {RUN_ID}\nManifest: {DIGEST}\n")
    return path


def test_publish_companion_and_email_auto_discovers_exact_guide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path, monkeypatch)
    source = _source(tmp_path)
    handoff = publish_validated_behavior_handoff(
        publication_root=tmp_path,
        export_run_id=RUN_ID,
        source=source,
        version="v001",
    )
    assert handoff.path == (
        tmp_path / "handoffs" / f"export_run_id={RUN_ID}" / "versions" / "v001.md"
    )
    assert handoff.path.read_bytes() == source.read_bytes()
    assert read_validated_behavior_handoff(dataset) == handoff
    assert (
        publish_validated_behavior_handoff(
            publication_root=tmp_path,
            export_run_id=RUN_ID,
            source=source,
            version="v001",
        )
        == handoff
    )

    announcement = prepare_validated_behavior_export_announcement(
        publication_root=tmp_path,
        export_run_id=RUN_ID,
        to=["reader@example.org"],
    )
    assert f"Reading guide: {handoff.path}" in announcement.notification.text_body
    assert "Reading guide SHA-256: " + handoff.document_sha256 in (
        announcement.notification.text_body
    )
    assert str(handoff.path) in announcement.notification.html_body
    assert announcement.context["handoff_document_sha256"] == handoff.document_sha256


def test_guide_tamper_or_wrong_export_blocks_announcement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path, monkeypatch)
    handoff = publish_validated_behavior_handoff(
        publication_root=tmp_path,
        export_run_id=RUN_ID,
        source=_source(tmp_path),
        version="v001",
    )
    handoff.path.write_text("tampered")
    with pytest.raises(ValueError, match="bytes differ"):
        prepare_validated_behavior_export_announcement(
            publication_root=tmp_path,
            export_run_id=RUN_ID,
            to=["reader@example.org"],
        )
    handoff.path.write_bytes(_source(tmp_path).read_bytes())
    dataset.manifest["record_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="does not bind this exact export"):
        read_validated_behavior_handoff(dataset)


def test_new_version_keeps_old_guide_and_updates_explicit_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path, monkeypatch)
    first = publish_validated_behavior_handoff(
        publication_root=tmp_path,
        export_run_id=RUN_ID,
        source=_source(tmp_path),
        version="v001",
    )
    second = publish_validated_behavior_handoff(
        publication_root=tmp_path,
        export_run_id=RUN_ID,
        source=_source(tmp_path, text="Updated guide"),
        version="v002",
    )
    assert first.path.read_text().startswith("# Reviewed guide")
    assert second.path.read_text().startswith("# Updated guide")
    assert read_validated_behavior_handoff(dataset) == second
    record = json.loads(second.record_path.read_text())
    assert record["version"] == "v002"
    assert record["export_manifest_record_sha256"] == DIGEST
    with pytest.raises(ValueError, match="newer version"):
        publish_validated_behavior_handoff(
            publication_root=tmp_path,
            export_run_id=RUN_ID,
            source=_source(tmp_path),
            version="v001",
        )


def test_publisher_rejects_guide_for_another_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dataset(tmp_path, monkeypatch)
    source = tmp_path / "wrong.md"
    source.write_text("# Another dataset\n")
    with pytest.raises(ValueError, match="must name the exact export"):
        publish_validated_behavior_handoff(
            publication_root=tmp_path,
            export_run_id=RUN_ID,
            source=source,
            version="v001",
        )
    assert not (tmp_path / "handoffs").exists()
