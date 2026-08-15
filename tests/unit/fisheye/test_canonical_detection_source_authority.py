from __future__ import annotations

import json
from pathlib import Path

import pytest

import fisheye.shared.zarr.canonical_detection_manifest as canonical


class _Group(dict[str, object]):
    def __init__(self, values=None, *, attrs=None) -> None:
        super().__init__(values or {})
        self.attrs = dict(attrs or {})


def _active_root() -> tuple[_Group, dict[str, object]]:
    run_id = "detect-canonical-v3"
    digest = "a" * 64
    manifest = {
        "schema_version": 3,
        "payload_digest": digest,
        "payload": {
            "run_id": run_id,
            "publication": {"stage_selector_eligible": True},
        },
    }
    run = _Group(
        attrs={
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "run_manifest": manifest,
        }
    )
    parent = _Group(
        {run_id: run},
        attrs={
            "latest": run_id,
            "latest_complete": run_id,
            canonical.CANONICAL_DETECTION_AUTHORITY_CONTRACT_ATTR: (
                canonical.CANONICAL_DETECTION_AUTHORITY_CONTRACT_V3
            ),
            canonical.CANONICAL_DETECTION_AUTHORITY_DIGEST_ATTR: digest,
        },
    )
    return _Group({"detect_runs": parent}), manifest


def test_active_canonical_source_requires_planner_bound_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest = _active_root()
    monkeypatch.setattr(
        canonical,
        "validate_canonical_detection_run_manifest",
        lambda _manifest: (),
    )

    observed = canonical.require_active_coordinate_canonical_detection(
        root,
        group_path="detect_runs/detect-canonical-v3",
        expected_manifest_digest="a" * 64,
    )

    assert observed is manifest
    with pytest.raises(ValueError, match="planner-bound expectation"):
        canonical.require_active_coordinate_canonical_detection(
            root,
            group_path="detect_runs/detect-canonical-v3",
            expected_manifest_digest="b" * 64,
        )


def test_publication_receipt_resolves_exact_active_v3_digest(tmp_path: Path) -> None:
    receipt = tmp_path / "publication.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_id": "palette.native_canonical_detection_publication",
                "schema_version": 1,
                "status": "complete",
                "group_path": "detect_runs/detect-canonical-v3",
                "run_id": "detect-canonical-v3",
                "native_run_manifest_schema_version": 3,
                "selector_eligible": True,
                "selector_activation": "complete",
                "run_manifest_digest": "c" * 64,
            }
        ),
        encoding="utf-8",
    )

    assert (
        canonical.resolve_expected_canonical_detection_manifest_digest(
            expected_group_path="detect_runs/detect-canonical-v3",
            publication_receipt_path=receipt,
        )
        == "c" * 64
    )
    with pytest.raises(ValueError, match="expected source"):
        canonical.resolve_expected_canonical_detection_manifest_digest(
            expected_group_path="detect_runs/another-run",
            publication_receipt_path=receipt,
        )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expected source"):
        canonical.resolve_expected_canonical_detection_manifest_digest(
            expected_group_path="detect_runs/detect-canonical-v3",
            publication_receipt_path=receipt,
        )


def test_source_expectation_requires_exactly_one_authority(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        canonical.resolve_expected_canonical_detection_manifest_digest(
            expected_group_path="detect_runs/detect-canonical-v3",
        )
    with pytest.raises(ValueError, match="exactly one"):
        canonical.resolve_expected_canonical_detection_manifest_digest(
            expected_group_path="detect_runs/detect-canonical-v3",
            expected_manifest_digest="d" * 64,
            publication_receipt_path=tmp_path / "publication.json",
        )
