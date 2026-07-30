from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.shared.zarr.benchmark_runtime import sha256_file
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils import finalize_crimson_canonical_v3_companion as mod


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _manifest(*, version: int, storage_profile: str = "access_aware") -> dict:
    payload = {
        "run_id": f"canonical_v{version}",
        "logical_schema": {"dimensions": {"n_frames": 3, "n_instances": 2}},
        "storage_plan": {"profile": storage_profile},
        "logical_content": {"digest": "a" * 64, "document": {"arrays": {}}},
        "source_evidence": {"source_kind": "native_detector"},
        "publication": {
            "stage_selector_eligible": False,
            "metadata_declarations_digest": "b" * 64,
        },
    }
    if version == 3:
        payload.update(
            {
                "source_evidence_kind": "native_detection",
                "coordinate_contract": {
                    "digest": "c" * 64,
                    "document": {"schema_id": "palette.array_coordinate_catalog"},
                },
            }
        )
    return {
        "schema_id": "palette.canonical_detection.run_manifest",
        "schema_version": version,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }


def _base_handoff(tmp_path: Path) -> tuple[Path, dict]:
    dimensions = {
        "n_frames": 3,
        "n_instances": 2,
        "n_frame_boundaries": 4,
        "source_width": 640,
        "source_height": 480,
    }
    artifacts = {
        name: {
            "stage": name,
            "server_path": str(tmp_path / f"base_{name}.zarr"),
            "run_path": f"runs/{name}",
            "run_id": f"{name}_run",
            "dimensions": dimensions,
            "logical_content_digest": "a" * 64,
        }
        for name in mod._EXPECTED_ARTIFACTS
    }
    payload = {
        "status": "complete",
        "candidate_id": "full_v8",
        "classification": "full_duration_fixture",
        "benchmark_only": True,
        "selector_eligible": False,
        "registry_registered": False,
        "production_state_changes": [],
        "dimensions": {"n_frames": 3, "n_instances": 2},
        "artifacts": artifacts,
        "receipts": {},
        "crimson_contract": {"commit": "d" * 40},
    }
    handoff = {
        "schema_id": "palette.crimson.storage_candidate_handoff",
        "schema_version": 1,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    return _write(tmp_path / "base_handoff.json", handoff), handoff


def test_companion_replaces_only_logically_identical_canonical_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_path, base = _base_handoff(tmp_path)
    companion_root = tmp_path / ".palette_benchmarks" / "companion"
    canonical_archive = companion_root / "canonical_detection.zarr"
    canonical_archive.mkdir(parents=True)
    result_path = _write(
        companion_root / "canonical_detection_result.json",
        {
            "status": "complete",
            "selector_eligible": False,
            "registry_registered": False,
            "production_state_changes": [],
            "coordinate_catalog": True,
            "run_manifest_schema_version": 3,
            "output_archive": str(canonical_archive),
            "output_run_id": "canonical_v3",
            "adapter_receipt_path": str(canonical_archive / "receipt.json"),
            "adapter_receipt_digest": "e" * 64,
        },
    )
    manifests = iter((_manifest(version=2), _manifest(version=3)))
    monkeypatch.setattr(mod, "_manifest_for_artifact", lambda _: next(manifests))
    monkeypatch.setattr(mod, "validate_canonical_detection_run_manifest", lambda _: ())
    replacement = {
        **base["payload"]["artifacts"]["canonical_detection"],
        "server_path": str(canonical_archive),
        "run_id": "canonical_v3",
    }
    monkeypatch.setattr(mod, "_artifact", lambda **_: replacement)
    monkeypatch.setattr(
        mod,
        "_git_state",
        lambda: {"commit": "f" * 40, "branch": "agent/test", "worktree_clean": True},
    )

    output = companion_root / "handoff_manifest.json"
    handoff = mod.finalize_crimson_canonical_v3_companion(
        base_handoff_path=base_path,
        expected_base_handoff_sha256=sha256_file(base_path),
        canonical_result_path=result_path,
        canonical_archive=canonical_archive,
        canonical_run="canonical_v3",
        crimson_validation_commit="1" * 40,
        expected_palette_commit="f" * 40,
        output=output,
    )

    assert output.is_file()
    assert handoff["payload_digest"] == canonical_json_sha256(handoff["payload"])
    payload = handoff["payload"]
    assert payload["artifacts"]["canonical_detection"] == replacement
    for name in mod._EXPECTED_ARTIFACTS - {"canonical_detection"}:
        assert payload["artifacts"][name] == base["payload"]["artifacts"][name]
    assert payload["canonical_v3_companion"]["equivalence"][
        "companion_manifest_schema_version"
    ] == 3
    assert payload["crimson_validation"]["commit"] == "1" * 40


def test_companion_rejects_changed_physical_storage_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "validate_canonical_detection_run_manifest", lambda _: ())
    with pytest.raises(ValueError, match="outside the manifest/catalog envelope"):
        mod._require_equal_contract(
            _manifest(version=2),
            _manifest(version=3, storage_profile="changed"),
        )
