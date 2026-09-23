from __future__ import annotations

import json

import pytest

from fisheye.shared.pose_deployment_manifest import (
    canonical_manifest,
    load_pose_deployment_bundle,
    read_json_document,
    validate_pose_deployment_document,
)
from fisheye.utils.export_pose_deployment_bundle import export_bundle
from tests.unit.fisheye.pose_deployment_helpers import RUN_ID, _sha, source_bundle


@pytest.fixture
def bundle(tmp_path):
    registry, _ = source_bundle(tmp_path)
    root = tmp_path / "bundle"
    result = export_bundle(
        registry=registry, run_id=RUN_ID, destination=root, apply=True
    )
    manifest = root / f"{RUN_ID}.canonical.manifest.json"
    return root, manifest, json.loads(manifest.read_text()), result


@pytest.mark.parametrize("version", [1, 3, True, "2"])
def test_unsupported_manifest_versions_fail_closed(bundle, version):
    root, _, document, _ = bundle
    document["schema_version"] = version
    with pytest.raises(ValueError, match="schema/version"):
        validate_pose_deployment_document(document, root=root)


@pytest.mark.parametrize(
    "relative", ["../escape", "/tmp/escape", "a/../escape", "./model", "a\\b", "a//b"]
)
def test_paths_cannot_escape_or_use_ambiguous_spelling(bundle, relative):
    root, _, document, _ = bundle
    document["payload"]["onnx"]["relative_path"] = relative
    document = canonical_manifest(document["payload"])
    with pytest.raises(ValueError, match="Unsafe relative path"):
        validate_pose_deployment_document(document, root=root)


def test_reader_requires_the_trusted_handoff_manifest_digest(bundle):
    _, manifest, _, _ = bundle
    with pytest.raises(ValueError, match="SHA-256"):
        load_pose_deployment_bundle(manifest, expected_manifest_sha256="0" * 64)


def test_same_bytes_through_symlink_are_rejected(bundle):
    root, _, document, _ = bundle
    source = root / "pose_model_skeleton.json"
    moved = root.parent / "sidecar.json"
    source.rename(moved)
    source.symlink_to(moved)
    with pytest.raises(ValueError, match="Symlinked"):
        validate_pose_deployment_document(document, root=root)


def test_rehashed_but_contradictory_projection_is_rejected(bundle):
    root, _, document, _ = bundle
    sidecar = root / "pose_model_skeleton.json"
    skeleton = json.loads(sidecar.read_text())
    skeleton["nodes"][0]["name"] = "eye_left"
    sidecar.write_text(json.dumps(skeleton))
    document["payload"]["pose_model_skeleton"]["sha256"] = _sha(sidecar)
    document = canonical_manifest(document["payload"])
    with pytest.raises(ValueError, match="projection"):
        validate_pose_deployment_document(document, root=root)


@pytest.mark.parametrize(
    "change", ["extra", "missing", "activate", "stale_digest", "producer"]
)
def test_manifest_grammar_and_publication_claim_are_closed(bundle, change):
    root, _, document, _ = bundle
    payload = document["payload"]
    if change == "extra":
        payload["extra"] = True
    elif change == "missing":
        del payload["pose_model_skeleton"]
    elif change == "activate":
        payload["selector_activation"] = True
    elif change == "producer":
        payload["producer"] = {}
    else:
        document["payload_digest"] = "0" * 64
    if change != "stale_digest":
        document = canonical_manifest(payload)
    with pytest.raises(ValueError):
        validate_pose_deployment_document(document, root=root)


@pytest.mark.parametrize("content", ['{"a":1,"a":2}', '{"a":NaN}', "[]"])
def test_json_parser_rejects_ambiguous_or_nonfinite_documents(tmp_path, content):
    path = tmp_path / "invalid.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        read_json_document(path)
