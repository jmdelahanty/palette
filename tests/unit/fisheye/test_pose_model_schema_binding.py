from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from fisheye.detection.detect_keypoints_yolo import detect_keypoints_yolo
from fisheye.registry.db import Registry
from fisheye.shared.pose_model_schema_binding import (
    EXPLICIT_CONSISTENCY_POLICY,
    PoseModelSchemaBindingError,
    build_explicit_pose_model_schema_binding,
    resolve_registered_pose_model_schema_binding,
    validate_pose_model_schema_binding,
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _redigest(binding: dict) -> None:
    record = {key: value for key, value in binding.items() if key != "binding_sha256"}
    binding["binding_sha256"] = sha256(
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _registered_pose_fixture(
    tmp_path: Path,
    *,
    manifest_labels: list[str],
    registry_labels: list[str],
) -> tuple[Registry, Path, str]:
    registry = Registry(tmp_path / "registry.sqlite")
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"exact-pose-model")
    manifest_path = tmp_path / "pose.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "task": "pose",
                "set_id": "pose_set",
                "pose_schema": {
                    "skeleton_id": "pose_skel_exact",
                    "kpt_shape": [3, 3],
                    "keypoint_labels": manifest_labels,
                    "skeleton": [[0, 1], [0, 2], [1, 2]],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    skeleton_id = registry.upsert_pose_skeleton_spec(
        kpt_shape=[3, 3],
        keypoint_labels=registry_labels,
        edges=[[0, 1], [0, 2], [1, 2]],
        name="pose_schema",
    )
    assert skeleton_id is not None
    registry.record_training_run(
        run_id="pose_run",
        set_id="pose_set",
        task_type="pose",
        config_path=None,
        manifest_path=manifest_path,
        skeleton_id=skeleton_id,
        model_path=model_path,
        metrics_path=None,
        manifest_sha256=_sha(manifest_path),
        model_sha256=_sha(model_path),
        status="success",
    )
    return registry, model_path, _sha(model_path)


def test_registered_binding_uses_hash_verified_ordered_manifest_schema(
    tmp_path: Path,
) -> None:
    labels = ["swim_bladder", "eye_left", "eye_right"]
    registry, model_path, model_sha256 = _registered_pose_fixture(
        tmp_path,
        manifest_labels=labels,
        registry_labels=labels,
    )
    try:
        binding = resolve_registered_pose_model_schema_binding(
            registry,
            run_id="pose_run",
            expected_set_id="pose_set",
            expected_model_path=str(model_path),
            expected_model_sha256=model_sha256,
        )
    finally:
        registry.close()

    assert binding["model"]["sha256"] == model_sha256
    assert binding["pose_schema"]["keypoint_labels"] == labels
    assert binding["pose_schema"]["kpt_shape"] == [3, 2]
    assert binding["pose_schema"]["metadata"]["model_kpt_shape"] == [3, 3]
    assert binding["authority"]["training_manifest_sha256"]


def test_registered_binding_uses_digest_identical_packaged_manifest_when_registry_path_is_missing(
    tmp_path: Path,
) -> None:
    labels = ["swim_bladder", "eye_left", "eye_right"]
    registry = Registry(tmp_path / "registry.sqlite")
    model_path = tmp_path / "model_run" / "weights" / "best.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"exact-pose-model")
    packaged_manifest = model_path.parent.parent / "inputs" / "pose.manifest.json"
    packaged_manifest.parent.mkdir()
    packaged_manifest.write_text(
        json.dumps(
            {
                "task": "pose",
                "set_id": "pose_set",
                "pose_schema": {
                    "skeleton_id": "pose_skel_exact",
                    "kpt_shape": [3, 3],
                    "keypoint_labels": labels,
                    "skeleton": [[0, 1], [0, 2], [1, 2]],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    missing_registered_path = tmp_path / "missing_workstation" / packaged_manifest.name
    skeleton_id = registry.upsert_pose_skeleton_spec(
        kpt_shape=[3, 3],
        keypoint_labels=labels,
        edges=[[0, 1], [0, 2], [1, 2]],
        name="pose_schema",
    )
    registry.record_training_run(
        run_id="pose_run",
        set_id="pose_set",
        task_type="pose",
        config_path=None,
        manifest_path=missing_registered_path,
        skeleton_id=skeleton_id,
        model_path=model_path,
        metrics_path=None,
        manifest_sha256=_sha(packaged_manifest),
        model_sha256=_sha(model_path),
        status="success",
    )
    try:
        binding = resolve_registered_pose_model_schema_binding(
            registry,
            run_id="pose_run",
            expected_set_id="pose_set",
            expected_model_path=str(model_path),
            expected_model_sha256=_sha(model_path),
        )
    finally:
        registry.close()

    assert binding["authority"]["training_manifest_path"] == str(
        packaged_manifest.resolve()
    )
    assert binding["authority"]["training_manifest_sha256"] == _sha(
        packaged_manifest
    )


def test_registered_binding_rejects_bad_packaged_manifest_digest(
    tmp_path: Path,
) -> None:
    labels = ["swim_bladder", "eye_left", "eye_right"]
    registry = Registry(tmp_path / "registry.sqlite")
    model_path = tmp_path / "model_run" / "weights" / "best.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"exact-pose-model")
    packaged_manifest = model_path.parent.parent / "inputs" / "pose.manifest.json"
    packaged_manifest.parent.mkdir()
    packaged_manifest.write_text("{}", encoding="utf-8")
    skeleton_id = registry.upsert_pose_skeleton_spec(
        kpt_shape=[3, 3],
        keypoint_labels=labels,
        edges=[[0, 1], [0, 2], [1, 2]],
        name="pose_schema",
    )
    registry.record_training_run(
        run_id="pose_run",
        set_id="pose_set",
        task_type="pose",
        config_path=None,
        manifest_path=tmp_path / "missing_workstation" / packaged_manifest.name,
        skeleton_id=skeleton_id,
        model_path=model_path,
        metrics_path=None,
        manifest_sha256="f" * 64,
        model_sha256=_sha(model_path),
        status="success",
    )
    try:
        with pytest.raises(
            PoseModelSchemaBindingError,
            match="Packaged pose training manifest content disagrees",
        ):
            resolve_registered_pose_model_schema_binding(
                registry,
                run_id="pose_run",
                expected_set_id="pose_set",
                expected_model_path=str(model_path),
                expected_model_sha256=_sha(model_path),
            )
    finally:
        registry.close()


def test_registered_binding_rejects_same_cardinality_reordered_labels(
    tmp_path: Path,
) -> None:
    registry, model_path, model_sha256 = _registered_pose_fixture(
        tmp_path,
        manifest_labels=["eye_left", "swim_bladder", "eye_right"],
        registry_labels=["swim_bladder", "eye_left", "eye_right"],
    )
    try:
        with pytest.raises(
            PoseModelSchemaBindingError,
            match="ordered keypoint labels disagree",
        ):
            resolve_registered_pose_model_schema_binding(
                registry,
                run_id="pose_run",
                expected_set_id="pose_set",
                expected_model_path=str(model_path),
                expected_model_sha256=model_sha256,
            )
    finally:
        registry.close()


def test_registered_binding_rejects_registry_spec_digest_not_matching_payload(
    tmp_path: Path,
) -> None:
    labels = ["swim_bladder", "eye_left", "eye_right"]
    registry, model_path, model_sha256 = _registered_pose_fixture(
        tmp_path,
        manifest_labels=labels,
        registry_labels=labels,
    )
    registry.conn.execute(
        "UPDATE pose_skeleton_specs SET spec_sha256 = ?",
        ("0" * 64,),
    )
    registry.conn.commit()
    try:
        with pytest.raises(
            PoseModelSchemaBindingError,
            match="spec_sha256 disagrees with the exact parsed spec_json payload",
        ):
            resolve_registered_pose_model_schema_binding(
                registry,
                run_id="pose_run",
                expected_set_id="pose_set",
                expected_model_path=str(model_path),
                expected_model_sha256=model_sha256,
            )
    finally:
        registry.close()


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("registry_skeleton_id", "pose_skel_other", "registry skeleton identity differs"),
        ("registry_skeleton_spec_sha256", "f" * 64, "registry skeleton digest differs"),
    ],
)
def test_registered_binding_rejects_duplicated_registry_authority_disagreement(
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
) -> None:
    labels = ["swim_bladder", "eye_left", "eye_right"]
    registry, model_path, model_sha256 = _registered_pose_fixture(
        tmp_path,
        manifest_labels=labels,
        registry_labels=labels,
    )
    try:
        binding = resolve_registered_pose_model_schema_binding(
            registry,
            run_id="pose_run",
            expected_set_id="pose_set",
            expected_model_path=str(model_path),
            expected_model_sha256=model_sha256,
        )
    finally:
        registry.close()

    binding["pose_schema"]["metadata"][field] = replacement
    _redigest(binding)
    with pytest.raises(PoseModelSchemaBindingError, match=message):
        validate_pose_model_schema_binding(
            binding,
            expected_model_sha256=model_sha256,
        )


def test_binding_digest_rejects_relabeling_after_resolution(tmp_path: Path) -> None:
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"exact-pose-model")
    digest = _sha(model_path)
    binding = build_explicit_pose_model_schema_binding(
        model_sha256=digest,
        assertion_id="reviewed-model-card-001",
        skeleton_id="pose_skel_exact",
        model_kpt_shape=[3, 3],
        keypoint_labels=["swim_bladder", "eye_left", "eye_right"],
        edges=[[0, 1], [0, 2], [1, 2]],
    )
    binding["pose_schema"]["keypoint_labels"] = [
        "eye_left",
        "swim_bladder",
        "eye_right",
    ]

    with pytest.raises(PoseModelSchemaBindingError, match="digest does not match"):
        validate_pose_model_schema_binding(
            binding,
            expected_model_sha256=digest,
        )


def test_explicit_binding_uses_distinct_assertion_consistency_policy(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"exact-pose-model")
    binding = build_explicit_pose_model_schema_binding(
        model_sha256=_sha(model_path),
        assertion_id="reviewed-model-card-001",
        skeleton_id="pose_skel_exact",
        model_kpt_shape=[3, 3],
        keypoint_labels=["swim_bladder", "eye_left", "eye_right"],
        edges=[[0, 1], [0, 2], [1, 2]],
    )

    assert binding["authority"]["consistency_policy"] == EXPLICIT_CONSISTENCY_POLICY


def test_direct_canonical_inference_rejects_missing_binding_before_model_load(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "archive.zarr"
    zarr_path.mkdir()
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"exact-pose-model")

    with pytest.raises(ValueError, match="model_pose_schema_binding"):
        detect_keypoints_yolo(str(zarr_path), str(model_path))


def test_same_cardinality_package_assertion_cannot_override_model_binding(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "archive.zarr"
    zarr_path.mkdir()
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"exact-pose-model")
    binding = build_explicit_pose_model_schema_binding(
        model_sha256=_sha(model_path),
        assertion_id="reviewed-reordered-schema",
        skeleton_id="pose_skel_reordered",
        model_kpt_shape=[3, 3],
        keypoint_labels=["eye_left", "swim_bladder", "eye_right"],
        edges=[[0, 1], [0, 2], [1, 2]],
    )

    with pytest.raises(ValueError, match="package disagrees"):
        detect_keypoints_yolo(
            str(zarr_path),
            str(model_path),
            model_pose_schema_binding=binding,
            pose_schema="traditional_v1",
        )
