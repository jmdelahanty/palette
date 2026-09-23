"""Package a registered pose ONNX and its exact skeleton for acquisition.

This final export step reads the registry and immutable promoted package only.
It never rebuilds models, edits historical evidence, registers a deployment,
or activates a selector. A fresh destination is mandatory; its canonical
manifest is installed last, without replacement, as the completion marker.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from types import SimpleNamespace
from typing import Any, Sequence

from fisheye.shared.artifact_fingerprint import require_artifact_content_identity
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.pose_deployment_manifest import (
    ARTIFACT_ROLES,
    CANONICAL_ONNX_MANIFEST_SCHEMA_ID,
    artifact_reference,
    canonical_manifest,
    load_pose_deployment_bundle,
    read_json_document,
    require_sha256,
    resolve_artifact,
    validate_pose_deployment_document,
)
from fisheye.shared.pose_model_input_contract import load_pose_model_input_contract
from fisheye.shared.pose_model_schema_binding import (
    resolve_registered_pose_model_schema_binding,
)
from fisheye.shared.pose_model_skeleton import (
    POSE_MODEL_SKELETON_FILENAME,
    build_pose_model_skeleton,
)
from fisheye.shared.pose_onnx_interface import inspect_pose_onnx_interface
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    canonical_json_sha256,
)


def _prepare(registry_path: Path, run_id: str) -> dict[str, Any]:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) is None:
        raise ValueError("model-run-id must be one safe filename component")
    connection = sqlite3.connect(
        registry_path.resolve().as_uri() + "?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        row = connection.execute(
            """SELECT tr.set_id, tr.model_path, tr.model_sha256, tr.status,
                      tm.status AS model_status, tr.task_type,
                      om.set_id AS onnx_set_id, om.path AS onnx_path,
                      om.sha256 AS onnx_sha256, om.manifest_path,
                      om.manifest_sha256
               FROM training_runs tr
               JOIN training_models tm ON tm.run_id = tr.run_id
               JOIN onnx_models om ON om.run_id = tr.run_id
               WHERE tr.run_id = ?""",
            (run_id,),
        ).fetchone()
        if (
            row is None
            or row["status"] != "success"
            or row["model_status"] != "success"
            or row["task_type"] != "pose"
        ):
            raise ValueError(
                "Expected one successful registered pose model and ONNX export"
            )
        if not row["set_id"] or row["set_id"] != row["onnx_set_id"]:
            raise ValueError("ONNX and training set identities disagree")
        binding = resolve_registered_pose_model_schema_binding(
            SimpleNamespace(conn=connection),
            run_id=run_id,
            expected_set_id=row["set_id"],
            expected_model_path=row["model_path"],
            expected_model_sha256=row["model_sha256"],
        )
        row = dict(row)
    finally:
        connection.close()

    weights_path = Path(row["model_path"])
    if weights_path.name != "best.pt" or weights_path.parent.name != "weights":
        raise ValueError(
            "Source must be a promoted pose model package with weights/best.pt"
        )
    package_root = weights_path.resolve().parent.parent
    source_canonical = Path(row["manifest_path"])
    if source_canonical.is_symlink() or not source_canonical.resolve().is_relative_to(
        package_root
    ):
        raise ValueError(
            "Registered canonical manifest must belong to the source package"
        )
    require_artifact_content_identity(
        source_canonical,
        role="registered canonical manifest",
        expected_sha256=require_sha256(
            row["manifest_sha256"], field="registered manifest SHA-256"
        ),
    )
    original = read_json_document(source_canonical)
    payload = original.get("payload", {})
    if (
        original.get("schema_id") != CANONICAL_ONNX_MANIFEST_SCHEMA_ID
        or type(original.get("schema_version")) is not int
        or original["schema_version"] != 1
        or type(payload) is not dict
        or original.get("payload_digest") != canonical_json_sha256(payload)
        or payload.get("status") != "complete"
        or payload.get("task") != "pose"
        or payload.get("run_id") != run_id
        or payload.get("set_id") != row["set_id"]
    ):
        raise ValueError("Source requires a valid promoted v1 canonical pose manifest")
    onnx_path = resolve_artifact(package_root, payload.get("onnx"), role="source ONNX")
    if (
        onnx_path.resolve() != Path(row["onnx_path"]).resolve()
        or payload["onnx"]["sha256"] != row["onnx_sha256"]
    ):
        raise ValueError("Registered ONNX and canonical manifest disagree")
    if payload.get("weights", {}).get("sha256") != row["model_sha256"]:
        raise ValueError("Canonical manifest and registered weights disagree")
    contract_path = resolve_artifact(
        package_root,
        payload.get("pose_model_input_contract"),
        role="source input contract",
    )
    export_path = resolve_artifact(
        package_root,
        payload.get("source_export_manifest"),
        role="source export manifest",
    )
    load_pose_model_input_contract(
        contract_path,
        model_path=weights_path,
        expected_set_id=row["set_id"],
        expected_run_id=run_id,
        expected_model_sha256=row["model_sha256"],
    )
    contract = read_json_document(contract_path)
    if contract["schema_version"] != 3 or contract["payload_digest"] != payload[
        "pose_model_input_contract"
    ].get("payload_digest"):
        raise ValueError(
            "Deployment packaging requires the bound runtime-receipt input contract v3"
        )
    training_path = Path(binding["authority"]["training_manifest_path"])
    if (
        contract["payload"]["evidence"]["training_manifest"]["sha256"]
        != binding["authority"]["training_manifest_sha256"]
    ):
        raise ValueError(
            "Input preprocessing and skeleton bindings select different training manifests"
        )
    skeleton = build_pose_model_skeleton(binding, onnx_sha256=row["onnx_sha256"])
    interface = inspect_pose_onnx_interface(
        onnx_path,
        kpt_shape=skeleton["kpt_shape"],
        network_shape_hw=contract["payload"]["training_input"]["network_shape_hw"],
    )
    paths = {
        "onnx": onnx_path,
        "source_export_manifest": export_path,
        "source_canonical_manifest": source_canonical,
        "training_manifest": training_path,
        "pose_model_input_contract": contract_path,
    }
    names = {
        "onnx": f"{run_id}.onnx",
        "source_export_manifest": f"{run_id}.onnx.manifest.json",
        "source_canonical_manifest": "source.canonical.manifest.json",
        "training_manifest": "training.manifest.json",
        "pose_model_input_contract": "pose_model_input_contract.json",
    }
    references = {
        role: artifact_reference(path, relative_path=names[role])
        for role, path in paths.items()
    }
    references["pose_model_input_contract"]["payload_digest"] = contract[
        "payload_digest"
    ]
    # Rehash against the already-established authorities, not new ad hoc values.
    expected = {
        "onnx": row["onnx_sha256"],
        "source_canonical_manifest": row["manifest_sha256"],
        "source_export_manifest": payload["source_export_manifest"]["sha256"],
        "training_manifest": binding["authority"]["training_manifest_sha256"],
        "pose_model_input_contract": payload["pose_model_input_contract"]["sha256"],
    }
    if any(references[role]["sha256"] != digest for role, digest in expected.items()):
        raise ValueError("Source artifacts changed during deployment inspection")
    # Catch missing/contradictory export evidence before creating output state.
    export = read_json_document(export_path)
    if (
        export.get("run_id") != run_id
        or export.get("weights", {}).get("sha256") != row["model_sha256"]
        or export.get("onnx", {}).get("sha256") != row["onnx_sha256"]
    ):
        raise ValueError("Source export manifest disagrees with registered identities")
    return {
        "source_package_root": package_root,
        "paths": paths,
        "skeleton": skeleton,
        "identity": {
            "run_id": run_id,
            "set_id": row["set_id"],
            "weights": {"sha256": row["model_sha256"]},
            "references": references,
            "onnx_interface": interface,
            "model_schema_binding_sha256": binding["binding_sha256"],
        },
    }


def _producer() -> dict[str, Any]:
    import onnx

    modules = [
        "fisheye.utils.export_pose_deployment_bundle",
        "fisheye.shared.pose_model_skeleton",
        "fisheye.shared.pose_deployment_manifest",
        "fisheye.shared.pose_onnx_interface",
        "fisheye.shared.pose_model_schema_binding",
        "fisheye.shared.pose_model_input_contract",
    ]
    return {
        "module": "fisheye.utils.export_pose_deployment_bundle",
        "git": get_git_info(),
        "python_version": sys.version,
        "onnx_version": onnx.__version__,
        "sqlite_version": sqlite3.sqlite_version,
        "source_sha256": {
            name: require_artifact_content_identity(
                Path(importlib.import_module(name).__file__), role="producer source"
            )["sha256"]
            for name in modules
        },
    }


def export_bundle(
    *, registry: Path, run_id: str, destination: Path, apply: bool = False
) -> dict[str, Any]:
    """Plan by default; export an explicitly requested new immutable revision."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Deployment destination already exists: {destination}")
    destination = destination.expanduser().resolve()
    plan = _prepare(registry, run_id)
    if destination.is_relative_to(plan["source_package_root"]):
        raise ValueError("Deployment output must not mutate the source model package")
    identity = plan["identity"]
    summary = {
        "status": "planned",
        "run_id": run_id,
        "set_id": identity["set_id"],
        "destination": str(destination),
        "selector_activation": False,
        "source_onnx_sha256": identity["references"]["onnx"]["sha256"],
        "skeleton_id": plan["skeleton"]["skeleton_id"],
        "kpt_shape": plan["skeleton"]["kpt_shape"],
    }
    if not apply:
        return summary
    # Reserving the directory excludes concurrent compliant writers. On failure
    # retain the incomplete directory for inspection; a retry uses a new path.
    destination.mkdir(parents=True, exist_ok=False)
    owner_stat = destination.stat()
    owner = (owner_stat.st_dev, owner_stat.st_ino)

    def require_owned():
        current = destination.stat()
        if destination.is_symlink() or (current.st_dev, current.st_ino) != owner:
            raise RuntimeError("Deployment output directory ownership changed")

    for role, source in plan["paths"].items():
        require_owned()
        shutil.copy2(
            source, destination / identity["references"][role]["relative_path"]
        )
        require_owned()
    skeleton_path = destination / POSE_MODEL_SKELETON_FILENAME
    require_owned()
    write_json_atomic(skeleton_path, plan["skeleton"], overwrite=False)
    references = dict(identity["references"])
    references["pose_model_skeleton"] = artifact_reference(
        skeleton_path, relative_path=skeleton_path.name
    )
    document = canonical_manifest(
        {
            "status": "complete",
            "task": "pose",
            "run_id": run_id,
            "set_id": identity["set_id"],
            "path_base": "manifest_directory",
            "selector_activation": False,
            "weights": identity["weights"],
            "onnx_interface": identity["onnx_interface"],
            "producer": _producer(),
            **references,
        }
    )
    # Both sides of copying must still establish the same admitted identity.
    if canonical_json_bytes(
        _prepare(registry, run_id)["identity"]
    ) != canonical_json_bytes(identity):
        raise ValueError("Registered pose source changed during deployment publication")
    validate_pose_deployment_document(document, root=destination)
    require_owned()
    manifest_path = destination / f"{run_id}.canonical.manifest.json"
    pending = destination / ".canonical.manifest.pending.json"
    write_json_atomic(pending, document, overwrite=False)
    require_owned()
    # link(), unlike rename(), cannot replace a competing existing publication.
    os.link(pending, manifest_path)
    require_owned()
    pending.unlink()
    manifest_sha = artifact_reference(manifest_path, relative_path=manifest_path.name)[
        "sha256"
    ]
    return {
        **summary,
        "status": "complete",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "pose_model_skeleton_sha256": references["pose_model_skeleton"]["sha256"],
        "artifacts": {role: references[role] for role in ARTIFACT_ROLES},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser(
        "export", help="Create a new transfer package; read-only unless --apply"
    )
    export.add_argument("--registry", type=Path, required=True)
    export.add_argument("--model-run-id", required=True)
    export.add_argument("--destination", type=Path, required=True)
    mode = export.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    validate = commands.add_parser(
        "validate", help="Verify a received bundle against its handoff digest"
    )
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_bundle(
            registry=args.registry,
            run_id=args.model_run_id,
            destination=args.destination,
            apply=args.apply,
        )
    else:
        result = load_pose_deployment_bundle(
            args.manifest, expected_manifest_sha256=args.expected_manifest_sha256
        )
        result = {
            "status": "valid",
            "run_id": result["manifest"]["payload"]["run_id"],
            "selector_activation": False,
        }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
