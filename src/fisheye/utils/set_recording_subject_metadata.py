#!/usr/bin/env python3
"""Publish audited count-only subject metadata for organized recordings.

This command is for recordings whose species, age, and expected subject count
are known but whose individual biological identities are not.  It publishes
canonical immutable subject/setup authorities without manufacturing subject
IDs, mirrors compatibility metadata, patches recording manifests atomically,
and can refresh the Palette registry.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import zarr

from fisheye.registry.db import Registry
from fisheye.registry.prune_stale_datasets import create_backup
from fisheye.shared.batch_logging import make_run_id, utc_now
from fisheye.shared.experiment_setup import (
    MissingExperimentSetupError,
    build_experiment_setup_record,
    experiment_setup_sha256,
    publish_experiment_setup,
    resolve_experiment_setup,
)
from fisheye.shared.import_source_fingerprint import optional_source_stat_fingerprint_attrs
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.subject_metadata import (
    MissingSubjectMetadataError,
    build_subject_metadata_record,
    publish_subject_metadata,
    resolve_subject_metadata,
    subject_metadata_sha256,
)
from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_IDENTITY_PROFILE,
    SourceRecordingIdentityError,
    load_source_recording_identity_profile,
)


TOOL_NAME = "fisheye.utils.set_recording_subject_metadata"
REPAIR_TYPE = "manual_recording_subject_metadata_v1"


def _open_root(path: Path, *, mode: str) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode=mode, use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode=mode)


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload, sha256(raw).hexdigest()


def _analysis_zarr(recording_dir: Path) -> Path:
    candidates = sorted((recording_dir / "zarr").glob("*_analysis.zarr"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one *_analysis.zarr under {recording_dir / 'zarr'}, "
            f"found {len(candidates)}"
        )
    if not (candidates[0] / "zarr.json").is_file():
        raise ValueError(f"analysis Zarr lacks root zarr.json: {candidates[0]}")
    return candidates[0].resolve()


def _subject_metadata(*, species: str, dpf: int, subject_count: int) -> dict[str, Any]:
    return {
        "species": species,
        "subject_count": subject_count,
        "dpf_at_acquisition": dpf,
        "days_post_fertilization": dpf,
        "subject_type": "individual" if subject_count == 1 else "group",
        "identity_scope": "count_only_no_subject_ids",
        "source": "recording_manifest_manual_assertion",
        "status": "user_asserted",
    }


def _setup_source() -> dict[str, str]:
    return {
        "kind": "recording_manifest_subject_metadata",
        "file_name": "recording_manifest.json",
        "json_pointer": "/subject_metadata",
        "count_field": "subject_count",
    }


def _desired(metadata: Mapping[str, Any]) -> dict[str, Any]:
    subject_record = build_subject_metadata_record(metadata)
    subject_digest = subject_metadata_sha256(subject_record)
    subject_run = f"subject_metadata_{subject_digest[:16]}"
    subject_ref = f"analysis/subject_metadata_runs/{subject_run}"
    setup_record = build_experiment_setup_record(
        metadata,
        source=_setup_source(),
        subject_metadata_sha256=subject_digest,
        subject_metadata_ref=subject_ref,
    )
    setup_digest = experiment_setup_sha256(setup_record)
    return {
        "subject_metadata": dict(metadata),
        "subject_metadata_run": subject_run,
        "subject_metadata_sha256": subject_digest,
        "experiment_setup_run": f"experiment_setup_{setup_digest[:16]}",
        "experiment_setup_sha256": setup_digest,
        "experiment_setup_record": setup_record,
    }


def _conflicts(existing: Mapping[str, Any], desired: Mapping[str, Any]) -> list[str]:
    conflicts: list[str] = []
    for key, value in desired.items():
        current = existing.get(key)
        if current not in (None, "") and current != value:
            conflicts.append(f"{key}: existing={current!r}, desired={value!r}")
    return conflicts


def plan_recording(
    recording_dir: Path,
    *,
    species: str,
    dpf: int,
    subject_count: int,
) -> dict[str, Any]:
    root_dir = recording_dir.expanduser().resolve()
    manifest_path = root_dir / "recording_manifest.json"
    manifest, manifest_sha256 = _load_manifest(manifest_path)
    zarr_path = _analysis_zarr(root_dir)
    metadata = _subject_metadata(species=species, dpf=dpf, subject_count=subject_count)
    desired = _desired(metadata)

    manifest_fields = {
        "species": species,
        "dpf_at_acquisition": dpf,
        "subject_count": subject_count,
        "num_dishes": 1,
        "fish_per_dish": subject_count,
        "subject_metadata": metadata,
    }
    conflicts = _conflicts(manifest, manifest_fields)
    if (
        load_source_recording_identity_profile(zarr_path)
        == SOURCE_RECORDING_IDENTITY_PROFILE
    ):
        conflicts.append(
            "current-profile source recordings are unsupported by this legacy tool"
        )

    root = _open_root(zarr_path, mode="r")
    try:
        current_subject = resolve_subject_metadata(root, allow_legacy=False)
    except MissingSubjectMetadataError:
        current_subject = None
    except Exception as exc:
        conflicts.append(f"subject authority invalid: {type(exc).__name__}: {exc}")
        current_subject = None
    if current_subject is not None and dict(current_subject.metadata) != metadata:
        conflicts.append("selected canonical subject metadata differs from requested metadata")

    try:
        current_setup = resolve_experiment_setup(root, allow_legacy=False)
    except MissingExperimentSetupError:
        current_setup = None
    except Exception as exc:
        conflicts.append(f"setup authority invalid: {type(exc).__name__}: {exc}")
        current_setup = None
    if current_setup is not None:
        if current_setup.expected_subject_count != subject_count:
            conflicts.append(
                "selected canonical expected_subject_count differs from requested count"
            )
        if current_setup.record_sha256 != desired["experiment_setup_sha256"]:
            conflicts.append("selected canonical experiment setup differs from requested setup")

    already_present = (
        not conflicts
        and all(manifest.get(key) == value for key, value in manifest_fields.items())
        and current_subject is not None
        and current_subject.record_sha256 == desired["subject_metadata_sha256"]
        and current_setup is not None
        and current_setup.record_sha256 == desired["experiment_setup_sha256"]
    )
    return {
        "recording": root_dir.name,
        "recording_dir": str(root_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256_before": manifest_sha256,
        "zarr_path": str(zarr_path),
        "status": "conflict" if conflicts else ("unchanged" if already_present else "planned"),
        "conflicts": conflicts,
        "manifest_fields": manifest_fields,
        "desired": desired,
    }


def _patch_manifest(plan: Mapping[str, Any], *, repair_id: str, reason: str) -> None:
    manifest_path = Path(str(plan["manifest_path"]))
    payload, current_sha256 = _load_manifest(manifest_path)
    if current_sha256 != plan["manifest_sha256_before"]:
        raise ValueError(f"manifest changed after preflight: {manifest_path}")
    fields = plan["manifest_fields"]
    assert isinstance(fields, Mapping)
    previous = {key: payload.get(key) for key in fields}
    payload.update(fields)
    repairs = payload.get("metadata_repairs")
    if repairs is None:
        repairs = []
        payload["metadata_repairs"] = repairs
    if not isinstance(repairs, list):
        raise ValueError(f"metadata_repairs is not a list: {manifest_path}")
    repairs.append(
        {
            "repair_type": REPAIR_TYPE,
            "repair_id": repair_id,
            "created_at_utc": utc_now(),
            "tool": TOOL_NAME,
            "reason": reason,
            "manifest_sha256_before": current_sha256,
            "previous": previous,
            "updated": dict(fields),
        }
    )
    write_json_atomic(manifest_path, payload)


def _source_artifact(manifest_path: Path) -> dict[str, Any]:
    fingerprint = optional_source_stat_fingerprint_attrs(
        manifest_path,
        attr_prefix="source_manifest",
    ).get("source_manifest_fingerprint")
    return {
        "kind": "recording_manifest_subject_metadata",
        "path": str(manifest_path),
        "stat_fingerprint": fingerprint,
        "json_pointer": "/subject_metadata",
    }


def _write_compatibility_metadata(root: zarr.Group, metadata: Mapping[str, Any]) -> None:
    root.attrs["species"] = metadata["species"]
    root.attrs["dpf_at_acquisition"] = metadata["dpf_at_acquisition"]
    analysis_metadata = root.require_group("analysis_metadata")
    analysis_metadata.attrs["subject_metadata"] = json.dumps(metadata, sort_keys=True)
    raw_context = analysis_metadata.attrs.get("session_context")
    try:
        context = json.loads(raw_context) if isinstance(raw_context, str) else dict(raw_context or {})
    except (TypeError, ValueError):
        context = {}
    context.update(
        {
            "species": metadata["species"],
            "dpf_at_acquisition": metadata["dpf_at_acquisition"],
            "subject_count": metadata["subject_count"],
        }
    )
    analysis_metadata.attrs["session_context"] = json.dumps(context, sort_keys=True)


def apply_plan(
    plan: Mapping[str, Any],
    *,
    repair_id: str,
    reason: str,
    registry: Registry | None,
) -> dict[str, Any]:
    if plan.get("status") == "conflict":
        raise ValueError(f"refusing conflicting plan for {plan['recording']}")
    if plan.get("status") == "unchanged":
        return {**dict(plan), "status": "unchanged", "registry_dataset_id": None}
    zarr_path = Path(str(plan["zarr_path"]))
    try:
        source_profile = load_source_recording_identity_profile(zarr_path)
    except SourceRecordingIdentityError as exc:
        raise ValueError(
            "current-profile source classification is invalid; legacy mutation "
            "is forbidden"
        ) from exc
    if source_profile == SOURCE_RECORDING_IDENTITY_PROFILE:
        raise ValueError(
            "legacy manual subject metadata tool does not mutate "
            "current-profile source recordings"
        )

    fresh = plan_recording(
        Path(str(plan["recording_dir"])),
        species=str(plan["manifest_fields"]["species"]),
        dpf=int(plan["manifest_fields"]["dpf_at_acquisition"]),
        subject_count=int(plan["manifest_fields"]["subject_count"]),
    )
    if fresh != plan:
        raise ValueError(f"apply-time plan changed for {plan['recording']}")

    manifest_path = Path(str(plan["manifest_path"]))
    _patch_manifest(plan, repair_id=repair_id, reason=reason)
    metadata = dict(plan["desired"]["subject_metadata"])
    artifact = _source_artifact(manifest_path)
    root = _open_root(zarr_path, mode="r+")
    subject = publish_subject_metadata(
        root,
        metadata,
        source_artifact=artifact,
        provenance_command=TOOL_NAME,
    )
    setup_record = build_experiment_setup_record(
        metadata,
        source=_setup_source(),
        subject_metadata_sha256=subject.record_sha256,
        subject_metadata_ref=subject.group_path,
    )
    setup = publish_experiment_setup(
        root,
        setup_record,
        source_artifact=artifact,
        provenance_command=TOOL_NAME,
    )
    _write_compatibility_metadata(root, metadata)

    verified_subject = resolve_subject_metadata(root, allow_legacy=False)
    verified_setup = resolve_experiment_setup(root, allow_legacy=False)
    if verified_subject.subject_ids:
        raise ValueError("count-only publication unexpectedly created subject IDs")
    if verified_setup.expected_subject_count != metadata["subject_count"]:
        raise ValueError("published expected subject count failed validation")

    registry_dataset_id = None
    if registry is not None:
        registry_dataset_id = registry.scan_zarr(zarr_path)
    return {
        **dict(plan),
        "status": "applied",
        "subject_metadata_path": subject.group_path,
        "experiment_setup_path": setup.group_path,
        "registry_dataset_id": registry_dataset_id,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_dirs", nargs="+", type=Path)
    parser.add_argument("--species", required=True)
    parser.add_argument("--dpf-at-acquisition", required=True, type=int)
    parser.add_argument("--subject-count", required=True, type=int)
    parser.add_argument("--reason", default="subject metadata supplied by the recording owner")
    parser.add_argument(
        "--repair-id",
        default=f"manual_recording_subject_metadata_{make_run_id()}",
    )
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    species = str(args.species).strip()
    if not species:
        raise ValueError("--species must not be empty")
    if args.dpf_at_acquisition < 0:
        raise ValueError("--dpf-at-acquisition must be >= 0")
    if args.subject_count < 1:
        raise ValueError("--subject-count must be >= 1")

    plans = [
        plan_recording(
            path,
            species=species,
            dpf=int(args.dpf_at_acquisition),
            subject_count=int(args.subject_count),
        )
        for path in args.recording_dirs
    ]
    if args.apply and any(plan["status"] == "conflict" for plan in plans):
        raise ValueError("one or more recording plans conflict; nothing applied")
    if args.apply and args.registry is not None:
        if args.backup is None:
            raise ValueError("--apply with --registry requires --backup")
        if args.backup.exists():
            raise FileExistsError(f"backup already exists: {args.backup}")
        create_backup(args.registry, args.backup)

    registry = Registry(args.registry) if args.apply and args.registry is not None else None
    try:
        results = (
            [
                apply_plan(
                    plan,
                    repair_id=str(args.repair_id),
                    reason=str(args.reason),
                    registry=registry,
                )
                for plan in plans
            ]
            if args.apply
            else plans
        )
    finally:
        if registry is not None:
            registry.close()

    report = {
        "schema_id": "palette.manual_recording_subject_metadata_report.v1",
        "mode": "apply" if args.apply else "dry_run",
        "recording_count": len(results),
        "status_counts": {
            status: sum(result["status"] == status for result in results)
            for status in sorted({str(result["status"]) for result in results})
        },
        "registry_backup": str(args.backup) if args.apply and args.backup else None,
        "recordings": results,
    }
    if args.output:
        write_json_atomic(args.output, report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if any(result["status"] == "conflict" for result in results) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
