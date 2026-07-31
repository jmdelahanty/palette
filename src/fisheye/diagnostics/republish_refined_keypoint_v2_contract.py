"""Republish unchanged refined-keypoint arrays under the current exact contract."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping
from uuid import uuid4

import zarr

from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.keypoint_quality_schema import KEYPOINT_QUALITY_SCHEMA_V1
from fisheye.shared.zarr.keypoint_schema import (
    KEYPOINT_SCHEMA_V2,
    REFINED_KEYPOINT_SCHEMA_V2,
)
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr.refined_keypoint_publication import (
    republish_selector_ineligible_refined_keypoint_snapshot,
)

SCHEMA_ID = "palette.refined_keypoint.contract_republication_receipt"
SCHEMA_VERSION = 1


def _strict_manifest(group: Any) -> dict[str, Any]:
    value = group.attrs.get("run_manifest")
    if not isinstance(value, Mapping):
        raise ValueError("Source run does not contain an object run_manifest.")
    canonical_json_bytes(value)
    return dict(value)


def _metadata_fingerprint(path: Path) -> str:
    digest = sha256()
    for metadata in sorted(path.rglob("zarr.json")):
        relative = metadata.relative_to(path).as_posix().encode("utf-8")
        payload = metadata.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _artifact_stats(path: Path) -> dict[str, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "file_count": len(files),
        "apparent_bytes": sum(item.stat().st_size for item in files),
    }


def _palette_revision(*, allow_dirty: bool) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[3]

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = git("status", "--short")
    if status and not allow_dirty:
        raise RuntimeError(
            "Palette worktree is dirty; commit before producing reproducible evidence."
        )
    return {
        "repository": str(repository),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "worktree_clean": not status,
        "dirty_status": [] if not status else status.splitlines(),
        "driver": (
            "src/fisheye/diagnostics/" "republish_refined_keypoint_v2_contract.py"
        ),
    }


def _require_benchmark_destination(path: Path) -> Path:
    destination = path.expanduser().resolve()
    if ".palette_benchmarks" not in destination.parts:
        raise ValueError("Destination must be below a .palette_benchmarks namespace.")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    return destination


def republish(args: argparse.Namespace) -> dict[str, object]:
    destination = _require_benchmark_destination(args.destination)
    palette = _palette_revision(allow_dirty=args.allow_dirty)
    destination.parent.mkdir(parents=True, exist_ok=True)
    hidden = destination.parent / f".{destination.name}.partial.{uuid4().hex}"
    hidden.mkdir()

    source_paths = {
        "refined": args.source_refined_zarr.expanduser().resolve(),
        "raw": args.raw_zarr.expanduser().resolve(),
        "quality": args.quality_zarr.expanduser().resolve(),
        "crop": args.crop_zarr.expanduser().resolve(),
    }
    fingerprints_before = {
        name: _metadata_fingerprint(path) for name, path in source_paths.items()
    }
    started = time.perf_counter()
    try:
        roots = {
            name: zarr.open_group(str(path), mode="r", use_consolidated=False)
            for name, path in source_paths.items()
        }
        refined_group = roots["refined"]["refined_keypoints_runs"][
            args.source_refined_run
        ]
        raw_group = roots["raw"]["keypoints_runs"][args.raw_run]
        quality_group = roots["quality"]["keypoint_quality_runs"][args.quality_run]
        crop_group = roots["crop"]["crop_runs"][args.crop_run]
        manifests = {
            "refined": _strict_manifest(refined_group),
            "raw": _strict_manifest(raw_group),
            "quality": _strict_manifest(quality_group),
            "crop": _strict_manifest(crop_group),
        }
        refined_arrays = {
            path: refined_group[path]
            for path in REFINED_KEYPOINT_SCHEMA_V2.binding_paths
        }
        raw_arrays = {
            path: raw_group[path] for path in KEYPOINT_SCHEMA_V2.binding_paths
        }
        quality_arrays = {
            path: quality_group[path]
            for path in KEYPOINT_QUALITY_SCHEMA_V1.binding_paths
        }
        crop_arrays = {path: crop_group[path] for path in crop_group.array_keys()}

        publication = republish_selector_ineligible_refined_keypoint_snapshot(
            source_refined_manifest=manifests["refined"],
            source_refined_arrays=refined_arrays,
            raw_manifest=manifests["raw"],
            quality_manifest=manifests["quality"],
            crop_manifest=manifests["crop"],
            raw_arrays=raw_arrays,
            quality_arrays=quality_arrays,
            source_crop_arrays=crop_arrays,
            destination=hidden / "refined_keypoints.zarr",
            run_id=args.destination_run,
            shadow_root=hidden,
            created_by="republish_refined_keypoint_v2_contract",
        )
        fingerprints_after = {
            name: _metadata_fingerprint(path) for name, path in source_paths.items()
        }
        if fingerprints_after != fingerprints_before:
            raise RuntimeError("A source metadata tree changed during republication.")
        source_content = manifests["refined"]["payload"]["logical_content"]
        result: dict[str, object] = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "created_at_utc": utc_now(),
            "benchmark_only": True,
            "selector_eligible": False,
            "registry_registered": False,
            "palette": palette,
            "source": {
                name: {
                    "path": str(path),
                    "run_id": getattr(args, f"{name}_run", None)
                    or args.source_refined_run,
                    "manifest_digest": canonical_json_sha256(manifests[name]),
                    "metadata_fingerprint_before": fingerprints_before[name],
                    "metadata_fingerprint_after": fingerprints_after[name],
                }
                for name, path in source_paths.items()
            },
            "artifact": {
                "path": "refined_keypoints.zarr",
                "run_id": publication.run_id,
                "manifest_digest": canonical_json_sha256(publication.manifest),
                "logical_content_digest": publication.manifest["payload"][
                    "logical_content"
                ]["digest"],
                "logical_content_equal": (
                    publication.manifest["payload"]["logical_content"] == source_content
                ),
                "snapshot_identity_preserved": True,
                "skeleton_semantics_inlined": True,
                "storage": publication.plans.as_manifest()["object_estimate"],
                "timing_seconds": dict(publication.phase_seconds),
                **_artifact_stats(publication.output_path),
            },
            "elapsed_seconds": time.perf_counter() - started,
            "production_state": {
                "selectors_written": False,
                "registry_written": False,
                "source_archives_mutated": False,
                "training_artifacts_written": False,
            },
        }
        (hidden / "handoff_manifest.json").write_bytes(
            canonical_json_bytes(result) + b"\n"
        )
        os.replace(hidden, destination)
        return result
    except Exception:
        if args.remove_failed_partial and hidden.exists():
            shutil.rmtree(hidden)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-refined-zarr", type=Path, required=True)
    parser.add_argument("--source-refined-run", required=True)
    parser.add_argument("--raw-zarr", type=Path, required=True)
    parser.add_argument("--raw-run", required=True)
    parser.add_argument("--quality-zarr", type=Path, required=True)
    parser.add_argument("--quality-run", required=True)
    parser.add_argument("--crop-zarr", type=Path, required=True)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--destination-run", required=True)
    parser.add_argument("--remove-failed-partial", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main() -> None:
    result = republish(_parser().parse_args())
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
