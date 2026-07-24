#!/usr/bin/env python3
"""Run one scale/repetition detection storage block through local scratch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from typing import Any, Mapping, Sequence

from fisheye.cluster.lsf.bundle import write_json_snapshot
from fisheye.shared.zarr.benchmark_environment import (
    STORAGE_BENCHMARK_THREAD_ENVIRONMENT,
)
from fisheye.shared.zarr.benchmark_fixture import (
    inventory_tree,
    thaw_tree_for_cleanup,
)
from fisheye.shared.zarr.benchmark_matrix import (
    require_storage_benchmark_matrix_manifest,
)
from fisheye.shared.zarr.benchmark_publication import publish_benchmark_candidate
from fisheye.shared.zarr.canonical_detection_benchmark import (
    load_detection_benchmark_input,
)
from fisheye.shared.zarr.detection_benchmark_access import (
    require_detection_consumer_workloads,
)
from fisheye.shared.zarr.detection_benchmark_staging import (
    prepare_canonical_detection_benchmark_staging,
)


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_repo(repo: Path, expected_commit: str) -> dict[str, object]:
    resolved = repo.expanduser().resolve()
    actual_commit = subprocess.run(
        ["git", "-C", str(resolved), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"Palette commit mismatch: expected {expected_commit}, got {actual_commit}."
        )
    dirty = subprocess.run(
        ["git", "-C", str(resolved), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"Palette benchmark checkout is dirty: {resolved}")
    import fisheye

    imported = Path(str(fisheye.__file__)).resolve()
    expected_import = (resolved / "src" / "fisheye" / "__init__.py").resolve()
    if imported != expected_import:
        raise RuntimeError(
            f"fisheye imported from {imported}, expected {expected_import}."
        )
    return {
        "repo": str(resolved),
        "commit": actual_commit,
        "clean": True,
        "fisheye_import": str(imported),
    }


def _resolve_scratch_root(
    *,
    workflow_id: str,
    scale_id: str,
    repetition_index: int,
    configured_base: Path | None,
    allow_local: bool,
) -> Path:
    job_id = str(os.environ.get("LSB_JOBID") or "").strip()
    job_index = str(os.environ.get("LSB_JOBINDEX") or "").strip()
    user = str(os.environ.get("USER") or "unknown")
    if not job_id and not allow_local:
        raise RuntimeError("Refusing benchmark block execution outside LSF.")
    if configured_base is not None:
        base = configured_base.expanduser().resolve()
    else:
        work_unit = f"{job_id}_{job_index}" if job_index not in {"", "0"} else job_id
        base = Path("/scratch") / user / work_unit
    if str(base).startswith("/groups/"):
        raise ValueError("Node-local scratch cannot be under /groups.")
    namespace = (base / "canonical_detection_storage").resolve()
    root = (
        namespace
        / workflow_id
        / f"{scale_id}_repetition_{repetition_index:03d}"
    ).resolve()
    if root == namespace or not root.is_relative_to(namespace):
        raise ValueError("Resolved scratch work unit escaped its benchmark namespace.")
    if root.exists():
        raise FileExistsError(f"Scratch root already exists: {root}")
    return root


def _select_block(
    matrix: Mapping[str, Any],
    *,
    scale_id: str,
    repetition_index: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if matrix.get("schema_id") != "palette.storage_benchmark_matrix":
        raise ValueError("Unsupported storage benchmark matrix.")
    matches = [
        item
        for item in matrix.get("repetitions", [])
        if isinstance(item, dict)
        and item.get("scale_id") == scale_id
        and item.get("repetition_index") == repetition_index
    ]
    if len(matches) != 1:
        raise ValueError("Matrix does not contain exactly one requested block.")
    candidates = {
        str(item["candidate_id"]): item
        for item in matrix.get("candidates", [])
        if isinstance(item, dict) and item.get("scale_id") == scale_id
    }
    trials = matches[0].get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("Matrix block contains no trials.")
    if any(str(trial.get("candidate_id")) not in candidates for trial in trials):
        raise ValueError("Matrix block references an unknown candidate.")
    if any(bool(trial.get("destination_collision")) for trial in trials):
        raise FileExistsError("Matrix block contains a destination collision.")
    return matches[0], candidates


def _scale_dimensions(matrix: Mapping[str, Any], scale_id: str) -> dict[str, int]:
    matches = [
        item
        for item in matrix.get("scales", [])
        if isinstance(item, dict) and item.get("scale_id") == scale_id
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("dimensions"), dict):
        raise ValueError("Matrix does not contain exact scale dimensions.")
    return {str(key): int(value) for key, value in matches[0]["dimensions"].items()}


def run_benchmark_block(
    *,
    matrix_path: Path,
    fixture_root: Path,
    workflow_root: Path,
    block_report: Path,
    scale_id: str,
    repetition_index: int,
    recording_identity: str,
    palette_repo: Path,
    expected_commit: str,
    scratch_base: Path | None = None,
    keep_scratch: bool = False,
    allow_local: bool = False,
) -> dict[str, object]:
    """Execute one same-host block and publish every planned candidate."""

    workflow = workflow_root.expanduser().resolve()
    report_path = block_report.expanduser().resolve()
    if report_path.exists():
        raise FileExistsError(f"Block report already exists: {report_path}")
    if not report_path.is_relative_to(workflow):
        raise ValueError("Block report must be below the benchmark workflow root.")
    matrix = _read_json(matrix_path.expanduser().resolve())
    require_storage_benchmark_matrix_manifest(matrix)
    block, candidates = _select_block(
        matrix,
        scale_id=scale_id,
        repetition_index=repetition_index,
    )
    dimensions = _scale_dimensions(matrix, scale_id)
    fixture = fixture_root.expanduser().resolve()
    fixture_manifest_path = fixture / "fixture_manifest.json"
    fixture_manifest_bytes_sha = _sha256_file(fixture_manifest_path)
    fixture_manifest = _read_json(fixture_manifest_path)
    if (
        fixture_manifest.get("status") != "published_immutable"
        or fixture_manifest.get("benchmark_only") is not True
        or fixture_manifest.get("canonical") is not False
    ):
        raise ValueError("Fixture is not an immutable noncanonical benchmark source.")
    repo_evidence = _verify_repo(palette_repo, expected_commit)
    effective_thread_environment = {
        key: os.environ.get(key) for key in STORAGE_BENCHMARK_THREAD_ENVIRONMENT
    }
    if (
        not allow_local
        and effective_thread_environment != STORAGE_BENCHMARK_THREAD_ENVIRONMENT
    ):
        raise RuntimeError(
            "LSF benchmark native-thread environment does not match the plan."
        )
    scratch_root = _resolve_scratch_root(
        workflow_id=str(matrix["matrix_id"]),
        scale_id=scale_id,
        repetition_index=repetition_index,
        configured_base=scratch_base,
        allow_local=allow_local,
    )
    started = time.perf_counter()
    partial: dict[str, object] = {
        "schema_id": "palette.canonical_detection_storage_benchmark_block",
        "schema_version": 1,
        "status": "running",
        "matrix": str(matrix_path.expanduser().resolve()),
        "matrix_fingerprint": matrix.get("matrix_fingerprint"),
        "fixture_id": fixture_manifest.get("fixture_id"),
        "fixture_root": str(fixture),
        "fixture_manifest_sha256": fixture_manifest_bytes_sha,
        "scale_id": scale_id,
        "repetition_index": repetition_index,
        "palette": repo_evidence,
        "native_thread_environment": effective_thread_environment,
        "scheduler": {
            "job_id": os.environ.get("LSB_JOBID"),
            "job_index": os.environ.get("LSB_JOBINDEX"),
            "queue": os.environ.get("LSB_QUEUE"),
            "hosts": os.environ.get("LSB_HOSTS"),
            "hostname": socket.gethostname(),
        },
        "scratch_root": str(scratch_root),
        "candidates": [],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True)
    try:
        local_source = scratch_root / "source.zarr"
        stage_started = time.perf_counter()
        shutil.copytree(fixture / str(fixture_manifest["copied_zarr_relative_path"]), local_source)
        stage_seconds = float(time.perf_counter() - stage_started)
        local_inventory = inventory_tree(local_source)
        expected_inventory = fixture_manifest["copied_inventory"]
        if (
            local_inventory.file_count != int(expected_inventory["file_count"])
            or local_inventory.apparent_bytes != int(expected_inventory["apparent_bytes"])
            or local_inventory.tree_sha256 != str(expected_inventory["tree_sha256"])
        ):
            raise RuntimeError("Node-local staged fixture does not match publication.")
        partial["stage_in"] = {
            "seconds": stage_seconds,
            "copy_method": "shutil.copytree",
            "returncode": 0,
            "source": str(fixture / str(fixture_manifest["copied_zarr_relative_path"])),
            "destination": str(local_source),
            "inventory": local_inventory.as_manifest(),
            "excluded_from_candidate_timings": True,
        }

        legacy_input = load_detection_benchmark_input(
            local_source,
            recording_identity=recording_identity,
            frame_limit=int(dimensions["n_frames"]),
        )
        if legacy_input.dimensions.as_manifest() != {
            **dimensions,
            "n_frame_boundaries": int(dimensions["n_frames"]) + 1,
        }:
            raise ValueError("Staged source cardinality does not match matrix scale.")
        canonical_staging = scratch_root / "canonical_staging.zarr"
        staging_report = prepare_canonical_detection_benchmark_staging(
            legacy_input,
            destination=canonical_staging,
            scratch_root=scratch_root,
        )
        del legacy_input
        partial["canonical_staging"] = staging_report

        candidate_records: list[dict[str, object]] = []
        for trial in block["trials"]:
            candidate_id = str(trial["candidate_id"])
            candidate = candidates[candidate_id]
            request = candidate["request"]
            local_candidate = scratch_root / "candidates" / f"{candidate_id}.zarr"
            local_report = scratch_root / "reports" / f"{candidate_id}.json"
            command = [
                str(palette_repo.expanduser().resolve() / "scripts" / "py"),
                "-m",
                "fisheye.diagnostics.run_canonical_detection_storage_candidate",
                str(canonical_staging),
                str(local_candidate),
                "--report",
                str(local_report),
                "--benchmark-root",
                str(scratch_root),
                "--chunk-bytes",
                str(request["target_chunk_bytes"]),
                "--layout",
                str(request["layout"]),
                "--apply",
            ]
            if request.get("target_shard_bytes") is not None:
                command.extend(["--shard-bytes", str(request["target_shard_bytes"])])
            subprocess_started = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=str(palette_repo.expanduser().resolve()),
                check=True,
                text=True,
                capture_output=True,
            )
            subprocess_seconds = float(time.perf_counter() - subprocess_started)
            local_result = _read_json(local_report)
            if local_result.get("storage_plan") != candidate.get("stage_plan"):
                raise RuntimeError(f"Resolved plan drift for candidate {candidate_id}.")
            if not all(
                bool(item.get("exact"))
                for item in local_result.get("digest_validation", {}).values()
            ):
                raise RuntimeError(f"Local candidate validation failed: {candidate_id}")

            local_read_report = (
                scratch_root / "reports" / f"{candidate_id}.local-read.json"
            )
            local_read_command = [
                str(palette_repo.expanduser().resolve() / "scripts" / "py"),
                "-m",
                "fisheye.diagnostics.run_canonical_detection_storage_reads",
                str(canonical_staging),
                str(local_candidate),
                "--report",
                str(local_read_report),
                "--storage-tier",
                "node_local_scratch",
                "--chunk-bytes",
                str(request["target_chunk_bytes"]),
                "--layout",
                str(request["layout"]),
                "--read-seed",
                str(matrix["seed"]),
            ]
            if request.get("target_shard_bytes") is not None:
                local_read_command.extend(
                    ["--shard-bytes", str(request["target_shard_bytes"])]
                )
            local_read_started = time.perf_counter()
            local_read_completed = subprocess.run(
                local_read_command,
                cwd=str(palette_repo.expanduser().resolve()),
                check=True,
                text=True,
                capture_output=True,
            )
            local_read_subprocess_seconds = float(
                time.perf_counter() - local_read_started
            )
            local_reads = _read_json(local_read_report)
            require_detection_consumer_workloads(local_reads["consumer_workloads"])
            if not all(bool(item.get("exact")) for item in local_reads["arrays"]):
                raise RuntimeError(
                    f"Local read candidate validation failed: {candidate_id}"
                )

            published = Path(str(trial["destination"])).expanduser().resolve()
            publication = publish_benchmark_candidate(
                source=local_candidate,
                destination=published,
                workflow_root=workflow,
            )
            prfs_read_report = (
                scratch_root / "reports" / f"{candidate_id}.prfs-read.json"
            )
            read_command = [
                str(palette_repo.expanduser().resolve() / "scripts" / "py"),
                "-m",
                "fisheye.diagnostics.run_canonical_detection_storage_reads",
                str(canonical_staging),
                str(published),
                "--report",
                str(prfs_read_report),
                "--storage-tier",
                "shared_prfs",
                "--chunk-bytes",
                str(request["target_chunk_bytes"]),
                "--layout",
                str(request["layout"]),
                "--read-seed",
                str(matrix["seed"]),
            ]
            if request.get("target_shard_bytes") is not None:
                read_command.extend(
                    ["--shard-bytes", str(request["target_shard_bytes"])]
                )
            read_started = time.perf_counter()
            read_completed = subprocess.run(
                read_command,
                cwd=str(palette_repo.expanduser().resolve()),
                check=True,
                text=True,
                capture_output=True,
            )
            read_subprocess_seconds = float(time.perf_counter() - read_started)
            prfs_reads = _read_json(prfs_read_report)
            require_detection_consumer_workloads(prfs_reads["consumer_workloads"])
            if not all(bool(item.get("exact")) for item in prfs_reads["arrays"]):
                raise RuntimeError(f"PRFS candidate validation failed: {candidate_id}")
            evidence_base = published.with_suffix("")
            local_evidence_path = Path(f"{evidence_base}.local-write.json")
            local_read_evidence_path = Path(f"{evidence_base}.local-read.json")
            publication_path = Path(f"{evidence_base}.publication.json")
            read_path = Path(f"{evidence_base}.prfs-read.json")
            for evidence_path in (
                local_evidence_path,
                local_read_evidence_path,
                publication_path,
                read_path,
            ):
                if evidence_path.exists():
                    raise FileExistsError(f"Candidate evidence exists: {evidence_path}")
            write_json_snapshot(local_evidence_path, local_result)
            write_json_snapshot(local_read_evidence_path, local_reads)
            write_json_snapshot(publication_path, publication)
            write_json_snapshot(read_path, prfs_reads)
            candidate_records.append(
                {
                    "position": int(trial["position"]),
                    "candidate_id": candidate_id,
                    "layout": request["layout"],
                    "physical_fingerprint": candidate["physical_fingerprint"],
                    "command": command,
                    "subprocess_seconds": subprocess_seconds,
                    "subprocess_stdout": completed.stdout,
                    "subprocess_stderr": completed.stderr,
                    "local_read_command": local_read_command,
                    "local_read_subprocess_seconds": local_read_subprocess_seconds,
                    "local_read_subprocess_stdout": local_read_completed.stdout,
                    "local_read_subprocess_stderr": local_read_completed.stderr,
                    "prfs_read_command": read_command,
                    "prfs_read_subprocess_seconds": read_subprocess_seconds,
                    "prfs_read_subprocess_stdout": read_completed.stdout,
                    "prfs_read_subprocess_stderr": read_completed.stderr,
                    "local_candidate": str(local_candidate),
                    "published_candidate": str(published),
                    "local_write_report": str(local_evidence_path),
                    "local_read_report": str(local_read_evidence_path),
                    "publication_report": str(publication_path),
                    "prfs_read_report": str(read_path),
                    "publication": publication,
                    "prfs_reads": {
                        "direct_open_seconds": prfs_reads["direct_open_seconds"],
                        "consolidated_open_seconds": prfs_reads[
                            "consolidated_open_seconds"
                        ],
                        "all_exact": all(
                            bool(item["exact"]) for item in prfs_reads["arrays"]
                        ),
                        "consumer_workloads_exact": True,
                    },
                }
            )
            partial["candidates"] = candidate_records

        final_fixture_inventory = inventory_tree(
            fixture / str(fixture_manifest["copied_zarr_relative_path"])
        )
        expected_fixture_inventory = fixture_manifest["copied_inventory"]
        fixture_unchanged = (
            _sha256_file(fixture_manifest_path) == fixture_manifest_bytes_sha
            and final_fixture_inventory.file_count
            == int(expected_fixture_inventory["file_count"])
            and final_fixture_inventory.apparent_bytes
            == int(expected_fixture_inventory["apparent_bytes"])
            and final_fixture_inventory.tree_sha256
            == str(expected_fixture_inventory["tree_sha256"])
        )
        if not fixture_unchanged:
            raise RuntimeError("Shared benchmark fixture changed during the block.")
        partial.update(
            {
                "status": "complete",
                "fixture_unchanged": True,
                "fixture_final_inventory": final_fixture_inventory.as_manifest(),
                "total_seconds": float(time.perf_counter() - started),
            }
        )
        write_json_snapshot(report_path, partial)
        return partial
    except BaseException as exc:
        partial.update(
            {
                "status": "failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "total_seconds": float(time.perf_counter() - started),
            }
        )
        write_json_snapshot(report_path, partial)
        raise
    finally:
        if not keep_scratch and scratch_root.exists():
            local_source = scratch_root / "source.zarr"
            if local_source.exists():
                thaw_tree_for_cleanup(local_source)
            shutil.rmtree(scratch_root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--fixture-root", required=True, type=Path)
    parser.add_argument("--workflow-root", required=True, type=Path)
    parser.add_argument("--block-report", required=True, type=Path)
    parser.add_argument("--scale-id", required=True)
    parser.add_argument("--repetition-index", required=True, type=int)
    parser.add_argument("--recording-identity", required=True)
    parser.add_argument("--palette-repo", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--scratch-base", type=Path)
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--allow-local", action="store_true")
    args = parser.parse_args(argv)
    report = run_benchmark_block(
        matrix_path=args.matrix,
        fixture_root=args.fixture_root,
        workflow_root=args.workflow_root,
        block_report=args.block_report,
        scale_id=args.scale_id,
        repetition_index=args.repetition_index,
        recording_identity=args.recording_identity,
        palette_repo=args.palette_repo,
        expected_commit=args.expected_commit,
        scratch_base=args.scratch_base,
        keep_scratch=bool(args.keep_scratch),
        allow_local=bool(args.allow_local),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "scale_id": report["scale_id"],
                "repetition_index": report["repetition_index"],
                "candidate_count": len(report["candidates"]),
                "block_report": str(args.block_report.expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
