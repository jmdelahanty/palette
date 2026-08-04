"""Trace benchmark process-tree reads against explicit storage roots.

This Linux-only diagnostic runs one command beneath ``strace -ff -yy`` and
GNU ``time -v``.  It attributes successful read-family syscalls only when
strace identifies the file descriptor as a path below one of the caller's
explicit target roots.  The resulting bytes are process-requested file bytes,
not guaranteed filesystem or network transfer bytes.  Raw traces and their
digests remain available for independent replay.

The command receives no shell and this module grants no authority to mutate a
Zarr archive, selector, registry, or production profile.  It is intended to
wrap read-only benchmark commands whose own contracts enforce those rules.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence
import uuid

from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr.benchmark_environment import (
    STORAGE_BENCHMARK_THREAD_ENVIRONMENT,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from fisheye.analysis_workflows.storage_benchmark_catalog import (
    DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE,
)

TRACE_SCHEMA_ID = "palette.process_tree_storage_io_trace"
TRACE_SCHEMA_VERSION = 1
MEASUREMENT_SCOPE = "process_tree_file_syscalls"

_READ_LINE = re.compile(
    r"^(?:\[pid\s+\d+\]\s+)?"
    r"(?P<syscall>read|pread64|readv|preadv|preadv2)"
    r"\((?P<fd>\d+)<(?P<path>[^>]*)>,.*\)\s+=\s+(?P<result>-?\d+)"
)
_TIME_FIELDS = {
    "user_seconds": "User time (seconds)",
    "system_seconds": "System time (seconds)",
    "elapsed": "Elapsed (wall clock) time (h:mm:ss or m:ss)",
    "peak_rss_kib": "Maximum resident set size (kbytes)",
    "filesystem_input_count": "File system inputs",
    "filesystem_output_count": "File system outputs",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_target(value: str | Path) -> Path:
    supplied = Path(value).expanduser()
    if supplied.is_symlink():
        raise ValueError(f"storage trace target is a symlink: {supplied}")
    target = supplied.resolve(strict=True)
    if not target.is_dir() and not target.is_file():
        raise ValueError(f"storage trace target is not a file or directory: {target}")
    if target in {Path("/"), Path.home().resolve()}:
        raise ValueError("storage trace target is too broad")
    return target


def _safe_output(value: str | Path, *, targets: Sequence[Path]) -> Path:
    output = Path(value).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"storage trace output already exists: {output}")
    if not any("benchmark" in part.lower() for part in output.parts):
        raise ValueError(
            "storage trace output requires a benchmark-only path component"
        )
    if output in {Path("/"), Path.home().resolve()}:
        raise ValueError("storage trace output is too broad")
    if any(
        output == target
        or output.is_relative_to(target)
        or target.is_relative_to(output)
        for target in targets
    ):
        raise ValueError("storage trace output must be disjoint from every target")
    return output


def _safe_command(command: Sequence[str]) -> tuple[str, ...]:
    if (
        not isinstance(command, Sequence)
        or isinstance(command, (str, bytes))
        or not command
        or any(type(item) is not str or not item for item in command)
    ):
        raise ValueError("storage trace command must be a nonempty exact string array")
    executable = shutil.which(command[0])
    if executable is None:
        candidate = Path(command[0]).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        executable = str(candidate) if candidate.is_file() else None
    if executable is None:
        raise FileNotFoundError(
            f"storage trace executable does not resolve: {command[0]}"
        )
    return (str(Path(executable).resolve()), *command[1:])


def _classified_path(
    captured: str,
    *,
    targets: Sequence[Path],
) -> tuple[Path, str] | None:
    cleaned = captured.removesuffix(" (deleted)")
    if not cleaned.startswith("/"):
        return None
    path = Path(cleaned)
    for target in targets:
        if path == target or path.is_relative_to(target):
            kind = "metadata" if path.name == "zarr.json" else "payload"
            return path, kind
    return None


def parse_strace_reads(
    trace_paths: Sequence[Path],
    *,
    target_roots: Sequence[Path],
) -> dict[str, object]:
    """Parse successful attributed read-family syscalls from raw strace files."""

    targets = tuple(_safe_target(path) for path in target_roots)
    by_kind: dict[str, dict[str, Any]] = {
        kind: {"read_operations": 0, "read_bytes": 0, "paths": set()}
        for kind in ("metadata", "payload")
    }
    syscall_counts = {
        name: 0 for name in ("read", "pread64", "readv", "preadv", "preadv2")
    }
    parsed_line_count = 0
    unattributed_successful_read_count = 0
    for trace_path in sorted(trace_paths):
        if not trace_path.is_file() or trace_path.is_symlink():
            raise ValueError(f"raw strace artifact is unsafe or absent: {trace_path}")
        with trace_path.open("r", encoding="utf-8", errors="strict") as stream:
            for line in stream:
                match = _READ_LINE.match(line.rstrip("\n"))
                if match is None:
                    continue
                result = int(match.group("result"))
                if result < 0:
                    continue
                parsed_line_count += 1
                classified = _classified_path(
                    match.group("path"),
                    targets=targets,
                )
                if classified is None:
                    unattributed_successful_read_count += 1
                    continue
                path, kind = classified
                record = by_kind[kind]
                record["read_operations"] += 1
                record["read_bytes"] += result
                record["paths"].add(str(path))
                syscall_counts[match.group("syscall")] += 1
    kinds = {
        kind: {
            "read_operations": record["read_operations"],
            "read_bytes": record["read_bytes"],
            "unique_object_count": len(record["paths"]),
            "paths": sorted(record["paths"]),
        }
        for kind, record in by_kind.items()
    }
    return {
        "read_operations": sum(
            int(record["read_operations"]) for record in kinds.values()
        ),
        "read_bytes": sum(int(record["read_bytes"]) for record in kinds.values()),
        "unique_object_count": len(
            {path for record in kinds.values() for path in record["paths"]}
        ),
        "by_storage_kind": kinds,
        "attributed_syscall_counts": syscall_counts,
        "parsed_successful_read_line_count": parsed_line_count,
        "unattributed_successful_read_count": unattributed_successful_read_count,
    }


def _parse_elapsed(value: str) -> float:
    pieces = value.split(":")
    try:
        numbers = [float(piece) for piece in pieces]
    except ValueError as exc:
        raise ValueError("GNU time elapsed value is invalid") from exc
    if len(numbers) == 2:
        return 60.0 * numbers[0] + numbers[1]
    if len(numbers) == 3:
        return 3600.0 * numbers[0] + 60.0 * numbers[1] + numbers[2]
    raise ValueError("GNU time elapsed value has an unsupported shape")


def parse_gnu_time(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("GNU time evidence file is unsafe or absent")
    observed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        for field, label in _TIME_FIELDS.items():
            prefix = f"{label}: "
            if stripped.startswith(prefix):
                observed[field] = stripped[len(prefix) :]
    if set(observed) != set(_TIME_FIELDS):
        raise ValueError("GNU time evidence omits required fields")
    result = {
        "user_seconds": float(observed["user_seconds"]),
        "system_seconds": float(observed["system_seconds"]),
        "elapsed_seconds": _parse_elapsed(observed["elapsed"]),
        "peak_rss_bytes": int(observed["peak_rss_kib"]) * 1024,
        "filesystem_input_count": int(observed["filesystem_input_count"]),
        "filesystem_output_count": int(observed["filesystem_output_count"]),
    }
    if any(value < 0 for value in result.values()):
        raise ValueError("GNU time evidence contains a negative measurement")
    return result


def _artifact(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(
                value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_storage_io_trace_receipt(
    value: Mapping[str, Any],
    *,
    evidence_root: Path | None = None,
    replay_benchmark_binding: bool = False,
) -> None:
    """Deeply validate one trace receipt and optionally rehash raw artifacts."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise ValueError("storage I/O trace envelope field set differs")
    if (
        value["schema_id"] != TRACE_SCHEMA_ID
        or value["schema_version"] != TRACE_SCHEMA_VERSION
    ):
        raise ValueError("storage I/O trace schema identity differs")
    payload = value["payload"]
    expected = {
        "measurement_scope",
        "measurement_limitations",
        "target_roots",
        "command",
        "started_at_utc",
        "finished_at_utc",
        "controller_wall_seconds",
        "return_code",
        "command_completed",
        "stdout",
        "stderr",
        "environment",
        "resource_usage",
        "reads",
        "raw_artifacts",
        "physical_io_measured",
        "target_reads_observed",
        "benchmark_binding",
        "benchmark_evidence_bound",
        "promotion_authorized",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("storage I/O trace payload field set differs")
    if value["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("storage I/O trace payload digest differs")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"storage I/O trace is not strict JSON: {exc}") from exc
    if payload["measurement_scope"] != MEASUREMENT_SCOPE:
        raise ValueError("storage I/O trace measurement scope differs")
    limitations = payload["measurement_limitations"]
    if limitations != [
        "file_syscall_bytes_are_not_filesystem_or_network_transfer_bytes",
        "mmap_page_fault_io_is_not_counted",
        "strace_overhead_invalidates_latency_comparison",
    ]:
        raise ValueError("storage I/O trace limitations differ")
    roots = payload["target_roots"]
    command = payload["command"]
    if (
        not isinstance(roots, list)
        or not roots
        or roots != sorted(set(roots))
        or not isinstance(command, list)
        or not command
        or any(type(item) is not str or not item for item in [*roots, *command])
    ):
        raise ValueError("storage I/O trace roots or command are invalid")
    for field in ("started_at_utc", "finished_at_utc"):
        value = payload[field]
        if type(value) is not str:
            raise ValueError(f"storage I/O trace {field} is invalid")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"storage I/O trace {field} is invalid") from exc
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(
            parsed
        ):
            raise ValueError(f"storage I/O trace {field} must use UTC")
    if (
        type(payload["controller_wall_seconds"]) not in {int, float}
        or payload["controller_wall_seconds"] < 0
    ):
        raise ValueError("storage I/O trace controller duration is invalid")
    if type(payload["return_code"]) is not int:
        raise ValueError("storage I/O trace return code is invalid")
    completed = payload["return_code"] == 0
    if payload["command_completed"] is not completed:
        raise ValueError("storage I/O trace completion classification differs")
    for stream_name in ("stdout", "stderr"):
        stream = payload[stream_name]
        if (
            not isinstance(stream, Mapping)
            or set(stream) != {"sha256", "bytes"}
            or type(stream["sha256"]) is not str
            or not _SHA256.fullmatch(stream["sha256"])
            or type(stream["bytes"]) is not int
            or stream["bytes"] < 0
        ):
            raise ValueError(f"storage I/O trace {stream_name} identity differs")
    resources = payload["resource_usage"]
    if not isinstance(resources, Mapping) or set(resources) != {
        "user_seconds",
        "system_seconds",
        "elapsed_seconds",
        "peak_rss_bytes",
        "filesystem_input_count",
        "filesystem_output_count",
    }:
        raise ValueError("storage I/O trace resource usage differs")
    if any(type(item) not in {int, float} or item < 0 for item in resources.values()):
        raise ValueError("storage I/O trace resource measurement is invalid")
    environment = payload["environment"]
    if not isinstance(environment, Mapping) or set(environment) != {
        "hostname",
        "platform",
        "python",
        "palette_git",
        "thread_environment",
        "strace",
        "gnu_time",
    }:
        raise ValueError("storage I/O trace environment differs")
    reads = payload["reads"]
    if not isinstance(reads, Mapping) or set(reads) != {
        "read_operations",
        "read_bytes",
        "unique_object_count",
        "by_storage_kind",
        "attributed_syscall_counts",
        "parsed_successful_read_line_count",
        "unattributed_successful_read_count",
    }:
        raise ValueError("storage I/O trace read summary differs")
    by_kind = reads["by_storage_kind"]
    if not isinstance(by_kind, Mapping) or set(by_kind) != {"metadata", "payload"}:
        raise ValueError("storage I/O trace storage-kind summary differs")
    for kind in ("metadata", "payload"):
        record = by_kind[kind]
        if not isinstance(record, Mapping) or set(record) != {
            "read_operations",
            "read_bytes",
            "unique_object_count",
            "paths",
        }:
            raise ValueError("storage I/O trace kind summary differs")
        paths = record["paths"]
        if (
            not isinstance(paths, list)
            or paths != sorted(set(paths))
            or record["unique_object_count"] != len(paths)
            or any(
                type(record[field]) is not int or record[field] < 0
                for field in ("read_operations", "read_bytes", "unique_object_count")
            )
        ):
            raise ValueError("storage I/O trace path inventory differs")
    syscall_counts = reads["attributed_syscall_counts"]
    if (
        not isinstance(syscall_counts, Mapping)
        or set(syscall_counts) != {"read", "pread64", "readv", "preadv", "preadv2"}
        or any(type(value) is not int or value < 0 for value in syscall_counts.values())
        or sum(syscall_counts.values()) != reads["read_operations"]
    ):
        raise ValueError("storage I/O trace syscall counts differ")
    for field in (
        "read_operations",
        "read_bytes",
        "unique_object_count",
        "parsed_successful_read_line_count",
        "unattributed_successful_read_count",
    ):
        if type(reads[field]) is not int or reads[field] < 0:
            raise ValueError(f"storage I/O trace {field} is invalid")
    expected_operations = sum(int(by_kind[kind]["read_operations"]) for kind in by_kind)
    expected_bytes = sum(int(by_kind[kind]["read_bytes"]) for kind in by_kind)
    expected_paths = {path for kind in by_kind for path in by_kind[kind]["paths"]}
    if (
        reads["read_operations"] != expected_operations
        or reads["read_bytes"] != expected_bytes
        or reads["unique_object_count"] != len(expected_paths)
        or payload["physical_io_measured"] is not True
        or payload["target_reads_observed"] is not (expected_operations > 0)
    ):
        raise ValueError("storage I/O trace aggregate or safety classification differs")
    binding = payload["benchmark_binding"]
    bound = payload["benchmark_evidence_bound"]
    if binding is None:
        if bound is not False:
            raise ValueError("unbound storage trace claims benchmark evidence")
    else:
        if not isinstance(binding, Mapping) or set(binding) != {
            "stage_id",
            "matrix_path",
            "matrix_sha256",
            "matrix_payload_digest",
            "normalized_identity",
            "matrix_validated",
        }:
            raise ValueError("storage trace benchmark binding field set differs")
        if (
            bound is not True
            or binding["matrix_validated"] is not True
            or binding["stage_id"] not in DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE
            or type(binding["matrix_path"]) is not str
            or not Path(binding["matrix_path"]).is_absolute()
            or type(binding["matrix_sha256"]) is not str
            or not _SHA256.fullmatch(binding["matrix_sha256"])
            or type(binding["matrix_payload_digest"]) is not str
            or not _SHA256.fullmatch(binding["matrix_payload_digest"])
            or not isinstance(binding["normalized_identity"], Mapping)
            or binding["normalized_identity"].get("stage_id") != binding["stage_id"]
            or not completed
        ):
            raise ValueError("storage trace benchmark binding is invalid")
        if replay_benchmark_binding:
            supplied_matrix = Path(binding["matrix_path"])
            if supplied_matrix.is_symlink() or not supplied_matrix.is_file():
                raise ValueError("bound benchmark matrix is unsafe or absent")
            matrix_path = supplied_matrix.resolve(strict=True)
            matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
            if not isinstance(matrix, Mapping):
                raise ValueError("bound benchmark matrix is not one object")
            normalized = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE[
                binding["stage_id"]
            ].validated_matrix_identity(matrix)
            if (
                _sha256_file(matrix_path) != binding["matrix_sha256"]
                or matrix.get("payload_digest") != binding["matrix_payload_digest"]
                or normalized != binding["normalized_identity"]
            ):
                raise ValueError("bound benchmark matrix digest or identity differs")
    if payload["promotion_authorized"] is not False:
        raise ValueError("storage trace cannot authorize promotion")
    artifacts = payload["raw_artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("storage I/O trace raw artifact inventory is absent")
    artifact_paths: list[str] = []
    for artifact in artifacts:
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != {"path", "sha256", "bytes"}
            or type(artifact["path"]) is not str
            or not artifact["path"]
            or type(artifact["sha256"]) is not str
            or not _SHA256.fullmatch(artifact["sha256"])
            or type(artifact["bytes"]) is not int
            or artifact["bytes"] < 0
        ):
            raise ValueError("storage I/O trace raw artifact identity differs")
        artifact_paths.append(artifact["path"])
        if evidence_root is not None:
            root = evidence_root.resolve(strict=True)
            supplied = evidence_root / artifact["path"]
            if supplied.is_symlink():
                raise ValueError("storage I/O raw artifact is a symlink")
            path = supplied.resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError("storage I/O raw artifact escapes its evidence root")
            if (
                _sha256_file(path) != artifact["sha256"]
                or path.stat().st_size != artifact["bytes"]
            ):
                raise ValueError("storage I/O raw artifact digest or size differs")
    if artifact_paths != sorted(set(artifact_paths)):
        raise ValueError("storage I/O raw artifact paths are not unique and sorted")


def run_traced_storage_command(
    command: Sequence[str],
    *,
    target_roots: Sequence[str | Path],
    output_dir: str | Path,
    stage_id: str | None = None,
    matrix_result_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one command and publish immutable process-tree storage-read evidence."""

    if platform.system() != "Linux":
        raise RuntimeError("process-tree strace evidence is Linux-only")
    strace = shutil.which("strace")
    gnu_time = shutil.which("time")
    if strace is None or gnu_time is None:
        raise RuntimeError("storage trace requires strace and GNU time")
    targets = tuple(sorted({_safe_target(path) for path in target_roots}))
    if not targets:
        raise ValueError("storage trace requires at least one explicit target root")
    output = _safe_output(output_dir, targets=targets)
    resolved_command = _safe_command(command)
    if (stage_id is None) != (matrix_result_path is None):
        raise ValueError("stage_id and matrix_result_path must be supplied together")
    if (
        stage_id is not None
        and stage_id not in DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE
    ):
        raise ValueError(f"unknown analytics benchmark stage {stage_id!r}")
    matrix_path = (
        None
        if matrix_result_path is None
        else Path(matrix_result_path).expanduser().resolve()
    )
    if matrix_path is not None and (
        matrix_path == output
        or matrix_path.is_relative_to(output)
        or output.is_relative_to(matrix_path)
        or any(
            matrix_path == target
            or matrix_path.is_relative_to(target)
            or target.is_relative_to(matrix_path)
            for target in targets
        )
    ):
        raise ValueError("matrix result must be disjoint from trace output and targets")
    output.mkdir(parents=True, exist_ok=False)
    trace_prefix = output / "strace"
    resource_path = output / "resource.txt"
    invoked = [
        strace,
        "-ff",
        "-qq",
        "-yy",
        "-s",
        "0",
        "-e",
        "trace=read,pread64,readv,preadv,preadv2",
        "-o",
        str(trace_prefix),
        "--",
        gnu_time,
        "-v",
        "-o",
        str(resource_path),
        "--",
        *resolved_command,
    ]
    environment = os.environ.copy()
    environment.update(STORAGE_BENCHMARK_THREAD_ENVIRONMENT)
    started_at = _utc_now()
    start = time.perf_counter()
    completed = subprocess.run(
        invoked,
        cwd=Path.cwd(),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    wall = time.perf_counter() - start
    finished_at = _utc_now()
    traces = sorted(output.glob("strace.*"))
    if not traces:
        raise RuntimeError("strace did not emit any raw process trace")
    reads = parse_strace_reads(traces, target_roots=targets)
    resources = parse_gnu_time(resource_path)
    artifacts = sorted(
        [_artifact(path, root=output) for path in [*traces, resource_path]],
        key=lambda record: str(record["path"]),
    )
    benchmark_binding: dict[str, object] | None = None
    if stage_id is not None:
        assert matrix_path is not None
        if not matrix_path.is_file() or matrix_path.is_symlink():
            raise ValueError(
                "benchmark matrix result is unsafe or absent after command"
            )
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        if not isinstance(matrix, Mapping):
            raise ValueError("benchmark matrix result must be one object")
        normalized = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE[
            stage_id
        ].validated_matrix_identity(matrix)
        archive = Path(str(normalized["archive_path"])).resolve()
        run_targets = tuple(
            archive.joinpath(*str(normalized[field]).split("/"))
            for field in ("source_run_path", "candidate_run_path")
        )
        if any(
            not any(
                run_target == target
                or run_target.is_relative_to(target)
                or target.is_relative_to(run_target)
                for target in targets
            )
            for run_target in run_targets
        ):
            raise ValueError(
                "storage trace targets do not cover both normalized benchmark runs"
            )
        benchmark_binding = {
            "stage_id": stage_id,
            "matrix_path": str(matrix_path),
            "matrix_sha256": _sha256_file(matrix_path),
            "matrix_payload_digest": str(matrix["payload_digest"]),
            "normalized_identity": normalized,
            "matrix_validated": True,
        }
    payload: dict[str, object] = {
        "measurement_scope": MEASUREMENT_SCOPE,
        "measurement_limitations": [
            "file_syscall_bytes_are_not_filesystem_or_network_transfer_bytes",
            "mmap_page_fault_io_is_not_counted",
            "strace_overhead_invalidates_latency_comparison",
        ],
        "target_roots": [str(path) for path in targets],
        "command": list(resolved_command),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "controller_wall_seconds": wall,
        "return_code": completed.returncode,
        "command_completed": completed.returncode == 0,
        "stdout": {
            "sha256": _sha256_bytes(completed.stdout),
            "bytes": len(completed.stdout),
        },
        "stderr": {
            "sha256": _sha256_bytes(completed.stderr),
            "bytes": len(completed.stderr),
        },
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "palette_git": get_git_info(repo_path=Path(__file__).resolve().parents[3]),
            "thread_environment": dict(STORAGE_BENCHMARK_THREAD_ENVIRONMENT),
            "strace": str(Path(strace).resolve()),
            "gnu_time": str(Path(gnu_time).resolve()),
        },
        "resource_usage": resources,
        "reads": reads,
        "raw_artifacts": artifacts,
        "physical_io_measured": True,
        "target_reads_observed": int(reads["read_operations"]) > 0,
        "benchmark_binding": benchmark_binding,
        "benchmark_evidence_bound": benchmark_binding is not None,
        "promotion_authorized": False,
    }
    receipt = {
        "schema_id": TRACE_SCHEMA_ID,
        "schema_version": TRACE_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }
    require_storage_io_trace_receipt(
        receipt,
        evidence_root=output,
        replay_benchmark_binding=benchmark_binding is not None,
    )
    _write_json_exclusive(output / "receipt.json", receipt)
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage-id")
    parser.add_argument("--matrix-result")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    receipt = run_traced_storage_command(
        command,
        target_roots=args.target_root,
        output_dir=args.output_dir,
        stage_id=args.stage_id,
        matrix_result_path=args.matrix_result,
    )
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0 if receipt["payload"]["command_completed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MEASUREMENT_SCOPE",
    "TRACE_SCHEMA_ID",
    "TRACE_SCHEMA_VERSION",
    "parse_gnu_time",
    "parse_strace_reads",
    "require_storage_io_trace_receipt",
    "run_traced_storage_command",
]
