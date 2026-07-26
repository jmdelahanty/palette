"""Build immutable paired full-analysis fixtures for Crimson storage gates.

The source analysis archive is always read-only.  The builder copies an
explicit allowlist of maintained nondetection product trees into a fresh
benchmark namespace, installs one canonical detection candidate per layout,
consolidates metadata only after the direct hierarchy is complete, validates
the pair, and atomically publishes the containing directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np
import zarr

from fisheye.shared.zarr.benchmark_fixture import (
    TreeInventory,
    freeze_tree,
    inventory_tree,
    thaw_tree_for_cleanup,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.detection_schema import (
    CANONICAL_DETECTION_SCHEMA_V1,
    CanonicalDetectionDimensions,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)


FIXTURE_SPEC_SCHEMA_ID = "palette.canonical_detection_full_analysis_fixture_spec"
FIXTURE_SPEC_SCHEMA_VERSION = 1
PAIR_MANIFEST_SCHEMA_ID = "palette.canonical_detection_full_analysis_fixture_pair"
PAIR_MANIFEST_SCHEMA_VERSION = 1
FIXTURE_RUN_NAME = "crimson_storage_fixture_sleepyfish_cam2010095_v1"
CRIMSON_CONTRACT_COMMIT = "dadd9d779f0737c9643f15e3831a7c514bf99665"
CRIMSON_CONTRACT_SHA256 = (
    "aa64a94de7096b6a22e53d76357a619ca92bc5296b38f0549202fd67aee36a86"
)
_LAYOUTS = ("regular", "hybrid")
_PHYSICAL_ARRAY_FIELDS = {"chunk_grid", "chunk_key_encoding", "codecs", "storage_transformers"}
_PHYSICAL_ARRAY_ATTRIBUTE_FIELDS = {
    "access_pattern",
    "codec_profile_id",
    "storage_policy_version",
    "storage_profile_id",
    "write_mode",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _palette_code_identity() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[4]
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "repository": str(repository),
        "commit": commit,
        "clean": not status,
        "dirty_path_count": len(status),
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(rendered)


def _normalize_relative_group_path(value: str) -> str:
    raw = str(value).strip().strip("/")
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"Invalid relative Zarr group path: {value!r}.")
    return path.as_posix()


def _ancestors(relative_path: str) -> tuple[str, ...]:
    parts = PurePosixPath(relative_path).parts
    return tuple(PurePosixPath(*parts[:index]).as_posix() for index in range(1, len(parts)))


def _require_group(path: Path, *, label: str) -> dict[str, Any]:
    metadata_path = path / "zarr.json"
    if not metadata_path.is_file():
        raise ValueError(f"{label} is not a Zarr v3 group: {path}")
    metadata = _read_json(metadata_path)
    if metadata.get("zarr_format") != 3 or metadata.get("node_type") != "group":
        raise ValueError(f"{label} is not a Zarr v3 group: {path}")
    return metadata


def _assert_no_overlapping_paths(paths: Sequence[str]) -> None:
    normalized = tuple(PurePosixPath(path) for path in paths)
    for index, left in enumerate(normalized):
        for right in normalized[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    "Included product paths must be unique nonoverlapping trees; "
                    f"got {left.as_posix()!r} and {right.as_posix()!r}."
                )


@dataclass(frozen=True)
class SelectedProduct:
    product: str
    path: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SelectedProduct":
        product = str(payload.get("product", "")).strip()
        if not product:
            raise ValueError("Every selected product requires a nonempty product name.")
        return cls(
            product=product,
            path=_normalize_relative_group_path(str(payload.get("path", ""))),
        )

    def as_manifest(self) -> dict[str, str]:
        return {"product": self.product, "path": self.path}


@dataclass(frozen=True)
class CandidateSpec:
    path: Path
    expected_profile_id: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CandidateSpec":
        raw_path = str(payload.get("path", "")).strip()
        profile = str(payload.get("expected_profile_id", "")).strip()
        if not raw_path or not profile:
            raise ValueError("Candidate path and expected_profile_id are required.")
        return cls(
            path=Path(raw_path).expanduser().resolve(),
            expected_profile_id=profile,
        )

    def as_manifest(self) -> dict[str, str]:
        return {
            "path": str(self.path),
            "expected_profile_id": self.expected_profile_id,
        }


@dataclass(frozen=True)
class FullAnalysisFixtureSpec:
    fixture_id: str
    recording_id: str
    source_archive: Path
    source_recording: Path
    source_video: Path
    source_video_relative_path: str
    detection_run_name: str
    selected_products: tuple[SelectedProduct, ...]
    node_expectations: tuple[Mapping[str, Any], ...]
    selector_overrides: Mapping[str, Mapping[str, Any]]
    source_expectations: Mapping[str, Any]
    candidates: Mapping[str, CandidateSpec]
    crimson_contract: Mapping[str, Any]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FullAnalysisFixtureSpec":
        if payload.get("schema_id") != FIXTURE_SPEC_SCHEMA_ID:
            raise ValueError(f"Fixture spec schema_id must be {FIXTURE_SPEC_SCHEMA_ID!r}.")
        if payload.get("schema_version") != FIXTURE_SPEC_SCHEMA_VERSION:
            raise ValueError(
                f"Fixture spec schema_version must be {FIXTURE_SPEC_SCHEMA_VERSION}."
            )
        fixture_id = str(payload.get("fixture_id", "")).strip()
        recording_id = str(payload.get("recording_id", "")).strip()
        if not fixture_id or not recording_id:
            raise ValueError("fixture_id and recording_id are required.")
        if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in fixture_id):
            raise ValueError(
                "fixture_id may contain only lowercase letters, digits, '_' and '-'."
            )

        selected_payload = payload.get("selected_products")
        if not isinstance(selected_payload, list) or not selected_payload:
            raise ValueError("selected_products must be a nonempty list.")
        selected = tuple(
            SelectedProduct.from_payload(item)
            for item in selected_payload
            if isinstance(item, Mapping)
        )
        if len(selected) != len(selected_payload):
            raise ValueError("Every selected_products entry must be an object.")
        if len({item.product for item in selected}) != len(selected):
            raise ValueError("Selected product names must be unique.")
        _assert_no_overlapping_paths([item.path for item in selected])
        if any(item.path == "detect_runs" or item.path.startswith("detect_runs/") for item in selected):
            raise ValueError("Source detect_runs cannot be copied into the paired fixture.")

        raw_node_expectations = payload.get("node_expectations", [])
        if not isinstance(raw_node_expectations, list) or any(
            not isinstance(item, Mapping) for item in raw_node_expectations
        ):
            raise ValueError("node_expectations must be a list of objects.")
        node_expectations: list[Mapping[str, Any]] = []
        for raw_expectation in raw_node_expectations:
            expectation = dict(raw_expectation)
            expectation["path"] = _normalize_relative_group_path(
                str(expectation.get("path", ""))
            )
            node_expectations.append(expectation)

        raw_overrides = payload.get("selector_overrides", {})
        if not isinstance(raw_overrides, Mapping):
            raise ValueError("selector_overrides must be an object.")
        selector_overrides: dict[str, Mapping[str, Any]] = {}
        for raw_path, raw_attrs in raw_overrides.items():
            path = _normalize_relative_group_path(str(raw_path))
            if not isinstance(raw_attrs, Mapping):
                raise ValueError(f"Selector override at {path!r} must be an object.")
            selector_overrides[path] = dict(raw_attrs)

        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, Mapping) or set(raw_candidates) != set(_LAYOUTS):
            raise ValueError("candidates must contain exactly regular and hybrid entries.")
        candidates = {
            layout: CandidateSpec.from_payload(raw_candidates[layout])
            for layout in _LAYOUTS
        }

        crimson_contract = payload.get("crimson_contract")
        if not isinstance(crimson_contract, Mapping):
            raise ValueError("crimson_contract must be an object.")
        if crimson_contract.get("commit") != CRIMSON_CONTRACT_COMMIT:
            raise ValueError("Fixture spec does not pin the frozen Crimson contract commit.")
        if crimson_contract.get("document_sha256") != CRIMSON_CONTRACT_SHA256:
            raise ValueError("Fixture spec does not pin the frozen Crimson contract digest.")

        detection_run_name = str(payload.get("detection_run_name", "")).strip()
        if detection_run_name != FIXTURE_RUN_NAME:
            raise ValueError(f"detection_run_name must be the frozen name {FIXTURE_RUN_NAME!r}.")

        source_expectations = payload.get("source_expectations")
        if not isinstance(source_expectations, Mapping) or not source_expectations:
            raise ValueError("source_expectations must be a nonempty object.")

        source_video_relative_path = PurePosixPath(
            str(payload.get("source_video_relative_path", ""))
        )
        if (
            not source_video_relative_path.parts
            or source_video_relative_path.is_absolute()
            or any(part in {"", ".", ".."} for part in source_video_relative_path.parts)
        ):
            raise ValueError("source_video_relative_path must be recording-relative.")

        raw_source_paths = {
            "source_archive": str(payload.get("source_archive", "")).strip(),
            "source_recording": str(payload.get("source_recording", "")).strip(),
            "source_video": str(payload.get("source_video", "")).strip(),
        }
        missing_source_paths = [
            name for name, value in raw_source_paths.items() if not value
        ]
        if missing_source_paths:
            raise ValueError(
                f"Fixture spec is missing source paths: {missing_source_paths!r}."
            )

        return cls(
            fixture_id=fixture_id,
            recording_id=recording_id,
            source_archive=Path(raw_source_paths["source_archive"])
            .expanduser()
            .resolve(),
            source_recording=Path(raw_source_paths["source_recording"])
            .expanduser()
            .resolve(),
            source_video=Path(raw_source_paths["source_video"])
            .expanduser()
            .resolve(),
            source_video_relative_path=source_video_relative_path.as_posix(),
            detection_run_name=detection_run_name,
            selected_products=selected,
            node_expectations=tuple(node_expectations),
            selector_overrides=selector_overrides,
            source_expectations=dict(source_expectations),
            candidates=candidates,
            crimson_contract=dict(crimson_contract),
        )

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_id": FIXTURE_SPEC_SCHEMA_ID,
            "schema_version": FIXTURE_SPEC_SCHEMA_VERSION,
            "fixture_id": self.fixture_id,
            "recording_id": self.recording_id,
            "source_archive": str(self.source_archive),
            "source_recording": str(self.source_recording),
            "source_video": str(self.source_video),
            "source_video_relative_path": self.source_video_relative_path,
            "detection_run_name": self.detection_run_name,
            "selected_products": [item.as_manifest() for item in self.selected_products],
            "node_expectations": [dict(item) for item in self.node_expectations],
            "selector_overrides": {
                path: dict(attrs) for path, attrs in self.selector_overrides.items()
            },
            "source_expectations": dict(self.source_expectations),
            "candidates": {
                layout: self.candidates[layout].as_manifest() for layout in _LAYOUTS
            },
            "crimson_contract": dict(self.crimson_contract),
        }


def load_full_analysis_fixture_spec(path: Path) -> FullAnalysisFixtureSpec:
    return FullAnalysisFixtureSpec.from_payload(_read_json(path.expanduser().resolve()))


def require_safe_full_analysis_destination(
    destination: Path,
    *,
    benchmark_root: Path,
    fixture_id: str,
) -> Path:
    root = benchmark_root.expanduser().resolve()
    resolved = destination.expanduser().resolve()
    expected_parent = root / "canonical_detection_storage" / "full_analysis"
    if resolved.parent != expected_parent or resolved.name != fixture_id:
        raise ValueError(
            "Full-analysis fixture destination must be exactly "
            f"{expected_parent / fixture_id}."
        )
    if resolved.exists():
        raise FileExistsError(f"Full-analysis fixture destination exists: {resolved}")
    return resolved


def require_safe_fixture_scratch_root(
    scratch_root: Path,
    *,
    benchmark_root: Path,
    source_recording: Path,
) -> Path:
    resolved = scratch_root.expanduser().resolve()
    benchmark = benchmark_root.expanduser().resolve()
    recording = source_recording.expanduser().resolve()
    if resolved == Path("/") or resolved in {benchmark, recording}:
        raise ValueError("Fixture scratch root is too broad or aliases protected data.")
    if resolved.is_relative_to(benchmark) or resolved.is_relative_to(recording):
        raise ValueError("Fixture scratch root cannot be inside benchmark or recording data.")
    if not resolved.is_dir():
        raise FileNotFoundError(f"Fixture scratch root does not exist: {resolved}")
    return resolved


def _validate_source_expectations(
    metadata: Mapping[str, Any],
    expectations: Mapping[str, Any],
) -> None:
    attributes = metadata.get("attributes")
    if not isinstance(attributes, Mapping):
        raise ValueError("Source archive root attributes are missing.")
    mismatches = {
        key: {"expected": expected, "actual": attributes.get(key)}
        for key, expected in expectations.items()
        if attributes.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Source archive expectation mismatch: {mismatches}")


def _validate_node_expectations(
    root: Path,
    expectations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    reports: list[dict[str, Any]] = []
    for expectation in expectations:
        path = str(expectation["path"])
        metadata_path = root / path / "zarr.json"
        absent = expectation.get("absent") is True
        if absent:
            if metadata_path.exists():
                raise ValueError(f"Expected source node to be absent: {path}")
            reports.append({"path": path, "absent": True, "validated": True})
            continue
        if not metadata_path.is_file():
            raise ValueError(f"Expected source node is missing: {path}")
        metadata = _read_json(metadata_path)
        actual = {
            "zarr_format": metadata.get("zarr_format"),
            "node_type": metadata.get("node_type"),
            "shape": metadata.get("shape"),
            "data_type": metadata.get("data_type"),
            "chunk_shape": (
                metadata.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape")
                if isinstance(metadata.get("chunk_grid"), Mapping)
                else None
            ),
            "codec_names": [
                codec.get("name")
                for codec in metadata.get("codecs", [])
                if isinstance(codec, Mapping)
            ],
        }
        checked = {
            key: value
            for key, value in expectation.items()
            if key
            in {
                "zarr_format",
                "node_type",
                "shape",
                "data_type",
                "chunk_shape",
                "codec_names",
            }
        }
        mismatches = {
            key: {"expected": expected, "actual": actual.get(key)}
            for key, expected in checked.items()
            if actual.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Source node expectation mismatch at {path}: {mismatches}")
        reports.append(
            {
                "path": path,
                "absent": False,
                "validated": True,
                "checked": checked,
                "metadata_sha256": _sha256_file(metadata_path),
            }
        )
    return tuple(reports)


def _metadata_file_paths(root: Path, selected_paths: Sequence[str]) -> tuple[Path, ...]:
    files: set[Path] = {root / "zarr.json"}
    for relative in selected_paths:
        selected_root = root / relative
        if selected_root.is_symlink():
            raise ValueError(
                f"Selected source trees cannot contain symlinks: {selected_root}"
            )
        for ancestor in _ancestors(relative):
            files.add(root / ancestor / "zarr.json")
        pending = [selected_root]
        while pending:
            candidate = pending.pop()
            metadata_path = candidate / "zarr.json"
            metadata = _require_group(candidate, label="Selected product")
            files.add(metadata_path)
            for child in candidate.iterdir():
                if child.is_symlink():
                    raise ValueError(f"Selected source trees cannot contain symlinks: {child}")
                child_metadata = child / "zarr.json"
                if not child.is_dir() or not child_metadata.is_file():
                    continue
                child_payload = _read_json(child_metadata)
                files.add(child_metadata)
                if child_payload.get("node_type") == "group":
                    pending.append(child)
            if metadata.get("node_type") != "group":  # pragma: no cover - guarded above
                raise AssertionError("unreachable")
    return tuple(sorted(files))


def _inventory_files(root: Path, files: Iterable[Path]) -> TreeInventory:
    resolved = root.expanduser().resolve()
    unresolved = {path.expanduser().absolute() for path in files}
    for path in unresolved:
        if path.is_symlink():
            raise ValueError(f"Fixture inventories cannot contain symlinks: {path}")
    unique = sorted({path.resolve() for path in unresolved})
    digest = hashlib.sha256()
    apparent = 0
    allocated = 0
    for path in unique:
        if not path.is_file() or not path.is_relative_to(resolved):
            raise ValueError(f"Inventory path is not a file below {resolved}: {path}")
        relative = path.relative_to(resolved).as_posix().encode("utf-8")
        stat = path.stat()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(int(stat.st_size).to_bytes(8, "little"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        apparent += int(stat.st_size)
        allocated += int(stat.st_blocks * 512)
    return TreeInventory(
        file_count=len(unique),
        apparent_bytes=apparent,
        allocated_bytes=allocated,
        tree_sha256=digest.hexdigest(),
    )


def _selected_tree_files(root: Path, selected_paths: Sequence[str]) -> tuple[Path, ...]:
    files: set[Path] = set()
    for relative in selected_paths:
        selected = root / relative
        if selected.is_symlink():
            raise ValueError(f"Selected source trees cannot contain symlinks: {selected}")
        if not selected.is_dir():
            raise FileNotFoundError(f"Selected product tree is absent: {selected}")
        for path in selected.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Selected source trees cannot contain symlinks: {path}")
            if path.is_file():
                files.add(path)
    return tuple(sorted(files))


def _selected_tree_inventory(root: Path, selected_paths: Sequence[str]) -> TreeInventory:
    return _inventory_files(root, _selected_tree_files(root, selected_paths))


def _same_inventory_content(left: TreeInventory, right: TreeInventory) -> bool:
    return (
        left.file_count == right.file_count
        and left.apparent_bytes == right.apparent_bytes
        and left.tree_sha256 == right.tree_sha256
    )


def _nondetection_inventory(root: Path, selected_paths: Sequence[str]) -> TreeInventory:
    files = set(_selected_tree_files(root, selected_paths))
    for relative in selected_paths:
        for ancestor in _ancestors(relative):
            files.add(root / ancestor / "zarr.json")
    return _inventory_files(root, files)


def _candidate_direct_metadata(candidate: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    root_metadata = _require_group(candidate, label="Detection candidate")
    result[""] = root_metadata
    for relative in ("instances", *CANONICAL_DETECTION_SCHEMA_V1.binding_paths):
        metadata_path = candidate / relative / "zarr.json"
        if not metadata_path.is_file():
            raise ValueError(f"Detection candidate is missing {relative}/zarr.json.")
        result[relative] = _read_json(metadata_path)
    return result


def _without_nested_consolidation(metadata: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    normalized.pop("consolidated_metadata", None)
    return normalized


def _validate_direct_consolidated_map(
    direct: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
) -> int:
    root = direct[""]
    envelope = root.get("consolidated_metadata")
    if not isinstance(envelope, Mapping) or envelope.get("kind") != "inline":
        raise ValueError(f"{label} lacks inline consolidated metadata.")
    consolidated = envelope.get("metadata")
    if not isinstance(consolidated, Mapping):
        raise ValueError(f"{label} consolidated metadata map is missing.")
    expected = set(direct) - {""}
    if set(consolidated) != expected:
        raise ValueError(
            f"{label} direct/consolidated path set mismatch: "
            f"direct={sorted(expected)!r}, consolidated={sorted(consolidated)!r}."
        )
    for path in sorted(expected):
        if _without_nested_consolidation(direct[path]) != _without_nested_consolidation(
            consolidated[path]
        ):
            raise ValueError(f"{label} direct/consolidated declaration mismatch at {path}.")
    return len(expected)


def _validate_candidate(
    candidate: CandidateSpec,
    *,
    layout: str,
    benchmark_root: Path,
) -> dict[str, Any]:
    if not candidate.path.is_relative_to(benchmark_root.expanduser().resolve()):
        raise ValueError(f"{layout} candidate must be below the benchmark root.")
    direct = _candidate_direct_metadata(candidate.path)
    root_attributes = direct[""].get("attributes")
    if not isinstance(root_attributes, Mapping):
        raise ValueError(f"{layout} candidate attributes are missing.")
    for field in ("benchmark_only",):
        if root_attributes.get(field) is not True:
            raise ValueError(f"{layout} candidate must declare {field}=true.")
    for field in ("canonical", "registry_registered", "selector_eligible"):
        if root_attributes.get(field) is not False:
            raise ValueError(f"{layout} candidate must declare {field}=false.")
    storage_plan = root_attributes.get("storage_plan")
    if not isinstance(storage_plan, Mapping):
        raise ValueError(f"{layout} candidate storage_plan is missing.")
    profile = storage_plan.get("storage_profile")
    if not isinstance(profile, Mapping) or profile.get("profile_id") != candidate.expected_profile_id:
        raise ValueError(f"{layout} candidate storage profile does not match the spec.")
    logical_schema = root_attributes.get("logical_schema")
    if not isinstance(logical_schema, Mapping):
        raise ValueError(f"{layout} candidate logical_schema is missing.")
    dimensions = logical_schema.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise ValueError(f"{layout} candidate dimensions are missing.")
    for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths:
        metadata = direct[path]
        if metadata.get("zarr_format") != 3 or metadata.get("node_type") != "array":
            raise ValueError(f"{layout} candidate path {path!r} is not a Zarr v3 array.")
    if layout == "regular":
        if any(
            any(codec.get("name") == "sharding_indexed" for codec in direct[path].get("codecs", []))
            for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths
        ):
            raise ValueError("Regular candidate unexpectedly contains indexed sharding.")
    else:
        if any(
            not direct[path].get("codecs")
            or direct[path]["codecs"][0].get("name") != "sharding_indexed"
            for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths
        ):
            raise ValueError("Hybrid candidate must shard every canonical array.")
    consolidated_count = _validate_direct_consolidated_map(
        direct,
        label=f"{layout} candidate",
    )
    return {
        "path": str(candidate.path),
        "expected_profile_id": candidate.expected_profile_id,
        "metadata_sha256": _sha256_file(candidate.path / "zarr.json"),
        "inventory": _inventory_files(
            candidate.path,
            (path for path in candidate.path.rglob("*") if path.is_file()),
        ).as_manifest(),
        "logical_schema": dict(logical_schema),
        "direct_consolidated_node_count": consolidated_count,
    }


def plan_full_analysis_fixture_pair(
    spec: FullAnalysisFixtureSpec,
    *,
    destination: Path,
    benchmark_root: Path,
    pair_copy_mode: str = "auto",
    expected_palette_commit: str | None = None,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    if pair_copy_mode not in {"auto", "copy", "reflink"}:
        raise ValueError("pair_copy_mode must be auto, copy, or reflink.")
    resolved_root = benchmark_root.expanduser().resolve()
    resolved_destination = require_safe_full_analysis_destination(
        destination,
        benchmark_root=resolved_root,
        fixture_id=spec.fixture_id,
    )
    resolved_scratch = (
        require_safe_fixture_scratch_root(
            scratch_root,
            benchmark_root=resolved_root,
            source_recording=spec.source_recording,
        )
        if scratch_root is not None
        else None
    )
    source_metadata = _require_group(spec.source_archive, label="Source analysis archive")
    if spec.source_archive.is_relative_to(resolved_root):
        raise ValueError("Production source archive cannot be inside the benchmark root.")
    if not spec.source_recording.is_dir():
        raise FileNotFoundError(f"Source recording does not exist: {spec.source_recording}")
    expected_video = spec.source_recording / spec.source_video_relative_path
    if expected_video.resolve() != spec.source_video or not spec.source_video.is_file():
        raise ValueError("Source video association does not resolve to the declared file.")
    _validate_source_expectations(source_metadata, spec.source_expectations)
    node_expectation_reports = _validate_node_expectations(
        spec.source_archive,
        spec.node_expectations,
    )

    selected_paths = tuple(item.path for item in spec.selected_products)
    for relative in selected_paths:
        _require_group(spec.source_archive / relative, label=f"Selected product {relative!r}")
        for ancestor in _ancestors(relative):
            _require_group(spec.source_archive / ancestor, label=f"Ancestor {ancestor!r}")
    for override_path in spec.selector_overrides:
        if override_path != "detect_runs":
            _require_group(
                spec.source_archive / override_path,
                label=f"Selector override group {override_path!r}",
            )

    candidate_reports = {
        layout: _validate_candidate(
            spec.candidates[layout],
            layout=layout,
            benchmark_root=resolved_root,
        )
        for layout in _LAYOUTS
    }
    regular_schema = candidate_reports["regular"]["logical_schema"]
    hybrid_schema = candidate_reports["hybrid"]["logical_schema"]
    if regular_schema != hybrid_schema:
        raise ValueError("Regular and hybrid candidates do not share one logical schema.")
    dimensions = regular_schema.get("dimensions")
    expected_dimensions = {
        "n_frames": spec.source_expectations.get("n_frames"),
        "source_width": spec.source_expectations.get("source_video_width"),
        "source_height": spec.source_expectations.get("source_video_height"),
    }
    if not isinstance(dimensions, Mapping) or any(
        dimensions.get(key) != value for key, value in expected_dimensions.items()
    ):
        raise ValueError("Candidate dimensions do not match the maintained recording.")

    metadata_inventory = _inventory_files(
        spec.source_archive,
        _metadata_file_paths(spec.source_archive, selected_paths),
    )
    video_stat = spec.source_video.stat()
    palette_code = _palette_code_identity()
    expected_commit = str(expected_palette_commit or "").strip() or None
    if expected_commit is not None and palette_code["commit"] != expected_commit:
        raise ValueError(
            "Palette commit mismatch: "
            f"expected {expected_commit}, found {palette_code['commit']}."
        )
    return {
        "schema_id": PAIR_MANIFEST_SCHEMA_ID,
        "schema_version": PAIR_MANIFEST_SCHEMA_VERSION,
        "status": "planned",
        "payload_io_performed": False,
        "benchmark_only": True,
        "canonical": False,
        "registry_registered": False,
        "selector_eligible": False,
        "fixture_id": spec.fixture_id,
        "destination": str(resolved_destination),
        "benchmark_root": str(resolved_root),
        "pair_copy_mode_requested": pair_copy_mode,
        "pair_copy_mode_resolved": None,
        "scratch_root": str(resolved_scratch) if resolved_scratch is not None else None,
        "source": {
            "recording_id": spec.recording_id,
            "recording": str(spec.source_recording),
            "archive": str(spec.source_archive),
            "selected_products": [item.as_manifest() for item in spec.selected_products],
            "node_expectations": list(node_expectation_reports),
            "direct_metadata_inventory": metadata_inventory.as_manifest(),
            "video": {
                "path": str(spec.source_video),
                "recording_relative_path": spec.source_video_relative_path,
                "copied": False,
                "size_bytes": int(video_stat.st_size),
                "mtime_ns": int(video_stat.st_mtime_ns),
            },
        },
        "detection_run": spec.detection_run_name,
        "candidates": candidate_reports,
        "selector_overrides": {
            path: dict(attrs) for path, attrs in spec.selector_overrides.items()
        },
        "crimson_contract": dict(spec.crimson_contract),
        "palette_code": {
            **palette_code,
            "expected_commit": expected_commit,
            "expected_commit_match": (
                None if expected_commit is None else True
            ),
        },
        "publication": {
            "regular": str(resolved_destination / "regular.zarr"),
            "hybrid": str(resolved_destination / "hybrid.zarr"),
            "pair_manifest": str(resolved_destination / "pair_manifest.json"),
            "atomic_unit": "containing_pair_directory",
            "lifecycle": "source_to_node_local_scratch_then_verified_copy_to_shared_storage",
            "failed_attempt_policy": "preserve_explicit_incomplete_sibling",
        },
        "safety": {
            "production_archive_open_mode": "read_only_filesystem_copy_source",
            "hardlinks": "forbidden",
            "source_reflinks": "forbidden",
            "pair_reflink_source": "independent_incomplete_benchmark_base_only",
            "registry_updates": 0,
            "production_selector_updates": 0,
            "training_artifacts": 0,
        },
    }


def _reject_symlinks(directory: str, names: list[str]) -> set[str]:
    for name in names:
        path = Path(directory) / name
        if path.is_symlink():
            raise ValueError(f"Fixture copies cannot contain symlinks: {path}")
    return set()


def _copy_tree(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise ValueError(f"Fixture copies cannot use symlink roots: {source}")
    shutil.copytree(source, destination, ignore=_reject_symlinks)


def _copy_group_metadata(source: Path, destination: Path) -> None:
    metadata = _require_group(source, label="Source group envelope")
    destination.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(destination / "zarr.json", metadata)


def _update_group_attributes(path: Path, attributes: Mapping[str, Any]) -> None:
    metadata_path = path / "zarr.json"
    metadata = _read_json(metadata_path)
    current = metadata.get("attributes")
    if not isinstance(current, Mapping):
        current = {}
    metadata["attributes"] = {**dict(current), **dict(attributes)}
    metadata.pop("consolidated_metadata", None)
    _write_json_atomic(metadata_path, metadata)


def _assemble_nondetection_base(
    spec: FullAnalysisFixtureSpec,
    *,
    destination: Path,
    created_at_utc: str,
) -> None:
    destination.mkdir(parents=True)
    source_root_metadata = _read_json(spec.source_archive / "zarr.json")
    source_root_metadata.pop("consolidated_metadata", None)
    source_attributes = source_root_metadata.get("attributes")
    if not isinstance(source_attributes, Mapping):
        source_attributes = {}
    source_root_metadata["attributes"] = {
        **dict(source_attributes),
        "benchmark_only": True,
        "canonical": False,
        "registry_registered": False,
        "selector_eligible": False,
        "fixture_id": spec.fixture_id,
        "fixture_schema_id": PAIR_MANIFEST_SCHEMA_ID,
        "fixture_schema_version": PAIR_MANIFEST_SCHEMA_VERSION,
        "fixture_created_at_utc": created_at_utc,
        "fixture_detection_run": spec.detection_run_name,
        "fixture_source_archive": str(spec.source_archive),
        "fixture_source_video_relative_path": spec.source_video_relative_path,
    }
    _write_json_exclusive(destination / "zarr.json", source_root_metadata)

    copied_ancestors: set[str] = set()
    for product in spec.selected_products:
        for ancestor in _ancestors(product.path):
            if ancestor in copied_ancestors:
                continue
            _copy_group_metadata(
                spec.source_archive / ancestor,
                destination / ancestor,
            )
            copied_ancestors.add(ancestor)
        _copy_tree(spec.source_archive / product.path, destination / product.path)

    if "detect_runs" not in copied_ancestors:
        _copy_group_metadata(spec.source_archive / "detect_runs", destination / "detect_runs")
    for path, attributes in spec.selector_overrides.items():
        _update_group_attributes(destination / path, attributes)
    _update_group_attributes(
        destination / "detect_runs",
        {
            "latest": spec.detection_run_name,
            "latest_complete": spec.detection_run_name,
            "benchmark_only": True,
            "canonical": False,
            "registry_registered": False,
            "selector_eligible": False,
        },
    )


def _probe_reflink_isolation(directory: Path) -> dict[str, Any]:
    source = directory / ".reflink_probe_source"
    destination = directory / ".reflink_probe_destination"
    source.write_bytes(b"palette-reflink-source")
    try:
        completed = subprocess.run(
            ["cp", "--reflink=always", "--", str(source), str(destination)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return {
                "supported": False,
                "returncode": int(completed.returncode),
                "stderr": completed.stderr.strip(),
            }
        inode_distinct = source.stat().st_ino != destination.stat().st_ino
        destination.write_bytes(b"palette-reflink-destination")
        mutation_isolated = source.read_bytes() == b"palette-reflink-source"
        supported = bool(inode_distinct and mutation_isolated)
        return {
            "supported": supported,
            "returncode": 0,
            "inode_distinct": inode_distinct,
            "mutation_isolated": mutation_isolated,
        }
    finally:
        source.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)


def _clone_pair_base(source: Path, destination: Path, *, mode: str) -> dict[str, Any]:
    probe = _probe_reflink_isolation(source.parent)
    use_reflink = mode == "reflink" or (mode == "auto" and probe["supported"])
    if mode == "reflink" and not probe["supported"]:
        raise RuntimeError(f"Required reflink isolation probe failed: {probe}")
    if use_reflink:
        completed = subprocess.run(
            ["cp", "--archive", "--reflink=always", "--", str(source), str(destination)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Pair-base reflink failed: {completed.stderr.strip()}")
        return {"method": "reflink", "probe": probe}
    _copy_tree(source, destination)
    return {"method": "copy", "probe": probe}


def _normalized_detection_run_attributes(candidate_metadata: Mapping[str, Any]) -> dict[str, Any]:
    attributes = candidate_metadata.get("attributes")
    if not isinstance(attributes, Mapping):
        raise ValueError("Candidate root attributes are missing.")
    logical_schema = attributes.get("logical_schema")
    storage_plan = attributes.get("storage_plan")
    if not isinstance(logical_schema, Mapping) or not isinstance(storage_plan, Mapping):
        raise ValueError("Candidate logical schema or storage plan is missing.")
    return {
        "benchmark_only": True,
        "canonical": False,
        "registry_registered": False,
        "selector_eligible": False,
        "schema_id": "palette.canonical_detection_full_analysis_fixture_run",
        "schema_version": 1,
        "logical_schema": dict(logical_schema),
        "storage_plan": dict(storage_plan),
    }


def _install_detection_candidate(
    candidate: Path,
    *,
    archive: Path,
    run_name: str,
) -> None:
    destination = archive / "detect_runs" / run_name
    _copy_tree(candidate, destination)
    os.chmod(destination, 0o755)
    metadata_path = destination / "zarr.json"
    os.chmod(metadata_path, 0o644)
    metadata = _read_json(metadata_path)
    metadata.pop("consolidated_metadata", None)
    metadata["attributes"] = _normalized_detection_run_attributes(metadata)
    _write_json_atomic(metadata_path, metadata)


def _direct_metadata_map(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {"": _read_json(root / "zarr.json")}
    pending = [root]
    while pending:
        group = pending.pop()
        for child in group.iterdir():
            if child.is_symlink():
                raise ValueError(f"Published fixture contains a symlink: {child}")
            metadata_path = child / "zarr.json"
            if not child.is_dir() or not metadata_path.is_file():
                continue
            relative = child.relative_to(root).as_posix()
            metadata = _read_json(metadata_path)
            result[relative] = metadata
            if metadata.get("node_type") == "group":
                pending.append(child)
    return result


def _consolidate_and_validate(root: Path) -> dict[str, Any]:
    warning_report = consolidate_metadata_capture_expected_warnings(root)
    direct = _direct_metadata_map(root)
    count = _validate_direct_consolidated_map(direct, label=str(root))
    zarr.open_group(str(root), mode="r", use_consolidated=False)
    zarr.open_group(str(root), mode="r", use_consolidated=True)
    return {
        "direct_consolidated_node_count": count,
        "direct_open": True,
        "consolidated_open": True,
        "warning_report": warning_report,
    }


def _candidate_dimensions(run: Any) -> CanonicalDetectionDimensions:
    logical_schema = run.attrs.get("logical_schema")
    if not isinstance(logical_schema, Mapping):
        raise ValueError("Fixture detection run lacks logical_schema.")
    dimensions = logical_schema.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise ValueError("Fixture detection run lacks logical dimensions.")
    return CanonicalDetectionDimensions(
        n_frames=int(dimensions["n_frames"]),
        n_instances=int(dimensions["n_instances"]),
        source_width=int(dimensions["source_width"]),
        source_height=int(dimensions["source_height"]),
    )


def _validate_detection_archive(root: Path, *, run_name: str) -> dict[str, Any]:
    archive = zarr.open_group(str(root), mode="r", use_consolidated=True)
    run = archive[f"detect_runs/{run_name}"]
    dimensions = _candidate_dimensions(run)
    arrays = {path: run[path] for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths}
    CANONICAL_DETECTION_SCHEMA_V1.require(arrays, dimensions=dimensions)
    digests = {
        path: sha256_array(np.asarray(array[:])) for path, array in arrays.items()
    }
    return {
        "dimensions": dimensions.as_manifest(),
        "arrays": {
            path: {
                "shape": list(arrays[path].shape),
                "dtype": str(arrays[path].dtype),
                "sha256": digests[path],
            }
            for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths
        },
        "all_logical_invariants_valid": True,
    }


def _normalize_detection_metadata_for_logical_comparison(
    path: str,
    metadata: Mapping[str, Any],
    *,
    run_path: str,
) -> dict[str, Any]:
    normalized = _without_nested_consolidation(metadata)
    if path == run_path:
        attributes = dict(normalized.get("attributes", {}))
        attributes.pop("storage_plan", None)
        normalized["attributes"] = attributes
        return normalized
    if path.startswith(f"{run_path}/instances/"):
        for field in _PHYSICAL_ARRAY_FIELDS:
            normalized.pop(field, None)
        attributes = dict(normalized.get("attributes", {}))
        for field in _PHYSICAL_ARRAY_ATTRIBUTE_FIELDS:
            attributes.pop(field, None)
        normalized["attributes"] = attributes
    return normalized


def _validate_pair_metadata_difference(
    regular: Path,
    hybrid: Path,
    *,
    run_name: str,
) -> dict[str, Any]:
    regular_direct = _direct_metadata_map(regular)
    hybrid_direct = _direct_metadata_map(hybrid)
    if set(regular_direct) != set(hybrid_direct):
        raise RuntimeError("Paired fixture direct metadata path sets differ.")
    run_path = f"detect_runs/{run_name}"
    physical_difference_paths: list[str] = []
    for path in sorted(regular_direct):
        left = regular_direct[path]
        right = hybrid_direct[path]
        if path == "":
            left = _without_nested_consolidation(left)
            right = _without_nested_consolidation(right)
        if left != right:
            if path != run_path and not path.startswith(f"{run_path}/instances/") and path != "":
                raise RuntimeError(f"Unexpected nondetection metadata difference at {path}.")
            physical_difference_paths.append(path)
        if _normalize_detection_metadata_for_logical_comparison(
            path,
            left,
            run_path=run_path,
        ) != _normalize_detection_metadata_for_logical_comparison(
            path,
            right,
            run_path=run_path,
        ):
            raise RuntimeError(f"Paired logical metadata differs at {path}.")

    regular_consolidated = regular_direct[""]["consolidated_metadata"]["metadata"]
    hybrid_consolidated = hybrid_direct[""]["consolidated_metadata"]["metadata"]
    regular_nondetection = {
        path: value
        for path, value in regular_consolidated.items()
        if path != "detect_runs" and not path.startswith("detect_runs/")
    }
    hybrid_nondetection = {
        path: value
        for path, value in hybrid_consolidated.items()
        if path != "detect_runs" and not path.startswith("detect_runs/")
    }
    if regular_nondetection != hybrid_nondetection:
        raise RuntimeError("Paired normalized nondetection consolidated metadata differs.")
    return {
        "direct_path_count": len(regular_direct),
        "physical_difference_paths": physical_difference_paths,
        "nondetection_consolidated_metadata_exact": True,
    }


def _fixture_manifest(
    *,
    layout: str,
    archive: Path,
    plan: Mapping[str, Any],
    source_pre: TreeInventory,
    source_post: TreeInventory,
    source_metadata_pre: TreeInventory,
    source_metadata_post: TreeInventory,
    selected_inventory: TreeInventory,
    nondetection_inventory: TreeInventory,
    detection_validation: Mapping[str, Any],
    metadata_validation: Mapping[str, Any],
    copy_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_id": "palette.canonical_detection_full_analysis_fixture",
        "schema_version": 1,
        "status": "published_immutable",
        "layout": layout,
        "archive": str(archive),
        "benchmark_only": True,
        "canonical": False,
        "registry_registered": False,
        "selector_eligible": False,
        "source": plan["source"],
        "source_selected_inventory_before": source_pre.as_manifest(),
        "source_selected_inventory_after": source_post.as_manifest(),
        "source_direct_metadata_inventory_before": source_metadata_pre.as_manifest(),
        "source_direct_metadata_inventory_after": source_metadata_post.as_manifest(),
        "source_unchanged": _same_inventory_content(
            source_pre, source_post
        ) and _same_inventory_content(source_metadata_pre, source_metadata_post),
        "selected_product_inventory": selected_inventory.as_manifest(),
        "nondetection_inventory": nondetection_inventory.as_manifest(),
        "detection_run": plan["detection_run"],
        "detection_candidate": plan["candidates"][layout],
        "detection_validation": dict(detection_validation),
        "metadata_validation": dict(metadata_validation),
        "pair_base_copy": dict(copy_report),
        "crimson_contract": plan["crimson_contract"],
        "palette_code": plan["palette_code"],
        "video_copied": False,
        "immutability": {
            "files_mode": "0444",
            "directories_mode": "0555",
            "jobs_must_open_read_only": True,
        },
    }


def publish_full_analysis_fixture_pair(
    spec: FullAnalysisFixtureSpec,
    *,
    destination: Path,
    benchmark_root: Path,
    pair_copy_mode: str = "auto",
    expected_palette_commit: str | None = None,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Build, validate, atomically publish, and freeze one paired fixture."""

    plan = plan_full_analysis_fixture_pair(
        spec,
        destination=destination,
        benchmark_root=benchmark_root,
        pair_copy_mode=pair_copy_mode,
        expected_palette_commit=expected_palette_commit,
        scratch_root=scratch_root,
    )
    if not expected_palette_commit:
        raise ValueError("Apply mode requires an explicit expected Palette commit.")
    if plan["palette_code"]["clean"] is not True:
        raise ValueError("Apply mode requires a clean commit-pinned Palette worktree.")
    if plan["scratch_root"] is None:
        raise ValueError("Apply mode requires an existing node-local scratch root.")
    resolved_destination = Path(str(plan["destination"]))
    resolved_destination.parent.mkdir(parents=True, exist_ok=True)
    scratch_base = Path(str(plan["scratch_root"]))
    temporary = scratch_base / (
        f"palette_canonical_detection_full_analysis_{spec.fixture_id}."
        f"incomplete.{uuid.uuid4().hex}"
    )
    temporary.mkdir()
    publication_temporary = resolved_destination.parent / (
        f".{resolved_destination.name}.incomplete.{uuid.uuid4().hex}"
    )
    selected_paths = tuple(item.path for item in spec.selected_products)
    created_at = _utc_now()

    try:
        source_selected_pre = _selected_tree_inventory(spec.source_archive, selected_paths)
        source_metadata_pre = _inventory_files(
            spec.source_archive,
            _metadata_file_paths(spec.source_archive, selected_paths),
        )
        video_pre = spec.source_video.stat()

        regular = temporary / "regular.zarr"
        hybrid = temporary / "hybrid.zarr"
        _assemble_nondetection_base(
            spec,
            destination=regular,
            created_at_utc=created_at,
        )
        copy_report = _clone_pair_base(regular, hybrid, mode=pair_copy_mode)
        _install_detection_candidate(
            spec.candidates["regular"].path,
            archive=regular,
            run_name=spec.detection_run_name,
        )
        _install_detection_candidate(
            spec.candidates["hybrid"].path,
            archive=hybrid,
            run_name=spec.detection_run_name,
        )

        metadata_validations = {
            "regular": _consolidate_and_validate(regular),
            "hybrid": _consolidate_and_validate(hybrid),
        }
        detection_validations = {
            "regular": _validate_detection_archive(
                regular,
                run_name=spec.detection_run_name,
            ),
            "hybrid": _validate_detection_archive(
                hybrid,
                run_name=spec.detection_run_name,
            ),
        }
        if detection_validations["regular"] != detection_validations["hybrid"]:
            raise RuntimeError("Regular and hybrid decoded detection content differs.")

        source_selected_post = _selected_tree_inventory(spec.source_archive, selected_paths)
        source_metadata_post = _inventory_files(
            spec.source_archive,
            _metadata_file_paths(spec.source_archive, selected_paths),
        )
        video_post = spec.source_video.stat()
        if not _same_inventory_content(
            source_selected_pre, source_selected_post
        ) or not _same_inventory_content(source_metadata_pre, source_metadata_post):
            raise RuntimeError("Maintained source archive changed while fixtures were built.")
        if (
            video_pre.st_size != video_post.st_size
            or video_pre.st_mtime_ns != video_post.st_mtime_ns
        ):
            raise RuntimeError("Source video identity changed while fixtures were built.")

        selected_inventories = {
            "regular": _selected_tree_inventory(regular, selected_paths),
            "hybrid": _selected_tree_inventory(hybrid, selected_paths),
        }
        if any(
            not _same_inventory_content(value, source_selected_pre)
            for value in selected_inventories.values()
        ):
            raise RuntimeError("A copied nondetection product tree differs from the source.")
        nondetection_inventories = {
            "regular": _nondetection_inventory(regular, selected_paths),
            "hybrid": _nondetection_inventory(hybrid, selected_paths),
        }
        if not _same_inventory_content(
            nondetection_inventories["regular"],
            nondetection_inventories["hybrid"],
        ):
            raise RuntimeError("Paired nondetection direct metadata or payload bytes differ.")
        pair_metadata = _validate_pair_metadata_difference(
            regular,
            hybrid,
            run_name=spec.detection_run_name,
        )

        manifests = {
            layout: _fixture_manifest(
                layout=layout,
                archive=resolved_destination / f"{layout}.zarr",
                plan=plan,
                source_pre=source_selected_pre,
                source_post=source_selected_post,
                source_metadata_pre=source_metadata_pre,
                source_metadata_post=source_metadata_post,
                selected_inventory=selected_inventories[layout],
                nondetection_inventory=nondetection_inventories[layout],
                detection_validation=detection_validations[layout],
                metadata_validation=metadata_validations[layout],
                copy_report=copy_report,
            )
            for layout in _LAYOUTS
        }
        for layout, manifest in manifests.items():
            _write_json_exclusive(temporary / f"{layout}_manifest.json", manifest)

        pair_manifest = {
            **plan,
            "status": "published_immutable",
            "payload_io_performed": True,
            "created_at_utc": created_at,
            "pair_copy_mode_resolved": copy_report["method"],
            "source_selected_inventory": source_selected_pre.as_manifest(),
            "source_unchanged": True,
            "nondetection_pair_exact": True,
            "decoded_detection_pair_exact": True,
            "metadata_difference_validation": pair_metadata,
            "publication_receipt_relative_path": "publication_receipt.json",
            "manifests": {
                layout: {
                    "relative_path": f"{layout}_manifest.json",
                    "sha256": hashlib.sha256(
                        (
                            json.dumps(manifests[layout], allow_nan=False, indent=2, sort_keys=True)
                            + "\n"
                        ).encode("utf-8")
                    ).hexdigest(),
                }
                for layout in _LAYOUTS
            },
            "summary": {
                "archives": 2,
                "selected_nondetection_products": len(selected_paths),
                "registry_updates": 0,
                "production_selector_updates": 0,
                "training_artifacts": 0,
                "profile_promoted": False,
            },
        }
        _write_json_exclusive(temporary / "pair_manifest.json", pair_manifest)
        scratch_inventory = inventory_tree(temporary)
        _copy_tree(temporary, publication_temporary)
        published_inventory = inventory_tree(publication_temporary)
        if not _same_inventory_content(scratch_inventory, published_inventory):
            raise RuntimeError("Shared-storage publication copy differs from scratch.")
        for layout in _LAYOUTS:
            zarr.open_group(
                str(publication_temporary / f"{layout}.zarr"),
                mode="r",
                use_consolidated=False,
            )
            zarr.open_group(
                str(publication_temporary / f"{layout}.zarr"),
                mode="r",
                use_consolidated=True,
            )
        publication_receipt = {
            "schema_id": "palette.canonical_detection_full_analysis_fixture_publication",
            "schema_version": 1,
            "status": "validated_before_atomic_install",
            "copy_method": "node_local_scratch_to_shared_copytree",
            "scratch_source": str(temporary),
            "destination": str(resolved_destination),
            "scratch_inventory": scratch_inventory.as_manifest(),
            "shared_inventory_before_receipt": published_inventory.as_manifest(),
            "exact_relative_path_size_content_match": True,
            "direct_and_consolidated_open": True,
        }
        _write_json_exclusive(
            publication_temporary / "publication_receipt.json",
            publication_receipt,
        )
        thaw_tree_for_cleanup(temporary)
        shutil.rmtree(temporary)
        freeze_tree(publication_temporary)
        publication_temporary.rename(resolved_destination)
        return {**pair_manifest, "publication_receipt": publication_receipt}
    except BaseException as exc:
        failure = {
            "schema_id": PAIR_MANIFEST_SCHEMA_ID,
            "schema_version": PAIR_MANIFEST_SCHEMA_VERSION,
            "status": "incomplete_failed",
            "failed_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "destination_not_published": str(resolved_destination),
            "scratch_incomplete_evidence": str(temporary),
            "shared_incomplete_evidence": str(publication_temporary),
            "cleanup_policy": "explicit_manual_cleanup_only",
        }
        try:
            _write_json_exclusive(temporary / "failure.json", failure)
        except Exception:
            pass
        try:
            publication_temporary.mkdir(exist_ok=True)
            _write_json_exclusive(publication_temporary / "failure.json", failure)
        except Exception:
            pass
        raise RuntimeError(
            "Paired fixture publication failed; incomplete evidence remains at "
            f"{temporary} and {publication_temporary}: {exc}"
        ) from exc


__all__ = [
    "CRIMSON_CONTRACT_COMMIT",
    "CRIMSON_CONTRACT_SHA256",
    "FIXTURE_RUN_NAME",
    "FIXTURE_SPEC_SCHEMA_ID",
    "FIXTURE_SPEC_SCHEMA_VERSION",
    "PAIR_MANIFEST_SCHEMA_ID",
    "PAIR_MANIFEST_SCHEMA_VERSION",
    "CandidateSpec",
    "FullAnalysisFixtureSpec",
    "SelectedProduct",
    "load_full_analysis_fixture_spec",
    "plan_full_analysis_fixture_pair",
    "publish_full_analysis_fixture_pair",
    "require_safe_full_analysis_destination",
    "require_safe_fixture_scratch_root",
]
