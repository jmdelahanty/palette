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
from itertools import product as cartesian_product
import json
import math
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
    thaw_tree_for_cleanup,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array, storage_stats
from fisheye.shared.zarr.canonical_detection_benchmark import (
    write_detection_benchmark_candidate,
)
from fisheye.shared.zarr.canonical_detection_benchmark_input import (
    CanonicalDetectionBenchmarkInput,
)
from fisheye.shared.zarr.detection_schema import (
    CANONICAL_DETECTION_SCHEMA_V1,
    CanonicalDetectionDimensions,
)
from fisheye.shared.zarr.detection_storage import (
    plan_canonical_detection_storage,
)
from fisheye.shared.zarr.storage_profiles import StorageProfile
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
_PHYSICAL_ARRAY_FIELDS = {
    "chunk_grid",
    "chunk_key_encoding",
    "codecs",
    "storage_transformers",
}
_PHYSICAL_ARRAY_ATTRIBUTE_FIELDS = {
    "access_pattern",
    "codec_profile_id",
    "storage_policy_version",
    "storage_profile_id",
    "write_mode",
}
_INTEGRATION_UNDECLARED_LEADING_AXIS_LIMIT = 2_048


@dataclass(frozen=True)
class AdditionalPrefixAxis:
    """One explicitly bounded non-frame/non-observation leading axis."""

    name: str
    source_length: int
    selected_length: int
    index_path: str | None = None
    index_validation: str = "identity_prefix"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AdditionalPrefixAxis":
        name = str(payload.get("name", "")).strip()
        source_length = int(payload.get("source_length", -1))
        selected_length = int(payload.get("selected_length", -1))
        if not name:
            raise ValueError("Every additional prefix axis requires a name.")
        if source_length < 0 or not 0 <= selected_length <= source_length:
            raise ValueError(f"Invalid prefix lengths for additional axis {name!r}.")
        raw_index_path = str(payload.get("index_path", "")).strip()
        index_validation = str(
            payload.get("index_validation", "identity_prefix")
        ).strip()
        if index_validation not in {"identity_prefix", "monotonic_unique"}:
            raise ValueError(f"Invalid index_validation for additional axis {name!r}.")
        if not raw_index_path and "index_validation" in payload:
            raise ValueError(
                f"Additional axis {name!r} cannot validate an absent index path."
            )
        return cls(
            name=name,
            source_length=source_length,
            selected_length=selected_length,
            index_path=(
                _normalize_relative_group_path(raw_index_path)
                if raw_index_path
                else None
            ),
            index_validation=index_validation,
        )

    def as_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_length": self.source_length,
            "selected_length": self.selected_length,
            "index_path": self.index_path,
            "index_validation": self.index_validation,
        }


@dataclass(frozen=True)
class IntegrationWindow:
    """Declared frame/row axes for bounded or full-duration fixture evidence."""

    classification: str
    camera_frame_start: int
    camera_frame_stop: int
    source_observation_rows: int
    frame_counts_path: str
    frame_indices_path: str
    additional_prefix_axes: tuple[AdditionalPrefixAxis, ...]
    csr_group_paths: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "IntegrationWindow":
        classification = str(payload.get("classification", "")).strip()
        if classification not in {
            "integration_fixture",
            "full_duration_promotion_fixture",
        }:
            raise ValueError(
                "integration_window.classification must be 'integration_fixture' "
                "or 'full_duration_promotion_fixture'."
            )
        start = int(payload.get("camera_frame_start", -1))
        stop = int(payload.get("camera_frame_stop", -1))
        if start != 0 or stop <= start:
            raise ValueError(
                "Integration fixtures must use a nonempty camera prefix beginning at zero."
            )
        source_rows = int(payload.get("source_observation_rows", -1))
        if source_rows < 0:
            raise ValueError("source_observation_rows cannot be negative.")
        raw_axes = payload.get("additional_prefix_axes", [])
        if not isinstance(raw_axes, list) or any(
            not isinstance(item, Mapping) for item in raw_axes
        ):
            raise ValueError("additional_prefix_axes must be a list of objects.")
        axes = tuple(AdditionalPrefixAxis.from_payload(item) for item in raw_axes)
        if len({axis.name for axis in axes}) != len(axes):
            raise ValueError("Additional prefix axis names must be unique.")
        source_lengths = [axis.source_length for axis in axes]
        if len(set(source_lengths)) != len(source_lengths):
            raise ValueError("Additional prefix source lengths must be unique.")
        raw_csr = payload.get("csr_group_paths", [])
        if not isinstance(raw_csr, list):
            raise ValueError("csr_group_paths must be a list.")
        csr_paths = tuple(_normalize_relative_group_path(str(path)) for path in raw_csr)
        if len(set(csr_paths)) != len(csr_paths):
            raise ValueError("CSR group paths must be unique.")
        return cls(
            classification=classification,
            camera_frame_start=start,
            camera_frame_stop=stop,
            source_observation_rows=source_rows,
            frame_counts_path=_normalize_relative_group_path(
                str(payload.get("frame_counts_path", ""))
            ),
            frame_indices_path=_normalize_relative_group_path(
                str(payload.get("frame_indices_path", ""))
            ),
            additional_prefix_axes=axes,
            csr_group_paths=csr_paths,
        )

    def as_manifest(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "camera_frame_start": self.camera_frame_start,
            "camera_frame_stop": self.camera_frame_stop,
            "source_observation_rows": self.source_observation_rows,
            "frame_counts_path": self.frame_counts_path,
            "frame_indices_path": self.frame_indices_path,
            "additional_prefix_axes": [
                axis.as_manifest() for axis in self.additional_prefix_axes
            ],
            "csr_group_paths": list(self.csr_group_paths),
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
    return tuple(
        PurePosixPath(*parts[:index]).as_posix() for index in range(1, len(parts))
    )


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
    integration_window: IntegrationWindow | None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FullAnalysisFixtureSpec":
        if payload.get("schema_id") != FIXTURE_SPEC_SCHEMA_ID:
            raise ValueError(
                f"Fixture spec schema_id must be {FIXTURE_SPEC_SCHEMA_ID!r}."
            )
        if payload.get("schema_version") != FIXTURE_SPEC_SCHEMA_VERSION:
            raise ValueError(
                f"Fixture spec schema_version must be {FIXTURE_SPEC_SCHEMA_VERSION}."
            )
        fixture_id = str(payload.get("fixture_id", "")).strip()
        recording_id = str(payload.get("recording_id", "")).strip()
        if not fixture_id or not recording_id:
            raise ValueError("fixture_id and recording_id are required.")
        if any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
            for character in fixture_id
        ):
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
        if any(
            item.path == "detect_runs" or item.path.startswith("detect_runs/")
            for item in selected
        ):
            raise ValueError(
                "Source detect_runs cannot be copied into the paired fixture."
            )

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
        if not isinstance(raw_candidates, Mapping) or set(raw_candidates) != set(
            _LAYOUTS
        ):
            raise ValueError(
                "candidates must contain exactly regular and hybrid entries."
            )
        candidates = {
            layout: CandidateSpec.from_payload(raw_candidates[layout])
            for layout in _LAYOUTS
        }

        crimson_contract = payload.get("crimson_contract")
        if not isinstance(crimson_contract, Mapping):
            raise ValueError("crimson_contract must be an object.")
        if crimson_contract.get("commit") != CRIMSON_CONTRACT_COMMIT:
            raise ValueError(
                "Fixture spec does not pin the frozen Crimson contract commit."
            )
        if crimson_contract.get("document_sha256") != CRIMSON_CONTRACT_SHA256:
            raise ValueError(
                "Fixture spec does not pin the frozen Crimson contract digest."
            )

        detection_run_name = str(payload.get("detection_run_name", "")).strip()
        if detection_run_name != FIXTURE_RUN_NAME:
            raise ValueError(
                f"detection_run_name must be the frozen name {FIXTURE_RUN_NAME!r}."
            )

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

        raw_integration_window = payload.get("integration_window")
        if raw_integration_window is not None and not isinstance(
            raw_integration_window, Mapping
        ):
            raise ValueError("integration_window must be an object when present.")
        integration_window = (
            IntegrationWindow.from_payload(raw_integration_window)
            if isinstance(raw_integration_window, Mapping)
            else None
        )
        if integration_window is not None:
            source_frames = int(source_expectations.get("n_frames", -1))
            if not 0 < integration_window.camera_frame_stop <= source_frames:
                raise ValueError(
                    "The integration camera-frame stop must be within source n_frames."
                )
            reserved_lengths = {
                source_frames,
                integration_window.source_observation_rows,
            }
            additional_lengths = {
                axis.source_length for axis in integration_window.additional_prefix_axes
            }
            if reserved_lengths & additional_lengths:
                raise ValueError(
                    "Additional prefix axes cannot reuse frame or observation lengths."
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
            source_video=Path(raw_source_paths["source_video"]).expanduser().resolve(),
            source_video_relative_path=source_video_relative_path.as_posix(),
            detection_run_name=detection_run_name,
            selected_products=selected,
            node_expectations=tuple(node_expectations),
            selector_overrides=selector_overrides,
            source_expectations=dict(source_expectations),
            candidates=candidates,
            crimson_contract=dict(crimson_contract),
            integration_window=integration_window,
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
            "selected_products": [
                item.as_manifest() for item in self.selected_products
            ],
            "node_expectations": [dict(item) for item in self.node_expectations],
            "selector_overrides": {
                path: dict(attrs) for path, attrs in self.selector_overrides.items()
            },
            "source_expectations": dict(self.source_expectations),
            "candidates": {
                layout: self.candidates[layout].as_manifest() for layout in _LAYOUTS
            },
            "crimson_contract": dict(self.crimson_contract),
            "integration_window": (
                self.integration_window.as_manifest()
                if self.integration_window is not None
                else None
            ),
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
        raise ValueError(
            "Fixture scratch root cannot be inside benchmark or recording data."
        )
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
                metadata.get("chunk_grid", {})
                .get("configuration", {})
                .get("chunk_shape")
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
            raise ValueError(
                f"Source node expectation mismatch at {path}: {mismatches}"
            )
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
                    raise ValueError(
                        f"Selected source trees cannot contain symlinks: {child}"
                    )
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


def _same_inventory_content(left: TreeInventory, right: TreeInventory) -> bool:
    return (
        left.file_count == right.file_count
        and left.apparent_bytes == right.apparent_bytes
        and left.tree_sha256 == right.tree_sha256
    )


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
            raise ValueError(
                f"{label} direct/consolidated declaration mismatch at {path}."
            )
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
    if (
        not isinstance(profile, Mapping)
        or profile.get("profile_id") != candidate.expected_profile_id
    ):
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
            raise ValueError(
                f"{layout} candidate path {path!r} is not a Zarr v3 array."
            )
    if layout == "regular":
        if any(
            any(
                codec.get("name") == "sharding_indexed"
                for codec in direct[path].get("codecs", [])
            )
            for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths
        ):
            raise ValueError(
                "Regular candidate unexpectedly contains indexed sharding."
            )
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
    source_metadata = _require_group(
        spec.source_archive, label="Source analysis archive"
    )
    if spec.source_archive.is_relative_to(resolved_root):
        raise ValueError(
            "Production source archive cannot be inside the benchmark root."
        )
    if not spec.source_recording.is_dir():
        raise FileNotFoundError(
            f"Source recording does not exist: {spec.source_recording}"
        )
    expected_video = spec.source_recording / spec.source_video_relative_path
    if expected_video.resolve() != spec.source_video or not spec.source_video.is_file():
        raise ValueError(
            "Source video association does not resolve to the declared file."
        )
    _validate_source_expectations(source_metadata, spec.source_expectations)
    node_expectation_reports = _validate_node_expectations(
        spec.source_archive,
        spec.node_expectations,
    )

    selected_paths = tuple(item.path for item in spec.selected_products)
    for relative in selected_paths:
        _require_group(
            spec.source_archive / relative, label=f"Selected product {relative!r}"
        )
        for ancestor in _ancestors(relative):
            _require_group(
                spec.source_archive / ancestor, label=f"Ancestor {ancestor!r}"
            )
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
        raise ValueError(
            "Regular and hybrid candidates do not share one logical schema."
        )
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

    bounded_integration = (
        spec.integration_window is not None
        and spec.integration_window.classification == "integration_fixture"
    )
    evidence_scope = (
        {
            "classification": "integration_fixture",
            "camera_frame_range": [
                spec.integration_window.camera_frame_start,
                spec.integration_window.camera_frame_stop,
            ],
            "valid_for": [
                "schema_open",
                "required_product_readiness",
                "overlay_correctness",
                "seek_cancellation",
            ],
            "not_valid_for": [
                "frozen_full_duration_promotion_gate",
                "full_duration_startup",
                "full_duration_object_count",
                "full_duration_cache_pressure",
                "long_traversal",
            ],
        }
        if bounded_integration
        else {
            "classification": "full_duration_promotion_fixture",
            "valid_for": ["frozen_full_duration_promotion_gate"],
            "not_valid_for": [],
        }
    )
    frame_relationship_plan = (
        _resolve_integration_cardinalities(spec)
        if spec.integration_window is not None
        else None
    )

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
        "evidence_scope": evidence_scope,
        "integration_window": (
            spec.integration_window.as_manifest()
            if spec.integration_window is not None
            else None
        ),
        "frame_relationship_plan": frame_relationship_plan,
        "destination": str(resolved_destination),
        "benchmark_root": str(resolved_root),
        "pair_copy_mode_requested": pair_copy_mode,
        "pair_copy_mode_resolved": None,
        "scratch_root": str(resolved_scratch) if resolved_scratch is not None else None,
        "source": {
            "recording_id": spec.recording_id,
            "recording": str(spec.source_recording),
            "archive": str(spec.source_archive),
            "selected_products": [
                item.as_manifest() for item in spec.selected_products
            ],
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
            "expected_commit_match": (None if expected_commit is None else True),
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
) -> dict[str, Any]:
    destination.mkdir(parents=True)
    root_normalizations: list[dict[str, Any]] = []
    source_root_metadata = _sanitized_direct_metadata(
        _read_json(spec.source_archive / "zarr.json"),
        source_metadata_path="zarr.json",
        records=root_normalizations,
        omit_consolidated=True,
    )
    source_attributes = source_root_metadata.get("attributes")
    if not isinstance(source_attributes, Mapping):
        source_attributes = {}
    window = spec.integration_window
    classification = (
        window.classification
        if window is not None
        else "full_duration_promotion_fixture"
    )
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
        "fixture_evidence_classification": classification,
        "fixture_nonfinite_normalizations": root_normalizations,
    }
    if window is not None:
        source_root_metadata["attributes"]["fixture_camera_frame_range"] = [
            window.camera_frame_start,
            window.camera_frame_stop,
        ]
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
        _copy_group_metadata(
            spec.source_archive / "detect_runs", destination / "detect_runs"
        )
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
    return {
        "nonfinite_normalizations": root_normalizations,
        "source_modified": False,
        "payload_copy": "whole_selected_product_trees",
        "source_video": "reference_only_not_copied",
    }


def _json_pointer_component(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _normalize_nonfinite_json(
    value: Any,
    *,
    source_metadata_path: str,
    pointer: str,
    records: list[dict[str, Any]],
) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        original = (
            "nan"
            if math.isnan(value)
            else "positive_infinity"
            if value > 0
            else "negative_infinity"
        )
        records.append(
            {
                "source_metadata_path": source_metadata_path,
                "json_pointer": pointer or "/",
                "original_value": original,
                "fixture_value": None,
                "scope": "benchmark_copy_only",
                "source_modified": False,
            }
        )
        return None
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_nonfinite_json(
                item,
                source_metadata_path=source_metadata_path,
                pointer=f"{pointer}/{_json_pointer_component(key)}",
                records=records,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _normalize_nonfinite_json(
                item,
                source_metadata_path=source_metadata_path,
                pointer=f"{pointer}/{index}",
                records=records,
            )
            for index, item in enumerate(value)
        ]
    return value


def _sanitized_direct_metadata(
    metadata: Mapping[str, Any],
    *,
    source_metadata_path: str,
    records: list[dict[str, Any]],
    omit_consolidated: bool,
) -> dict[str, Any]:
    direct = dict(metadata)
    if omit_consolidated and "consolidated_metadata" in direct:
        direct.pop("consolidated_metadata")
        records.append(
            {
                "source_metadata_path": source_metadata_path,
                "json_pointer": "/consolidated_metadata",
                "original_value": "source_inline_consolidated_metadata",
                "fixture_value": "omitted_then_regenerated_from_sliced_direct_metadata",
                "scope": "benchmark_copy_only",
                "source_modified": False,
            }
        )
    normalized = _normalize_nonfinite_json(
        direct,
        source_metadata_path=source_metadata_path,
        pointer="",
        records=records,
    )
    if not isinstance(normalized, dict):  # pragma: no cover - mapping input
        raise AssertionError("Normalized Zarr metadata must remain an object.")
    return normalized


def _open_direct_array(path: Path, *, mode: str) -> Any:
    return zarr.open_array(str(path), mode=mode)


def _require_array_metadata(
    root: Path,
    relative_path: str,
    *,
    expected_leading_length: int | None = None,
) -> dict[str, Any]:
    metadata_path = root / relative_path / "zarr.json"
    metadata = _read_json(metadata_path)
    if metadata.get("zarr_format") != 3 or metadata.get("node_type") != "array":
        raise ValueError(f"Expected a Zarr v3 array at {relative_path!r}.")
    shape = metadata.get("shape")
    if not isinstance(shape, list):
        raise ValueError(f"Array shape is missing at {relative_path!r}.")
    if expected_leading_length is not None and (
        not shape or int(shape[0]) != expected_leading_length
    ):
        raise ValueError(
            f"Array {relative_path!r} does not have expected leading length "
            f"{expected_leading_length}."
        )
    return metadata


def _validate_csr_vectors(
    pointers: np.ndarray,
    lengths: np.ndarray,
    *,
    label: str,
) -> tuple[int, int]:
    if pointers.shape != lengths.shape or pointers.ndim != 1:
        raise ValueError(f"CSR ptr/len shapes differ at {label!r}.")
    valid = (pointers >= 0) & (lengths > 0)
    empty = (pointers == -1) & (lengths == 0)
    if not np.all(valid | empty):
        raise ValueError(
            f"CSR rows at {label!r} must be positive spans or (-1, 0) empties."
        )
    valid_pointers = pointers[valid]
    valid_lengths = lengths[valid]
    if valid_pointers.size:
        if int(valid_pointers[0]) != 0 or not np.array_equal(
            valid_pointers[1:], valid_pointers[:-1] + valid_lengths[:-1]
        ):
            raise ValueError(f"CSR positive spans are not contiguous at {label!r}.")
        endpoint = int(valid_pointers[-1] + valid_lengths[-1])
    else:
        endpoint = 0
    return endpoint, int(empty.sum())


def _resolve_integration_cardinalities(
    spec: FullAnalysisFixtureSpec,
) -> dict[str, Any]:
    window = spec.integration_window
    if window is None:  # pragma: no cover - caller guard
        raise ValueError("An integration window is required.")
    source_frame_count = int(spec.source_expectations["n_frames"])
    if window.classification == "full_duration_promotion_fixture":
        if window.camera_frame_stop != source_frame_count:
            raise ValueError(
                "A full-duration fixture window must cover every camera frame."
            )
        incomplete_axes = [
            axis.name
            for axis in window.additional_prefix_axes
            if axis.selected_length != axis.source_length
        ]
        if incomplete_axes:
            raise ValueError(
                "A full-duration fixture cannot truncate additional axes: "
                f"{incomplete_axes}."
            )
    if source_frame_count == window.source_observation_rows:
        raise ValueError(
            "Frame and observation source cardinalities collide; declare an "
            "explicit per-array slice map before using this source."
        )
    _require_array_metadata(
        spec.source_archive,
        window.frame_counts_path,
        expected_leading_length=source_frame_count,
    )
    counts_array = _open_direct_array(
        spec.source_archive / window.frame_counts_path,
        mode="r",
    )
    all_counts = np.asarray(counts_array[:], dtype=np.int64)
    if np.any(all_counts < 0):
        raise ValueError("The integration frame-count reference contains negatives.")
    if int(all_counts.sum(dtype=np.int64)) != window.source_observation_rows:
        raise ValueError(
            "The complete frame-count reference does not match source_observation_rows."
        )
    selected_counts = np.asarray(
        all_counts[window.camera_frame_start : window.camera_frame_stop],
        dtype=np.int64,
    )
    selected_rows = int(selected_counts.sum(dtype=np.int64))

    _require_array_metadata(
        spec.source_archive,
        window.frame_indices_path,
        expected_leading_length=window.source_observation_rows,
    )
    selected_frame_indices = np.asarray(
        _open_direct_array(
            spec.source_archive / window.frame_indices_path,
            mode="r",
        )[:selected_rows],
        dtype=np.int64,
    )
    if selected_frame_indices.shape != (selected_rows,):
        raise ValueError("The frame-index reference is not one-dimensional.")
    if selected_rows and (
        int(selected_frame_indices[0]) < window.camera_frame_start
        or int(selected_frame_indices[-1]) >= window.camera_frame_stop
        or np.any(np.diff(selected_frame_indices) < 0)
    ):
        raise ValueError(
            "The selected frame-index reference escapes the prefix window."
        )
    observed_counts = np.bincount(
        selected_frame_indices,
        minlength=window.camera_frame_stop,
    )[window.camera_frame_start : window.camera_frame_stop]
    if not np.array_equal(observed_counts, selected_counts):
        raise ValueError("frame_counts does not exactly describe frame_indices.")

    leading_axes: dict[int, dict[str, Any]] = {
        source_frame_count: {
            "axis": "camera_frames",
            "source_length": source_frame_count,
            "selected_length": window.camera_frame_stop,
        },
        window.source_observation_rows: {
            "axis": "observation_rows",
            "source_length": window.source_observation_rows,
            "selected_length": selected_rows,
        },
    }
    additional_reports: list[dict[str, Any]] = []
    for axis in window.additional_prefix_axes:
        report = axis.as_manifest()
        if axis.index_path is not None:
            _require_array_metadata(
                spec.source_archive,
                axis.index_path,
                expected_leading_length=axis.source_length,
            )
            selected_index = np.asarray(
                _open_direct_array(
                    spec.source_archive / axis.index_path,
                    mode="r",
                )[: axis.selected_length],
                dtype=np.int64,
            )
            if selected_index.shape != (axis.selected_length,):
                raise ValueError(
                    f"Additional axis index {axis.index_path!r} has an "
                    "unexpected shape."
                )
            if axis.index_validation == "identity_prefix":
                expected_index = np.arange(axis.selected_length, dtype=np.int64)
                if not np.array_equal(selected_index, expected_index):
                    raise ValueError(
                        f"Additional axis index {axis.index_path!r} is not an "
                        "identity prefix."
                    )
                report["identity_prefix_valid"] = True
            else:
                if selected_index.size and (
                    int(selected_index[0]) < 0 or np.any(np.diff(selected_index) <= 0)
                ):
                    raise ValueError(
                        f"Additional axis index {axis.index_path!r} is not "
                        "strictly increasing and nonnegative."
                    )
                report.update(
                    {
                        "monotonic_unique_valid": True,
                        "selected_index_min": (
                            int(selected_index[0]) if selected_index.size else None
                        ),
                        "selected_index_max": (
                            int(selected_index[-1]) if selected_index.size else None
                        ),
                        "selected_index_gap_count": (
                            int(selected_index[-1] - selected_index[0] + 1)
                            - int(selected_index.size)
                            if selected_index.size
                            else 0
                        ),
                    }
                )
        leading_axes[axis.source_length] = {
            "axis": axis.name,
            "source_length": axis.source_length,
            "selected_length": axis.selected_length,
        }
        additional_reports.append(report)

    point_axes: dict[str, dict[str, Any]] = {}
    csr_reports: list[dict[str, Any]] = []
    for group_path in window.csr_group_paths:
        ptr_path = f"{group_path}/ptr"
        len_path = f"{group_path}/len"
        points_path = f"{group_path}/points_xy"
        _require_array_metadata(
            spec.source_archive,
            ptr_path,
            expected_leading_length=window.source_observation_rows,
        )
        _require_array_metadata(
            spec.source_archive,
            len_path,
            expected_leading_length=window.source_observation_rows,
        )
        points_metadata = _require_array_metadata(spec.source_archive, points_path)
        points_shape = tuple(int(value) for value in points_metadata["shape"])
        if len(points_shape) != 2 or points_shape[1] != 2:
            raise ValueError(f"CSR points array has invalid shape at {points_path!r}.")
        ptr_array = _open_direct_array(spec.source_archive / ptr_path, mode="r")
        len_array = _open_direct_array(spec.source_archive / len_path, mode="r")
        pointers = np.asarray(ptr_array[:selected_rows], dtype=np.int64)
        lengths = np.asarray(len_array[:selected_rows], dtype=np.int64)
        selected_points, selected_empty_rows = _validate_csr_vectors(
            pointers,
            lengths,
            label=group_path,
        )
        full_pointers = np.asarray(ptr_array[:], dtype=np.int64)
        full_lengths = np.asarray(len_array[:], dtype=np.int64)
        source_endpoint, source_empty_rows = _validate_csr_vectors(
            full_pointers,
            full_lengths,
            label=f"{group_path}:complete_source",
        )
        if source_endpoint != points_shape[0]:
            raise ValueError(
                f"Complete CSR endpoint does not match points at {group_path!r}."
            )
        point_axes[points_path] = {
            "axis": f"csr_points:{group_path}",
            "source_length": points_shape[0],
            "selected_length": selected_points,
        }
        csr_reports.append(
            {
                "group_path": group_path,
                "row_count": selected_rows,
                "point_count": selected_points,
                "empty_row_count": selected_empty_rows,
                "source_empty_row_count": source_empty_rows,
                "contiguous_prefix": True,
                "complete_source_endpoint_valid": True,
            }
        )
    return {
        "camera_frame_range": [
            window.camera_frame_start,
            window.camera_frame_stop,
        ],
        "source_frame_count": source_frame_count,
        "selected_frame_count": window.camera_frame_stop,
        "source_observation_rows": window.source_observation_rows,
        "selected_observation_rows": selected_rows,
        "frame_counts_path": window.frame_counts_path,
        "frame_indices_path": window.frame_indices_path,
        "frame_counts_exact": True,
        "leading_axes": leading_axes,
        "point_axes": point_axes,
        "additional_prefix_axes": additional_reports,
        "csr_groups": csr_reports,
    }


def _target_array_shape(
    relative_path: str,
    source_shape: tuple[int, ...],
    *,
    cardinalities: Mapping[str, Any],
) -> tuple[tuple[int, ...], str]:
    if not source_shape:
        return source_shape, "scalar"
    point_axis = cardinalities["point_axes"].get(relative_path)
    if point_axis is not None:
        return (
            int(point_axis["selected_length"]),
            *source_shape[1:],
        ), str(point_axis["axis"])
    leading_axis = cardinalities["leading_axes"].get(source_shape[0])
    if leading_axis is not None:
        return (
            int(leading_axis["selected_length"]),
            *source_shape[1:],
        ), str(leading_axis["axis"])
    if source_shape[0] > _INTEGRATION_UNDECLARED_LEADING_AXIS_LIMIT:
        raise ValueError(
            f"Array {relative_path!r} has undeclared large leading axis "
            f"{source_shape[0]}."
        )
    return source_shape, "constant_or_small_index"


def _payload_unit_shape(
    metadata: Mapping[str, Any],
    target_shape: tuple[int, ...],
) -> tuple[int, ...]:
    if not target_shape:
        return ()
    chunk_grid = metadata.get("chunk_grid")
    if not isinstance(chunk_grid, Mapping):
        raise ValueError("Array chunk_grid metadata is missing.")
    configuration = chunk_grid.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("Array chunk_grid configuration is missing.")
    raw_shape = configuration.get("chunk_shape")
    if not isinstance(raw_shape, list) or len(raw_shape) != len(target_shape):
        raise ValueError("Array payload-unit shape is invalid.")
    unit_shape = tuple(int(value) for value in raw_shape)
    if any(value <= 0 for value in unit_shape):
        raise ValueError("Array payload-unit dimensions must be positive.")
    return unit_shape


def _tile_selections(
    shape: tuple[int, ...],
    unit_shape: tuple[int, ...],
) -> Iterable[tuple[slice, ...]]:
    if not shape:
        yield ()
        return
    if any(value == 0 for value in shape):
        return
    ranges = [range(0, length, unit) for length, unit in zip(shape, unit_shape)]
    for starts in cartesian_product(*ranges):
        yield tuple(
            slice(start, min(start + unit, length))
            for start, unit, length in zip(starts, unit_shape, shape)
        )


def _logical_digest_header(
    *,
    metadata: Mapping[str, Any],
    shape: tuple[int, ...],
    unit_shape: tuple[int, ...],
) -> hashlib._Hash:
    digest = hashlib.sha256()
    header = {
        "schema_id": "palette.logical_array_slice_digest",
        "schema_version": 1,
        "shape": list(shape),
        "data_type": metadata.get("data_type"),
        "block_shape": list(unit_shape),
    }
    encoded = json.dumps(header, allow_nan=False, sort_keys=True).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    return digest


def _update_logical_digest(
    digest: Any,
    *,
    selection: tuple[slice, ...],
    values: np.ndarray,
    string_payload: bool,
) -> int:
    bounds = [[item.start, item.stop] for item in selection]
    encoded_bounds = json.dumps(bounds, separators=(",", ":")).encode("utf-8")
    digest.update(len(encoded_bounds).to_bytes(8, "little"))
    digest.update(encoded_bounds)
    array = np.asarray(values)
    if string_payload or array.dtype.kind in {"O", "S", "U", "T"}:
        encoded_bytes = 0
        for item in array.ravel(order="C"):
            if item is None:
                tag = b"n"
                encoded = b""
            elif isinstance(item, bytes):
                tag = b"b"
                encoded = item
            else:
                tag = b"s"
                encoded = str(item).encode("utf-8")
            digest.update(tag)
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
            encoded_bytes += len(encoded)
        return encoded_bytes
    contiguous = np.ascontiguousarray(array)
    payload = memoryview(contiguous).cast("B")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(payload)
    return int(payload.nbytes)


def _hash_logical_array_slice(
    array: Any,
    *,
    metadata: Mapping[str, Any],
    shape: tuple[int, ...],
    unit_shape: tuple[int, ...],
) -> dict[str, Any]:
    digest = _logical_digest_header(
        metadata=metadata,
        shape=shape,
        unit_shape=unit_shape,
    )
    block_count = 0
    logical_payload_bytes = 0
    string_payload = metadata.get("data_type") == "string"
    for selection in _tile_selections(shape, unit_shape):
        values = np.asarray(array[selection])
        logical_payload_bytes += _update_logical_digest(
            digest,
            selection=selection,
            values=values,
            string_payload=string_payload,
        )
        block_count += 1
    return {
        "sha256": digest.hexdigest(),
        "block_count": block_count,
        "logical_payload_bytes": logical_payload_bytes,
    }


def _copy_integration_array(
    source: Path,
    destination: Path,
    *,
    relative_path: str,
    metadata: Mapping[str, Any],
    cardinalities: Mapping[str, Any],
    normalization_records: list[dict[str, Any]],
) -> dict[str, Any]:
    source_shape = tuple(int(value) for value in metadata["shape"])
    target_shape, axis = _target_array_shape(
        relative_path,
        source_shape,
        cardinalities=cardinalities,
    )
    source_metadata_path = f"{relative_path}/zarr.json"
    copied_metadata = _sanitized_direct_metadata(
        metadata,
        source_metadata_path=source_metadata_path,
        records=normalization_records,
        omit_consolidated=False,
    )
    copied_metadata["shape"] = list(target_shape)
    destination.mkdir(parents=True)
    _write_json_exclusive(destination / "zarr.json", copied_metadata)
    source_array = _open_direct_array(source, mode="r")
    destination_array = _open_direct_array(destination, mode="r+")
    unit_shape = _payload_unit_shape(copied_metadata, target_shape)
    string_payload = copied_metadata.get("data_type") == "string"
    source_digest = _logical_digest_header(
        metadata=copied_metadata,
        shape=target_shape,
        unit_shape=unit_shape,
    )
    source_blocks = 0
    source_payload_bytes = 0
    for selection in _tile_selections(target_shape, unit_shape):
        values = np.asarray(source_array[selection])
        destination_array[selection] = values
        source_payload_bytes += _update_logical_digest(
            source_digest,
            selection=selection,
            values=values,
            string_payload=string_payload,
        )
        source_blocks += 1
    destination_digest = _hash_logical_array_slice(
        destination_array,
        metadata=copied_metadata,
        shape=target_shape,
        unit_shape=unit_shape,
    )
    if source_digest.hexdigest() != destination_digest["sha256"]:
        raise RuntimeError(f"Copied logical array differs at {relative_path!r}.")
    return {
        "path": relative_path,
        "axis": axis,
        "source_shape": list(source_shape),
        "fixture_shape": list(target_shape),
        "data_type": copied_metadata.get("data_type"),
        "payload_unit_shape": list(unit_shape),
        "source_metadata_sha256": _sha256_file(source / "zarr.json"),
        "fixture_direct_metadata_sha256": _sha256_file(destination / "zarr.json"),
        "source_logical_sha256": source_digest.hexdigest(),
        "regular_logical_sha256": destination_digest["sha256"],
        "block_count": source_blocks,
        "logical_payload_bytes": source_payload_bytes,
        "source_to_regular_exact": True,
    }


def _copy_integration_group_tree(
    source: Path,
    destination: Path,
    *,
    source_root: Path,
    cardinalities: Mapping[str, Any],
    normalization_records: list[dict[str, Any]],
    group_reports: list[dict[str, Any]],
    array_reports: list[dict[str, Any]],
) -> None:
    relative_path = source.relative_to(source_root).as_posix()
    metadata = _read_json(source / "zarr.json")
    if metadata.get("node_type") == "array":
        array_reports.append(
            _copy_integration_array(
                source,
                destination,
                relative_path=relative_path,
                metadata=metadata,
                cardinalities=cardinalities,
                normalization_records=normalization_records,
            )
        )
        return
    if metadata.get("zarr_format") != 3 or metadata.get("node_type") != "group":
        raise ValueError(f"Selected node is not a Zarr v3 node: {source}")
    copied_metadata = _sanitized_direct_metadata(
        metadata,
        source_metadata_path=f"{relative_path}/zarr.json",
        records=normalization_records,
        omit_consolidated=True,
    )
    destination.mkdir(parents=True)
    _write_json_exclusive(destination / "zarr.json", copied_metadata)
    group_reports.append(
        {
            "path": relative_path,
            "source_metadata_sha256": _sha256_file(source / "zarr.json"),
            "fixture_direct_metadata_sha256": _sha256_file(destination / "zarr.json"),
        }
    )
    for child in sorted(source.iterdir()):
        if child.is_symlink():
            raise ValueError(f"Selected source trees cannot contain symlinks: {child}")
        if not child.is_dir() or not (child / "zarr.json").is_file():
            continue
        _copy_integration_group_tree(
            child,
            destination / child.name,
            source_root=source_root,
            cardinalities=cardinalities,
            normalization_records=normalization_records,
            group_reports=group_reports,
            array_reports=array_reports,
        )


def _copy_sanitized_group_envelope(
    source: Path,
    destination: Path,
    *,
    source_root: Path,
    normalization_records: list[dict[str, Any]],
) -> dict[str, Any]:
    relative_path = source.relative_to(source_root).as_posix()
    metadata = _require_group(source, label="Source group envelope")
    copied = _sanitized_direct_metadata(
        metadata,
        source_metadata_path=f"{relative_path}/zarr.json",
        records=normalization_records,
        omit_consolidated=True,
    )
    destination.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(destination / "zarr.json", copied)
    return {
        "path": relative_path,
        "source_metadata_sha256": _sha256_file(source / "zarr.json"),
        "fixture_direct_metadata_sha256": _sha256_file(destination / "zarr.json"),
    }


def _assemble_integration_nondetection_base(
    spec: FullAnalysisFixtureSpec,
    *,
    destination: Path,
    created_at_utc: str,
    cardinalities: Mapping[str, Any],
) -> dict[str, Any]:
    window = spec.integration_window
    if window is None:  # pragma: no cover - caller guard
        raise ValueError("An integration window is required.")
    destination.mkdir(parents=True)
    normalization_records: list[dict[str, Any]] = []
    source_root_metadata = _read_json(spec.source_archive / "zarr.json")
    copied_root = _sanitized_direct_metadata(
        source_root_metadata,
        source_metadata_path="zarr.json",
        records=normalization_records,
        omit_consolidated=True,
    )
    source_attributes = copied_root.get("attributes")
    if not isinstance(source_attributes, Mapping):
        source_attributes = {}
    copied_root["attributes"] = {
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
        "fixture_evidence_classification": "integration_fixture",
        "fixture_camera_frame_range": [
            window.camera_frame_start,
            window.camera_frame_stop,
        ],
        "fixture_selected_observation_rows": cardinalities["selected_observation_rows"],
        "fixture_source_metadata_semantics": (
            "source publication provenance retained; array payloads are exact "
            "prefix slices and source aggregate summaries are not recomputed"
        ),
    }
    _write_json_exclusive(destination / "zarr.json", copied_root)

    group_reports: list[dict[str, Any]] = [
        {
            "path": "",
            "source_metadata_sha256": _sha256_file(spec.source_archive / "zarr.json"),
            "fixture_direct_metadata_sha256": _sha256_file(destination / "zarr.json"),
        }
    ]
    array_reports: list[dict[str, Any]] = []
    copied_ancestors: set[str] = set()
    for selected in spec.selected_products:
        for ancestor in _ancestors(selected.path):
            if ancestor in copied_ancestors:
                continue
            group_reports.append(
                _copy_sanitized_group_envelope(
                    spec.source_archive / ancestor,
                    destination / ancestor,
                    source_root=spec.source_archive,
                    normalization_records=normalization_records,
                )
            )
            copied_ancestors.add(ancestor)
        _copy_integration_group_tree(
            spec.source_archive / selected.path,
            destination / selected.path,
            source_root=spec.source_archive,
            cardinalities=cardinalities,
            normalization_records=normalization_records,
            group_reports=group_reports,
            array_reports=array_reports,
        )

    if "detect_runs" not in copied_ancestors:
        group_reports.append(
            _copy_sanitized_group_envelope(
                spec.source_archive / "detect_runs",
                destination / "detect_runs",
                source_root=spec.source_archive,
                normalization_records=normalization_records,
            )
        )
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
    for report in group_reports:
        relative_path = str(report["path"])
        metadata_path = (
            destination / relative_path / "zarr.json"
            if relative_path
            else destination / "zarr.json"
        )
        report["fixture_direct_metadata_sha256"] = _sha256_file(metadata_path)
    return {
        "schema_id": "palette.integration_fixture_logical_slice_manifest",
        "schema_version": 1,
        "cardinalities": dict(cardinalities),
        "groups": group_reports,
        "arrays": array_reports,
        "nonfinite_normalizations": normalization_records,
        "source_modified": False,
    }


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
            [
                "cp",
                "--archive",
                "--reflink=always",
                "--",
                str(source),
                str(destination),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Pair-base reflink failed: {completed.stderr.strip()}")
        return {"method": "reflink", "probe": probe}
    _copy_tree(source, destination)
    return {"method": "copy", "probe": probe}


def _normalized_detection_run_attributes(
    candidate_metadata: Mapping[str, Any],
) -> dict[str, Any]:
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


def _storage_profile_from_candidate(candidate: CandidateSpec) -> StorageProfile:
    metadata = _read_json(candidate.path / "zarr.json")
    attributes = metadata.get("attributes")
    if not isinstance(attributes, Mapping):
        raise ValueError("Detection candidate attributes are missing.")
    storage_plan = attributes.get("storage_plan")
    if not isinstance(storage_plan, Mapping):
        raise ValueError("Detection candidate storage plan is missing.")
    profile = storage_plan.get("storage_profile")
    if not isinstance(profile, Mapping):
        raise ValueError("Detection candidate storage profile is missing.")
    overrides = profile.get("target_chunk_bytes_by_access", {})
    if not isinstance(overrides, Mapping):
        raise ValueError(
            "Detection candidate access-specific chunk targets are invalid."
        )
    restored = StorageProfile(
        profile_id=str(profile["profile_id"]),
        target_chunk_bytes=int(profile["target_chunk_bytes"]),
        min_chunk_bytes=int(profile["min_chunk_bytes"]),
        max_chunk_bytes=int(profile["max_chunk_bytes"]),
        eager_max_bytes=int(profile["eager_max_bytes"]),
        target_shard_bytes=int(profile["target_shard_bytes"]),
        per_row_target_shard_bytes=int(profile["per_row_target_shard_bytes"]),
        max_shard_bytes=int(profile["max_shard_bytes"]),
        max_payload_objects=int(profile["max_payload_objects"]),
        codec_profile_id=str(profile["codec_profile_id"]),
        shard_immutable=bool(profile.get("shard_immutable", True)),
        shard_owned_appends=bool(profile.get("shard_owned_appends", True)),
        target_chunk_bytes_by_access={
            str(access): int(value) for access, value in overrides.items()
        },
    )
    if restored.profile_id != candidate.expected_profile_id:
        raise ValueError(
            "Restored detection storage profile ID does not match the spec."
        )
    return restored


def _load_detection_integration_prefix(
    candidate: CandidateSpec,
    *,
    frame_stop: int,
) -> CanonicalDetectionBenchmarkInput:
    source = zarr.open_group(
        str(candidate.path),
        mode="r",
        use_consolidated=True,
    )
    logical_schema = source.attrs.get("logical_schema")
    if not isinstance(logical_schema, Mapping):
        raise ValueError("Canonical candidate logical schema is missing.")
    raw_dimensions = logical_schema.get("dimensions")
    if not isinstance(raw_dimensions, Mapping):
        raise ValueError("Canonical candidate dimensions are missing.")
    source_dimensions = CanonicalDetectionDimensions(
        n_frames=int(raw_dimensions["n_frames"]),
        n_instances=int(raw_dimensions["n_instances"]),
        source_width=int(raw_dimensions["source_width"]),
        source_height=int(raw_dimensions["source_height"]),
    )
    if not 0 < frame_stop <= source_dimensions.n_frames:
        raise ValueError("Detection integration prefix is outside the source domain.")
    offsets = np.asarray(
        source["instances/frame_row_offsets"][: frame_stop + 1],
        dtype=np.int64,
    )
    if offsets.shape != (frame_stop + 1,):
        raise ValueError("Detection prefix offsets do not have F+1 elements.")
    selected_rows = int(offsets[-1])
    arrays: dict[str, np.ndarray] = {}
    for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths:
        if path == "instances/frame_row_offsets":
            arrays[path] = offsets
        else:
            arrays[path] = np.asarray(source[path][:selected_rows])
    dimensions = CanonicalDetectionDimensions(
        n_frames=frame_stop,
        n_instances=selected_rows,
        source_width=source_dimensions.source_width,
        source_height=source_dimensions.source_height,
    )
    return CanonicalDetectionBenchmarkInput(
        dimensions=dimensions,
        arrays=arrays,
        source_identity={
            "canonical_full_duration_candidate": str(candidate.path),
            "canonical_full_duration_candidate_metadata_sha256": _sha256_file(
                candidate.path / "zarr.json"
            ),
            "source_profile_id": candidate.expected_profile_id,
            "source_frame_count": source_dimensions.n_frames,
            "selected_frame_range": [0, frame_stop],
            "selected_detection_rows": selected_rows,
            "selection": "exact_identity_preserving_prefix",
        },
    )


def _require_detection_prefix_equivalence(
    regular: CanonicalDetectionBenchmarkInput,
    hybrid: CanonicalDetectionBenchmarkInput,
) -> dict[str, Any]:
    if regular.dimensions != hybrid.dimensions:
        raise ValueError("Detection candidate prefix dimensions differ.")
    arrays: dict[str, Any] = {}
    for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths:
        regular_digest = sha256_array(regular.arrays[path])
        hybrid_digest = sha256_array(hybrid.arrays[path])
        if regular_digest != hybrid_digest:
            raise ValueError(f"Detection candidate prefixes differ at {path!r}.")
        arrays[path] = {
            "shape": list(regular.arrays[path].shape),
            "dtype": str(regular.arrays[path].dtype),
            "sha256": regular_digest,
        }
    return {
        "dimensions": regular.dimensions.as_manifest(),
        "arrays": arrays,
        "regular_hybrid_exact": True,
        "frame_row_offsets": {
            "shape": [regular.dimensions.n_frames + 1],
            "starts_at_zero": True,
            "monotonic": True,
            "ends_at_n_instances": True,
            "exactly_matches_frame_indices": True,
        },
    }


def _write_detection_integration_prefix(
    benchmark_input: CanonicalDetectionBenchmarkInput,
    *,
    candidate: CandidateSpec,
    archive: Path,
    run_name: str,
) -> dict[str, Any]:
    profile = _storage_profile_from_candidate(candidate)
    plans = plan_canonical_detection_storage(
        benchmark_input.dimensions,
        profile=profile,
    )
    destination = archive / "detect_runs" / run_name
    report = write_detection_benchmark_candidate(
        benchmark_input,
        destination=destination,
        plans=plans,
        benchmark_root=archive,
    )
    metadata_path = destination / "zarr.json"
    metadata = _read_json(metadata_path)
    metadata.pop("consolidated_metadata", None)
    metadata["attributes"] = _normalized_detection_run_attributes(metadata)
    _write_json_atomic(metadata_path, metadata)
    return {
        "source_candidate": str(candidate.path),
        "profile_id": profile.profile_id,
        "dimensions": benchmark_input.dimensions.as_manifest(),
        "storage_plan": plans.as_manifest(),
        "write_timing": dict(report["timing"]),
        "physical": dict(report["physical"]),
    }


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


def _validate_existing_consolidated_metadata(root: Path) -> dict[str, Any]:
    direct = _direct_metadata_map(root)
    count = _validate_direct_consolidated_map(direct, label=str(root))
    zarr.open_group(str(root), mode="r", use_consolidated=False)
    zarr.open_group(str(root), mode="r", use_consolidated=True)
    return {
        "direct_consolidated_node_count": count,
        "direct_open": True,
        "consolidated_open": True,
        "metadata_mutated": False,
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
            if (
                path != run_path
                and not path.startswith(f"{run_path}/instances/")
                and path != ""
            ):
                raise RuntimeError(
                    f"Unexpected nondetection metadata difference at {path}."
                )
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
        raise RuntimeError(
            "Paired normalized nondetection consolidated metadata differs."
        )
    return {
        "direct_path_count": len(regular_direct),
        "physical_difference_paths": physical_difference_paths,
        "nondetection_consolidated_metadata_exact": True,
    }


def _raise_nonfinite_json(value: str) -> None:
    raise ValueError(f"Non-finite JSON token is forbidden: {value}")


def _validate_strict_json_files(root: Path) -> dict[str, Any]:
    checked = 0
    for path in sorted(root.rglob("*.json")):
        json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_raise_nonfinite_json,
        )
        checked += 1
    return {"strict_json": True, "json_file_count": checked}


def _validate_strict_direct_metadata(root: Path) -> dict[str, Any]:
    """Validate Zarr declarations without walking payload chunk directories."""

    direct = _direct_metadata_map(root)
    for relative_path in direct:
        metadata_path = (
            root / relative_path / "zarr.json" if relative_path else root / "zarr.json"
        )
        json.loads(
            metadata_path.read_text(encoding="utf-8"),
            parse_constant=_raise_nonfinite_json,
        )
    return {
        "strict_json": True,
        "json_file_count": len(direct),
        "scope": "direct_zarr_metadata_only",
        "payload_directories_walked": False,
    }


def _selected_array_metadata_paths(
    root: Path,
    selected_paths: Sequence[str],
) -> tuple[tuple[str, Path, dict[str, Any]], ...]:
    selected_prefixes = tuple(f"{path}/" for path in selected_paths)
    reports: list[tuple[str, Path, dict[str, Any]]] = []
    for metadata_path in _metadata_file_paths(root, selected_paths):
        relative_metadata = metadata_path.relative_to(root).as_posix()
        if not relative_metadata.endswith("/zarr.json"):
            continue
        relative_path = relative_metadata.removesuffix("/zarr.json")
        if not any(
            relative_path == selected or relative_path.startswith(prefix)
            for selected, prefix in zip(selected_paths, selected_prefixes)
        ):
            continue
        metadata = _read_json(metadata_path)
        if metadata.get("node_type") == "array":
            reports.append((relative_path, metadata_path.parent, metadata))
    return tuple(sorted(reports, key=lambda item: item[0]))


def _sample_coordinates(shape: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    if any(length == 0 for length in shape):
        return ()
    if not shape:
        return ((),)
    coordinates = {
        tuple(0 for _ in shape),
        tuple(length // 2 for length in shape),
        tuple(length - 1 for length in shape),
    }
    return tuple(sorted(coordinates))


def _sample_selected_arrays(
    root: Path,
    selected_paths: Sequence[str],
) -> dict[str, Any]:
    """Hash deterministic boundary samples without enumerating every chunk."""

    arrays: list[dict[str, Any]] = []
    sample_count = 0
    for relative_path, array_path, metadata in _selected_array_metadata_paths(
        root,
        selected_paths,
    ):
        shape = tuple(int(value) for value in metadata.get("shape", []))
        coordinates = _sample_coordinates(shape)
        digest = _logical_digest_header(
            metadata=metadata,
            shape=shape,
            unit_shape=tuple(1 for _ in shape),
        )
        array = _open_direct_array(array_path, mode="r")
        string_payload = metadata.get("data_type") == "string"
        for coordinate in coordinates:
            selection = tuple(slice(index, index + 1) for index in coordinate)
            _update_logical_digest(
                digest,
                selection=selection,
                values=np.asarray(array[coordinate]),
                string_payload=string_payload,
            )
        arrays.append(
            {
                "path": relative_path,
                "shape": list(shape),
                "data_type": metadata.get("data_type"),
                "direct_metadata_sha256": _sha256_file(array_path / "zarr.json"),
                "sample_coordinates": [list(value) for value in coordinates],
                "sample_sha256": digest.hexdigest(),
            }
        )
        sample_count += len(coordinates)
    return {
        "schema_id": "palette.deterministic_array_sample_ledger",
        "schema_version": 1,
        "array_count": len(arrays),
        "sample_count": sample_count,
        "coordinate_policy": "origin_midpoint_endpoint_per_array",
        "payload_chunk_directories_enumerated": False,
        "arrays": arrays,
    }


def _require_matching_sample_ledgers(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if expected != actual:
        raise RuntimeError(f"Deterministic array sample ledger differs for {label}.")


def _validate_integration_logical_slices(
    root: Path,
    *,
    slice_manifest: Mapping[str, Any],
    digest_field: str,
    source_root: bool,
) -> dict[str, Any]:
    arrays = slice_manifest.get("arrays")
    if not isinstance(arrays, list):
        raise ValueError("Integration slice manifest arrays are missing.")
    checked = 0
    logical_payload_bytes = 0
    for raw_report in arrays:
        if not isinstance(raw_report, dict):
            raise ValueError("Integration array report must be mutable objects.")
        relative_path = str(raw_report["path"])
        metadata = _read_json(root / relative_path / "zarr.json")
        fixture_shape = tuple(int(value) for value in raw_report["fixture_shape"])
        unit_shape = tuple(int(value) for value in raw_report["payload_unit_shape"])
        if (
            not source_root
            and tuple(int(value) for value in metadata["shape"]) != fixture_shape
        ):
            raise ValueError(f"Fixture shape drifted at {relative_path!r}.")
        digest = _hash_logical_array_slice(
            _open_direct_array(root / relative_path, mode="r"),
            metadata=metadata,
            shape=fixture_shape,
            unit_shape=unit_shape,
        )
        if digest["sha256"] != raw_report["source_logical_sha256"]:
            label = "source" if source_root else digest_field
            raise RuntimeError(
                f"Integration logical slice differs from {label} at {relative_path!r}."
            )
        raw_report[digest_field] = digest["sha256"]
        raw_report[f"{digest_field}_exact"] = True
        checked += 1
        logical_payload_bytes += int(digest["logical_payload_bytes"])
    return {
        "array_count": checked,
        "logical_payload_bytes": logical_payload_bytes,
        "all_exact": True,
        "digest_schema_id": "palette.logical_array_slice_digest",
        "digest_schema_version": 1,
    }


def _validate_integration_row_relationships(
    root: Path,
    *,
    spec: FullAnalysisFixtureSpec,
    cardinalities: Mapping[str, Any],
) -> dict[str, Any]:
    window = spec.integration_window
    if window is None:  # pragma: no cover - caller guard
        raise ValueError("An integration window is required.")
    counts = np.asarray(
        _open_direct_array(root / window.frame_counts_path, mode="r")[:],
        dtype=np.int64,
    )
    frame_indices = np.asarray(
        _open_direct_array(root / window.frame_indices_path, mode="r")[:],
        dtype=np.int64,
    )
    selected_frames = int(cardinalities["selected_frame_count"])
    selected_rows = int(cardinalities["selected_observation_rows"])
    if counts.shape != (selected_frames,) or frame_indices.shape != (selected_rows,):
        raise ValueError("Integration frame/row reference shapes are invalid.")
    expected_counts = np.bincount(frame_indices, minlength=selected_frames)
    if not np.array_equal(counts, expected_counts):
        raise ValueError("Integration frame-count reference no longer matches rows.")

    observation_frame_paths: list[str] = []
    frame_count_paths: list[str] = []
    for relative_path, metadata in _direct_metadata_map(root).items():
        if metadata.get("node_type") != "array":
            continue
        if relative_path.startswith("detect_runs/"):
            continue
        shape = tuple(int(value) for value in metadata.get("shape", []))
        leaf = PurePosixPath(relative_path).name
        if leaf in {"frame_indices", "frame_index"} and shape == (selected_rows,):
            values = np.asarray(
                _open_direct_array(root / relative_path, mode="r")[:],
                dtype=np.int64,
            )
            if not np.array_equal(values, frame_indices):
                raise ValueError(
                    f"Observation frame identity differs at {relative_path!r}."
                )
            observation_frame_paths.append(relative_path)
        if leaf in {"frame_counts", "n_rois"} and shape == (selected_frames,):
            values = np.asarray(
                _open_direct_array(root / relative_path, mode="r")[:],
                dtype=np.int64,
            )
            if not np.array_equal(values, counts):
                raise ValueError(f"Frame counts differ at {relative_path!r}.")
            frame_count_paths.append(relative_path)

    csr_paths: list[str] = []
    for raw_report in cardinalities["csr_groups"]:
        group_path = str(raw_report["group_path"])
        pointers = np.asarray(
            _open_direct_array(root / group_path / "ptr", mode="r")[:],
            dtype=np.int64,
        )
        lengths = np.asarray(
            _open_direct_array(root / group_path / "len", mode="r")[:],
            dtype=np.int64,
        )
        points_metadata = _read_json(root / group_path / "points_xy" / "zarr.json")
        expected_points = int(raw_report["point_count"])
        if pointers.shape != (selected_rows,) or lengths.shape != (selected_rows,):
            raise ValueError(f"Sliced CSR row shape differs at {group_path!r}.")
        endpoint, empty_rows = _validate_csr_vectors(
            pointers,
            lengths,
            label=group_path,
        )
        if endpoint != expected_points or int(points_metadata["shape"][0]) != endpoint:
            raise ValueError(f"Sliced CSR endpoint differs at {group_path!r}.")
        if empty_rows != int(raw_report["empty_row_count"]):
            raise ValueError(f"Sliced CSR empty-row count differs at {group_path!r}.")
        csr_paths.append(group_path)

    return {
        "camera_frame_identity": "source_indices_preserved_no_rebase",
        "frame_range": [0, selected_frames],
        "observation_rows": selected_rows,
        "frame_count_reference": window.frame_counts_path,
        "frame_index_reference": window.frame_indices_path,
        "frame_counts_exact": True,
        # Directory iteration order is not a logical property of an archive and
        # may differ after the regular tree is cloned and its detection run is
        # rewritten with a different physical layout.  Keep relationship
        # evidence canonical so equivalent stores compare and hash identically.
        "observation_frame_identity_paths": sorted(observation_frame_paths),
        "frame_count_paths": sorted(frame_count_paths),
        "csr_group_paths": sorted(csr_paths),
        "all_relationships_valid": True,
    }


def _fixture_manifest(
    *,
    layout: str,
    archive: Path,
    plan: Mapping[str, Any],
    source_metadata_pre: TreeInventory,
    source_metadata_post: TreeInventory,
    source_samples_pre: Mapping[str, Any],
    source_samples_post: Mapping[str, Any],
    copied_samples: Mapping[str, Any],
    detection_validation: Mapping[str, Any],
    metadata_validation: Mapping[str, Any],
    strict_json_validation: Mapping[str, Any],
    row_relationship_validation: Mapping[str, Any] | None,
    assembly_report: Mapping[str, Any],
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
        "source_direct_metadata_inventory_before": source_metadata_pre.as_manifest(),
        "source_direct_metadata_inventory_after": source_metadata_post.as_manifest(),
        "source_sample_ledger_before": dict(source_samples_pre),
        "source_sample_ledger_after": dict(source_samples_post),
        "copied_sample_ledger": dict(copied_samples),
        "source_unchanged": _same_inventory_content(
            source_metadata_pre,
            source_metadata_post,
        )
        and source_samples_pre == source_samples_post,
        "nondetection_payload_validation": {
            "method": "copy_completion_plus_exact_metadata_and_deterministic_samples",
            "complete_payload_hashing": False,
            "rationale": (
                "benchmark fixture avoids repeated enumeration and hashing of "
                "high-fanout dense-mask chunks"
            ),
            "sample_ledger": dict(copied_samples),
        },
        "detection_run": plan["detection_run"],
        "detection_candidate": plan["candidates"][layout],
        "detection_validation": dict(detection_validation),
        "metadata_validation": dict(metadata_validation),
        "strict_json_validation": dict(strict_json_validation),
        "row_relationship_validation": (
            dict(row_relationship_validation)
            if row_relationship_validation is not None
            else None
        ),
        "assembly": dict(assembly_report),
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


def _publish_integration_fixture_pair(
    spec: FullAnalysisFixtureSpec,
    *,
    plan: Mapping[str, Any],
    pair_copy_mode: str,
) -> dict[str, Any]:
    """Publish a bounded prefix fixture without full-tree payload hashing."""

    window = spec.integration_window
    if window is None:  # pragma: no cover - caller guard
        raise ValueError("An integration window is required.")
    resolved_destination = Path(str(plan["destination"]))
    resolved_destination.parent.mkdir(parents=True, exist_ok=True)
    scratch_base = Path(str(plan["scratch_root"]))
    temporary = scratch_base / (
        f"palette_canonical_detection_integration_{spec.fixture_id}."
        f"incomplete.{uuid.uuid4().hex}"
    )
    temporary.mkdir()
    publication_temporary = resolved_destination.parent / (
        f".{resolved_destination.name}.incomplete.{uuid.uuid4().hex}"
    )
    selected_paths = tuple(item.path for item in spec.selected_products)
    created_at = _utc_now()

    try:
        source_metadata_pre = _inventory_files(
            spec.source_archive,
            _metadata_file_paths(spec.source_archive, selected_paths),
        )
        video_pre = spec.source_video.stat()
        cardinalities = _resolve_integration_cardinalities(spec)

        detection_inputs = {
            layout: _load_detection_integration_prefix(
                spec.candidates[layout],
                frame_stop=window.camera_frame_stop,
            )
            for layout in _LAYOUTS
        }
        detection_prefix_equivalence = _require_detection_prefix_equivalence(
            detection_inputs["regular"],
            detection_inputs["hybrid"],
        )

        regular = temporary / "regular.zarr"
        hybrid = temporary / "hybrid.zarr"
        slice_manifest = _assemble_integration_nondetection_base(
            spec,
            destination=regular,
            created_at_utc=created_at,
            cardinalities=cardinalities,
        )
        copy_report = _clone_pair_base(regular, hybrid, mode=pair_copy_mode)

        copied_arrays = slice_manifest["arrays"]
        local_logical_validations = {
            "regular": {
                "array_count": len(copied_arrays),
                "logical_payload_bytes": sum(
                    int(report["logical_payload_bytes"]) for report in copied_arrays
                ),
                "all_exact": True,
                "validation_phase": "source_read_write_and_immediate_readback",
                "digest_schema_id": "palette.logical_array_slice_digest",
                "digest_schema_version": 1,
            },
            "hybrid": _validate_integration_logical_slices(
                hybrid,
                slice_manifest=slice_manifest,
                digest_field="hybrid_logical_sha256",
                source_root=False,
            ),
            "source_after_copy": _validate_integration_logical_slices(
                spec.source_archive,
                slice_manifest=slice_manifest,
                digest_field="source_postcopy_logical_sha256",
                source_root=True,
            ),
        }

        detection_write_reports = {
            layout: _write_detection_integration_prefix(
                detection_inputs[layout],
                candidate=spec.candidates[layout],
                archive=(regular if layout == "regular" else hybrid),
                run_name=spec.detection_run_name,
            )
            for layout in _LAYOUTS
        }
        metadata_validations = {
            "regular": _consolidate_and_validate(regular),
            "hybrid": _consolidate_and_validate(hybrid),
        }
        strict_json_validations = {
            "regular": _validate_strict_json_files(regular),
            "hybrid": _validate_strict_json_files(hybrid),
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
        row_relationship_validations = {
            "regular": _validate_integration_row_relationships(
                regular,
                spec=spec,
                cardinalities=cardinalities,
            ),
            "hybrid": _validate_integration_row_relationships(
                hybrid,
                spec=spec,
                cardinalities=cardinalities,
            ),
        }
        if (
            row_relationship_validations["regular"]
            != row_relationship_validations["hybrid"]
        ):
            raise RuntimeError("Paired integration row relationships differ.")

        source_metadata_post = _inventory_files(
            spec.source_archive,
            _metadata_file_paths(spec.source_archive, selected_paths),
        )
        video_post = spec.source_video.stat()
        if not _same_inventory_content(source_metadata_pre, source_metadata_post):
            raise RuntimeError(
                "Maintained source metadata changed during fixture build."
            )
        if (
            video_pre.st_size != video_post.st_size
            or video_pre.st_mtime_ns != video_post.st_mtime_ns
        ):
            raise RuntimeError("Source video identity changed during fixture build.")
        pair_metadata = _validate_pair_metadata_difference(
            regular,
            hybrid,
            run_name=spec.detection_run_name,
        )

        manifests: dict[str, dict[str, Any]] = {}
        for layout in _LAYOUTS:
            manifests[layout] = {
                "schema_id": "palette.canonical_detection_integration_fixture",
                "schema_version": 1,
                "status": "published_immutable",
                "layout": layout,
                "archive": str(resolved_destination / f"{layout}.zarr"),
                "benchmark_only": True,
                "canonical": False,
                "registry_registered": False,
                "selector_eligible": False,
                "evidence_scope": plan["evidence_scope"],
                "source": plan["source"],
                "source_direct_metadata_inventory_before": (
                    source_metadata_pre.as_manifest()
                ),
                "source_direct_metadata_inventory_after": (
                    source_metadata_post.as_manifest()
                ),
                "source_unchanged": True,
                "logical_slice_manifest": slice_manifest,
                "logical_slice_validation": local_logical_validations[layout],
                "row_relationship_validation": row_relationship_validations[layout],
                "detection_run": plan["detection_run"],
                "detection_candidate": plan["candidates"][layout],
                "detection_prefix_write": detection_write_reports[layout],
                "detection_validation": detection_validations[layout],
                "metadata_validation": metadata_validations[layout],
                "strict_json_validation": strict_json_validations[layout],
                "pair_base_copy": copy_report,
                "crimson_contract": plan["crimson_contract"],
                "palette_code": plan["palette_code"],
                "video_copied": False,
                "immutability": {
                    "files_mode": "0444",
                    "directories_mode": "0555",
                    "jobs_must_open_read_only": True,
                },
            }
            _write_json_exclusive(
                temporary / f"{layout}_manifest.json",
                manifests[layout],
            )

        pair_manifest = {
            **plan,
            "status": "published_immutable",
            "payload_io_performed": True,
            "created_at_utc": created_at,
            "pair_copy_mode_resolved": copy_report["method"],
            "source_unchanged": True,
            "source_payload_validation": {
                "method": "exact_declared_logical_slices_only",
                **local_logical_validations["source_after_copy"],
            },
            "logical_slice_manifest": slice_manifest,
            "nondetection_pair_exact": True,
            "decoded_detection_pair_exact": True,
            "detection_source_prefix_equivalence": detection_prefix_equivalence,
            "metadata_difference_validation": pair_metadata,
            "row_relationship_validation": row_relationship_validations["regular"],
            "publication_receipt_relative_path": "publication_receipt.json",
            "manifests": {
                layout: {
                    "relative_path": f"{layout}_manifest.json",
                    "sha256": hashlib.sha256(
                        (
                            json.dumps(
                                manifests[layout],
                                allow_nan=False,
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n"
                        ).encode("utf-8")
                    ).hexdigest(),
                }
                for layout in _LAYOUTS
            },
            "summary": {
                "archives": 2,
                "selected_nondetection_products": len(selected_paths),
                "selected_camera_frames": window.camera_frame_stop,
                "selected_observation_rows": cardinalities["selected_observation_rows"],
                "registry_updates": 0,
                "production_selector_updates": 0,
                "training_artifacts": 0,
                "profile_promoted": False,
                "full_duration_gate_satisfied": False,
            },
        }
        _write_json_exclusive(temporary / "pair_manifest.json", pair_manifest)
        scratch_stats = storage_stats(temporary)
        _copy_tree(temporary, publication_temporary)
        shared_stats = storage_stats(publication_temporary)
        for field in (
            "file_count",
            "metadata_file_count",
            "payload_file_count",
            "apparent_bytes",
        ):
            if scratch_stats[field] != shared_stats[field]:
                raise RuntimeError(
                    f"Shared-storage copy stat mismatch for {field}: "
                    f"{scratch_stats[field]} != {shared_stats[field]}."
                )

        published_logical_validations: dict[str, Any] = {}
        for layout in _LAYOUTS:
            archive = publication_temporary / f"{layout}.zarr"
            zarr.open_group(str(archive), mode="r", use_consolidated=False)
            zarr.open_group(str(archive), mode="r", use_consolidated=True)
            validation_manifest = json.loads(
                json.dumps(slice_manifest, allow_nan=False)
            )
            published_logical_validations[layout] = (
                _validate_integration_logical_slices(
                    archive,
                    slice_manifest=validation_manifest,
                    digest_field="published_logical_sha256",
                    source_root=False,
                )
            )
            _validate_integration_row_relationships(
                archive,
                spec=spec,
                cardinalities=cardinalities,
            )
            _validate_detection_archive(
                archive,
                run_name=spec.detection_run_name,
            )
        published_pair_metadata = _validate_pair_metadata_difference(
            publication_temporary / "regular.zarr",
            publication_temporary / "hybrid.zarr",
            run_name=spec.detection_run_name,
        )
        published_strict_json = _validate_strict_json_files(publication_temporary)
        publication_receipt = {
            "schema_id": "palette.canonical_detection_integration_fixture_publication",
            "schema_version": 1,
            "status": "validated_before_atomic_install",
            "copy_method": "node_local_scratch_to_shared_copytree",
            "scratch_source": str(temporary),
            "destination": str(resolved_destination),
            "scratch_storage_stats": scratch_stats,
            "shared_storage_stats_before_receipt": shared_stats,
            "exact_relative_path_size_match": True,
            "payload_verification": "exact_logical_slice_hashes",
            "published_logical_validations": published_logical_validations,
            "published_metadata_difference_validation": published_pair_metadata,
            "direct_and_consolidated_open": True,
            "strict_json_validation": published_strict_json,
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
            "evidence_scope": "integration_fixture",
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
            "Paired integration fixture publication failed; incomplete evidence "
            f"remains at {temporary} and {publication_temporary}: {exc}"
        ) from exc


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
    if (
        spec.integration_window is not None
        and spec.integration_window.classification == "integration_fixture"
    ):
        return _publish_integration_fixture_pair(
            spec,
            plan=plan,
            pair_copy_mode=pair_copy_mode,
        )
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
        source_metadata_pre = _inventory_files(
            spec.source_archive,
            _metadata_file_paths(spec.source_archive, selected_paths),
        )
        source_samples_pre = _sample_selected_arrays(
            spec.source_archive,
            selected_paths,
        )
        video_pre = spec.source_video.stat()
        raw_cardinalities = plan.get("frame_relationship_plan")
        cardinalities = (
            raw_cardinalities if isinstance(raw_cardinalities, Mapping) else None
        )
        if spec.integration_window is not None and cardinalities is None:
            raise ValueError("The planned frame relationship evidence is missing.")

        regular = temporary / "regular.zarr"
        hybrid = temporary / "hybrid.zarr"
        assembly_report = _assemble_nondetection_base(
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
        strict_json_validations = {
            "regular": _validate_strict_direct_metadata(regular),
            "hybrid": _validate_strict_direct_metadata(hybrid),
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

        row_relationship_validations = (
            {
                layout: _validate_integration_row_relationships(
                    regular if layout == "regular" else hybrid,
                    spec=spec,
                    cardinalities=cardinalities,
                )
                for layout in _LAYOUTS
            }
            if cardinalities is not None
            else None
        )
        if (
            row_relationship_validations is not None
            and row_relationship_validations["regular"]
            != row_relationship_validations["hybrid"]
        ):
            raise RuntimeError("Paired full-duration row relationships differ.")

        copied_samples = {
            "regular": _sample_selected_arrays(regular, selected_paths),
            "hybrid": _sample_selected_arrays(hybrid, selected_paths),
        }
        for layout in _LAYOUTS:
            _require_matching_sample_ledgers(
                source_samples_pre,
                copied_samples[layout],
                label=f"scratch {layout}",
            )

        source_metadata_post = _inventory_files(
            spec.source_archive,
            _metadata_file_paths(spec.source_archive, selected_paths),
        )
        source_samples_post = _sample_selected_arrays(
            spec.source_archive,
            selected_paths,
        )
        video_post = spec.source_video.stat()
        if not _same_inventory_content(source_metadata_pre, source_metadata_post):
            raise RuntimeError(
                "Maintained source metadata changed while fixtures were built."
            )
        _require_matching_sample_ledgers(
            source_samples_pre,
            source_samples_post,
            label="source after copy",
        )
        if (
            video_pre.st_size != video_post.st_size
            or video_pre.st_mtime_ns != video_post.st_mtime_ns
        ):
            raise RuntimeError(
                "Source video identity changed while fixtures were built."
            )

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
                source_metadata_pre=source_metadata_pre,
                source_metadata_post=source_metadata_post,
                source_samples_pre=source_samples_pre,
                source_samples_post=source_samples_post,
                copied_samples=copied_samples[layout],
                detection_validation=detection_validations[layout],
                metadata_validation=metadata_validations[layout],
                strict_json_validation=strict_json_validations[layout],
                row_relationship_validation=(
                    row_relationship_validations[layout]
                    if row_relationship_validations is not None
                    else None
                ),
                assembly_report=assembly_report,
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
            "source_unchanged": True,
            "source_payload_validation": {
                "method": (
                    "copy_completion_plus_exact_direct_metadata_and_"
                    "deterministic_array_samples"
                ),
                "complete_payload_hashing": False,
                "sample_ledger": source_samples_post,
                "source_direct_metadata_inventory_before": (
                    source_metadata_pre.as_manifest()
                ),
                "source_direct_metadata_inventory_after": (
                    source_metadata_post.as_manifest()
                ),
            },
            "assembly": assembly_report,
            "nondetection_pair_exact": True,
            "decoded_detection_pair_exact": True,
            "metadata_difference_validation": pair_metadata,
            "row_relationship_validation": (
                row_relationship_validations["regular"]
                if row_relationship_validations is not None
                else None
            ),
            "publication_receipt_relative_path": "publication_receipt.json",
            "manifests": {
                layout: {
                    "relative_path": f"{layout}_manifest.json",
                    "sha256": hashlib.sha256(
                        (
                            json.dumps(
                                manifests[layout],
                                allow_nan=False,
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n"
                        ).encode("utf-8")
                    ).hexdigest(),
                }
                for layout in _LAYOUTS
            },
            "summary": {
                "archives": 2,
                "selected_nondetection_products": len(selected_paths),
                "camera_frames": int(spec.source_expectations["n_frames"]),
                "detection_instances": int(
                    detection_validations["regular"]["dimensions"]["n_instances"]
                ),
                "registry_updates": 0,
                "production_selector_updates": 0,
                "training_artifacts": 0,
                "profile_promoted": False,
                "full_duration_fixture_published": True,
                "full_duration_gate_satisfied": False,
            },
        }
        _write_json_exclusive(temporary / "pair_manifest.json", pair_manifest)
        _copy_tree(temporary, publication_temporary)
        published_metadata_validations: dict[str, Any] = {}
        published_strict_json_validations: dict[str, Any] = {}
        published_detection_validations: dict[str, Any] = {}
        published_sample_validations: dict[str, Any] = {}
        published_row_relationship_validations: dict[str, Any] = {}
        for layout in _LAYOUTS:
            archive = publication_temporary / f"{layout}.zarr"
            published_metadata_validations[layout] = (
                _validate_existing_consolidated_metadata(archive)
            )
            published_strict_json_validations[layout] = (
                _validate_strict_direct_metadata(archive)
            )
            published_detection_validations[layout] = _validate_detection_archive(
                archive,
                run_name=spec.detection_run_name,
            )
            if published_detection_validations[layout] != detection_validations[layout]:
                raise RuntimeError(f"Published detection payload differs for {layout}.")
            published_samples = _sample_selected_arrays(archive, selected_paths)
            _require_matching_sample_ledgers(
                source_samples_pre,
                published_samples,
                label=f"published {layout}",
            )
            published_sample_validations[layout] = published_samples
            if cardinalities is not None:
                published_row_relationship_validations[layout] = (
                    _validate_integration_row_relationships(
                        archive,
                        spec=spec,
                        cardinalities=cardinalities,
                    )
                )
                if (
                    published_row_relationship_validations[layout]
                    != row_relationship_validations[layout]
                ):
                    raise RuntimeError(
                        f"Published row relationships differ for {layout}."
                    )
        published_pair_metadata = _validate_pair_metadata_difference(
            publication_temporary / "regular.zarr",
            publication_temporary / "hybrid.zarr",
            run_name=spec.detection_run_name,
        )
        if published_pair_metadata != pair_metadata:
            raise RuntimeError("Published pair metadata evidence differs from scratch.")
        publication_receipt = {
            "schema_id": "palette.canonical_detection_full_analysis_fixture_publication",
            "schema_version": 1,
            "status": "validated_before_atomic_install",
            "copy_method": "node_local_scratch_to_shared_copytree",
            "scratch_source": str(temporary),
            "destination": str(resolved_destination),
            "payload_verification": (
                "copy_completion_exact_metadata_full_detection_hashes_"
                "and_deterministic_nondetection_samples"
            ),
            "complete_tree_payload_hashing": False,
            "published_metadata_validations": published_metadata_validations,
            "published_strict_json_validations": (published_strict_json_validations),
            "published_detection_validations": (published_detection_validations),
            "published_sample_validations": published_sample_validations,
            "published_row_relationship_validations": (
                published_row_relationship_validations
            ),
            "published_metadata_difference_validation": published_pair_metadata,
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
    "AdditionalPrefixAxis",
    "CandidateSpec",
    "FullAnalysisFixtureSpec",
    "IntegrationWindow",
    "SelectedProduct",
    "load_full_analysis_fixture_spec",
    "plan_full_analysis_fixture_pair",
    "publish_full_analysis_fixture_pair",
    "require_safe_full_analysis_destination",
    "require_safe_fixture_scratch_root",
]
