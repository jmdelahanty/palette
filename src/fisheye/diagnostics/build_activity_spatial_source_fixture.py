"""Build an isolated exact source archive for activity/spatial export benchmarks.

Historical compact-v8 swim-bout runs may predate the executable
``array_schema_manifest``.  This tool never stamps or otherwise mutates that
historical authority.  It copies the exact persisted run into an isolated
benchmark archive, attests the copied logical surface with the current exact
manifest, and proves every decoded array remains identical.  This is a
lossless storage/contract projection, not a scientific recomputation.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any
import uuid

import numpy as np
import zarr

from fisheye.analysis.swim_bout_schema import (
    MANIFEST_ATTRIBUTE,
    SWIM_BOUT_LAYOUT,
    SWIM_BOUT_RUN_SCHEMA_ID,
    SWIM_BOUT_RUN_SCHEMA_VERSION,
    build_swim_bout_columnar_field_dtypes,
    validate_swim_bout_array_manifest,
    write_swim_bout_array_manifest,
)
from fisheye.analytics_exports.activity_spatial_time_bins import (
    bind_activity_spatial_sources,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.run_provenance import git_identity
from fisheye.shared.zarr.benchmark_fixture import inventory_tree
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import require_runs_parent

FIXTURE_SCHEMA_ID = "palette.activity_spatial.source_fixture"
FIXTURE_SCHEMA_VERSION = 2
ARCHIVE_NAME = "analysis.zarr"
MANIFEST_NAME = "fixture_manifest.json"
_FIXED_UTF8_WIDTH_PROJECTION = {
    "indexes/candidates/candidate_name": 256,
    "indexes/candidates/parameters_json": 8192,
    "indexes/signal_variants/parameters_json": 8192,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_component(value: str, *, label: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"{label} must be one safe path component.")
    return text


def _require_source_archive(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_dir():
        raise FileNotFoundError(
            f"Source archive is not a regular directory: {resolved}"
        )
    metadata = resolved / "zarr.json"
    if metadata.is_symlink() or not metadata.is_file():
        raise ValueError("Source archive must be one direct Zarr-v3 group.")
    document = json.loads(metadata.read_text(encoding="utf-8"))
    if document.get("zarr_format") != 3 or document.get("node_type") != "group":
        raise ValueError("Source archive must be Zarr format 3.")
    return resolved


def _require_destination(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if ".palette_benchmarks" not in resolved.parts or "fixtures" not in resolved.parts:
        raise ValueError(
            "Activity/spatial source fixtures must be below a "
            ".palette_benchmarks/.../fixtures namespace."
        )
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError(f"Fixture destination already exists: {resolved}")
    return resolved


def _require_work_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_dir():
        raise FileNotFoundError(f"Work root is not a regular directory: {resolved}")
    if not (
        resolved.is_relative_to(Path("/tmp").resolve())
        or ".palette_scratch" in resolved.parts
    ):
        raise ValueError("Work root must be below /tmp or .palette_scratch.")
    return resolved


def _require_nonsymlink_tree(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be one nonsymlink directory.")
    for child in path.rglob("*"):
        if child.is_symlink():
            raise ValueError(f"{label} contains a symlink: {child}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_strings(child)
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for child in value:
            yield from _iter_strings(child)


def _iter_semantic_group_refs(value: object) -> Iterable[str]:
    """Expand named legacy collection dependencies that are not stored as refs."""

    if isinstance(value, Mapping):
        source_proxy_runs = value.get("source_proxy_crop_runs")
        if isinstance(source_proxy_runs, Sequence) and not isinstance(
            source_proxy_runs,
            (str, bytes, bytearray),
        ):
            for name in source_proxy_runs:
                if isinstance(name, str):
                    yield f"/crop_runs/{name}"
        source_refined_paths = value.get("source_refined_run_paths")
        if isinstance(source_refined_paths, Sequence) and not isinstance(
            source_refined_paths,
            (str, bytes, bytearray),
        ):
            for path in source_refined_paths:
                if isinstance(path, str):
                    yield f"/{path.strip('/')}"
        for child in value.values():
            yield from _iter_semantic_group_refs(child)
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for child in value:
            yield from _iter_semantic_group_refs(child)


def _referenced_dependency_nodes(
    source_archive: Path,
    *,
    track_manifest: Mapping[str, Any],
    track_run_path: str,
) -> tuple[tuple[str, str], ...]:
    """Resolve the transitive archive-internal Zarr-node reference closure."""

    run_prefix = track_run_path.strip("/")
    nodes: dict[str, str] = {}
    pending = [(value, "node") for value in _iter_strings(track_manifest)]
    pending.extend(
        (value, "whole_group") for value in _iter_semantic_group_refs(track_manifest)
    )
    while pending:
        value, requested_mode = pending.pop()
        if not value.startswith("/"):
            continue
        node_ref = value.split("@", 1)[0].strip("/")
        parts = tuple(node_ref.split("/"))
        if (
            not node_ref
            or node_ref == run_prefix
            or node_ref.startswith(f"{run_prefix}/")
            or any(part in {"", ".", ".."} for part in parts)
        ):
            continue
        candidate = source_archive.joinpath(*parts)
        metadata = candidate / "zarr.json"
        if not (
            candidate.is_dir()
            and not candidate.is_symlink()
            and metadata.is_file()
            and not metadata.is_symlink()
        ):
            continue
        existing_mode = nodes.get(node_ref)
        node_mode = (
            "whole_group"
            if requested_mode == "whole_group"
            else existing_mode or "node"
        )
        nodes[node_ref] = node_mode
        if existing_mode is not None:
            continue
        document = json.loads(metadata.read_text(encoding="utf-8"))
        pending.extend((child, "node") for child in _iter_strings(document))
        pending.extend(
            (child, "whole_group") for child in _iter_semantic_group_refs(document)
        )
    whole_groups = sorted(path for path, mode in nodes.items() if mode == "whole_group")
    filtered = {
        path: mode
        for path, mode in nodes.items()
        if not any(
            path != root and path.startswith(f"{root}/") for root in whole_groups
        )
    }
    return tuple(sorted(filtered.items()))


def _copy_ancestor_group_metadata(
    source_archive: Path,
    local_archive: Path,
    *,
    node_path: str,
) -> None:
    parts = tuple(node_path.split("/"))
    for length in range(1, len(parts)):
        relative = parts[:length]
        source_group = source_archive.joinpath(*relative)
        source_metadata = source_group / "zarr.json"
        if not source_metadata.is_file() or source_metadata.is_symlink():
            raise ValueError(
                "Referenced dependency has a missing or symlinked ancestor: "
                f"{'/'.join(relative)!r}."
            )
        document = json.loads(source_metadata.read_text(encoding="utf-8"))
        if document.get("node_type") != "group":
            raise ValueError(
                f"Referenced dependency ancestor is not a group: {'/'.join(relative)!r}."
            )
        target_group = local_archive.joinpath(*relative)
        target_group.mkdir(parents=True, exist_ok=True)
        target_metadata = target_group / "zarr.json"
        if not target_metadata.exists():
            shutil.copy2(source_metadata, target_metadata)


def _copy_track_dependency_projection(
    source_archive: Path,
    local_archive: Path,
    *,
    track_manifest: Mapping[str, Any],
    track_run_path: str,
) -> dict[str, Any]:
    node_specs = _referenced_dependency_nodes(
        source_archive,
        track_manifest=track_manifest,
        track_run_path=track_run_path,
    )
    receipts: dict[str, dict[str, Any]] = {}
    for node_path, copy_mode in node_specs:
        source_node = source_archive.joinpath(*node_path.split("/"))
        source_metadata = source_node / "zarr.json"
        if (
            source_node.is_symlink()
            or not source_node.is_dir()
            or source_metadata.is_symlink()
            or not source_metadata.is_file()
        ):
            raise ValueError(
                f"Track dependency {node_path!r} is not one direct Zarr node."
            )
        document = json.loads(source_metadata.read_text(encoding="utf-8"))
        node_type = document.get("node_type")
        if node_type not in {"group", "array"}:
            raise ValueError(f"Track dependency {node_path!r} has invalid node_type.")
        _copy_ancestor_group_metadata(
            source_archive,
            local_archive,
            node_path=node_path,
        )
        target_node = local_archive.joinpath(*node_path.split("/"))
        if node_type == "array":
            if copy_mode != "node":
                raise ValueError(
                    f"Track dependency {node_path!r} cannot use whole-group mode."
                )
            _require_nonsymlink_tree(
                source_node,
                label=f"Track dependency {node_path!r}",
            )
            if target_node.exists():
                raise RuntimeError(
                    f"Track dependency projection collides at {node_path!r}."
                )
            source_inventory = inventory_tree(source_node)
            shutil.copytree(source_node, target_node)
            copied_inventory = inventory_tree(target_node)
            if (
                source_inventory.file_count != copied_inventory.file_count
                or source_inventory.apparent_bytes != copied_inventory.apparent_bytes
                or source_inventory.tree_sha256 != copied_inventory.tree_sha256
            ):
                raise RuntimeError(f"Copied track dependency differs at {node_path!r}.")
            receipts[node_path] = {
                "node_type": "array",
                "source": source_inventory.as_manifest(),
                "copied": copied_inventory.as_manifest(),
                "exact_tree_equality": True,
            }
        elif copy_mode == "whole_group":
            if target_node.exists():
                raise RuntimeError(
                    f"Whole-group dependency projection collides at {node_path!r}."
                )
            source_inventory = inventory_tree(source_node)
            shutil.copytree(source_node, target_node)
            copied_inventory = inventory_tree(target_node)
            if (
                source_inventory.file_count != copied_inventory.file_count
                or source_inventory.apparent_bytes != copied_inventory.apparent_bytes
                or source_inventory.tree_sha256 != copied_inventory.tree_sha256
            ):
                raise RuntimeError(
                    f"Copied whole-group dependency differs at {node_path!r}."
                )
            receipts[node_path] = {
                "node_type": "group_tree",
                "source": source_inventory.as_manifest(),
                "copied": copied_inventory.as_manifest(),
                "exact_tree_equality": True,
            }
        else:
            target_node.mkdir(parents=True, exist_ok=True)
            target_metadata = target_node / "zarr.json"
            if not target_metadata.exists():
                shutil.copy2(source_metadata, target_metadata)
            source_sha = _sha256_file(source_metadata)
            copied_sha = _sha256_file(target_metadata)
            if source_sha != copied_sha:
                raise RuntimeError(
                    f"Copied track dependency metadata differs at {node_path!r}."
                )
            receipts[node_path] = {
                "node_type": "group",
                "source_zarr_json_sha256": source_sha,
                "copied_zarr_json_sha256": copied_sha,
                "exact_metadata_equality": True,
            }
    return {
        "projection_policy": "transitive_manifest_referenced_nontrack_zarr_nodes_v1",
        "node_count": len(receipts),
        "nodes": receipts,
    }


def _validate_dependency_sources_unchanged(
    source_archive: Path,
    projection: Mapping[str, Any],
) -> bool:
    nodes = projection.get("nodes")
    if not isinstance(nodes, Mapping):
        raise ValueError("Dependency projection receipt lacks its node map.")
    for node_path, raw in nodes.items():
        if not isinstance(node_path, str) or not isinstance(raw, Mapping):
            raise ValueError("Dependency projection receipt is malformed.")
        source_node = source_archive.joinpath(*node_path.split("/"))
        if raw.get("node_type") == "array":
            expected = raw.get("source")
            if not isinstance(expected, Mapping):
                return False
            observed = inventory_tree(source_node)
            if (
                observed.file_count != expected.get("file_count")
                or observed.apparent_bytes != expected.get("apparent_bytes")
                or observed.tree_sha256 != expected.get("tree_sha256")
            ):
                return False
        elif raw.get("node_type") == "group_tree":
            expected = raw.get("source")
            if not isinstance(expected, Mapping):
                return False
            observed = inventory_tree(source_node)
            if (
                observed.file_count != expected.get("file_count")
                or observed.apparent_bytes != expected.get("apparent_bytes")
                or observed.tree_sha256 != expected.get("tree_sha256")
            ):
                return False
        elif raw.get("node_type") == "group":
            if _sha256_file(source_node / "zarr.json") != raw.get(
                "source_zarr_json_sha256"
            ):
                return False
        else:
            raise ValueError("Dependency projection receipt has invalid node_type.")
    return True


def _iter_arrays(
    group: zarr.Group,
    prefix: str = "",
) -> Iterable[tuple[str, zarr.Array]]:
    for name in sorted(group.array_keys()):
        path = f"{prefix}/{name}" if prefix else str(name)
        yield path, group[name]
    for name in sorted(group.group_keys()):
        path = f"{prefix}/{name}" if prefix else str(name)
        yield from _iter_arrays(group[name], path)


def _array_receipts(group: zarr.Group) -> dict[str, dict[str, object]]:
    receipts: dict[str, dict[str, object]] = {}
    for path, array in _iter_arrays(group):
        values = np.asarray(array[:])
        receipts[path] = {
            "dtype": values.dtype.str,
            "shape": [int(value) for value in values.shape],
            "array_values_sha256": array_values_sha256(values),
        }
    return receipts


def _array_at_path(group: zarr.Group, path: str) -> zarr.Array:
    node: Any = group
    for component in path.split("/"):
        node = node[component]
    if not isinstance(node, zarr.Array):
        raise TypeError(f"Expected array at {path!r}.")
    return node


def _parent_and_leaf(group: zarr.Group, path: str) -> tuple[zarr.Group, str]:
    components = path.split("/")
    node = group
    for component in components[:-1]:
        child = node[component]
        if not isinstance(child, zarr.Group):
            raise TypeError(f"Expected group while resolving {path!r}.")
        node = child
    return node, components[-1]


def _decoded_fixed_utf8_rows(values: np.ndarray, *, path: str) -> tuple[str, ...]:
    if values.dtype != np.dtype("uint8") or values.ndim != 2:
        raise ValueError(f"{path}: fixed UTF-8 storage must be uint8[row, byte].")
    decoded: list[str] = []
    for row_index, row in enumerate(values):
        payload, separator, padding = bytes(row).partition(b"\0")
        if separator and any(padding):
            raise ValueError(f"{path}: row {row_index} has nonzero bytes after NUL.")
        try:
            decoded.append(payload.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{path}: row {row_index} is not canonical UTF-8."
            ) from exc
    return tuple(decoded)


def _widen_fixed_utf8_columns(run: zarr.Group) -> tuple[dict[str, object], ...]:
    """Apply the one closed lossless compact-v8 fixed-byte projection."""

    receipts: list[dict[str, object]] = []
    physically_widened: set[str] = set()
    expected_tables = build_swim_bout_columnar_field_dtypes()
    fixed_fields = {
        f"{table_path}/{field_name}": logical_dtype
        for table_path, fields in expected_tables.items()
        for field_name, logical_dtype in fields.items()
        if np.dtype(logical_dtype).kind == "S"
    }
    for path, exact_logical_dtype in sorted(fixed_fields.items()):
        exact_width = int(np.dtype(exact_logical_dtype).itemsize)
        source_array = _array_at_path(run, path)
        source = np.asarray(source_array[:])
        decoded_before = _decoded_fixed_utf8_rows(source, path=path)
        source_width = int(source.shape[1])
        table_path, field_name = path.rsplit("/", 1)
        table: Any = run
        for component in table_path.split("/"):
            table = table[component]
        field_dtypes = dict(table.attrs.get("field_dtypes", {}))
        if field_name not in field_dtypes:
            raise ValueError(f"{path}: source table omits its logical dtype.")
        source_logical_dtype = str(field_dtypes[field_name])
        parsed_source_logical_dtype = np.dtype(source_logical_dtype)
        if (
            parsed_source_logical_dtype.kind != "S"
            or parsed_source_logical_dtype.itemsize > exact_width
        ):
            raise ValueError(
                f"{path}: source logical fixed-byte dtype is incompatible."
            )
        logical_dtype_changed = source_logical_dtype != exact_logical_dtype
        physical_width_changed = logical_dtype_changed and source_width < exact_width
        if physical_width_changed:
            if _FIXED_UTF8_WIDTH_PROJECTION.get(path) != exact_width:
                raise ValueError(
                    f"{path}: physical widening is outside the closed compatibility "
                    "projection."
                )
            padded = np.zeros((int(source.shape[0]), exact_width), dtype=np.uint8)
            padded[:, :source_width] = source
            attributes = dict(source_array.attrs)
            parent, leaf = _parent_and_leaf(run, path)
            del parent[leaf]
            replacement = parent.create_array(
                leaf,
                data=padded,
                chunks=(
                    max(1, min(int(source.shape[0]) or 1, 65_536)),
                    exact_width,
                ),
                overwrite=False,
            )
            replacement.attrs.update(attributes)
            physically_widened.add(path)
        else:
            replacement = source_array
        observed = np.asarray(replacement[:])
        decoded_after = _decoded_fixed_utf8_rows(observed, path=path)
        if (
            decoded_after != decoded_before
            or not np.array_equal(observed[:, :source_width], source)
            or np.any(observed[:, source_width:])
        ):
            raise RuntimeError(f"{path}: fixed UTF-8 widening changed decoded values.")
        if logical_dtype_changed:
            field_dtypes[field_name] = exact_logical_dtype
            table.attrs["field_dtypes"] = field_dtypes
        if physical_width_changed or logical_dtype_changed:
            receipts.append(
                {
                    "path": path,
                    "source_physical_width": source_width,
                    "source_logical_dtype": source_logical_dtype,
                    "exact_physical_width": exact_width,
                    "exact_logical_dtype": exact_logical_dtype,
                    "physical_width_changed": physical_width_changed,
                    "logical_dtype_changed": logical_dtype_changed,
                    "row_count": int(source.shape[0]),
                    "decoded_utf8_values_preserved": True,
                    "source_bytes_preserved_as_prefix": True,
                    "added_bytes_are_zero_padding": True,
                }
            )
    if physically_widened != set(_FIXED_UTF8_WIDTH_PROJECTION):
        raise RuntimeError(
            "Historical source does not match the closed three-column physical "
            "widening projection."
        )
    return tuple(receipts)


def _validate_lossless_array_projection(
    source: zarr.Group,
    projected: zarr.Group,
    *,
    widened_columns: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    source_receipts = _array_receipts(source)
    projected_receipts = _array_receipts(projected)
    if set(source_receipts) != set(projected_receipts):
        raise RuntimeError("Lossless projection changed the array inventory.")
    widened_paths = {
        str(item["path"])
        for item in widened_columns
        if item.get("physical_width_changed") is True
    }
    if widened_paths != set(_FIXED_UTF8_WIDTH_PROJECTION):
        raise RuntimeError("Lossless projection did not apply the exact widening set.")
    for path in sorted(source_receipts):
        if path in widened_paths:
            before = np.asarray(_array_at_path(source, path)[:])
            after = np.asarray(_array_at_path(projected, path)[:])
            exact_width = _FIXED_UTF8_WIDTH_PROJECTION[path]
            if (
                after.dtype != before.dtype
                or before.ndim != 2
                or after.shape != (before.shape[0], exact_width)
                or not np.array_equal(after[:, : before.shape[1]], before)
                or np.any(after[:, before.shape[1] :])
                or _decoded_fixed_utf8_rows(before, path=path)
                != _decoded_fixed_utf8_rows(after, path=path)
            ):
                raise RuntimeError(f"{path}: widened projection is not lossless.")
        elif source_receipts[path] != projected_receipts[path]:
            raise RuntimeError(f"{path}: non-widened decoded array changed.")
    return {
        "equal": True,
        "equality_semantics": (
            "exact dtype/shape/value equality except the three closed fixed-UTF8 "
            "width widenings, which preserve every source byte and add zero padding"
        ),
        "array_count": len(projected_receipts),
        "exact_array_count": len(projected_receipts) - len(widened_paths),
        "widened_array_count": len(widened_paths),
        "source_arrays": source_receipts,
        "projected_arrays": projected_receipts,
        "widened_columns": [dict(item) for item in widened_columns],
    }


def _copy_group_attrs(source: zarr.Group, destination: zarr.Group) -> None:
    destination.attrs.put(dict(source.attrs))


def _seed_track_authority(
    source_archive: Path,
    local_archive: Path,
    *,
    track_run: str,
    final_archive_path: Path,
    code_identity: Mapping[str, Any],
) -> dict[str, object]:
    source_root = open_zarr_root(source_archive, mode="r")
    source_analysis = source_root["analysis"]
    source_tracks = source_analysis["track_kinematics_runs"]
    source_offline = source_tracks["offline"]
    if track_run not in source_offline:
        raise KeyError(f"Source track authority does not exist: {track_run!r}.")
    source_run = source_offline[track_run]
    track_manifest = source_run.attrs.get("track_motion_publication_manifest")
    if not isinstance(track_manifest, Mapping):
        raise ValueError("Source track authority lacks its exact publication manifest.")
    relative_track_path = f"analysis/track_kinematics_runs/offline/{track_run}"
    source_run_path = (
        source_archive / "analysis" / "track_kinematics_runs" / "offline" / track_run
    )
    _require_nonsymlink_tree(source_run_path, label="Source track authority")
    source_inventory = inventory_tree(source_run_path)

    local_root = open_zarr_root(local_archive, mode="w")
    root_attrs = dict(source_root.attrs)
    root_attrs.update(
        {
            "benchmark_only": True,
            "canonical": False,
            "registry_registered": False,
            "selector_eligible": False,
            "fixture_schema_id": FIXTURE_SCHEMA_ID,
            "fixture_schema_version": FIXTURE_SCHEMA_VERSION,
            "fixture_final_archive_path": str(final_archive_path),
            "fixture_source_archive": str(source_archive),
            "fixture_palette_commit": code_identity["git_sha"],
            "fixture_palette_git_dirty": code_identity["git_dirty"],
        }
    )
    local_root.attrs.put(root_attrs)
    local_analysis = local_root.create_group("analysis")
    _copy_group_attrs(source_analysis, local_analysis)
    local_tracks = require_runs_parent(local_analysis, "track_kinematics_runs")
    _copy_group_attrs(source_tracks, local_tracks)
    require_runs_parent(local_analysis, "track_kinematics_runs")
    local_offline = local_tracks.create_group("offline")
    _copy_group_attrs(source_offline, local_offline)
    local_offline.attrs["latest"] = track_run
    local_offline.attrs["latest_complete"] = track_run

    target_run_path = (
        local_archive / "analysis" / "track_kinematics_runs" / "offline" / track_run
    )
    shutil.copytree(source_run_path, target_run_path)
    copied_inventory = inventory_tree(target_run_path)
    if (
        source_inventory.file_count != copied_inventory.file_count
        or source_inventory.apparent_bytes != copied_inventory.apparent_bytes
        or source_inventory.tree_sha256 != copied_inventory.tree_sha256
    ):
        raise RuntimeError("Copied track authority differs from its source tree.")
    dependencies = _copy_track_dependency_projection(
        source_archive,
        local_archive,
        track_manifest=track_manifest,
        track_run_path=relative_track_path,
    )
    return {
        "run_name": track_run,
        "source_path": str(source_run_path),
        "source_inventory": source_inventory.as_manifest(),
        "copied_inventory": copied_inventory.as_manifest(),
        "exact_tree_equality": True,
        "dependency_projection": dependencies,
    }


def _seed_lossless_swim_bout_projection(
    source_archive: Path,
    local_archive: Path,
    *,
    historical_run_name: str,
    projected_run_name: str,
) -> dict[str, object]:
    """Copy and attest one historical run only inside the isolated archive."""

    source_root = open_zarr_root(source_archive, mode="r")
    source_parent = source_root["analysis/swim_bout_runs"]
    historical = source_parent[historical_run_name]
    attrs = dict(historical.attrs)
    if (
        attrs.get("schema_id") != SWIM_BOUT_RUN_SCHEMA_ID
        or attrs.get("schema_version") != SWIM_BOUT_RUN_SCHEMA_VERSION
        or attrs.get("layout") != SWIM_BOUT_LAYOUT
    ):
        raise ValueError(
            "Historical swim-bout source is not the maintained compact-v8 logical "
            "surface; older schemas require a separate compatibility migration."
        )
    if attrs.get(MANIFEST_ATTRIBUTE) is not None:
        raise ValueError(
            "Historical source already has an exact manifest; use it directly "
            "instead of creating a compatibility projection."
        )
    if attrs.get("palette_run_completion_status") != "complete":
        raise ValueError("Historical swim-bout source is not explicitly complete.")
    if attrs.get("stage_selector_eligible") is not True:
        raise ValueError("Historical swim-bout source is not selector eligible.")

    source_run_path = (
        source_archive / "analysis" / "swim_bout_runs" / historical_run_name
    )
    _require_nonsymlink_tree(source_run_path, label="Historical swim-bout source")
    source_inventory = inventory_tree(source_run_path)

    root = open_zarr_root(local_archive, mode="a")
    analysis = root["analysis"]
    parent = require_runs_parent(analysis, "swim_bout_runs")
    _copy_group_attrs(source_parent, parent)
    require_runs_parent(analysis, "swim_bout_runs")
    parent.attrs["latest"] = projected_run_name
    parent.attrs["latest_complete"] = projected_run_name
    target_run_path = local_archive / "analysis" / "swim_bout_runs" / projected_run_name
    shutil.copytree(source_run_path, target_run_path)
    copied_before_attestation = inventory_tree(target_run_path)
    if (
        source_inventory.file_count != copied_before_attestation.file_count
        or source_inventory.apparent_bytes != copied_before_attestation.apparent_bytes
        or source_inventory.tree_sha256 != copied_before_attestation.tree_sha256
    ):
        raise RuntimeError(
            "Copied historical swim-bout tree differs before attestation."
        )

    root = open_zarr_root(local_archive, mode="a")
    parent = root["analysis/swim_bout_runs"]
    run = parent[projected_run_name]
    run.attrs["palette_run_name"] = projected_run_name
    run.attrs["stage_selector_eligible"] = True
    projection = {
        "schema_id": "palette.swim_bout.lossless_contract_projection",
        "schema_version": 1,
        "source_archive": str(source_archive),
        "source_run": historical_run_name,
        "source_run_path": f"analysis/swim_bout_runs/{historical_run_name}",
        "source_tree_sha256": source_inventory.tree_sha256,
        "scientific_recomputation_performed": False,
        "decoded_values_copied_losslessly": True,
        "scope": "isolated_benchmark_fixture_only",
    }
    widened_columns = _widen_fixed_utf8_columns(run)
    projection["fixed_utf8_policy"] = "lossless_fixed_utf8_width_widening_v1"
    projection["widened_columns"] = [dict(item) for item in widened_columns]
    projection["payload_digest"] = canonical_json_sha256(projection)
    run.attrs["benchmark_contract_projection"] = projection
    write_swim_bout_array_manifest(run, byte_planner_adopted=False)
    errors = validate_swim_bout_array_manifest(run, byte_planner_adopted=False)
    if errors:
        raise RuntimeError(
            "Lossless swim-bout projection manifest is invalid: " + "; ".join(errors)
        )
    if not isinstance(run.attrs.get(MANIFEST_ATTRIBUTE), Mapping):
        raise RuntimeError("Lossless swim-bout projection lacks its exact manifest.")
    equality = _validate_lossless_array_projection(
        historical,
        run,
        widened_columns=widened_columns,
    )
    return {
        "run_name": projected_run_name,
        "source_run_name": historical_run_name,
        "projection_policy": "lossless_decoded_contract_attestation_v1",
        "scientific_recomputation_performed": False,
        "byte_planner_adopted": False,
        "array_manifest_sha256": canonical_json_sha256(run.attrs[MANIFEST_ATTRIBUTE]),
        "source_inventory": source_inventory.as_manifest(),
        "copied_before_attestation_inventory": copied_before_attestation.as_manifest(),
        "source_tree_copy_exact_before_attestation": True,
        "source_completion_status": attrs["palette_run_completion_status"],
        "source_selector_eligible": attrs["stage_selector_eligible"],
        "selector_scope": "benchmark_archive_only",
        "stage_selector_eligible": True,
        "logical_array_equality": equality,
    }


def _envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    return {
        "schema_id": FIXTURE_SCHEMA_ID,
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "payload": body,
        "payload_digest": canonical_json_sha256(body),
    }


def build_activity_spatial_source_fixture(
    *,
    source_zarr: Path,
    source_track_run: str,
    historical_swim_bout_run: str,
    exact_swim_bout_run: str,
    destination: Path,
    work_root: Path,
) -> dict[str, Any]:
    """Publish one immutable benchmark-only exact source archive."""

    source = _require_source_archive(source_zarr)
    code_identity = git_identity()
    if (
        not isinstance(code_identity.get("git_sha"), str)
        or len(code_identity["git_sha"]) != 40
        or type(code_identity.get("git_dirty")) is not bool
    ):
        raise RuntimeError("Fixture builder requires an exact Palette git identity.")
    evidence_eligible = code_identity["git_dirty"] is False
    track_run = _require_component(source_track_run, label="Source track run")
    historical_name = _require_component(
        historical_swim_bout_run,
        label="Historical swim-bout run",
    )
    exact_name = _require_component(exact_swim_bout_run, label="Exact swim-bout run")
    if exact_name == historical_name:
        raise ValueError("Exact and historical swim-bout run names must differ.")
    output = _require_destination(destination)
    scratch_parent = _require_work_root(work_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    hidden_output = output.parent / f".{output.name}.incomplete.{uuid.uuid4().hex}"
    scratch = scratch_parent / f"activity-spatial-fixture-{uuid.uuid4().hex}"
    local_archive = scratch / ARCHIVE_NAME
    final_archive = output / ARCHIVE_NAME

    source_root = open_zarr_root(source, mode="r")
    historical = source_root["analysis/swim_bout_runs"][historical_name]
    if historical.attrs.get(MANIFEST_ATTRIBUTE) is not None:
        raise ValueError(
            "Historical source already has an exact manifest; use it directly instead "
            "of creating a lossless compatibility-projection fixture."
        )
    if historical.attrs.get("source_track_kinematics_run") != track_run:
        raise ValueError("Historical swim-bout source binds a different track run.")
    historical_path = source / "analysis" / "swim_bout_runs" / historical_name
    _require_nonsymlink_tree(historical_path, label="Historical swim-bout source")
    source_track_path = (
        source / "analysis" / "track_kinematics_runs" / "offline" / track_run
    )
    source_before = {
        "track": inventory_tree(source_track_path),
        "historical_swim_bout": inventory_tree(historical_path),
    }
    historical_arrays = _array_receipts(historical)

    scratch.mkdir(parents=False)
    started = perf_counter()
    try:
        seed_started = perf_counter()
        track_receipt = _seed_track_authority(
            source,
            local_archive,
            track_run=track_run,
            final_archive_path=final_archive,
            code_identity=code_identity,
        )
        seed_seconds = perf_counter() - seed_started

        projection_started = perf_counter()
        activation = _seed_lossless_swim_bout_projection(
            source,
            local_archive,
            historical_run_name=historical_name,
            projected_run_name=exact_name,
        )
        projection_seconds = perf_counter() - projection_started
        consolidate_metadata_capture_expected_warnings(local_archive)
        local_root = open_zarr_root(local_archive, mode="r")
        exact = local_root["analysis/swim_bout_runs"][exact_name]
        exact_arrays = _array_receipts(exact)
        logical_equality = activation["logical_array_equality"]
        if (
            not isinstance(logical_equality, Mapping)
            or logical_equality.get("equal") is not True
            or logical_equality.get("projected_arrays") != exact_arrays
            or logical_equality.get("source_arrays") != historical_arrays
        ):
            raise RuntimeError("Lossless swim-bout projection receipt is inconsistent.")
        metadata = validate_direct_consolidated_subtree(
            local_archive,
            subtree_path=f"analysis/swim_bout_runs/{exact_name}",
        )
        recording_id = str(local_root.attrs.get("recording_id") or source.name)
        binding = bind_activity_spatial_sources(
            local_root,
            zarr_path=final_archive,
            recording_id=recording_id,
            track_kinematics_run=track_run,
            track_scope="offline",
            swim_bout_runs_by_track={int(historical.attrs["track_id"]): exact_name},
        )
        local_inventory = inventory_tree(local_archive)

        copy_started = perf_counter()
        hidden_output.mkdir()
        hidden_archive = hidden_output / ARCHIVE_NAME
        shutil.copytree(local_archive, hidden_archive)
        copied_inventory = inventory_tree(hidden_archive)
        if (
            copied_inventory.file_count != local_inventory.file_count
            or copied_inventory.apparent_bytes != local_inventory.apparent_bytes
            or copied_inventory.tree_sha256 != local_inventory.tree_sha256
        ):
            raise RuntimeError("Published fixture archive differs from local source.")
        copy_seconds = perf_counter() - copy_started

        source_after = {
            "track": inventory_tree(source_track_path),
            "historical_swim_bout": inventory_tree(historical_path),
        }
        source_unchanged = all(
            source_before[name].tree_sha256 == source_after[name].tree_sha256
            and source_before[name].file_count == source_after[name].file_count
            and source_before[name].apparent_bytes == source_after[name].apparent_bytes
            for name in source_before
        )
        if not source_unchanged:
            raise RuntimeError("A production source tree changed during fixture build.")
        dependency_sources_unchanged = _validate_dependency_sources_unchanged(
            source,
            track_receipt["dependency_projection"],
        )
        if not dependency_sources_unchanged:
            raise RuntimeError(
                "A projected track dependency changed during fixture build."
            )
        payload = {
            "created_at_utc": _utc_now(),
            "status": "published_immutable_benchmark_source",
            "benchmark_only": True,
            "canonical": False,
            "registry_registered": False,
            "selector_eligible": False,
            "code_identity": code_identity,
            "evidence_eligible": evidence_eligible,
            "source_archive": str(source),
            "destination": str(output),
            "archive_path": str(final_archive),
            "source_track": track_receipt,
            "historical_swim_bout_run": historical_name,
            "exact_swim_bout": activation,
            "projection_contract": {
                "operation": "lossless_decoded_contract_attestation",
                "scientific_recomputation_performed": False,
                "source_authority_modified": False,
                "isolated_copy_manifest_attested": True,
                "byte_planner_adopted": False,
                "physical_profile_evidence_scope": (
                    "separate_exact_tabular_candidate_benchmark"
                ),
            },
            "logical_array_equality": logical_equality,
            "direct_consolidated_metadata_equivalence": metadata.to_json(),
            "activity_spatial_source_binding_digest": binding.binding["payload_sha256"],
            "archive_inventory": copied_inventory.as_manifest(),
            "source_nonmutation": {
                "unchanged": True,
                "dependency_projection_unchanged": True,
                "before": {
                    name: receipt.as_manifest()
                    for name, receipt in source_before.items()
                },
                "after": {
                    name: receipt.as_manifest()
                    for name, receipt in source_after.items()
                },
            },
            "timing_seconds": {
                "seed_track_copy_and_validation": seed_seconds,
                "lossless_swim_bout_projection": projection_seconds,
                "scratch_to_hidden_publication": copy_seconds,
                "total_before_visibility": perf_counter() - started,
            },
            "production_state_changes": [],
        }
        document = _envelope(payload)
        write_json_atomic(hidden_output / MANIFEST_NAME, document)
        hidden_output.rename(output)

        final_root = open_zarr_root(final_archive, mode="r")
        final_binding = bind_activity_spatial_sources(
            final_root,
            zarr_path=final_archive,
            recording_id=recording_id,
            track_kinematics_run=track_run,
            track_scope="offline",
            swim_bout_runs_by_track={int(historical.attrs["track_id"]): exact_name},
        )
        if (
            final_binding.binding["payload_sha256"]
            != payload["activity_spatial_source_binding_digest"]
        ):
            raise RuntimeError("Published fixture source binding changed after rename.")
        return document
    except BaseException:
        if hidden_output.exists():
            shutil.rmtree(hidden_output)
        if output.exists():
            shutil.rmtree(output)
        raise
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zarr", type=Path, required=True)
    parser.add_argument("--source-track-run", required=True)
    parser.add_argument("--historical-swim-bout-run", required=True)
    parser.add_argument("--exact-swim-bout-run", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_activity_spatial_source_fixture(
        source_zarr=args.source_zarr,
        source_track_run=args.source_track_run,
        historical_swim_bout_run=args.historical_swim_bout_run,
        exact_swim_bout_run=args.exact_swim_bout_run,
        destination=args.destination,
        work_root=args.work_root,
    )
    if args.report is not None:
        write_json_atomic(args.report.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_NAME",
    "FIXTURE_SCHEMA_ID",
    "FIXTURE_SCHEMA_VERSION",
    "MANIFEST_NAME",
    "build_activity_spatial_source_fixture",
]
