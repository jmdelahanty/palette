"""Canonical mask-local QC after a browser checkpoint Apply receipt."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from fisheye.refinement import finalize_subject_masks as finalizer
from fisheye.shared.detect_reason_codec import read_reason_labels
from fisheye.shared.mask_geometry import MASK_ELLIPSE_METHOD
from fisheye.shared.mask_store import open_mask_store
from fisheye.shared.refined_subject_component_contours import (
    COMPONENT_CONTOUR_SCHEMA_ID,
    DEFAULT_BOUNDARY_POLICY,
    DEFAULT_CONTOUR_COORDINATE_SPACE,
    DEFAULT_CONTOUR_METHOD,
    DEFAULT_CONTOUR_METHOD_VERSION,
    extract_largest_external_contour,
    read_component_contours,
)
from fisheye.shared.refined_subject_mask_mutation import resolve_mutable_refined_subject_mask_run
from fisheye.shared import refined_subject_eye_geometry as eye_geometry
from fisheye.shared.zarr.local_group_staging import stage_group, sync_group_back


QC_POLICY_ID = "palette.browser_subject_mask_apply_full_qc_v1"
QC_POLICY_VERSION = 1
QC_ROW_CHUNK = 32
CHUNK_DIGESTS_KEY = "row_chunk_mask_sha256"
DERIVED_DIGESTS_KEY = "row_chunk_derived_sha256"
LOCAL_STAGING_ENV = "PALETTE_MASK_QC_LOCAL_STAGING"
STAGING_DIR_ENV = "PALETTE_MASK_QC_STAGING_DIR"
_BODY_CONTOUR_COMPONENTS = frozenset({"subject_body", "swim_bladder"})
_EYE_COMPONENTS = frozenset(eye_geometry.EYE_COMPONENTS)
_CONTOUR_COMPONENTS = _BODY_CONTOUR_COMPONENTS | _EYE_COMPONENTS


def _require_attr_compatible(group: zarr.Group, key: str, expected: object, description: str) -> None:
    actual = group.attrs.get(key)
    if actual is not None and actual != expected:
        raise RuntimeError(f"Browser Apply QC cannot change {description} {key}: {actual!r}.")


def _require_compatible_contract(run: zarr.Group) -> tuple[str, ...]:
    prior_policy = run.attrs.get("browser_apply_qc_policy")
    if prior_policy is not None and (
        not isinstance(prior_policy, dict)
        or prior_policy.get("id") != QC_POLICY_ID
        or type(prior_policy.get("version")) is not int
        or prior_policy.get("version") != QC_POLICY_VERSION
    ):
        raise RuntimeError("Browser Apply QC cannot replace a different declared browser QC policy.")
    metric_level = run.attrs.get("component_metric_level")
    if metric_level not in (None, finalizer.EDITABLE_REVIEW_METRIC_LEVEL):
        raise RuntimeError(
            f"Browser Apply QC requires full component metrics; this run declares {metric_level!r}."
        )
    labels = run.attrs.get("mask_labels")
    if not isinstance(labels, (list, tuple)) or not labels:
        raise RuntimeError("Browser Apply QC requires declared mask_labels.")
    names = tuple(str(value) for value in labels)
    has_both_eyes = _EYE_COMPONENTS.issubset(names)
    components = run.get("components")
    schema_attrs = (
        (run, "component_metrics_schema_id", finalizer._COMPONENT_METRICS_SCHEMA_ID),
        (run, "metric_qc_schema_id", finalizer._COMPONENT_METRIC_QC_SCHEMA_ID),
    )
    metrics_group = run.get("metrics")
    if isinstance(metrics_group, zarr.Group):
        schema_attrs += (
            (metrics_group, "schema_id", finalizer._COMPONENT_METRICS_SCHEMA_ID),
            (metrics_group, "qc_schema_id", finalizer._COMPONENT_METRIC_QC_SCHEMA_ID),
        )
    for group, key, expected in schema_attrs:
        actual = group.attrs.get(key)
        if actual is not None and actual != expected:
            raise RuntimeError(f"Browser Apply QC cannot change declared {key}: {actual!r}.")
    if isinstance(components, zarr.Group):
        for name in names:
            component = components.get(name)
            if not isinstance(component, zarr.Group):
                continue
            for key, expected in (
                ("component_metric_level", finalizer.EDITABLE_REVIEW_METRIC_LEVEL),
                ("component_metrics_schema_id", finalizer._COMPONENT_METRICS_SCHEMA_ID),
                ("metric_qc_schema_id", finalizer._COMPONENT_METRIC_QC_SCHEMA_ID),
            ):
                actual = component.attrs.get(key)
                if actual is not None and actual != expected:
                    raise RuntimeError(f"Browser Apply QC cannot change {name} {key}: {actual!r}.")
            component_metrics = component.get("metrics")
            if isinstance(component_metrics, zarr.Group):
                for key, expected in (
                    ("schema_id", finalizer._COMPONENT_METRICS_SCHEMA_ID),
                    ("qc_schema_id", finalizer._COMPONENT_METRIC_QC_SCHEMA_ID),
                    ("metric_level", finalizer.EDITABLE_REVIEW_METRIC_LEVEL),
                    ("qc_policy", finalizer._component_metric_qc_policy_payload(name)),
                ):
                    actual = component_metrics.attrs.get(key)
                    if actual is not None and actual != expected:
                        raise RuntimeError(
                            f"Browser Apply QC cannot change {name} metrics {key}: {actual!r}."
                        )
            if (name not in _CONTOUR_COMPONENTS or (name in _EYE_COMPONENTS and not has_both_eyes)) and isinstance(component.get("contours"), zarr.Group):
                raise RuntimeError(f"Browser Apply QC cannot refresh existing {name} contours.")
            if name in _EYE_COMPONENTS:
                geometry = component.get("geometry")
                if isinstance(geometry, zarr.Group):
                    if not has_both_eyes:
                        raise RuntimeError("Browser Apply QC cannot refresh incomplete eye-pair geometry.")
                    for key, expected in (
                        ("geometry_schema_id", eye_geometry.EYE_GEOMETRY_SCHEMA_ID),
                        ("geometry_method", MASK_ELLIPSE_METHOD),
                        ("source_mask_component", name),
                    ):
                        _require_attr_compatible(geometry, key, expected, f"{name} geometry")
    eye_pair = run.get("relations/eye_pair/metrics")
    if isinstance(eye_pair, zarr.Group):
        if not has_both_eyes:
            raise RuntimeError("Browser Apply QC cannot refresh incomplete eye-pair relation.")
        for key, expected in (
            ("relation_schema_id", eye_geometry.EYE_PAIR_RELATION_SCHEMA_ID),
            ("relation_components", list(eye_geometry.EYE_COMPONENTS)),
            ("relation_method", "ellipse_centroid_distance"),
        ):
            _require_attr_compatible(eye_pair, key, expected, "eye-pair relation")
    if has_both_eyes:
        _require_attr_compatible(run, "eye_geometry_schema_id", eye_geometry.EYE_GEOMETRY_SCHEMA_ID, "run eye geometry")
    declared_contours = run.attrs.get("component_contours_components")
    if declared_contours is not None and (
        not isinstance(declared_contours, (list, tuple))
        or any(str(name) not in _CONTOUR_COMPONENTS for name in declared_contours)
    ):
        raise RuntimeError("Browser Apply QC cannot refresh declared noncanonical component contours.")
    for name in _CONTOUR_COMPONENTS.intersection(names):
        group = components.get(name) if isinstance(components, zarr.Group) else None
        contours = group.get("contours") if isinstance(group, zarr.Group) else None
        if not isinstance(contours, zarr.Group):
            continue
        expected = {
            "schema_id": COMPONENT_CONTOUR_SCHEMA_ID,
            "contour_schema_id": COMPONENT_CONTOUR_SCHEMA_ID,
            "method": DEFAULT_CONTOUR_METHOD,
            "method_version": DEFAULT_CONTOUR_METHOD_VERSION,
            "boundary_policy": DEFAULT_BOUNDARY_POLICY,
            "coordinate_space": DEFAULT_CONTOUR_COORDINATE_SPACE,
            "point_order": "xy",
            "min_points": 1 if name in _EYE_COMPONENTS else 2,
        }
        for key, value in expected.items():
            actual = contours.attrs.get(key)
            if actual is not None and (type(actual) is not type(value) or actual != value):
                raise RuntimeError(
                    f"Browser Apply QC cannot change {name} contour {key}: {actual!r}."
                )
    return names


def _validate_metric_and_contour_rows(
    run: zarr.Group,
    names: tuple[str, ...],
    original_reasons: dict[str, np.ndarray],
    chunk_starts: frozenset[int] | None = None,
) -> None:
    """Recompute and compare every QC chunk, or only ``chunk_starts``."""
    masks = run["masks_roi"]
    metrics = run["metrics"]
    row_count = int(masks.shape[0])
    refreshed_reasons = {}
    for name in names:
        labels = read_reason_labels(run["components"][name])
        if labels is None or len(labels) != row_count:
            raise RuntimeError(f"Browser Apply QC lost {name} reason rows.")
        refreshed_reasons[name] = labels
    for start in range(0, row_count, QC_ROW_CHUNK):
        if chunk_starts is not None and start not in chunk_starts:
            continue
        stop = min(start + QC_ROW_CHUNK, row_count)
        mask_chunk = np.asarray(masks[start:stop], dtype=np.uint8)
        for comp_idx, name in enumerate(names):
            computed = finalizer._compute_mask_local_metric_payload(
                component_name=name, masks=mask_chunk[:, comp_idx], metric_level=finalizer.EDITABLE_REVIEW_METRIC_LEVEL,
            )
            component = run["components"][name]
            for key, expected in computed.spatial_metrics.items():
                actual = np.asarray(metrics[key][start:stop, comp_idx])
                if not np.array_equal(actual, expected, equal_nan=True):
                    raise RuntimeError(f"Browser Apply QC readback differs at metrics/{key}, {name} rows {start}:{stop}.")
            for key in ("mask_present", "area_px"):
                actual = np.asarray(component[key][start:stop])
                if not np.array_equal(actual, computed.spatial_metrics[key], equal_nan=True):
                    raise RuntimeError(f"Browser Apply QC readback differs at {name}/{key} rows {start}:{stop}.")
            for key, expected in computed.component_metrics.items():
                actual = np.asarray(component["metrics"][key][start:stop])
                if not np.array_equal(actual, expected, equal_nan=True):
                    raise RuntimeError(f"Browser Apply QC readback differs at {name}/metrics/{key} rows {start}:{stop}.")
            expected_reasons = finalizer._replace_metric_qc_reason_labels(
                original_reasons[name][start:stop], computed.reason_labels,
            )
            actual_reasons = refreshed_reasons[name][start:stop]
            if not np.array_equal(actual_reasons, expected_reasons):
                raise RuntimeError(f"Browser Apply QC reason tags differ at {name} rows {start}:{stop}.")

    for name in _BODY_CONTOUR_COMPONENTS.intersection(names):
        component = run["components"][name]
        contours = component.get("contours")
        if not isinstance(contours, zarr.Group):
            raise RuntimeError(f"Browser Apply QC did not create {name} contours.")
        ptr_array = contours["ptr"]
        length_array = contours["len"]
        points_array = contours["points_xy"]
        if ptr_array.shape != (row_count,) or length_array.shape != (row_count,):
            raise RuntimeError(f"Browser Apply QC {name} contour row shape differs.")
        comp_idx = names.index(name)
        for start in range(0, row_count, QC_ROW_CHUNK):
            if chunk_starts is not None and start not in chunk_starts:
                continue
            stop = min(start + QC_ROW_CHUNK, row_count)
            mask_chunk = np.asarray(masks[start:stop, comp_idx], dtype=np.uint8)
            ptr = np.asarray(ptr_array[start:stop], dtype=np.int64)
            length = np.asarray(length_array[start:stop], dtype=np.int32)
            for offset, mask in enumerate(mask_chunk):
                row = start + offset
                expected = extract_largest_external_contour(mask, min_points=2)
                if expected is None:
                    if int(ptr[offset]) != -1 or int(length[offset]) != 0:
                        raise RuntimeError(f"Browser Apply QC {name} contour row {row} differs.")
                else:
                    begin, count = int(ptr[offset]), int(length[offset])
                    if begin < 0 or count != len(expected) or not np.array_equal(
                        np.asarray(points_array[begin:begin + count], dtype=np.float32),
                        np.asarray(expected, dtype=np.float32),
                    ):
                        raise RuntimeError(f"Browser Apply QC {name} contour row {row} differs.")
    if _EYE_COMPONENTS.issubset(names):
        _validate_eye_geometry_rows(run, names, chunk_starts)


def _validate_eye_geometry_rows(
    run: zarr.Group, names: tuple[str, ...], chunk_starts: frozenset[int] | None = None,
) -> None:
    masks = run["masks_roi"]
    row_count = int(masks.shape[0])
    available_array = run.get("available_channels")
    available = (
        np.asarray(available_array[:], dtype=bool).reshape(-1)
        if available_array is not None else np.ones((len(names),), dtype=bool)
    )
    pair = run["relations/eye_pair/metrics"]
    for start in range(0, row_count, QC_ROW_CHUNK):
        if chunk_starts is not None and start not in chunk_starts:
            continue
        stop = min(start + QC_ROW_CHUNK, row_count)
        n_rows = stop - start
        success = np.zeros((n_rows, 2), dtype=bool)
        centroids = np.full((n_rows, 2, 2), np.nan, dtype=np.float32)
        for eye_idx, name in enumerate(eye_geometry.EYE_COMPONENTS):
            comp_idx = names.index(name)
            component = run["components"][name]
            geometry = component["geometry"]
            contours = component["contours"]
            ellipse_values = np.asarray(geometry["ellipse_params"][start:stop], dtype=np.float32)
            ellipse_success = np.asarray(geometry["ellipse_success"][start:stop], dtype=bool)
            ptr = np.asarray(contours["ptr"][start:stop], dtype=np.int64)
            length = np.asarray(contours["len"][start:stop], dtype=np.int32)
            if ellipse_values.shape != (n_rows, 5) or ellipse_success.shape != (n_rows,):
                raise RuntimeError(f"Browser Apply QC {name} eye geometry row shape differs.")
            mask_chunk = np.asarray(masks[start:stop, comp_idx], dtype=np.uint8)
            for offset, mask in enumerate(mask_chunk):
                expected_ellipse = np.full((5,), np.nan, dtype=np.float32)
                expected_contour = None
                if comp_idx < len(available) and bool(available[comp_idx]):
                    measured_success, ellipse, centroid, expected_contour, _failure = eye_geometry._measure_eye_mask(mask)
                    success[offset, eye_idx] = bool(measured_success)
                    expected_ellipse = np.asarray(ellipse, dtype=np.float32)
                    centroids[offset, eye_idx] = np.asarray(centroid, dtype=np.float32)
                if bool(ellipse_success[offset]) != bool(success[offset, eye_idx]) or not np.array_equal(
                    ellipse_values[offset], expected_ellipse, equal_nan=True,
                ):
                    raise RuntimeError(f"Browser Apply QC {name} eye ellipse row {start + offset} differs.")
                begin, count = int(ptr[offset]), int(length[offset])
                if expected_contour is None:
                    if begin != -1 or count != 0:
                        raise RuntimeError(f"Browser Apply QC {name} eye contour row {start + offset} differs.")
                elif begin < 0 or count != len(expected_contour) or not np.array_equal(
                    np.asarray(contours["points_xy"][begin:begin + count], dtype=np.float32),
                    np.asarray(expected_contour, dtype=np.float32),
                ):
                    raise RuntimeError(f"Browser Apply QC {name} eye contour row {start + offset} differs.")
        separation = np.full((n_rows,), np.nan, dtype=np.float32)
        separation_valid = np.zeros((n_rows,), dtype=bool)
        for offset in range(n_rows):
            if bool(np.all(success[offset])) and bool(np.all(np.isfinite(centroids[offset]))):
                separation[offset] = np.float32(np.linalg.norm(centroids[offset, 0] - centroids[offset, 1]))
                separation_valid[offset] = True
        if not np.array_equal(np.asarray(pair["separation_px"][start:stop]), separation, equal_nan=True):
            raise RuntimeError(f"Browser Apply QC eye separation rows {start}:{stop} differ.")
        if not np.array_equal(np.asarray(pair["separation_valid"][start:stop]), separation_valid):
            raise RuntimeError(f"Browser Apply QC eye separation validity rows {start}:{stop} differ.")


def _archive_path(root: zarr.Group) -> Path | None:
    """The archive directory of a root opened on a local/NFS filesystem store."""

    store_root = getattr(root.store, "root", None)
    if store_root is None or str(getattr(root, "path", "") or "").strip("/"):
        return None
    path = Path(str(store_root))
    return path if path.is_dir() else None


def _staging_scratch() -> Path | None:
    if os.environ.get(LOCAL_STAGING_ENV, "1").strip() == "0":
        return None
    scratch = Path(os.environ.get(STAGING_DIR_ENV) or tempfile.gettempdir())
    return scratch if scratch.is_dir() else None


def refresh_subject_mask_apply_qc_locked(
    *,
    root: zarr.Group,
    refined_run: str,
    expected_edit_revision: int,
) -> dict[str, Any]:
    """Refresh and verify full mask-local QC; caller holds the refined-run lock.

    The refresh and its verification run on a node-local copy of the run
    (many small reads and read-modify-writes are slow on NFS), and only the
    changed derived files are synced back: dense ``masks_roi`` must be
    byte-identical, the run's attributes are written last, and every written
    file is checked by digest. The archive is marked stale before the sync,
    so an interrupted sync fails closed exactly as an interrupted in-place
    refresh does. ``PALETTE_MASK_QC_LOCAL_STAGING=0`` refreshes in place.
    """

    archive = _archive_path(root)
    scratch = _staging_scratch()
    if archive is None or scratch is None:
        return {**_refresh_and_verify(root, refined_run, expected_edit_revision), "qc_execution": "in_place"}
    group_path = f"refined_subject_masks_runs/{refined_run}"
    with tempfile.TemporaryDirectory(prefix="palette-mask-qc-", dir=scratch) as tmp:
        staged = stage_group(archive, group_path, Path(tmp))
        local_root = zarr.open_group(str(staged.local_root), mode="a", use_consolidated=False)
        try:
            result = _refresh_and_verify(local_root, refined_run, expected_edit_revision)
        except BaseException:
            # Mirror the in-place path: once the refresh has marked the run
            # stale it stays stale on the archive; an earlier refusal leaves
            # the archive untouched.
            local_run = zarr.open_group(str(staged.local_group), mode="r", use_consolidated=False)
            if bool(local_run.attrs.get("metrics_stale")):
                _mark_archive_stale(root, refined_run)
            raise
        _mark_archive_stale(root, refined_run)
        sync = sync_group_back(staged, protected=("masks_roi",))
    return {**result, "qc_execution": "local_staging", "qc_files_written": sync["files_written"],
            "qc_files_removed": sync["files_removed"]}


def _mark_archive_stale(root: zarr.Group, refined_run: str) -> None:
    run = resolve_mutable_refined_subject_mask_run(root, str(refined_run))
    run.attrs.update({"metrics_stale": True, "contours_stale": True})
    run.attrs.pop("browser_apply_qc_policy", None)


def _refresh_and_verify(root: zarr.Group, refined_run: str, expected_edit_revision: int) -> dict[str, Any]:
    run = resolve_mutable_refined_subject_mask_run(root, str(refined_run))
    actual_revision = run.attrs.get("edit_revision")
    if type(actual_revision) is not int or actual_revision != int(expected_edit_revision):
        raise RuntimeError(
            f"Pending mask QC is bound to edit revision {expected_edit_revision}, found {actual_revision!r}."
        )
    names = _require_compatible_contract(run)
    mask_store = open_mask_store(run, source_path=f"refined_subject_masks_runs/{refined_run}", prefer="dense")
    if mask_store.storage_surface != "masks_roi":
        raise RuntimeError("Browser Apply QC requires authoritative dense masks_roi.")
    original_reasons = {}
    for name in names:
        labels = read_reason_labels(run["components"][name])
        if labels is None or len(labels) != int(run["masks_roi"].shape[0]):
            raise RuntimeError(f"Browser Apply QC requires valid {name} reason rows.")
        original_reasons[name] = labels
    row_count = int(run["masks_roi"].shape[0])
    chunk_digests = _chunk_mask_digests(run["masks_roi"])
    before = _derived_row_snapshot(run, row_count)
    changed = _changed_chunk_starts(
        run.attrs.get("browser_apply_qc_policy"), chunk_digests, _chunk_derived_digests(before, row_count),
        int(expected_edit_revision),
    )
    # A retry may start from a previously refreshed surface whose registry or
    # audit effect failed. Any failure during the new refresh must fail closed.
    run.attrs.update({"metrics_stale": True, "contours_stale": True})
    run.attrs.pop("browser_apply_qc_policy", None)
    scope = "full"
    fallback_reason = None
    if changed is not None:
        rows = [row for start in sorted(changed) for row in range(start, min(start + QC_ROW_CHUNK, row_count))]
        run = _refresh_rows(root, refined_run, names, expected_edit_revision, rows)
        _validate_metric_and_contour_rows(run, names, original_reasons, frozenset(changed))
        fallback_reason = _first_unchanged_row_difference(before, _derived_row_snapshot(run, row_count), set(rows))
        scope = "chunks" if fallback_reason is None else "full"
    if scope == "full":
        # No usable earlier verification, or (fallback) the stored values of
        # unchanged rows did not match their pixels: refresh every row.
        run = _refresh_rows(root, refined_run, names, expected_edit_revision, None)
        _validate_metric_and_contour_rows(run, names, original_reasons)
    run.attrs["browser_apply_qc_policy"] = {
        "id": QC_POLICY_ID,
        "version": QC_POLICY_VERSION,
        "metric_level": finalizer.EDITABLE_REVIEW_METRIC_LEVEL,
        "row_chunk": QC_ROW_CHUNK,
        "contours": "full_applicable_components",
        "edit_revision": int(expected_edit_revision),
        # Optional (version 1 readers ignore them): digests of each verified
        # QC chunk's pixels and derived values, so the next Apply recomputes
        # only chunks where either has changed since this verification.
        CHUNK_DIGESTS_KEY: chunk_digests,
        DERIVED_DIGESTS_KEY: _chunk_derived_digests(_derived_row_snapshot(run, row_count), row_count),
    }
    run.attrs["metrics_stale"] = False
    run.attrs["contours_stale"] = False
    return {
        "qc_status": "complete",
        "qc_policy_id": QC_POLICY_ID,
        "qc_edit_revision": int(expected_edit_revision),
        "qc_component_count": len(names),
        "qc_row_count": row_count,
        "qc_scope": scope,
        "qc_chunks_refreshed": len(changed) if scope == "chunks" else len(chunk_digests),
        "qc_chunk_count": len(chunk_digests),
        **({"qc_scope_fallback_reason": fallback_reason} if fallback_reason else {}),
    }


def _refresh_rows(root, refined_run, names, expected_edit_revision, rows):
    summary = finalizer.refresh_refined_subject_mask_metrics_run(
        root,
        refined_run=str(refined_run),
        components=None,
        metric_level=finalizer.EDITABLE_REVIEW_METRIC_LEVEL,
        chunk_size=QC_ROW_CHUNK,
        refresh_reason_tags=True,
        write_eye_geometry=_EYE_COMPONENTS.issubset(names),
        write_component_contours=True,
        rows=rows,
    )
    run = resolve_mutable_refined_subject_mask_run(root, str(refined_run))
    if type(run.attrs.get("edit_revision")) is not int or run.attrs["edit_revision"] != int(expected_edit_revision):
        raise RuntimeError("Refined mask edit revision changed during QC refresh.")
    if tuple(summary.get("components") or ()) != names:
        raise RuntimeError("Browser Apply QC did not refresh every component.")
    return run


def _chunk_mask_digests(masks) -> list[str]:
    """SHA-256 of each QC chunk's dense pixels (shape, dtype and bytes)."""

    digests = []
    for start in range(0, int(masks.shape[0]), QC_ROW_CHUNK):
        chunk = np.ascontiguousarray(masks[start:start + QC_ROW_CHUNK])
        digest = hashlib.sha256(f"{chunk.shape}|{chunk.dtype}".encode())
        digest.update(chunk.tobytes())
        digests.append(digest.hexdigest())
    return digests


def _changed_chunk_starts(prior, digests: list[str], derived: list[str], revision: int) -> set[int] | None:
    """QC chunks whose pixels or derived values differ from the previous revision's verified QC.

    Every other chunk is byte-identical, pixels and derived values, to a
    state that a full-equivalent QC verified. None means refresh every row:
    no usable earlier verification, or a rerun at the same revision (which
    keeps its full refresh).
    """

    if not (
        isinstance(prior, dict)
        and prior.get("id") == QC_POLICY_ID
        and prior.get("version") == QC_POLICY_VERSION
        and prior.get("row_chunk") == QC_ROW_CHUNK
        and prior.get("metric_level") == finalizer.EDITABLE_REVIEW_METRIC_LEVEL
        and prior.get("contours") == "full_applicable_components"
        and type(prior.get("edit_revision")) is int
        and prior["edit_revision"] < int(revision)
        and isinstance(prior.get(CHUNK_DIGESTS_KEY), list)
        and len(prior[CHUNK_DIGESTS_KEY]) == len(digests)
        and isinstance(prior.get(DERIVED_DIGESTS_KEY), list)
        and len(prior[DERIVED_DIGESTS_KEY]) == len(derived)
    ):
        return None
    return {
        index * QC_ROW_CHUNK
        for index in range(len(digests))
        if prior[CHUNK_DIGESTS_KEY][index] != digests[index] or prior[DERIVED_DIGESTS_KEY][index] != derived[index]
    }


def _chunk_derived_digests(snapshot: dict[str, object], row_count: int) -> list[str]:
    """SHA-256 of each QC chunk's derived values (every path of the snapshot)."""

    digests = []
    for start in range(0, row_count, QC_ROW_CHUNK):
        stop = min(start + QC_ROW_CHUNK, row_count)
        digest = hashlib.sha256()
        for path in sorted(snapshot):
            value = snapshot[path]
            digest.update(path.encode() + b"\0")
            if value is None:
                digest.update(b"<unreadable>")
            elif isinstance(value, list):
                for row in value[start:stop]:
                    digest.update(b"-" if row is None else _array_bytes(np.asarray(row)))
            else:
                digest.update(_array_bytes(np.asarray(value)[start:stop]))
        digests.append(digest.hexdigest())
    return digests


def _array_bytes(array: np.ndarray) -> bytes:
    header = f"{array.shape}|{array.dtype}|".encode()
    if array.dtype.kind in "OU":
        return header + "\0".join(str(item) for item in array.reshape(-1).tolist()).encode()
    return header + np.ascontiguousarray(array).tobytes()


def _derived_row_snapshot(run: zarr.Group, row_count: int) -> dict[str, object]:
    """Every row-indexed derived value of the run (contours and reasons decoded)."""

    snapshot: dict[str, object] = {}

    def visit(group: zarr.Group, prefix: str) -> None:
        for name, child in group.members():
            path = f"{prefix}/{name}" if prefix else name
            if isinstance(child, zarr.Group):
                if name == "contours":
                    snapshot[path] = read_component_contours(group, row_count)
                elif name != "sampled_contours":
                    visit(child, path)
            elif name == "reason_bytes":
                snapshot[path] = read_reason_labels(group)
            elif path != "masks_roi" and child.shape and int(child.shape[0]) == row_count:
                snapshot[path] = np.asarray(child[...])

    visit(run, "")
    return snapshot


def _first_unchanged_row_difference(before, after, rows: set[int]) -> str | None:
    """Where a row outside ``rows`` changed (None when every such row is identical)."""

    if before.keys() != after.keys():
        return "the run's derived arrays changed"
    keep = np.asarray([row for row in range(_row_count(before)) if row not in rows], dtype=np.int64)
    for path in sorted(before):
        old, new = before[path], after[path]
        if old is None or new is None:
            if old is not new:
                return f"{path} could not be read"
        elif isinstance(old, list):
            for row in keep.tolist():
                a, b = old[row], new[row]
                if (a is None) != (b is None) or (a is not None and not np.array_equal(a, b)):
                    return f"{path} row {row}"
        else:
            a, b = np.asarray(old)[keep], np.asarray(new)[keep]
            if a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
                return f"{path} (unchanged rows)"
    return None


def _row_count(snapshot) -> int:
    for value in snapshot.values():
        if value is not None:
            return len(value)
    return 0
