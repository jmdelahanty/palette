"""Pose observation v3: the skeleton sidecar in the file and recomputed headings.

agent-contracts ``citrus-pose-observation-v3`` (PR 104): every distinct
sidecar named by ``contract_json.metadata_cameras`` is stored verbatim at
``/observations/pose/sidecars/<sidecar_sha256>``, and every object's
``heading_status`` and derived heading follow from its keypoints by
``contract_json.heading_arithmetic`` (``pose_heading_arithmetic_v3``). This
module re-derives both from the file alone: the status, origin, unit
direction and eye check bit for bit (binary32 steps reproduced with numpy
float32, each operation rounded on its own as Citrus builds with
``-ffp-contract=off``), the angle within ``ANGLE_TOLERANCE_DEG`` by
``wrap_delta`` because it is a library transcendental.

The executed form is Citrus ``src/ipc/pose_metadata.cpp`` (``ParseCamera``,
``PointFor``, ``EvaluateHeading``) at 1855e374.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from fisheye.shared.heading_convention import heading_degrees, wrap_delta_degrees

from .common import parse_json, require
from .hdf5_types import dataset_bytes, iter_blocks

SIDECARS = "/observations/pose/sidecars"
ARITHMETIC_VERSION = "pose_heading_arithmetic_v3"
SIDECAR_SCHEMA_ID = "palette.pose_model_skeleton"
EYE_LEFT, EYE_RIGHT = "eye_left", "eye_right"
MAX_KEYPOINTS = 16
# shaman_v2.h: PoseStatus::kPoses, kObjectHasPose, kKeypointVisible.
POSE_STATUS_POSES = 3
OBJECT_HAS_POSE = 1 << 1
KEYPOINT_VISIBLE = 1 << 0
# kHeadingDegenerateEpsilonPx is the binary32 1e-6f, compared as a double.
DEGENERATE_EPSILON = float(np.float32(1e-6))
ANGLE_TOLERANCE_DEG = 1e-4
(
    NO_POSE,
    UNRESOLVED_METADATA,
    IDENTITY_MISMATCH,
    MISSING_POINT,
    INVALID_POINT,
    LOW_CONFIDENCE,
    DEGENERATE,
    VALID,
    POLICY_DISABLED,
    LEFT_RIGHT_UNRESOLVED,
) = range(10)
F32 = np.float32


@dataclass(frozen=True)
class Skeleton:
    """One camera's resolved sidecar: labels and the heading recipe."""

    labels: tuple[str, ...]
    skeleton64: int
    model64: int
    heading_enabled: bool
    direction_from: tuple[int, ...] = ()
    direction_to: tuple[int, ...] = ()
    origin: tuple[int, ...] = ()
    eye_left: int | None = None
    eye_right: int | None = None


def _prefix64(digest: object, reason: str) -> int:
    require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest),
        reason,
    )
    return int(digest[:16], 16)


def _unsigned(value: object, reason: str) -> int:
    require(type(value) is int and value >= 0, reason)
    return value


def _expression(expression: object, ids: dict[str, int]) -> tuple[int, ...]:
    require(isinstance(expression, Mapping), "pose_sidecar_heading_expression")
    op = expression.get("op")
    if op == "keypoint":
        names = [expression.get("label")]
    elif op == "midpoint":
        names = expression.get("labels")
        require(
            isinstance(names, list) and len(names) == 2 and names[0] != names[1],
            "pose_sidecar_heading_midpoint_labels",
        )
    else:
        require(False, "pose_sidecar_heading_op_unsupported")
    require(all(name in ids for name in names), "pose_sidecar_heading_label_unknown")
    return tuple(ids[name] for name in names)


def _skeleton(text: bytes, camera: Mapping) -> Skeleton:
    """Parse a sidecar as Citrus ParseCamera does; any defect refuses the file."""

    document = parse_json(text, label="pose_sidecar")
    require(
        document.get("schema_id") == SIDECAR_SCHEMA_ID
        and document.get("schema_version") == 1
        and type(document.get("schema_version")) is int,
        "pose_sidecar_schema",
    )
    nodes, shape = document.get("nodes"), document.get("kpt_shape")
    require(
        isinstance(nodes, list)
        and 0 < len(nodes) <= MAX_KEYPOINTS
        and isinstance(shape, list)
        and len(shape) == 2
        and _unsigned(shape[0], "pose_sidecar_shape") == len(nodes)
        and _unsigned(shape[1], "pose_sidecar_shape") == 3,
        "pose_sidecar_shape",
    )
    ids: dict[str, int] = {}
    labels = []
    for index, node in enumerate(nodes):
        require(isinstance(node, Mapping), "pose_sidecar_node")
        require(_unsigned(node.get("id"), "pose_sidecar_node_id") == index, "pose_sidecar_node_ids_noncontiguous")
        name = node.get("name")
        require(isinstance(name, str) and name != "" and name not in ids, "pose_sidecar_label_duplicate_or_empty")
        ids[name] = index
        labels.append(name)
    edges = document.get("edges")
    require(isinstance(edges, list), "pose_sidecar_edges")
    for edge in edges:
        require(
            isinstance(edge, list)
            and len(edge) == 2
            and _unsigned(edge[0], "pose_sidecar_edge") < len(nodes)
            and _unsigned(edge[1], "pose_sidecar_edge") < len(nodes)
            and edge[0] != edge[1],
            "pose_sidecar_edge",
        )
    binding = document.get("model_schema_binding")
    pose_schema = binding.get("pose_schema") if isinstance(binding, Mapping) else None
    require(isinstance(pose_schema, Mapping), "pose_sidecar_pose_schema_missing")
    metadata = pose_schema.get("metadata")
    require(
        isinstance(metadata, Mapping)
        and pose_schema.get("skeleton_id") == document.get("skeleton_id")
        and metadata.get("model_kpt_shape") == shape
        and pose_schema.get("nodes") == nodes
        and pose_schema.get("edges") == edges
        and pose_schema.get("keypoint_labels") == labels
        and pose_schema.get("kpt_shape") == [len(nodes), 2],
        "pose_sidecar_nested_schema_mismatch",
    )
    policy = metadata.get("heading_computation")
    require(
        isinstance(policy, Mapping)
        and type(policy.get("enabled")) is bool
        and type(policy.get("version")) is int
        and 0 <= policy["version"] <= 2**32 - 1,
        "pose_sidecar_heading_policy",
    )
    # The contract's per-camera summary is the sidecar's own policy.
    require(
        camera.get("heading_enabled") is policy["enabled"]
        and camera.get("heading_policy_version") == policy["version"],
        "pose_sidecar_heading_policy_mismatch",
    )
    common = {
        "labels": tuple(labels),
        "skeleton64": _prefix64(camera.get("sidecar_sha256"), "pose_sidecar_digest_malformed"),
        "model64": _prefix64(camera.get("engine_sha256"), "pose_engine_digest_malformed"),
        "eye_left": ids.get(EYE_LEFT),
        "eye_right": ids.get(EYE_RIGHT),
    }
    if not policy["enabled"]:
        return Skeleton(heading_enabled=False, **common)
    require(policy["version"] == 1, "pose_sidecar_heading_policy_unsupported")
    expressions = {
        name: _expression(policy.get(name), ids)
        for name in ("direction_from", "direction_to", "origin")
    }
    used = {labels[i] for ids_ in expressions.values() for i in ids_}
    declared = policy.get("dependent_keypoints")
    require(
        isinstance(declared, list)
        and len(declared) == len(used)
        and set(declared) == used,
        "pose_sidecar_heading_dependencies_mismatch",
    )
    return Skeleton(heading_enabled=True, **expressions, **common)


def resolve_sidecars(h5, contract: Mapping) -> dict[str, Skeleton]:
    """Every metadata camera's sidecar, verbatim in the file and bound by digest."""

    arithmetic = contract.get("heading_arithmetic")
    require(
        isinstance(arithmetic, Mapping) and arithmetic.get("version") == ARITHMETIC_VERSION,
        "pose_heading_arithmetic_version",
    )
    cameras = contract["metadata_cameras"]
    referenced = set()
    skeletons = {}
    for serial, camera in cameras.items():
        digest = camera["sidecar_sha256"]
        path = f"{SIDECARS}/{digest}"
        require(camera["sidecar_ref"] == path, "pose_sidecar_ref_mismatch")
        require(path in h5, f"pose_sidecar_missing:{digest}")
        dataset = h5[path]
        require(dataset.shape == () and dataset.dtype.kind == "O", f"pose_sidecar_not_scalar_string:{digest}")
        text = dataset_bytes(dataset)
        require(sha256(text).hexdigest() == digest, f"pose_sidecar_digest_mismatch:{digest}")
        require(
            int(digest[:16], 16) == camera["ipc_skeleton_id_hash"],
            "pose_sidecar_skeleton_hash_mismatch",
        )
        require(
            _prefix64(camera["engine_sha256"], "pose_engine_digest_malformed") == camera["ipc_model_id_hash"],
            "pose_engine_model_hash_mismatch",
        )
        referenced.add(digest)
        skeletons[serial] = _skeleton(text, camera)
    stored = set(h5[SIDECARS].keys()) if SIDECARS in h5 else set()
    require(stored == referenced, "pose_sidecar_set_mismatch")
    return skeletons


def _bits(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)


class _Points:
    """Per-object keypoints of the labels a recipe reads (dense by label)."""

    def __init__(self, count: int, keypoints, owner: np.ndarray, labels: set[int]):
        self.present = {label: np.zeros(count, dtype=bool) for label in labels}
        self.x = {label: np.zeros(count, dtype=np.float32) for label in labels}
        self.y = {label: np.zeros(count, dtype=np.float32) for label in labels}
        self.confidence = {label: np.zeros(count, dtype=np.float32) for label in labels}
        self.flags = {label: np.zeros(count, dtype=np.uint16) for label in labels}
        label_ids = keypoints["label_id"]
        for label in labels:
            hit = label_ids == label
            where = owner[hit]
            self.present[label][where] = True
            self.x[label][where] = keypoints["x_px"][hit]
            self.y[label][where] = keypoints["y_px"][hit]
            self.confidence[label][where] = keypoints["confidence"][hit]
            self.flags[label][where] = keypoints["flags"][hit]


def _resolve(
    expression: tuple[int, ...],
    points: _Points,
    gate: np.float32,
    status: np.ndarray,
    pending: np.ndarray,
):
    """PointFor: each label in order (absent 3, invalid 4, low 5), then the sum."""

    for label in expression:
        absent = pending & ~points.present[label]
        status[absent] = MISSING_POINT
        pending &= ~absent
        x, y, c = points.x[label], points.y[label], points.confidence[label]
        with np.errstate(invalid="ignore"):
            invalid = pending & ~(
                np.isfinite(x) & np.isfinite(y) & np.isfinite(c) & (c >= F32(0)) & (c <= F32(1))
            )
        status[invalid] = INVALID_POINT
        pending &= ~invalid
        low = pending & ((c < gate) | ((points.flags[label] & KEYPOINT_VISIBLE) == 0))
        status[low] = LOW_CONFIDENCE
        pending &= ~low
    with np.errstate(all="ignore"):
        sum_x = np.zeros(status.shape, dtype=np.float32)
        sum_y = np.zeros(status.shape, dtype=np.float32)
        for label in expression:
            sum_x = sum_x + points.x[label]
            sum_y = sum_y + points.y[label]
        count = F32(len(expression))
        return sum_x / count, sum_y / count


def check_v3_headings(h5, contract: Mapping, updates, objects, keypoints) -> None:
    """Every object's stored heading is what its keypoints give (pose v3)."""

    skeletons = resolve_sidecars(h5, contract)
    gate = F32(contract["minimum_keypoint_confidence"])
    pose_status = np.asarray(updates.fields("pose_status")[:])
    serials = np.asarray(updates.fields("camera_serial")[:])
    skeleton_hash = np.asarray(updates.fields("pose_skeleton_id_hash")[:], dtype=np.uint64)
    model_hash = np.asarray(updates.fields("pose_model_id_hash")[:], dtype=np.uint64)
    counts = np.asarray(objects.fields("keypoint_count")[:], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    for start, block in iter_blocks(objects):
        n = len(block)
        first, last = int(offsets[start]), int(offsets[start + n])
        points = keypoints[first:last]
        owner = (points["object_row"].astype(np.int64) - start) if last > first else np.zeros(0, np.int64)
        update = block["update_row"].astype(np.int64)
        expected = np.full(n, NO_POSE, dtype=np.int64)
        pending = (
            (pose_status[update] == POSE_STATUS_POSES)
            & ((block["flags"] & OBJECT_HAS_POSE) != 0)
            & (block["keypoint_count"] > 0)
        )
        origin_x = np.zeros(n, dtype=np.float32)
        origin_y = np.zeros(n, dtype=np.float32)
        direction_x = np.zeros(n, dtype=np.float32)
        direction_y = np.zeros(n, dtype=np.float32)
        serial_text = np.array(
            [bytes(value).split(b"\0", 1)[0].decode("utf-8", "replace") for value in serials[update]],
            dtype=object,
        )
        for serial in sorted(set(serial_text[pending])):
            selected = pending & (serial_text == serial)
            skeleton = skeletons.get(serial)
            if skeleton is None:
                expected[selected] = UNRESOLVED_METADATA
                continue
            mismatch = selected & (
                (skeleton_hash[update] != np.uint64(skeleton.skeleton64))
                | (model_hash[update] != np.uint64(skeleton.model64))
            )
            expected[mismatch] = IDENTITY_MISMATCH
            selected &= ~mismatch
            if not skeleton.heading_enabled:
                expected[selected] = POLICY_DISABLED
                continue
            status = expected  # assigned in place, first failure wins
            in_scope = np.zeros(n, dtype=bool)
            in_scope[selected] = True
            mine = in_scope[owner] if owner.size else np.zeros(0, dtype=bool)
            label_ids = points["label_id"].astype(np.int64)
            bad = np.zeros(n, dtype=bool)
            bad[owner[mine & (label_ids >= len(skeleton.labels))]] = True
            keys = owner[mine] * 65536 + label_ids[mine]
            unique, seen = np.unique(keys, return_counts=True)
            bad[(unique[seen > 1] // 65536).astype(np.int64)] = True
            bad &= selected
            status[bad] = INVALID_POINT
            selected &= ~bad
            if not (np.isfinite(gate) and F32(0) <= gate <= F32(1)):
                status[selected] = INVALID_POINT
                continue
            wanted = set(skeleton.direction_from + skeleton.direction_to + skeleton.origin)
            wanted |= {label for label in (skeleton.eye_left, skeleton.eye_right) if label is not None}
            dense = _Points(n, points[mine], owner[mine], wanted)
            from_x, from_y = _resolve(skeleton.direction_from, dense, gate, status, selected)
            to_x, to_y = _resolve(skeleton.direction_to, dense, gate, status, selected)
            point_x, point_y = _resolve(skeleton.origin, dense, gate, status, selected)
            with np.errstate(all="ignore"):
                dx = to_x - from_x
                dy = to_y - from_y
                dx64, dy64 = dx.astype(np.float64), dy.astype(np.float64)
                length = np.sqrt(dx64 * dx64 + dy64 * dy64)
                degenerate = selected & ~(np.isfinite(length) & (length > DEGENERATE_EPSILON))
                status[degenerate] = DEGENERATE
                selected &= ~degenerate
                forward_x = (dx64 / length).astype(np.float32)
                forward_y = (dy64 / length).astype(np.float32)
                left_x, left_y = forward_y, -forward_x
            if skeleton.eye_left is None or skeleton.eye_right is None:
                status[selected] = LEFT_RIGHT_UNRESOLVED
                continue
            el, er = skeleton.eye_left, skeleton.eye_right
            with np.errstate(all="ignore"):
                eyes = (
                    dense.present[el]
                    & dense.present[er]
                    & np.isfinite(dense.x[el])
                    & np.isfinite(dense.y[el])
                    & np.isfinite(dense.x[er])
                    & np.isfinite(dense.y[er])
                )
                product = ((dense.x[el] - dense.x[er]) * left_x) + ((dense.y[el] - dense.y[er]) * left_y)
                unresolved = selected & ~(eyes & (product > F32(0)))
            status[unresolved] = LEFT_RIGHT_UNRESOLVED
            selected &= ~unresolved
            status[selected] = VALID
            origin_x[selected], origin_y[selected] = point_x[selected], point_y[selected]
            direction_x[selected], direction_y[selected] = forward_x[selected], forward_y[selected]
        stored = block["heading_status"].astype(np.int64)
        wrong = np.flatnonzero(stored != expected)
        require(
            wrong.size == 0,
            "pose_heading_not_reproduced:status"
            + (f":row={start + int(wrong[0])}:stored={int(stored[wrong[0]])}:expected={int(expected[wrong[0]])}" if wrong.size else ""),
        )
        valid = expected == VALID
        for name, values in (
            ("heading_origin_x_px", origin_x),
            ("heading_origin_y_px", origin_y),
            ("heading_direction_x", direction_x),
            ("heading_direction_y", direction_y),
        ):
            differs = np.flatnonzero(valid & (_bits(block[name]) != _bits(values)))
            require(
                differs.size == 0,
                f"pose_heading_not_reproduced:{name}"
                + (f":row={start + int(differs[0])}" if differs.size else ""),
            )
        if valid.any():
            implied = heading_degrees(block["heading_direction_x"][valid], block["heading_direction_y"][valid])
            delta = np.abs(wrap_delta_degrees(block["heading_angle_degrees"][valid].astype(np.float64) - implied))
            require(bool(np.all(delta <= ANGLE_TOLERANCE_DEG)), "pose_heading_not_reproduced:heading_angle_degrees")


__all__ = ["SIDECARS", "Skeleton", "check_v3_headings", "resolve_sidecars"]
