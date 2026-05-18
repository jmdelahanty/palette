"""Browser-based keypoint review UI backed by a shared save/review session."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..registry.db import Registry, RegistryPaths

if TYPE_CHECKING:
    from . import keypoint_review_backend


def _parse_index_sequence(value: Optional[str]) -> Optional[list[int]]:
    if value is None:
        return None
    raw = str(value).replace(",", " ").strip()
    if not raw:
        return None
    out: list[int] = []
    for token in raw.split():
        try:
            out.append(int(token))
        except (TypeError, ValueError):
            continue
    return sorted(set(out)) if out else None


def _parse_roi_sequence(values: Optional[Sequence[int]]) -> Optional[list[int]]:
    if not values:
        return None
    out: list[int] = []
    for item in values:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(set(out)) if out else None


@dataclass(frozen=True)
class _ServerConfig:
    zarr_path: Optional[str]
    host: str
    port: int
    refined_run: Optional[str]
    crop_run: Optional[str]
    include_all: bool
    target_frames: Optional[list[int]]
    target_roi_indices: Optional[list[int]]
    review_method: str
    review_intended_use: Optional[str]
    reviewer: Optional[str]
    review_notes: Optional[str]
    frame_flag_file: Optional[str]
    auto_advance_on_save: bool
    lock: bool
    force_lock: bool
    registry_path: Optional[str]
    dataset_id: Optional[str]
    dataset_limit: int
    dataset_review_filter: str


@dataclass
class _ServerState:
    session: "keypoint_review_backend.ReviewSession"  # type: ignore[name-defined]
    position: int = 0
    filter_mode: str = "failed"
    search: str = ""
    review_method: str = "manual"
    review_intended_use: Optional[str] = None
    reviewer: Optional[str] = None
    review_notes: Optional[str] = None
    frame_flag_file: Optional[str] = None
    auto_advance_on_save: bool = False
    registry_path: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset: Optional[dict[str, object]] = None
    dataset_limit: int = 500
    dataset_review_filter: str = "needs_review"
    refined_run_arg: Optional[str] = None
    crop_run_arg: Optional[str] = None
    include_all_arg: bool = False
    target_frames: Optional[list[int]] = None
    target_roi_indices: Optional[list[int]] = None
    review_lock: Optional["_ReviewLock"] = None
    lock_enabled: bool = True
    force_lock: bool = False


_CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".txt": "text/plain; charset=utf-8",
}


@dataclass(frozen=True)
class _ReviewLock:
    path: Path

    def release(self) -> None:
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass


def _safe_run_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _acquire_review_lock(zarr_path: str, refined_run: str, *, force: bool = False) -> _ReviewLock:
    lock_dir = Path(zarr_path).expanduser() / ".palette_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"keypoint_review_web.{_safe_run_name(refined_run)}.lock"
    if lock_path.exists() and not force:
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        pid = int(existing.get("pid") or -1) if isinstance(existing, Mapping) else -1
        if _process_is_alive(pid):
            raise RuntimeError(
                "Another keypoint web review session appears active for this refined run "
                f"(pid={pid}, lock={lock_path}). Stop it or pass --force-lock."
            )
    payload = {
        "pid": os.getpid(),
        "zarr_path": str(zarr_path),
        "refined_run": str(refined_run),
    }
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return _ReviewLock(lock_path)


def _to_json_scalar(value: object, *, allow_nan: bool = False) -> object:
    if value is None:
        return None
    try:
        scalar = np.asarray(value).item()
    except Exception:
        return str(value)

    if isinstance(scalar, (bytes, bytearray, memoryview)):
        try:
            return bytes(scalar).decode("utf-8")
        except Exception:
            return str(scalar)
    if isinstance(scalar, np.ndarray):  # pragma: no cover - defensive
        return str(scalar.tolist())
    if isinstance(scalar, np.number):
        scalar = scalar.item()
    if isinstance(scalar, (float, int, str, bool, type(None))):
        if isinstance(scalar, float) and not allow_nan and (scalar != scalar or scalar == float("inf") or scalar == -float("inf")):
            return None
        return scalar
    return str(scalar)


def _to_json_float_or_none(value: object) -> Optional[float]:
    scalar = _to_json_scalar(value)
    try:
        if scalar is None:
            return None
        num = float(scalar)
    except (TypeError, ValueError):
        return None
    return num if np.isfinite(num) else None


def _row_get(row: Mapping[str, object], key: str) -> object:
    if isinstance(row, Mapping):
        return row.get(key)
    try:
        return row[key]  # type: ignore[index]
    except Exception:
        return None


def _dataset_row_payload(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "dataset_id": _to_json_scalar(_row_get(row, "dataset_id")),
        "session_uuid": _to_json_scalar(_row_get(row, "session_uuid")),
        "recording_id": _to_json_scalar(_row_get(row, "recording_id")),
        "zarr_path": _to_json_scalar(_row_get(row, "zarr_path")),
        "zarr_use": _to_json_scalar(_row_get(row, "zarr_use")),
        "zarr_origin": _to_json_scalar(_row_get(row, "zarr_origin")),
        "status": _to_json_scalar(_row_get(row, "status")),
        "source_layout": _to_json_scalar(_row_get(row, "source_layout")),
        "dish_design": _to_json_scalar(_row_get(row, "dish_design")),
        "fish_id": _to_json_scalar(_row_get(row, "fish_id")),
        "subject_id": _to_json_scalar(_row_get(row, "subject_id")),
        "camera_serial": _to_json_scalar(_row_get(row, "camera_serial")),
        "fps": _to_json_scalar(_row_get(row, "fps")),
    }


def _open_registry(registry_path: str) -> Registry:
    return Registry(Path(registry_path).expanduser())


def _zarr_child_group_exists(zarr_path: object, child: str) -> bool:
    if not zarr_path:
        return False
    child_path = Path(str(zarr_path)).expanduser() / child
    if not child_path.is_dir():
        return False
    if (child_path / "zarr.json").exists() or (child_path / ".zgroup").exists():
        return True
    try:
        next(child_path.iterdir())
    except StopIteration:
        return False
    except OSError:
        return False
    return True


def _is_keypoint_reviewable_training_dataset(dataset: Mapping[str, object]) -> bool:
    zarr_path = dataset.get("zarr_path")
    return _zarr_child_group_exists(zarr_path, "refined_keypoints_runs") and _zarr_child_group_exists(zarr_path, "crop_runs")


def _read_zarr_group_attrs(path: Path) -> dict[str, object]:
    try:
        data = json.loads((path / "zarr.json").read_text(encoding="utf-8"))
    except Exception:
        return {}
    attrs = data.get("attributes")
    return dict(attrs) if isinstance(attrs, Mapping) else {}


def _review_status_for_training_dataset(dataset: Mapping[str, object]) -> dict[str, object]:
    zarr_path = dataset.get("zarr_path")
    if not zarr_path:
        return {"approved": False, "reason": "missing_zarr_path"}
    refined_parent = Path(str(zarr_path)).expanduser() / "refined_keypoints_runs"
    parent_attrs = _read_zarr_group_attrs(refined_parent)
    candidate_names: list[str] = []
    for key in ("latest", "keypoint_review_status_latest"):
        value = parent_attrs.get(key)
        if value is not None:
            name = str(value).strip()
            if name and name not in candidate_names:
                candidate_names.append(name)

    if not candidate_names:
        try:
            candidate_names = sorted(path.name for path in refined_parent.iterdir() if path.is_dir())
        except OSError:
            candidate_names = []

    for run_name in candidate_names:
        run_attrs = _read_zarr_group_attrs(refined_parent / run_name)
        status = run_attrs.get("keypoint_review_status")
        if not isinstance(status, Mapping):
            continue
        state = str(status.get("state") or "").strip().lower()
        intended_use = str(status.get("intended_use") or "").strip().lower()
        approved = state == "approved" and (not intended_use or intended_use == "training")
        return {
            "approved": approved,
            "review_state": state or None,
            "review_intended_use": intended_use or None,
            "review_run": run_name,
            "review_status": dict(status),
            "reason": "approved" if approved else "not_approved",
        }

    return {
        "approved": False,
        "review_state": None,
        "review_intended_use": None,
        "review_run": candidate_names[0] if candidate_names else None,
        "review_status": None,
        "reason": "no_keypoint_review_status",
    }


def _dataset_matches_review_filter(dataset: Mapping[str, object], review_filter: str) -> bool:
    mode = str(review_filter or "needs_review").strip().lower()
    status = _review_status_for_training_dataset(dataset)
    if mode in {"all", "any"}:
        return True
    if mode in {"approved", "approved_training"}:
        return bool(status.get("approved"))
    if mode in {"needs_review", "unapproved", "not_approved", "without_approved"}:
        return not bool(status.get("approved"))
    raise ValueError(f"Unsupported dataset review filter: {review_filter}")


def _annotate_dataset_review_status(dataset: dict[str, object]) -> dict[str, object]:
    status = _review_status_for_training_dataset(dataset)
    out = dict(dataset)
    out.update(
        {
            "keypoint_review_approved": bool(status.get("approved")),
            "keypoint_review_state": status.get("review_state"),
            "keypoint_review_intended_use": status.get("review_intended_use"),
            "keypoint_review_run": status.get("review_run"),
            "keypoint_review_reason": status.get("reason"),
        }
    )
    return out


def _dedupe_datasets_by_zarr_path(datasets: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for dataset in datasets:
        zarr_path = str(dataset.get("zarr_path") or "")
        key = str(Path(zarr_path).expanduser()) if zarr_path else str(dataset.get("dataset_id") or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(dataset)
    return out


def _list_training_datasets(
    registry_path: str,
    *,
    limit: int = 500,
    keypoint_reviewable_only: bool = True,
    review_filter: str = "needs_review",
) -> list[dict[str, object]]:
    registry = _open_registry(registry_path)
    query_limit = max(1, int(limit))
    if keypoint_reviewable_only or review_filter not in {"all", "any"}:
        query_limit = max(query_limit * 10, query_limit + 100)
    rows = registry.query_datasets(
        zarr_use="training",
        exclude_status="missing",
        limit=query_limit,
    )
    datasets = [_dataset_row_payload(row) for row in rows]
    if keypoint_reviewable_only:
        datasets = [dataset for dataset in datasets if _is_keypoint_reviewable_training_dataset(dataset)]
    datasets = [_annotate_dataset_review_status(dataset) for dataset in datasets]
    datasets = [dataset for dataset in datasets if _dataset_matches_review_filter(dataset, review_filter)]
    datasets = _dedupe_datasets_by_zarr_path(datasets)
    return datasets[: max(1, int(limit))]


def _find_training_dataset(registry_path: str, dataset_id: str) -> dict[str, object]:
    registry = _open_registry(registry_path)
    row = registry.conn.execute(
        """
        SELECT dcc.dataset_id, dcc.session_uuid, dcc.zarr_path,
               dcc.recording_id, dcc.zarr_origin, dcc.zarr_use,
               dcc.dataset_status AS status, dcc.source_layout,
               dcc.dish_design, COALESCE(dcc.subject_id, dcc.legacy_fish_id) AS fish_id,
               dcc.subject_id AS subject_id, dcc.camera_serial, dcc.fps
        FROM dataset_context_current dcc
        WHERE dcc.dataset_id = ?
          AND dcc.zarr_use = 'training'
          AND (dcc.dataset_status IS NULL OR dcc.dataset_status != 'missing')
        LIMIT 1
        """,
        (str(dataset_id),),
    ).fetchone()
    if row is None:
        raise ValueError(f"Training dataset not found in registry: {dataset_id}")
    dataset = _dataset_row_payload(row)
    if not _is_keypoint_reviewable_training_dataset(dataset):
        raise ValueError(
            "Training dataset is not keypoint-reviewable: "
            f"{dataset_id} (requires refined_keypoints_runs and crop_runs)"
        )
    return _annotate_dataset_review_status(dataset)


def _resolve_initial_dataset(config: _ServerConfig) -> tuple[str, Optional[str], Optional[dict[str, object]]]:
    if config.zarr_path:
        dataset: Optional[dict[str, object]] = None
        if config.registry_path and config.dataset_id:
            dataset = _find_training_dataset(config.registry_path, config.dataset_id)
        return str(config.zarr_path), config.dataset_id, dataset

    if not config.registry_path:
        raise ValueError("Provide a zarr_path, or enable registry mode with --registry.")

    if config.dataset_id:
        dataset = _find_training_dataset(config.registry_path, config.dataset_id)
    else:
        datasets = _list_training_datasets(
            config.registry_path,
            limit=config.dataset_limit,
            review_filter=config.dataset_review_filter,
        )
        if not datasets:
            raise ValueError(f"No active training datasets found in registry: {config.registry_path}")
        dataset = datasets[0]

    zarr_path = dataset.get("zarr_path")
    if not zarr_path:
        raise ValueError(f"Registry dataset has no zarr_path: {dataset.get('dataset_id')}")
    return str(zarr_path), str(dataset.get("dataset_id") or ""), dataset


def _build_session_summary(state: _ServerState, backend_module: "Any" = None) -> Optional[dict[str, object]]:
    if backend_module is None:
        return None
    try:
        return dict(backend_module.review_session_summary(state.session))
    except Exception as exc:
        return {"error": str(exc)}


def _build_state_payload(state: _ServerState, backend_module: "Any" = None) -> dict[str, object]:
    total = int(state.session.failures.size)
    review_status = state.session.refined.attrs.get("keypoint_review_status")
    review_status_payload = review_status if isinstance(review_status, Mapping) else None
    dataset_context = {
        "registry_enabled": bool(state.registry_path),
        "registry_path": state.registry_path,
        "dataset_id": state.dataset_id,
        "dataset": state.dataset,
        "zarr_path": str(state.session.zarr_path),
        "dataset_summary": _build_session_summary(state, backend_module),
    }
    if total == 0:
        return {
            "ok": False,
            "position": 0,
            "total": 0,
            "roi_idx": None,
            "frame_idx": None,
            "labels": list(state.session.keypoint_labels),
            "reason": "",
            "status": {},
            "refined_run": str(state.session.refined_run),
            "crop_run": str(state.session.crop_run),
            "filter_mode": state.filter_mode,
            "search": state.search,
            "review_status": review_status_payload,
            "auto_advance_on_save": bool(state.auto_advance_on_save),
            **dataset_context,
        }

    roi_idx = int(state.session.failures[state.position])
    frame_idx = int(state.session.frame_indices[roi_idx])
    reason = ""
    if state.session.reason_arr is not None:
        reason = "" if state.session.reason_arr[roi_idx] is None else str(state.session.reason_arr[roi_idx])

    status: dict[str, object] = {}
    if state.session.heading_arr is not None:
        status["heading"] = _to_json_float_or_none(state.session.heading_arr[roi_idx])
    if state.session.refined_success_arr is not None:
        status["refined_success"] = bool(_to_json_scalar(np.asarray(state.session.refined_success_arr[roi_idx], dtype=bool).item()))
    if state.session.usable_arr is not None:
        status["usable_keypoints"] = bool(_to_json_scalar(np.asarray(state.session.usable_arr[roi_idx], dtype=bool).item()))
    if state.session.edit_applied_arr is not None:
        status["edit_applied"] = bool(_to_json_scalar(np.asarray(state.session.edit_applied_arr[roi_idx], dtype=bool).item()))

    return {
        "ok": True,
        "position": int(state.position),
        "total": total,
        "roi_idx": roi_idx,
        "frame_idx": frame_idx,
        "labels": list(state.session.keypoint_labels),
        "reason": reason,
        "status": status,
        "refined_run": str(state.session.refined_run),
        "crop_run": str(state.session.crop_run),
        "filter_mode": state.filter_mode,
        "search": state.search,
        "review_status": review_status_payload,
        "auto_advance_on_save": bool(state.auto_advance_on_save),
        **dataset_context,
    }

def _clamp_position(total: int, value: int) -> int:
    if total <= 0:
        return 0
    if value < 0:
        return 0
    if value >= total:
        return total - 1
    return value


def _format_error(error: str, *, details: Optional[str] = None, status: int = HTTPStatus.BAD_REQUEST) -> dict[str, object]:
    payload: dict[str, object] = {
        "ok": False,
        "error": error,
    }
    if details:
        payload["details"] = details
    payload["status"] = status
    return payload


def _read_json_body(handler: BaseHTTPRequestHandler) -> tuple[dict[str, object], Optional[dict[str, object]]]:
    content_length = int(handler.headers.get("Content-Length", "0") or 0)
    if content_length <= 0:
        return {}, None
    raw = handler.rfile.read(content_length)
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # pragma: no cover - exercised by runtime path
        return {}, {"error": "invalid_json", "details": str(exc)}
    if isinstance(body, dict):
        return body, None
    return {}, {"error": "invalid_payload", "details": "Expected JSON object."}


def _refresh_queue(state: _ServerState, backend_module: "Any") -> None:
    state.session.failures = backend_module.filter_review_rois(
        state.session,
        filter_mode=state.filter_mode,
        search=state.search,
    )
    state.position = _clamp_position(int(state.session.failures.size), state.position)


def _advance_after_action(state: _ServerState, *, advance: bool) -> None:
    if not advance:
        return
    total = int(state.session.failures.size)
    if total <= 0:
        state.position = 0
    elif state.position < total - 1:
        state.position += 1


def _jump_to_requested_position(state: _ServerState, backend_module: "Any", data: Mapping[str, object]) -> tuple[bool, str]:
    roi_value = data.get("roi_idx")
    frame_value = data.get("frame_idx")
    roi_idx: Optional[int] = None
    frame_idx: Optional[int] = None
    try:
        if roi_value is not None and str(roi_value).strip() != "":
            roi_idx = int(roi_value)
        if frame_value is not None and str(frame_value).strip() != "":
            frame_idx = int(frame_value)
    except (TypeError, ValueError):
        return False, "roi_idx/frame_idx must be integers."

    position = backend_module.find_queue_position(state.session, roi_idx=roi_idx, frame_idx=frame_idx)
    if position is not None:
        state.position = int(position)
        return True, "ok"

    include_missing = bool(data.get("include_if_missing", True))
    if not include_missing:
        return False, "requested ROI/frame is not in the current queue."

    total_rows = int(state.session.frame_indices.shape[0])
    resolved_roi: Optional[int] = None
    if roi_idx is not None and 0 <= roi_idx < total_rows:
        resolved_roi = int(roi_idx)
    elif frame_idx is not None:
        matches = np.where(np.asarray(state.session.frame_indices, dtype=np.int64) == int(frame_idx))[0]
        if matches.size > 0:
            resolved_roi = int(matches[0])
    if resolved_roi is None:
        return False, "requested ROI/frame does not exist."

    state.session.failures = np.asarray(
        sorted(set(np.asarray(state.session.failures, dtype="i4").tolist() + [resolved_roi])),
        dtype="i4",
    )
    position = backend_module.find_queue_position(state.session, roi_idx=resolved_roi)
    state.position = _clamp_position(int(state.session.failures.size), int(position or 0))
    return True, "ok"


def _resolve_review_session_for_state(
    state: _ServerState,
    backend_module: "Any",
    zarr_path: str,
) -> "keypoint_review_backend.ReviewSession":  # type: ignore[name-defined]
    return backend_module.resolve_review_session(
        zarr_path,
        refined_run=state.refined_run_arg,
        crop_run=state.crop_run_arg,
        include_all=state.include_all_arg,
        target_frames=state.target_frames,
        target_roi_indices=state.target_roi_indices,
    )


def _switch_training_dataset(
    state: _ServerState,
    backend_module: "Any",
    dataset_id: str,
) -> dict[str, object]:
    if not state.registry_path:
        raise ValueError("Registry mode is not enabled for this server.")
    dataset = _find_training_dataset(state.registry_path, dataset_id)
    zarr_path = dataset.get("zarr_path")
    if not zarr_path:
        raise ValueError(f"Registry dataset has no zarr_path: {dataset_id}")

    next_session = _resolve_review_session_for_state(state, backend_module, str(zarr_path))

    if state.lock_enabled:
        if state.review_lock is not None:
            state.review_lock.release()
            state.review_lock = None
        state.review_lock = _acquire_review_lock(str(zarr_path), next_session.refined_run, force=state.force_lock)

    state.session = next_session
    state.dataset_id = str(dataset.get("dataset_id") or dataset_id)
    state.dataset = dataset
    state.position = 0
    state.search = ""
    state.filter_mode = "all" if state.include_all_arg else "failed"
    _refresh_queue(state, backend_module)
    return dataset


def _build_handler(
    state: _ServerState,
    static_root: Path,
    backend_module: "Any",
) -> type[BaseHTTPRequestHandler]:
    class KeypointReviewRequestHandler(BaseHTTPRequestHandler):
        server_version = "PaletteKeypointReviewWeb/0.1"
        sys_version = ""

        def _write_bytes(
            self,
            payload: bytes,
            *,
            status: HTTPStatus = HTTPStatus.OK,
            content_type: str = "application/octet-stream",
        ) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _write_json(self, payload: object, *, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            self._write_bytes(data, status=status, content_type="application/json; charset=utf-8")

        def _write_not_found(self, message: str = "Not found") -> None:
            self._write_json(_format_error(message, status=HTTPStatus.NOT_FOUND), status=HTTPStatus.NOT_FOUND)

        def _write_bad_request(self, message: str, details: Optional[str] = None) -> None:
            self._write_json(_format_error(message, details=details, status=HTTPStatus.BAD_REQUEST), status=HTTPStatus.BAD_REQUEST)

        def _serve_static(self, relative_path: str) -> None:
            candidate = (static_root / relative_path).resolve()
            if not candidate.is_relative_to(static_root) or not candidate.is_file():
                self._write_not_found("Static asset not found.")
                return
            suffix = candidate.suffix.lower()
            content_type = _CONTENT_TYPES.get(suffix, "application/octet-stream")
            self._write_bytes(candidate.read_bytes(), content_type=content_type)

        def _handle_api_get(self, path: str, query_params: Mapping[str, list[str]]) -> bool:
            del query_params
            if path == "/api/state":
                self._write_json({"ok": True, "state": _build_state_payload(state, backend_module)})
                return True

            if path == "/api/registry/datasets":
                if not state.registry_path:
                    self._write_json({"ok": True, "enabled": False, "datasets": []})
                    return True
                try:
                    datasets = _list_training_datasets(
                        state.registry_path,
                        limit=state.dataset_limit,
                        review_filter=state.dataset_review_filter,
                    )
                except Exception as exc:
                    self._write_json(
                        _format_error("registry_dataset_list_error", details=str(exc), status=HTTPStatus.BAD_REQUEST),
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return True
                self._write_json(
                    {
                        "ok": True,
                        "enabled": True,
                        "registry_path": state.registry_path,
                        "dataset_id": state.dataset_id,
                        "review_filter": state.dataset_review_filter,
                        "datasets": datasets,
                    }
                )
                return True

            if path == "/api/roi/current":
                try:
                    payload = backend_module.load_roi_payload(state.session, position=state.position)
                    payload["state"] = _build_state_payload(state, backend_module)
                    payload["ok"] = True
                except Exception as exc:
                    self._write_json(
                        _format_error("roi_load_error", details=str(exc), status=HTTPStatus.NOT_FOUND),
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return True
                self._write_json(payload)
                return True

            return False

        def _handle_api_post(self, path: str) -> bool:
            data, payload_error = _read_json_body(self)
            if payload_error is not None:
                self._write_json({"ok": False, **payload_error}, status=HTTPStatus.BAD_REQUEST)
                return True

            if path == "/api/roi/current/save":
                if "points" not in data:
                    self._write_bad_request("payload_validation", "Missing `points` field.")
                    return True
                advance = bool(data.get("advance", state.auto_advance_on_save))
                try:
                    result = backend_module.save_roi_correction(
                        state.session,
                        position=state.position,
                        points=data["points"],
                    )
                except Exception as exc:
                    self._write_json(_format_error("save_error", details=str(exc), status=HTTPStatus.BAD_REQUEST), status=HTTPStatus.BAD_REQUEST)
                    return True

                _advance_after_action(state, advance=advance)
                self._write_json({"ok": True, "result": result, "state": _build_state_payload(state, backend_module)})
                return True

            if path == "/api/roi/current/action":
                action = str(data.get("action") or "").strip().lower()
                advance = bool(data.get("advance", state.auto_advance_on_save))
                try:
                    if action in {"x", "no_keypoints", "mark_no_keypoints"}:
                        result = backend_module.mark_no_keypoints(state.session, position=state.position)
                        if state.filter_mode == "failed":
                            _refresh_queue(state, backend_module)
                    elif action in {"d", "detection_issue", "mark_detection_issue"}:
                        result = backend_module.mark_detection_issue(state.session, position=state.position)
                        if state.filter_mode == "failed":
                            _refresh_queue(state, backend_module)
                    elif action in {"c", "clear", "clear_failure_label"}:
                        result = backend_module.clear_failure_label(state.session, position=state.position)
                        if state.filter_mode == "failed":
                            _refresh_queue(state, backend_module)
                    elif action in {"b", "flag", "flag_followup"}:
                        result = backend_module.flag_followup_frame(
                            state.session,
                            position=state.position,
                            flag_path=state.frame_flag_file,
                        )
                    else:
                        self._write_bad_request("payload_validation", f"Unsupported action: {action or '<empty>'}.")
                        return True
                except Exception as exc:
                    self._write_json(_format_error("action_error", details=str(exc), status=HTTPStatus.BAD_REQUEST), status=HTTPStatus.BAD_REQUEST)
                    return True
                _advance_after_action(state, advance=advance)
                self._write_json({"ok": True, "result": result, "state": _build_state_payload(state, backend_module)})
                return True

            if path == "/api/review_status":
                requested_state = str(data.get("state") or "").strip().lower()
                if requested_state not in {"approved", "pending", "rejected", "needs_review"}:
                    self._write_bad_request("payload_validation", "`state` must be approved, pending, rejected, or needs_review.")
                    return True
                try:
                    result = backend_module.apply_review_status(
                        state.session,
                        state=requested_state,
                        method=str(data.get("method") or state.review_method),
                        intended_use=data.get("intended_use") or state.review_intended_use,
                        reviewer=data.get("reviewer") or state.reviewer,
                        notes=data.get("notes") or state.review_notes,
                    )
                except Exception as exc:
                    self._write_json(_format_error("review_status_error", details=str(exc), status=HTTPStatus.BAD_REQUEST), status=HTTPStatus.BAD_REQUEST)
                    return True
                self._write_json({"ok": True, "result": result, "state": _build_state_payload(state, backend_module)})
                return True

            if path == "/api/filter":
                state.filter_mode = str(data.get("filter_mode") or data.get("mode") or "failed").strip().lower()
                state.search = str(data.get("search") or "").strip()
                state.position = 0
                try:
                    _refresh_queue(state, backend_module)
                except Exception as exc:
                    self._write_json(_format_error("filter_error", details=str(exc), status=HTTPStatus.BAD_REQUEST), status=HTTPStatus.BAD_REQUEST)
                    return True
                self._write_json({"ok": True, "state": _build_state_payload(state, backend_module)})
                return True

            if path == "/api/jump":
                ok, message = _jump_to_requested_position(state, backend_module, data)
                if not ok:
                    self._write_json(_format_error("jump_error", details=message, status=HTTPStatus.BAD_REQUEST), status=HTTPStatus.BAD_REQUEST)
                    return True
                self._write_json({"ok": True, "state": _build_state_payload(state, backend_module)})
                return True

            if path == "/api/nav":
                delta = data.get("delta")
                if delta is None:
                    direction = str(data.get("direction", "")).strip().lower()
                    if direction in {"next", "n", "+"}:
                        delta = 1
                    elif direction in {"prev", "p", "-"}:
                        delta = -1
                    else:
                        delta = 0
                try:
                    delta_i = int(delta)
                except (TypeError, ValueError):
                    self._write_bad_request("payload_validation", "`delta` must be an integer.")
                    return True

                old_pos = state.position
                total = int(state.session.failures.size)
                state.position = _clamp_position(total, old_pos + delta_i)
                moved = state.position != old_pos
                self._write_json({"ok": True, "moved": moved, "state": _build_state_payload(state, backend_module)})
                return True

            if path == "/api/registry/select":
                dataset_id = str(data.get("dataset_id") or "").strip()
                if not dataset_id:
                    self._write_bad_request("payload_validation", "Missing `dataset_id`.")
                    return True
                try:
                    dataset = _switch_training_dataset(state, backend_module, dataset_id)
                except Exception as exc:
                    self._write_json(
                        _format_error("registry_dataset_select_error", details=str(exc), status=HTTPStatus.BAD_REQUEST),
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return True
                self._write_json(
                    {
                        "ok": True,
                        "dataset": dataset,
                        "state": _build_state_payload(state, backend_module),
                    }
                )
                return True

            return False

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path or "/"
            query_params = parse_qs(parsed.query, keep_blank_values=False)

            if path == "/":
                self._serve_static("index.html")
                return
            if path in {"/app.js", "/style.css"}:
                relative = path.lstrip("/")
                self._serve_static(relative)
                return
            if path == "/healthz":
                self._write_json({"ok": True})
                return
            if path.startswith("/api/"):
                try:
                    handled = self._handle_api_get(path, query_params)
                except Exception as exc:
                    self._write_json(_format_error("internal_error", details=str(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR), status=HTTPStatus.INTERNAL_SERVER_ERROR)
                    return
                if handled:
                    return
            self._write_not_found()

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path or "/"
            if path.startswith("/api/"):
                try:
                    handled = self._handle_api_post(path)
                except Exception as exc:
                    self._write_json(_format_error("internal_error", details=str(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR), status=HTTPStatus.INTERNAL_SERVER_ERROR)
                    return
                if handled:
                    return
            self._write_not_found()

        def do_HEAD(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path or "/"
            if path in {"/", "/healthz"} or path.startswith("/api/") or path in {"/app.js", "/style.css"}:
                self.send_response(HTTPStatus.OK)
                self.end_headers()
                return
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

    return KeypointReviewRequestHandler


def run_server(config: _ServerConfig) -> int:
    from . import keypoint_review_backend as backend_module

    zarr_path, dataset_id, dataset = _resolve_initial_dataset(config)
    session = backend_module.resolve_review_session(
        zarr_path,
        refined_run=config.refined_run,
        crop_run=config.crop_run,
        include_all=config.include_all,
        target_frames=config.target_frames,
        target_roi_indices=config.target_roi_indices,
    )
    state = _ServerState(
        session=session,
        position=0,
        filter_mode="all" if config.include_all else "failed",
        search="",
        review_method=config.review_method,
        review_intended_use=config.review_intended_use,
        reviewer=config.reviewer,
        review_notes=config.review_notes,
        frame_flag_file=config.frame_flag_file,
        auto_advance_on_save=bool(config.auto_advance_on_save),
        registry_path=config.registry_path,
        dataset_id=dataset_id,
        dataset=dataset,
        dataset_limit=int(config.dataset_limit),
        dataset_review_filter=str(config.dataset_review_filter),
        refined_run_arg=config.refined_run,
        crop_run_arg=config.crop_run,
        include_all_arg=bool(config.include_all),
        target_frames=config.target_frames,
        target_roi_indices=config.target_roi_indices,
        lock_enabled=bool(config.lock),
        force_lock=bool(config.force_lock),
    )

    static_root = Path(__file__).resolve().parent / "keypoint_review_web" / "static"
    if not static_root.is_dir():
        raise RuntimeError(f"Missing static assets: {static_root}")

    state.review_lock = (
        _acquire_review_lock(zarr_path, session.refined_run, force=bool(config.force_lock))
        if config.lock
        else None
    )
    handler = _build_handler(state=state, static_root=static_root, backend_module=backend_module)
    server = ThreadingHTTPServer((config.host, int(config.port)), handler)
    print("Keypoint review web")
    print(f"  zarr:        {zarr_path}")
    if config.registry_path:
        print(f"  registry:    {config.registry_path}")
    if dataset_id:
        print(f"  dataset:     {dataset_id}")
    if config.registry_path:
        print(f"  dataset filter: {config.dataset_review_filter}")
    print(f"  refined run: {session.refined_run}")
    print(f"  crop run:    {session.crop_run}")
    print(f"  server:      http://{config.host}:{int(config.port)}")
    print(f"  ROI count:   {int(session.failures.size)}")
    print(f"  filter:      {state.filter_mode}")
    if state.review_lock is not None:
        print(f"  lock:        {state.review_lock.path}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if state.review_lock is not None:
            state.review_lock.release()
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser UI for manual keypoint corrections.")
    parser.add_argument("zarr_path", nargs="?", help="Path to Palette Zarr directory.")
    parser.add_argument("--manual", action="store_true", help="Compatibility flag; manual mode is always enabled.")
    parser.add_argument(
        "--registry",
        nargs="?",
        const="auto",
        dest="registry_path",
        help=(
            "Enable registry-backed training dataset dropdown. "
            "Optionally pass the registry SQLite path; without a value this uses PALETTE_REGISTRY_PATH/config/default."
        ),
    )
    parser.add_argument("--dataset-id", help="Initial training dataset_id to open from the registry.")
    parser.add_argument("--dataset-limit", type=int, default=500, help="Maximum training datasets to list in the dropdown.")
    parser.add_argument(
        "--dataset-review-filter",
        default="needs_review",
        choices=["needs_review", "all", "approved"],
        help="Registry dropdown filter for keypoint-reviewable training datasets.",
    )
    parser.add_argument("--refined-run", dest="refined_run", help="Refined keypoint run to review (defaults to latest).")
    parser.add_argument("--crop-run", dest="crop_run", help="Crop run to source ROI images from (defaults to source run).")
    parser.add_argument("--all", action="store_true", help="Review all keypoints instead of only failed keypoints.")
    parser.add_argument(
        "--frames",
        help="Comma/space-separated frame indices to review.",
    )
    parser.add_argument(
        "--roi",
        action="append",
        default=[],
        type=int,
        help="Specific ROI index to review. Repeatable.",
    )
    parser.add_argument("--review-method", default="manual", choices=["manual", "algorithmic", "hybrid", "spotcheck"], help="Review method label.")
    parser.add_argument("--review-intended-use", choices=["training", "full_recording"], help="Review intended-use label.")
    parser.add_argument("--reviewer", help="Reviewer name (defaults to existing metadata behavior).")
    parser.add_argument("--review-notes", help="Optional review notes stored with review status.")
    parser.add_argument("--frame-flag-file", help="JSON file for b/follow-up frame flags.")
    parser.add_argument("--auto-advance-on-save", action="store_true", help="Advance to the next queued ROI after save/actions.")
    parser.add_argument("--no-lock", action="store_true", help="Disable the per-Zarr/refined-run web review lock.")
    parser.add_argument("--force-lock", action="store_true", help="Overwrite a stale/active review lock.")
    parser.add_argument("--port", type=int, default=8787, help="Port to serve the web UI on.")
    parser.add_argument("--host", default="localhost", help="Host to bind the server to.")
    return parser.parse_args(argv)


def build_server_config(args: argparse.Namespace) -> _ServerConfig:
    target_frames = _parse_index_sequence(args.frames)
    target_roi_indices = _parse_roi_sequence(args.roi)
    registry_path = args.registry_path
    if registry_path == "auto" or (registry_path is None and args.dataset_id):
        registry_path = str(RegistryPaths.from_env(Path.cwd()).path)
    zarr_path = str(args.zarr_path).strip() if args.zarr_path is not None else None
    if not zarr_path:
        zarr_path = None
    return _ServerConfig(
        zarr_path=zarr_path,
        host=args.host,
        port=int(args.port),
        refined_run=args.refined_run,
        crop_run=args.crop_run,
        include_all=bool(args.all),
        target_frames=target_frames,
        target_roi_indices=target_roi_indices,
        review_method=str(args.review_method),
        review_intended_use=args.review_intended_use,
        reviewer=args.reviewer,
        review_notes=args.review_notes,
        frame_flag_file=args.frame_flag_file,
        auto_advance_on_save=bool(args.auto_advance_on_save),
        lock=not bool(args.no_lock),
        force_lock=bool(args.force_lock),
        registry_path=registry_path,
        dataset_id=args.dataset_id,
        dataset_limit=int(args.dataset_limit),
        dataset_review_filter=str(args.dataset_review_filter),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = build_server_config(args)
    return run_server(config)


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    raise SystemExit(main())
