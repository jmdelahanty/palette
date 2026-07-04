"""Browser-based refined detection review UI."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence
from urllib.parse import urlparse

if TYPE_CHECKING:
    from . import detect_review_backend


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


@dataclass(frozen=True)
class _ServerConfig:
    zarr_path: str
    host: str
    port: int
    refined_run: Optional[str]
    include_all: bool
    target_frames: Optional[list[int]]
    max_items: Optional[int]
    manual_score: float
    manual_class_id: int


@dataclass
class _ServerState:
    session: "detect_review_backend.DetectReviewSession"  # type: ignore[name-defined]
    position: int = 0


_CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}


def _format_error(error: str, *, details: Optional[str] = None, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> dict[str, object]:
    payload: dict[str, object] = {
        "ok": False,
        "error": error,
        "status": int(status),
    }
    if details:
        payload["details"] = details
    return payload


def _state_payload(state: _ServerState, backend_module) -> dict[str, object]:
    summary = backend_module.review_session_summary(state.session)
    return {
        "position": int(state.position),
        "summary": summary,
        "refined_run": state.session.refined_run_name,
        "zarr_path": state.session.zarr_path,
        "total": int(state.session.review_rows.shape[0]),
        "frame_width": int(state.session.width),
        "frame_height": int(state.session.height),
    }


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    raw_len = handler.headers.get("Content-Length")
    try:
        length = int(raw_len or "0")
    except ValueError:
        length = 0
    if length <= 0:
        return {}
    data = handler.rfile.read(length)
    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON request body must be an object.")
    return payload


def _make_handler(state: _ServerState, static_root: Path, backend_module):
    class DetectReviewRequestHandler(BaseHTTPRequestHandler):
        server_version = "PaletteDetectReviewWeb/0.1"
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
            content_type = _CONTENT_TYPES.get(candidate.suffix.lower(), "application/octet-stream")
            self._write_bytes(candidate.read_bytes(), content_type=content_type)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"", "/"}:
                self._serve_static("index.html")
                return
            if path.startswith("/static/"):
                self._serve_static(path[len("/static/") :])
                return
            if path == "/api/state":
                self._write_json({"ok": True, "state": _state_payload(state, backend_module)})
                return
            if path == "/api/frame/current":
                try:
                    payload = dict(backend_module.load_frame_payload(state.session, position=state.position))
                    payload["state"] = _state_payload(state, backend_module)
                    payload["ok"] = True
                except Exception as exc:
                    self._write_json(
                        _format_error("frame_load_error", details=str(exc), status=HTTPStatus.NOT_FOUND),
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                self._write_json(payload)
                return
            self._write_not_found()

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                body = _read_json_body(self)
            except Exception as exc:
                self._write_bad_request("Invalid JSON body.", details=str(exc))
                return

            if path == "/api/nav":
                try:
                    delta = int(body.get("delta") or 0)
                except (TypeError, ValueError):
                    self._write_bad_request("delta must be an integer.")
                    return
                old = state.position
                total = int(state.session.review_rows.shape[0])
                state.position = min(max(0, state.position + delta), max(0, total - 1))
                self._write_json(
                    {
                        "ok": True,
                        "moved": state.position != old,
                        "state": _state_payload(state, backend_module),
                    }
                )
                return

            if path == "/api/frame/current/save":
                try:
                    result = backend_module.apply_manual_edit(
                        state.session,
                        position=state.position,
                        bbox_norm=body.get("bbox_norm"),
                    )
                    if bool(body.get("advance")):
                        total = int(state.session.review_rows.shape[0])
                        state.position = min(state.position + 1, max(0, total - 1))
                except Exception as exc:
                    self._write_json(
                        _format_error("save_failed", details=str(exc), status=HTTPStatus.BAD_REQUEST),
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                self._write_json({"ok": True, "result": result, "state": _state_payload(state, backend_module)})
                return

            if path == "/api/review_status":
                try:
                    result = backend_module.apply_review_status(
                        state.session,
                        state=str(body.get("state") or "approved"),
                        method=str(body.get("method") or "manual"),
                        intended_use=str(body.get("intended_use") or "training"),
                        reviewer=str(body["reviewer"]) if body.get("reviewer") is not None else None,
                        notes=str(body["notes"]) if body.get("notes") is not None else None,
                    )
                except Exception as exc:
                    self._write_json(
                        _format_error("review_status_failed", details=str(exc), status=HTTPStatus.BAD_REQUEST),
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                status = HTTPStatus.OK if bool(result.get("changed")) else HTTPStatus.CONFLICT
                self._write_json({"ok": bool(result.get("changed")), "result": result, "state": _state_payload(state, backend_module)}, status=status)
                return

            self._write_not_found()

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return DetectReviewRequestHandler


def run_server(config: _ServerConfig) -> int:
    from . import detect_review_backend as backend_module

    session = backend_module.resolve_review_context(
        config.zarr_path,
        refined_run=config.refined_run,
        include_all=config.include_all,
        target_frames=config.target_frames,
        max_items=config.max_items,
        manual_score=config.manual_score,
        manual_class_id=config.manual_class_id,
    )
    state = _ServerState(session=session)
    static_root = Path(__file__).resolve().parent / "detect_review_web" / "static"
    handler = _make_handler(state, static_root, backend_module)
    server = ThreadingHTTPServer((config.host, int(config.port)), handler)
    url_host = "localhost" if config.host in {"0.0.0.0", "::"} else config.host
    print(f"Detection review web UI: http://{url_host}:{config.port}")
    print(f"zarr={config.zarr_path}")
    print(f"refined_run={session.refined_run_name} reviewable={int(session.review_rows.shape[0])}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", help="Training/analysis Zarr containing raw_video and refined_detect_runs.")
    parser.add_argument("--manual", action="store_true", help="Accepted for parity with detect_review; web mode is manual.")
    parser.add_argument("--refined-run", default=None, help="Refined detect run to edit; defaults to refined_detect_runs.latest.")
    parser.add_argument("--all", dest="include_all", action="store_true", help="Review all frames instead of missing/filtered frames only.")
    parser.add_argument("--frames", default=None, help="Comma/space-separated frame indices to review.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum review items to load.")
    parser.add_argument("--manual-score", type=float, default=1.0, help="Confidence score assigned to manual boxes.")
    parser.add_argument("--manual-class-id", type=int, default=0, help="Class id assigned to manual boxes.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8788, help="Bind port.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = _ServerConfig(
        zarr_path=args.zarr_path,
        host=args.host,
        port=int(args.port),
        refined_run=args.refined_run,
        include_all=bool(args.include_all),
        target_frames=_parse_index_sequence(args.frames),
        max_items=args.max_frames,
        manual_score=float(args.manual_score),
        manual_class_id=int(args.manual_class_id),
    )
    return run_server(config)


if __name__ == "__main__":
    raise SystemExit(main())
