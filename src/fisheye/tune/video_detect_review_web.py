"""Browser video-backed refined detection review server."""

from __future__ import annotations

import argparse
import json
import mimetypes
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    from . import video_detect_review_backend


@dataclass(frozen=True)
class _ServerConfig:
    zarr_path: str
    host: str
    port: int
    collection_id: Optional[str]
    refined_run: Optional[str]
    recording_frame_index: Optional[str]
    review_proxy_manifest: Optional[str]
    editable: bool
    manual_score: float
    manual_class_id: int


@dataclass
class _ServerState:
    session: "video_detect_review_backend.VideoDetectReviewSession"  # type: ignore[name-defined]
    current_frame: int = 0


_CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}
_MEDIA_COPY_CHUNK_BYTES = 1024 * 1024


def _format_error(error: str, *, details: Optional[str] = None, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> dict[str, object]:
    payload: dict[str, object] = {
        "ok": False,
        "error": error,
        "status": int(status),
    }
    if details:
        payload["details"] = details
    return payload


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    raw_len = handler.headers.get("Content-Length")
    try:
        length = int(raw_len or "0")
    except ValueError:
        length = 0
    if length <= 0:
        return {}
    payload = json.loads(handler.rfile.read(length).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON request body must be an object.")
    return payload


def _state_payload(state: _ServerState, backend_module) -> dict[str, object]:
    return {
        "current_frame": int(state.current_frame),
        "summary": backend_module.review_session_summary(state.session),
        "videos": backend_module.video_sources_payload(state.session),
    }


def _parse_frame_from_path(path: str, prefix: str) -> int | None:
    if not path.startswith(prefix):
        return None
    raw = path[len(prefix) :].strip("/")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _parse_save_frame_from_path(path: str) -> int | None:
    prefix = "/api/frame/"
    suffix = "/save"
    if not path.startswith(prefix) or not path.endswith(suffix):
        return None
    raw = path[len(prefix) : -len(suffix)].strip("/")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _parse_range_header(value: str | None, *, file_size: int) -> tuple[int, int] | None:
    if not value:
        return None
    if not value.startswith("bytes="):
        raise ValueError("Only byte ranges are supported.")
    spec = value[len("bytes=") :].split(",", 1)[0].strip()
    if "-" not in spec:
        raise ValueError("Invalid Range header.")
    start_raw, end_raw = spec.split("-", 1)
    if start_raw == "":
        suffix = int(end_raw)
        if suffix <= 0:
            raise ValueError("Invalid suffix byte range.")
        start = max(0, file_size - suffix)
        end = file_size - 1
    else:
        start = int(start_raw)
        end = int(end_raw) if end_raw else file_size - 1
    if start < 0 or end < start or start >= file_size:
        raise ValueError("Unsatisfiable byte range.")
    return start, min(end, file_size - 1)


def _make_handler(state: _ServerState, static_root: Path, backend_module):
    class VideoDetectReviewRequestHandler(BaseHTTPRequestHandler):
        server_version = "PaletteVideoDetectReviewWeb/0.1"
        sys_version = ""

        def _write_bytes(
            self,
            payload: bytes,
            *,
            status: HTTPStatus = HTTPStatus.OK,
            content_type: str = "application/octet-stream",
            extra_headers: Optional[dict[str, str]] = None,
        ) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    return

        def _send_media_headers(
            self,
            *,
            status: HTTPStatus,
            content_type: str,
            content_length: int,
            extra_headers: Optional[dict[str, str]] = None,
        ) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(content_length))
            self.send_header("Accept-Ranges", "bytes")
            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)
            self.end_headers()

        def _stream_file_range(self, path: Path, *, start: int, length: int) -> None:
            remaining = int(length)
            with path.open("rb") as handle:
                handle.seek(start)
                while remaining > 0:
                    chunk = handle.read(min(_MEDIA_COPY_CHUNK_BYTES, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(chunk)

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
            self._write_bytes(
                candidate.read_bytes(),
                content_type=content_type,
                extra_headers={"Cache-Control": "no-store"},
            )

        def _serve_media(self, video_id: str) -> None:
            source = state.session.videos.get(video_id)
            if source is None:
                self._write_not_found("Video source not found.")
                return
            if not source.path.is_file():
                self._write_not_found(f"Video file not found: {source.path}")
                return
            file_size = source.path.stat().st_size
            content_type = mimetypes.guess_type(source.path.name)[0] or "video/mp4"
            try:
                byte_range = _parse_range_header(self.headers.get("Range"), file_size=file_size)
            except Exception as exc:
                self.send_response(int(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE))
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes */{file_size}")
                payload = json.dumps(
                    _format_error(
                        "Invalid Range header.",
                        details=str(exc),
                        status=HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                    ),
                    allow_nan=False,
                ).encode("utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if self.command != "HEAD":
                    try:
                        self.wfile.write(payload)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                return
            start, end = byte_range if byte_range is not None else (0, file_size - 1)
            length = max(0, end - start + 1)
            headers = {}
            if byte_range is not None:
                headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            self._send_media_headers(
                status=HTTPStatus.PARTIAL_CONTENT if byte_range is not None else HTTPStatus.OK,
                content_type=content_type,
                content_length=length,
                extra_headers=headers,
            )
            if self.command != "HEAD":
                self._stream_file_range(source.path, start=start, length=length)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if path in {"", "/"}:
                self._serve_static("index.html")
                return
            if path.startswith("/static/"):
                self._serve_static(path[len("/static/") :])
                return
            if path.startswith("/media/"):
                self._serve_media(path[len("/media/") :].strip("/"))
                return
            if path == "/api/state":
                self._write_json({"ok": True, "state": _state_payload(state, backend_module)})
                return
            if path == "/api/frame/current":
                try:
                    payload = backend_module.load_frame_payload(state.session, state.current_frame)
                    payload["ok"] = True
                    payload["state"] = _state_payload(state, backend_module)
                except Exception as exc:
                    self._write_json(
                        _format_error("frame_load_error", details=str(exc), status=HTTPStatus.NOT_FOUND),
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                self._write_json(payload)
                return
            frame_index = _parse_frame_from_path(path, "/api/frame/")
            if frame_index is not None:
                try:
                    state.current_frame = frame_index
                    payload = backend_module.load_frame_payload(state.session, frame_index)
                    payload["ok"] = True
                    payload["state"] = _state_payload(state, backend_module)
                except Exception as exc:
                    self._write_json(
                        _format_error("frame_load_error", details=str(exc), status=HTTPStatus.NOT_FOUND),
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                self._write_json(payload)
                return
            if path == "/api/search":
                query = parse_qs(parsed.query)
                direction = str((query.get("direction") or ["next"])[0])
                start = int((query.get("start") or [state.current_frame])[0])
                total = len(state.session.frame_records)
                step = -1 if direction == "prev" else 1
                candidate = min(max(0, start + step), max(0, total - 1))
                while 0 <= candidate < total:
                    try:
                        payload = backend_module.load_frame_payload(state.session, candidate)
                    except Exception:
                        candidate += step
                        continue
                    if payload.get("bbox_norm") is None or direction in {"next", "prev"}:
                        state.current_frame = candidate
                        payload["ok"] = True
                        payload["state"] = _state_payload(state, backend_module)
                        self._write_json(payload)
                        return
                    candidate += step
                self._write_json(_format_error("No matching frame found.", status=HTTPStatus.NOT_FOUND), status=HTTPStatus.NOT_FOUND)
                return
            self._write_not_found()

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

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
                    target = body.get("frame")
                    if target is not None:
                        next_frame = int(target)
                    else:
                        next_frame = state.current_frame + delta
                except (TypeError, ValueError):
                    self._write_bad_request("frame/delta must be integers.")
                    return
                total = len(state.session.frame_records)
                state.current_frame = min(max(0, next_frame), max(0, total - 1))
                self._write_json({"ok": True, "state": _state_payload(state, backend_module)})
                return

            save_current = path == "/api/frame/current/save"
            frame_index = _parse_save_frame_from_path(path)
            save_by_frame = frame_index is not None
            if save_current or save_by_frame:
                target_frame = state.current_frame if save_current else int(frame_index)
                try:
                    result = backend_module.apply_manual_edit(
                        state.session,
                        parent_frame_index=target_frame,
                        bbox_norm=body.get("bbox_norm"),
                    )
                    state.current_frame = target_frame
                    if bool(body.get("advance")):
                        total = len(state.session.frame_records)
                        state.current_frame = min(state.current_frame + 1, max(0, total - 1))
                except Exception as exc:
                    self._write_json(
                        _format_error("save_failed", details=str(exc), status=HTTPStatus.BAD_REQUEST),
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                self._write_json({"ok": True, "result": result, "state": _state_payload(state, backend_module)})
                return

            self._write_not_found()

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return VideoDetectReviewRequestHandler


def run_server(config: _ServerConfig) -> int:
    from . import video_detect_review_backend as backend_module

    session = backend_module.resolve_video_detect_review_session(
        config.zarr_path,
        collection_id=config.collection_id,
        refined_run=config.refined_run,
        recording_frame_index=config.recording_frame_index,
        review_proxy_manifest=config.review_proxy_manifest,
        editable=config.editable,
        manual_score=config.manual_score,
        manual_class_id=config.manual_class_id,
    )
    state = _ServerState(session=session, current_frame=0)
    static_root = Path(__file__).resolve().parent / "video_detect_review_web" / "static"
    handler = _make_handler(state, static_root, backend_module)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    summary = backend_module.review_session_summary(session)
    print(
        f"Serving Palette video detect review at http://{config.host}:{config.port} "
        f"mode={summary['mode']} frames={summary['total_frames']} videos={summary['video_count']} "
        f"editable={summary['editable']}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping video detect review server.")
    finally:
        server.server_close()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a video-backed Palette detection review UI.")
    parser.add_argument("zarr_path", type=Path, help="Analysis Zarr path")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Use 0.0.0.0 only on trusted networks.")
    parser.add_argument("--port", type=int, default=8790, help="Bind port")
    parser.add_argument("--collection-id", default=None, help="Finalized clipped refined-detect collection id")
    parser.add_argument("--refined-run", default=None, help="Traditional refined detect run name")
    parser.add_argument("--recording-frame-index", type=Path, default=None, help="Override recording_frame_index.parquet")
    parser.add_argument("--review-proxy-manifest", type=Path, default=None, help="Use derived review-proxy videos for clipped media.")
    parser.add_argument("--edit", action="store_true", help="Allow saving bbox edits back into the analysis Zarr")
    parser.add_argument("--manual-score", type=float, default=1.0, help="Confidence score for manually added boxes")
    parser.add_argument("--manual-class-id", type=int, default=0, help="Class id for manually added boxes")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    return run_server(
        _ServerConfig(
            zarr_path=str(args.zarr_path),
            host=str(args.host),
            port=int(args.port),
            collection_id=args.collection_id,
            refined_run=args.refined_run,
            recording_frame_index=str(args.recording_frame_index) if args.recording_frame_index else None,
            review_proxy_manifest=str(args.review_proxy_manifest) if args.review_proxy_manifest else None,
            editable=bool(args.edit),
            manual_score=float(args.manual_score),
            manual_class_id=int(args.manual_class_id),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
