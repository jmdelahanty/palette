from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import numpy as np


_SCHEMA_ID = "palette.playground.fixed_video_roi.v1"
_MAX_REQUEST_BYTES = 64 * 1024


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    frame_count: int
    fps: float


def _probe_video(path: Path) -> VideoMetadata:
    import cv2

    source = Path(path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {source}")
    try:
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if width < 1 or height < 1 or frame_count < 1:
        raise ValueError(
            f"invalid video geometry/count for {source}: "
            f"width={width}, height={height}, frames={frame_count}"
        )
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"invalid video frame rate for {source}: {fps}")
    return VideoMetadata(width=width, height=height, frame_count=frame_count, fps=fps)


def _read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    import cv2

    source = Path(path)
    index = int(frame_index)
    if index < 0:
        raise ValueError("frame_index cannot be negative")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {source}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ValueError(f"could not decode frame {index} from {source}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"decoded frame {index} has unsupported shape {frame.shape}")
    return np.asarray(frame, dtype=np.uint8)


def _parse_roi(value: str) -> tuple[int, int, int, int]:
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x,y,width,height")
    try:
        numbers = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("ROI values must be integers") from exc
    return numbers


def _validate_roi(
    value: Any,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("roi_xywh must contain x, y, width, height")
    if any(isinstance(item, bool) for item in value):
        raise ValueError("roi_xywh values must be integer pixels")
    try:
        numeric = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("roi_xywh values must be finite numbers") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("roi_xywh values must be finite")
    rounded = tuple(int(round(item)) for item in numeric)
    if any(abs(item - integer) > 1e-6 for item, integer in zip(numeric, rounded)):
        raise ValueError("roi_xywh values must resolve to whole source pixels")
    x, y, box_width, box_height = rounded
    if x < 0 or y < 0:
        raise ValueError("ROI origin cannot be negative")
    if box_width < 1 or box_height < 1:
        raise ValueError("ROI width and height must be positive")
    if x + box_width > int(width) or y + box_height > int(height):
        raise ValueError(
            f"ROI {(x, y, box_width, box_height)} exceeds "
            f"frame bounds {(int(width), int(height))}"
        )
    return x, y, box_width, box_height


def _validate_frame_index(value: Any, *, frame_count: int) -> int:
    if isinstance(value, bool):
        raise ValueError("frame_index must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("frame_index must be an integer") from exc
    index = int(round(numeric))
    if not np.isfinite(numeric) or abs(numeric - index) > 1e-6:
        raise ValueError("frame_index must be an integer")
    if not 0 <= index < int(frame_count):
        raise ValueError(
            f"frame_index {index} is outside 0..{int(frame_count) - 1}"
        )
    return index


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _encode_png(frame_bgr: np.ndarray) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(".png", np.asarray(frame_bgr, dtype=np.uint8))
    if not ok:
        raise RuntimeError("OpenCV could not encode frame as PNG")
    return bytes(encoded)


def _selection_payload(
    *,
    video: Path,
    metadata: VideoMetadata,
    frame_index: int,
    roi_xywh: tuple[int, int, int, int],
    preview_output: Path,
) -> dict[str, Any]:
    source = Path(video).resolve()
    stat = source.stat()
    return {
        "schema_id": _SCHEMA_ID,
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_space": "source_video_frame_pixels",
        "roi_xywh": [int(value) for value in roi_xywh],
        "frame_index": int(frame_index),
        "frame_shape_hw": [int(metadata.height), int(metadata.width)],
        "video_frame_count": int(metadata.frame_count),
        "video_fps": float(metadata.fps),
        "source_video": str(source),
        "source_video_size_bytes": int(stat.st_size),
        "source_video_mtime_ns": int(stat.st_mtime_ns),
        "preview_png": str(Path(preview_output).resolve()),
        "measurement_note": (
            "The ROI is fixed in source top-camera frame pixels and must not be "
            "interpreted as a fish-attached or stabilized coordinate surface."
        ),
    }


def _preview_png(frame_bgr: np.ndarray, roi_xywh: tuple[int, int, int, int]) -> bytes:
    import cv2

    preview = np.asarray(frame_bgr, dtype=np.uint8).copy()
    x, y, width, height = roi_xywh
    cv2.rectangle(
        preview,
        (int(x), int(y)),
        (int(x + width - 1), int(y + height - 1)),
        (0, 0, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return _encode_png(preview)


@dataclass
class SelectorState:
    video: Path
    output_json: Path
    preview_output: Path
    metadata: VideoMetadata
    initial_frame_index: int
    initial_roi: tuple[int, int, int, int] | None
    frame_reader: Callable[[Path, int], np.ndarray] = _read_video_frame

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def state_payload(self) -> dict[str, Any]:
        return {
            "schema_id": _SCHEMA_ID,
            "video_name": self.video.name,
            "source_video": str(self.video.resolve()),
            "width": int(self.metadata.width),
            "height": int(self.metadata.height),
            "frame_count": int(self.metadata.frame_count),
            "fps": float(self.metadata.fps),
            "initial_frame_index": int(self.initial_frame_index),
            "initial_roi_xywh": (
                list(self.initial_roi) if self.initial_roi is not None else None
            ),
            "output_json": str(self.output_json.resolve()),
            "preview_png": str(self.preview_output.resolve()),
        }

    def frame_png(self, frame_index: int) -> bytes:
        index = _validate_frame_index(
            frame_index, frame_count=int(self.metadata.frame_count)
        )
        frame = self.frame_reader(self.video, index)
        expected = (int(self.metadata.height), int(self.metadata.width))
        if tuple(frame.shape[:2]) != expected:
            raise ValueError(
                f"decoded frame shape {tuple(frame.shape[:2])} does not match {expected}"
            )
        return _encode_png(frame)

    def save(self, request: dict[str, Any]) -> dict[str, Any]:
        index = _validate_frame_index(
            request.get("frame_index"), frame_count=int(self.metadata.frame_count)
        )
        roi = _validate_roi(
            request.get("roi_xywh"),
            width=int(self.metadata.width),
            height=int(self.metadata.height),
        )
        frame = self.frame_reader(self.video, index)
        if tuple(frame.shape[:2]) != (
            int(self.metadata.height),
            int(self.metadata.width),
        ):
            raise ValueError("decoded frame shape changed while saving ROI")
        payload = _selection_payload(
            video=self.video,
            metadata=self.metadata,
            frame_index=index,
            roi_xywh=roi,
            preview_output=self.preview_output,
        )
        json_bytes = (
            json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        preview_bytes = _preview_png(frame, roi)
        with self._lock:
            _atomic_write_bytes(self.preview_output, preview_bytes)
            _atomic_write_bytes(self.output_json, json_bytes)
            self.initial_frame_index = index
            self.initial_roi = roi
        return payload


_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Embedded Heart ROI</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f6f7;
      color: #1d2529;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-width: 320px; min-height: 100vh; }
    header {
      min-height: 58px; padding: 10px 18px; border-bottom: 1px solid #cbd2d6;
      background: #ffffff; display: flex; align-items: center; gap: 14px;
    }
    header h1 { margin: 0; font-size: 18px; font-weight: 680; letter-spacing: 0; }
    #videoName { color: #59666d; font-size: 13px; overflow-wrap: anywhere; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 278px; min-height: calc(100vh - 58px); }
    .viewer {
      min-width: 0; padding: 16px; display: flex; align-items: flex-start;
      justify-content: center; overflow: auto; background: #252b2e;
    }
    canvas {
      display: block; max-width: 100%; max-height: calc(100vh - 90px);
      width: auto; height: auto; background: #000000; cursor: crosshair;
      touch-action: none; box-shadow: 0 1px 5px rgba(0, 0, 0, 0.35);
    }
    aside { padding: 16px; border-left: 1px solid #cbd2d6; background: #ffffff; }
    .section { padding: 0 0 18px; margin: 0 0 18px; border-bottom: 1px solid #e1e5e7; }
    .section:last-child { border-bottom: 0; }
    h2 { margin: 0 0 12px; font-size: 13px; font-weight: 700; text-transform: uppercase; color: #47545a; }
    label { display: block; margin-bottom: 5px; font-size: 12px; font-weight: 650; color: #455158; }
    input[type="number"] {
      width: 100%; height: 36px; border: 1px solid #aeb8bd; border-radius: 4px;
      padding: 6px 8px; background: #ffffff; color: #152025; font: inherit;
    }
    input[type="number"]:focus { outline: 2px solid #197e85; outline-offset: 1px; }
    .row { display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: end; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    button {
      min-height: 36px; border: 1px solid #96a2a8; border-radius: 4px; padding: 7px 11px;
      background: #ffffff; color: #1c272c; font: inherit; font-size: 13px; font-weight: 650;
      cursor: pointer;
    }
    button:hover { background: #edf1f2; }
    button:focus-visible { outline: 2px solid #197e85; outline-offset: 1px; }
    button.primary { width: 100%; background: #b4232f; border-color: #941c26; color: #ffffff; }
    button.primary:hover { background: #941c26; }
    button:disabled { cursor: not-allowed; opacity: 0.45; }
    .buttonRow { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
    .meta { display: grid; grid-template-columns: auto 1fr; gap: 6px 10px; margin: 0; font-size: 12px; }
    .meta dt { color: #69767c; }
    .meta dd { margin: 0; text-align: right; overflow-wrap: anywhere; }
    #status { min-height: 36px; margin-top: 10px; font-size: 12px; line-height: 1.45; color: #526068; overflow-wrap: anywhere; }
    #status.error { color: #a31825; }
    #status.saved { color: #18724b; }
    @media (max-width: 760px) {
      main { grid-template-columns: 1fr; }
      aside { border-left: 0; border-top: 1px solid #cbd2d6; }
      .viewer { min-height: 55vh; }
      canvas { max-height: 70vh; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Embedded Heart ROI</h1>
    <div id="videoName"></div>
  </header>
  <main>
    <section class="viewer"><canvas id="canvas"></canvas></section>
    <aside>
      <div class="section">
        <h2>Reference Frame</h2>
        <div class="row">
          <div><label for="frameIndex">Frame index</label><input id="frameIndex" type="number" min="0" step="1"></div>
          <button id="loadFrame" type="button">Load</button>
        </div>
      </div>
      <div class="section">
        <h2>Source-Pixel Box</h2>
        <div class="grid">
          <div><label for="boxX">X</label><input id="boxX" type="number" min="0" step="1"></div>
          <div><label for="boxY">Y</label><input id="boxY" type="number" min="0" step="1"></div>
          <div><label for="boxW">Width</label><input id="boxW" type="number" min="1" step="1"></div>
          <div><label for="boxH">Height</label><input id="boxH" type="number" min="1" step="1"></div>
        </div>
        <div class="buttonRow">
          <button id="resetBox" type="button">Reset Box</button>
          <button id="fitBox" type="button">Full Frame</button>
        </div>
      </div>
      <div class="section">
        <h2>Video</h2>
        <dl class="meta">
          <dt>Dimensions</dt><dd id="dimensions"></dd>
          <dt>Frames</dt><dd id="frameCount"></dd>
          <dt>Frame rate</dt><dd id="fps"></dd>
        </dl>
      </div>
      <button id="save" class="primary" type="button" disabled>Save ROI</button>
      <div id="status"></div>
    </aside>
  </main>
  <script>
    "use strict";
    const canvas = document.getElementById("canvas");
    const context = canvas.getContext("2d");
    const frameInput = document.getElementById("frameIndex");
    const fields = ["boxX", "boxY", "boxW", "boxH"].map(id => document.getElementById(id));
    const saveButton = document.getElementById("save");
    const statusNode = document.getElementById("status");
    let appState = null;
    let image = new Image();
    let roi = null;
    let savedRoi = null;
    let drawingStart = null;

    function setStatus(message, kind = "") {
      statusNode.textContent = message;
      statusNode.className = kind;
    }

    function normalizeBox(x0, y0, x1, y1) {
      const left = Math.max(0, Math.min(canvas.width - 1, Math.round(Math.min(x0, x1))));
      const top = Math.max(0, Math.min(canvas.height - 1, Math.round(Math.min(y0, y1))));
      const right = Math.max(left + 1, Math.min(canvas.width, Math.round(Math.max(x0, x1))));
      const bottom = Math.max(top + 1, Math.min(canvas.height, Math.round(Math.max(y0, y1))));
      return [left, top, right - left, bottom - top];
    }

    function validBox(box) {
      if (!box || box.length !== 4 || !appState) return false;
      const [x, y, w, h] = box;
      return Number.isInteger(x) && Number.isInteger(y) && Number.isInteger(w) && Number.isInteger(h)
        && x >= 0 && y >= 0 && w > 0 && h > 0
        && x + w <= appState.width && y + h <= appState.height;
    }

    function syncFields() {
      const values = roi || [0, 0, 0, 0];
      fields.forEach((field, index) => { field.value = String(values[index]); });
      saveButton.disabled = !validBox(roi);
    }

    function draw() {
      context.clearRect(0, 0, canvas.width, canvas.height);
      if (image.complete && image.naturalWidth) context.drawImage(image, 0, 0);
      if (!validBox(roi)) return;
      const [x, y, w, h] = roi;
      context.save();
      context.fillStyle = "rgba(0, 0, 0, 0.34)";
      context.fillRect(0, 0, canvas.width, y);
      context.fillRect(0, y + h, canvas.width, canvas.height - y - h);
      context.fillRect(0, y, x, h);
      context.fillRect(x + w, y, canvas.width - x - w, h);
      context.strokeStyle = "#ff3545";
      context.lineWidth = Math.max(2, Math.round(canvas.width / 350));
      context.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
      context.restore();
    }

    function pointerPosition(event) {
      const bounds = canvas.getBoundingClientRect();
      return [
        (event.clientX - bounds.left) * canvas.width / bounds.width,
        (event.clientY - bounds.top) * canvas.height / bounds.height,
      ];
    }

    async function loadFrame() {
      if (!appState) return;
      const index = Math.round(Number(frameInput.value));
      if (!Number.isFinite(index) || index < 0 || index >= appState.frame_count) {
        setStatus(`Frame must be between 0 and ${appState.frame_count - 1}.`, "error");
        return;
      }
      saveButton.disabled = true;
      setStatus("Loading frame...");
      const next = new Image();
      next.onload = () => {
        image = next;
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        syncFields();
        draw();
        setStatus(`Frame ${index} loaded.`);
      };
      next.onerror = () => setStatus("Frame could not be loaded.", "error");
      next.src = `/api/frame?index=${index}&cache=${Date.now()}`;
    }

    async function saveRoi() {
      if (!validBox(roi)) return;
      saveButton.disabled = true;
      setStatus("Saving...");
      try {
        const response = await fetch("/api/save", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({frame_index: Number(frameInput.value), roi_xywh: roi}),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Save failed");
        savedRoi = roi.slice();
        setStatus(`Saved ${result.output_json}`, "saved");
      } catch (error) {
        setStatus(error.message, "error");
      } finally {
        saveButton.disabled = !validBox(roi);
      }
    }

    canvas.addEventListener("pointerdown", event => {
      if (!image.complete) return;
      canvas.setPointerCapture(event.pointerId);
      drawingStart = pointerPosition(event);
      roi = normalizeBox(drawingStart[0], drawingStart[1], drawingStart[0] + 1, drawingStart[1] + 1);
      syncFields(); draw();
    });
    canvas.addEventListener("pointermove", event => {
      if (!drawingStart) return;
      const current = pointerPosition(event);
      roi = normalizeBox(drawingStart[0], drawingStart[1], current[0], current[1]);
      syncFields(); draw();
    });
    canvas.addEventListener("pointerup", () => { drawingStart = null; });
    canvas.addEventListener("pointercancel", () => { drawingStart = null; });

    fields.forEach(field => field.addEventListener("change", () => {
      const candidate = fields.map(item => Math.round(Number(item.value)));
      if (validBox(candidate)) {
        roi = candidate; setStatus("Box updated.");
      } else {
        setStatus("Box is outside the source frame.", "error");
      }
      syncFields(); draw();
    }));
    document.getElementById("loadFrame").addEventListener("click", loadFrame);
    document.getElementById("resetBox").addEventListener("click", () => {
      roi = savedRoi ? savedRoi.slice() : null; syncFields(); draw(); setStatus("Box reset.");
    });
    document.getElementById("fitBox").addEventListener("click", () => {
      roi = [0, 0, appState.width, appState.height]; syncFields(); draw(); setStatus("Full frame selected.");
    });
    saveButton.addEventListener("click", saveRoi);
    window.addEventListener("keydown", event => {
      if (event.key.toLowerCase() === "s" && !event.ctrlKey && !event.metaKey && validBox(roi)) saveRoi();
    });

    fetch("/api/state").then(response => response.json()).then(state => {
      appState = state;
      document.getElementById("videoName").textContent = state.video_name;
      document.getElementById("dimensions").textContent = `${state.width} x ${state.height}`;
      document.getElementById("frameCount").textContent = String(state.frame_count);
      document.getElementById("fps").textContent = `${state.fps.toFixed(3)} fps`;
      frameInput.max = String(state.frame_count - 1);
      frameInput.value = String(state.initial_frame_index);
      roi = state.initial_roi_xywh ? state.initial_roi_xywh.slice() : null;
      savedRoi = roi ? roi.slice() : null;
      syncFields();
      loadFrame();
    }).catch(error => setStatus(error.message, "error"));
  </script>
</body>
</html>
"""


def _handler_class(state: SelectorState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send_bytes(
            self,
            content: bytes,
            *,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def _send_json(
            self,
            payload: dict[str, Any],
            *,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            content = json.dumps(payload, sort_keys=True).encode("utf-8")
            self._send_bytes(
                content,
                content_type="application/json; charset=utf-8",
                status=status,
            )

        def _error(self, error: Exception, status: HTTPStatus) -> None:
            self._send_json({"error": str(error)}, status=status)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_bytes(
                        _HTML.encode("utf-8"),
                        content_type="text/html; charset=utf-8",
                    )
                    return
                if parsed.path == "/api/state":
                    self._send_json(state.state_payload())
                    return
                if parsed.path == "/api/frame":
                    query = parse_qs(parsed.query)
                    raw_index = query.get("index", [state.initial_frame_index])[0]
                    index = _validate_frame_index(
                        raw_index, frame_count=int(state.metadata.frame_count)
                    )
                    self._send_bytes(
                        state.frame_png(index), content_type="image/png"
                    )
                    return
                self._error(FileNotFoundError(parsed.path), HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self._error(exc, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # pragma: no cover - defensive server boundary
                self._error(exc, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/save":
                self._error(FileNotFoundError(parsed.path), HTTPStatus.NOT_FOUND)
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if not 0 < content_length <= _MAX_REQUEST_BYTES:
                    raise ValueError("request body is empty or too large")
                content_type = self.headers.get("Content-Type", "")
                if "application/json" not in content_type:
                    raise ValueError("Content-Type must be application/json")
                request = json.loads(self.rfile.read(content_length))
                if not isinstance(request, dict):
                    raise ValueError("request JSON must be an object")
                payload = state.save(request)
                self._send_json(
                    {
                        "saved": True,
                        "output_json": str(state.output_json.resolve()),
                        "preview_png": str(state.preview_output.resolve()),
                        "selection": payload,
                    }
                )
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                self._error(exc, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # pragma: no cover - defensive server boundary
                self._error(exc, HTTPStatus.INTERNAL_SERVER_ERROR)

        def log_message(self, format_string: str, *args: Any) -> None:
            print(f"{self.client_address[0]} - {format_string % args}")

    return Handler


def _load_existing_roi(
    path: Path,
    *,
    video: Path,
    metadata: VideoMetadata,
) -> tuple[int, tuple[int, int, int, int]] | None:
    source = Path(path)
    if not source.exists():
        return None
    try:
        payload = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read existing ROI JSON {source}: {exc}") from exc
    if payload.get("schema_id") != _SCHEMA_ID:
        raise ValueError(f"existing ROI JSON has an unsupported schema: {source}")
    if Path(str(payload.get("source_video", ""))).resolve() != Path(video).resolve():
        raise ValueError(f"existing ROI JSON belongs to a different video: {source}")
    shape = payload.get("frame_shape_hw")
    if shape != [int(metadata.height), int(metadata.width)]:
        raise ValueError(f"existing ROI JSON frame shape does not match video: {source}")
    frame_index = _validate_frame_index(
        payload.get("frame_index"), frame_count=int(metadata.frame_count)
    )
    roi = _validate_roi(
        payload.get("roi_xywh"), width=int(metadata.width), height=int(metadata.height)
    )
    return frame_index, roi


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Serve a browser-based fixed ROI selector for a video frame. "
            "The server is designed for SSH local-port forwarding."
        )
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--initial-roi", type=str)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-output", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    metadata = _probe_video(video)
    output = args.output.resolve()
    preview = (
        args.preview_output.resolve()
        if args.preview_output is not None
        else output.with_name(f"{output.stem}_preview.png")
    )
    frame_index = _validate_frame_index(
        args.frame_index, frame_count=int(metadata.frame_count)
    )
    roi = None
    existing = _load_existing_roi(output, video=video, metadata=metadata)
    if existing is not None:
        frame_index, roi = existing
    if args.initial_roi is not None:
        roi = _validate_roi(
            _parse_roi(args.initial_roi),
            width=int(metadata.width),
            height=int(metadata.height),
        )

    state = SelectorState(
        video=video,
        output_json=output,
        preview_output=preview,
        metadata=metadata,
        initial_frame_index=frame_index,
        initial_roi=roi,
    )
    if not 1 <= int(args.port) <= 65535:
        raise ValueError("port must be between 1 and 65535")
    server = ThreadingHTTPServer((str(args.host), int(args.port)), _handler_class(state))
    host = str(args.host)
    url_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    print(f"video: {video}")
    print(
        f"video geometry: {metadata.width}x{metadata.height}, "
        f"frames={metadata.frame_count}, fps={metadata.fps:.6g}"
    )
    print(f"ROI JSON: {output}")
    print(f"preview PNG: {preview}")
    print(f"selector: http://{url_host}:{int(args.port)}")
    if host in {"127.0.0.1", "localhost", "::1"}:
        print("from a laptop terminal, forward the port with:")
        print(
            f"  ssh -N -L {int(args.port)}:127.0.0.1:{int(args.port)} "
            "<same-workstation-ssh-target>"
        )
        print(f"then open http://127.0.0.1:{int(args.port)}")
    print("press Ctrl-C in this terminal to stop the selector")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nstopping selector")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
