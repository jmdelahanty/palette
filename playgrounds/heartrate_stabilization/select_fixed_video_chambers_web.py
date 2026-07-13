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

import cv2
import numpy as np

from select_fixed_video_roi_web import (
    VideoMetadata,
    _atomic_write_bytes,
    _encode_png,
    _probe_video,
    _read_video_frame,
    _validate_frame_index,
)


_SCHEMA_ID = "palette.playground.fixed_video_chambers.v1"
_LABELS = ("chamber_a", "chamber_b")
_COLORS_BGR = ((40, 70, 255), (255, 120, 40))
_MAX_REQUEST_BYTES = 128 * 1024


def _validate_polygon(value: Any, *, width: int, height: int) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError("each chamber polygon needs at least three vertices")
    points: list[tuple[int, int]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("polygon vertices must contain x and y")
        if any(isinstance(item, bool) for item in point):
            raise ValueError("polygon coordinates must be integer pixels")
        try:
            numeric = [float(item) for item in point]
        except (TypeError, ValueError) as exc:
            raise ValueError("polygon coordinates must be finite numbers") from exc
        if not np.isfinite(numeric).all():
            raise ValueError("polygon coordinates must be finite")
        rounded = tuple(int(round(item)) for item in numeric)
        if any(abs(item - integer) > 1e-6 for item, integer in zip(numeric, rounded)):
            raise ValueError("polygon vertices must resolve to whole source pixels")
        x, y = rounded
        if not (0 <= x < int(width) and 0 <= y < int(height)):
            raise ValueError(f"polygon vertex {(x, y)} exceeds frame bounds {(width, height)}")
        points.append((x, y))
    if len(set(points)) < 3:
        raise ValueError("polygon needs at least three distinct vertices")
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if abs(float(cv2.contourArea(contour))) < 1.0:
        raise ValueError("polygon area must be at least one source pixel")
    return tuple(points)


def _validate_chambers(
    value: Any,
    *,
    width: int,
    height: int,
) -> dict[str, tuple[tuple[int, int], ...]]:
    if not isinstance(value, dict) or set(value) != set(_LABELS):
        raise ValueError(f"chambers must contain exactly {', '.join(_LABELS)}")
    return {
        label: _validate_polygon(value[label], width=width, height=height)
        for label in _LABELS
    }


def _preview_png(
    frame_bgr: np.ndarray,
    chambers: dict[str, tuple[tuple[int, int], ...]],
) -> bytes:
    preview = np.asarray(frame_bgr, dtype=np.uint8).copy()
    overlay = preview.copy()
    for label, color in zip(_LABELS, _COLORS_BGR):
        contour = np.asarray(chambers[label], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [contour], color)
        cv2.polylines(preview, [contour], True, color, 2, cv2.LINE_AA)
    preview = cv2.addWeighted(overlay, 0.20, preview, 0.80, 0.0)
    for label, color in zip(_LABELS, _COLORS_BGR):
        contour = np.asarray(chambers[label], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(preview, [contour], True, color, 2, cv2.LINE_AA)
    return _encode_png(preview)


def _selection_payload(
    *,
    video: Path,
    metadata: VideoMetadata,
    frame_index: int,
    chambers: dict[str, tuple[tuple[int, int], ...]],
    preview_output: Path,
) -> dict[str, Any]:
    source = Path(video).resolve()
    stat = source.stat()
    return {
        "schema_id": _SCHEMA_ID,
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_space": "source_video_frame_pixels",
        "anatomical_identity_status": "unassigned_chamber_a_chamber_b",
        "chambers": {
            label: {"polygon_xy": [list(point) for point in chambers[label]]}
            for label in _LABELS
        },
        "frame_index": int(frame_index),
        "frame_shape_hw": [int(metadata.height), int(metadata.width)],
        "video_frame_count": int(metadata.frame_count),
        "video_fps": float(metadata.fps),
        "source_video": str(source),
        "source_video_size_bytes": int(stat.st_size),
        "source_video_mtime_ns": int(stat.st_mtime_ns),
        "preview_png": str(Path(preview_output).resolve()),
        "measurement_note": (
            "Fixed source-pixel polygons covering the visually identified chamber regions; "
            "they are not per-frame segmentations or confirmed anatomical identities."
        ),
    }


@dataclass
class ChamberSelectorState:
    video: Path
    output_json: Path
    preview_output: Path
    metadata: VideoMetadata
    initial_frame_index: int
    initial_chambers: dict[str, tuple[tuple[int, int], ...]] | None
    frame_reader: Callable[[Path, int], np.ndarray] = _read_video_frame

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def state_payload(self) -> dict[str, Any]:
        return {
            "schema_id": _SCHEMA_ID,
            "video_name": self.video.name,
            "width": int(self.metadata.width),
            "height": int(self.metadata.height),
            "frame_count": int(self.metadata.frame_count),
            "fps": float(self.metadata.fps),
            "initial_frame_index": int(self.initial_frame_index),
            "initial_chambers": (
                {
                    label: [list(point) for point in self.initial_chambers[label]]
                    for label in _LABELS
                }
                if self.initial_chambers is not None
                else None
            ),
            "output_json": str(self.output_json.resolve()),
            "preview_png": str(self.preview_output.resolve()),
        }

    def frame_png(self, frame_index: int) -> bytes:
        index = _validate_frame_index(frame_index, frame_count=int(self.metadata.frame_count))
        frame = self.frame_reader(self.video, index)
        if tuple(frame.shape[:2]) != (int(self.metadata.height), int(self.metadata.width)):
            raise ValueError("decoded frame shape does not match video metadata")
        return _encode_png(frame)

    def save(self, request: dict[str, Any]) -> dict[str, Any]:
        index = _validate_frame_index(
            request.get("frame_index"), frame_count=int(self.metadata.frame_count)
        )
        chambers = _validate_chambers(
            request.get("chambers"),
            width=int(self.metadata.width),
            height=int(self.metadata.height),
        )
        frame = self.frame_reader(self.video, index)
        payload = _selection_payload(
            video=self.video,
            metadata=self.metadata,
            frame_index=index,
            chambers=chambers,
            preview_output=self.preview_output,
        )
        content = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        preview = _preview_png(frame, chambers)
        with self._lock:
            _atomic_write_bytes(self.preview_output, preview)
            _atomic_write_bytes(self.output_json, content)
            self.initial_frame_index = index
            self.initial_chambers = chambers
        return payload


_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Side-Camera Chambers</title>
  <style>
    :root { font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #20272b; background: #f2f4f5; }
    * { box-sizing: border-box; }
    body { margin: 0; min-width: 320px; }
    header { height: 58px; padding: 10px 18px; display: flex; align-items: center; gap: 14px; background: white; border-bottom: 1px solid #cbd2d6; }
    h1 { margin: 0; font-size: 18px; letter-spacing: 0; }
    #videoName { color: #59666d; font-size: 13px; overflow-wrap: anywhere; }
    main { min-height: calc(100vh - 58px); display: grid; grid-template-columns: minmax(0, 1fr) 280px; }
    .viewer { padding: 16px; display: flex; justify-content: center; align-items: flex-start; overflow: auto; background: #252b2e; }
    canvas { display: block; max-width: 100%; max-height: calc(100vh - 90px); width: auto; height: auto; cursor: crosshair; touch-action: none; background: black; }
    aside { padding: 16px; background: white; border-left: 1px solid #cbd2d6; }
    .section { padding-bottom: 18px; margin-bottom: 18px; border-bottom: 1px solid #e1e5e7; }
    h2 { margin: 0 0 10px; color: #47545a; font-size: 13px; text-transform: uppercase; letter-spacing: 0; }
    label { display: block; margin-bottom: 5px; font-size: 12px; font-weight: 650; }
    input { width: 100%; height: 36px; padding: 6px 8px; border: 1px solid #aeb8bd; border-radius: 4px; font: inherit; }
    button { min-height: 36px; padding: 7px 10px; border: 1px solid #96a2a8; border-radius: 4px; background: white; color: #1c272c; font: inherit; font-size: 13px; font-weight: 650; cursor: pointer; }
    button:hover { background: #edf1f2; }
    button:focus-visible, input:focus { outline: 2px solid #197e85; outline-offset: 1px; }
    button.active-a { color: white; background: #d32f3f; border-color: #a91f2d; }
    button.active-b { color: white; background: #2877c7; border-color: #1c5a99; }
    button.primary { width: 100%; color: white; background: #197e85; border-color: #12656b; }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    .row { display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: end; }
    .segments, .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .actions { margin-top: 8px; }
    .counts { margin-top: 10px; display: grid; grid-template-columns: 1fr auto; gap: 5px; font-size: 12px; }
    .a { color: #b42332; } .b { color: #236cad; }
    #status { min-height: 36px; margin-top: 10px; color: #526068; font-size: 12px; overflow-wrap: anywhere; }
    #status.error { color: #a31825; } #status.saved { color: #18724b; }
    @media (max-width: 760px) { main { grid-template-columns: 1fr; } aside { border-left: 0; border-top: 1px solid #cbd2d6; } canvas { max-height: 70vh; } }
  </style>
</head>
<body>
  <header><h1>Side-Camera Chambers</h1><div id="videoName"></div></header>
  <main>
    <section class="viewer"><canvas id="canvas"></canvas></section>
    <aside>
      <div class="section">
        <h2>Reference Frame</h2>
        <div class="row"><div><label for="frameIndex">Frame index</label><input id="frameIndex" type="number" min="0" step="1"></div><button id="loadFrame">Load</button></div>
      </div>
      <div class="section">
        <h2>Active Polygon</h2>
        <div class="segments"><button id="selectA" class="active-a">Chamber A</button><button id="selectB">Chamber B</button></div>
        <div class="actions"><button id="undo">Undo</button><button id="clear">Clear</button></div>
        <div class="counts"><span class="a">Chamber A</span><span id="countA">0 vertices</span><span class="b">Chamber B</span><span id="countB">0 vertices</span></div>
      </div>
      <button id="save" class="primary" disabled>Save Chamber ROIs</button>
      <div id="status"></div>
    </aside>
  </main>
  <script>
    "use strict";
    const canvas = document.getElementById("canvas"), context = canvas.getContext("2d");
    const frameInput = document.getElementById("frameIndex"), saveButton = document.getElementById("save"), statusNode = document.getElementById("status");
    const selectA = document.getElementById("selectA"), selectB = document.getElementById("selectB");
    let state = null, image = new Image(), active = "chamber_a";
    let polygons = {chamber_a: [], chamber_b: []};

    function status(message, kind = "") { statusNode.textContent = message; statusNode.className = kind; }
    function validPolygon(points) { return points.length >= 3; }
    function sync() {
      document.getElementById("countA").textContent = `${polygons.chamber_a.length} vertices`;
      document.getElementById("countB").textContent = `${polygons.chamber_b.length} vertices`;
      saveButton.disabled = !(validPolygon(polygons.chamber_a) && validPolygon(polygons.chamber_b));
      selectA.className = active === "chamber_a" ? "active-a" : "";
      selectB.className = active === "chamber_b" ? "active-b" : "";
    }
    function drawPolygon(points, stroke, fill) {
      if (!points.length) return;
      context.save(); context.beginPath(); context.moveTo(points[0][0], points[0][1]);
      points.slice(1).forEach(point => context.lineTo(point[0], point[1]));
      if (points.length >= 3) { context.closePath(); context.fillStyle = fill; context.fill(); }
      context.strokeStyle = stroke; context.lineWidth = 2; context.stroke();
      context.fillStyle = stroke; points.forEach(point => { context.beginPath(); context.arc(point[0], point[1], 3, 0, Math.PI * 2); context.fill(); });
      context.restore();
    }
    function draw() {
      context.clearRect(0, 0, canvas.width, canvas.height);
      if (image.complete && image.naturalWidth) context.drawImage(image, 0, 0);
      drawPolygon(polygons.chamber_a, "#ff3545", "rgba(255,53,69,0.18)");
      drawPolygon(polygons.chamber_b, "#328ee6", "rgba(50,142,230,0.18)");
    }
    function pointer(event) { const b = canvas.getBoundingClientRect(); return [Math.round((event.clientX-b.left)*canvas.width/b.width), Math.round((event.clientY-b.top)*canvas.height/b.height)]; }
    canvas.addEventListener("pointerdown", event => { if (!state) return; const p = pointer(event); p[0] = Math.max(0, Math.min(state.width-1, p[0])); p[1] = Math.max(0, Math.min(state.height-1, p[1])); polygons[active].push(p); sync(); draw(); });
    selectA.addEventListener("click", () => { active = "chamber_a"; sync(); });
    selectB.addEventListener("click", () => { active = "chamber_b"; sync(); });
    document.getElementById("undo").addEventListener("click", () => { polygons[active].pop(); sync(); draw(); });
    document.getElementById("clear").addEventListener("click", () => { polygons[active] = []; sync(); draw(); });
    async function loadFrame() {
      const index = Math.round(Number(frameInput.value));
      if (!Number.isFinite(index) || index < 0 || index >= state.frame_count) { status(`Frame must be between 0 and ${state.frame_count-1}.`, "error"); return; }
      const next = new Image(); next.onload = () => { image = next; canvas.width = image.naturalWidth; canvas.height = image.naturalHeight; draw(); status(`Frame ${index} loaded.`); };
      next.onerror = () => status("Frame could not be loaded.", "error"); next.src = `/api/frame?index=${index}&cache=${Date.now()}`;
    }
    document.getElementById("loadFrame").addEventListener("click", loadFrame);
    saveButton.addEventListener("click", async () => {
      saveButton.disabled = true; status("Saving...");
      try {
        const response = await fetch("/api/save", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({frame_index:Number(frameInput.value), chambers:polygons})});
        const result = await response.json(); if (!response.ok) throw new Error(result.error || "Save failed"); status(`Saved ${result.output_json}`, "saved");
      } catch (error) { status(error.message, "error"); } finally { sync(); }
    });
    fetch("/api/state").then(response => response.json()).then(value => { state=value; document.getElementById("videoName").textContent=value.video_name; frameInput.max=String(value.frame_count-1); frameInput.value=String(value.initial_frame_index); if(value.initial_chambers) polygons=value.initial_chambers; sync(); loadFrame(); }).catch(error => status(error.message,"error"));
  </script>
</body>
</html>"""


def _handler_class(state: ChamberSelectorState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, content: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_bytes(json.dumps(payload, sort_keys=True).encode(), "application/json", status)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self.send_bytes(_HTML.encode(), "text/html; charset=utf-8")
                elif parsed.path == "/api/state":
                    self.send_json(state.state_payload())
                elif parsed.path == "/api/frame":
                    index = parse_qs(parsed.query).get("index", [state.initial_frame_index])[0]
                    self.send_bytes(state.frame_png(index), "image/png")
                else:
                    self.send_json({"error": parsed.path}, HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # pragma: no cover
                self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/save":
                self.send_json({"error": self.path}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= _MAX_REQUEST_BYTES:
                    raise ValueError("request body is empty or too large")
                request = json.loads(self.rfile.read(length))
                payload = state.save(request)
                self.send_json({"saved": True, "output_json": str(state.output_json.resolve()), "preview_png": str(state.preview_output.resolve()), "selection": payload})
            except (json.JSONDecodeError, ValueError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # pragma: no cover
                self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def log_message(self, format_string: str, *args: Any) -> None:
            print(f"{self.client_address[0]} - {format_string % args}")

    return Handler


def _load_existing(
    path: Path,
    *,
    video: Path,
    metadata: VideoMetadata,
) -> tuple[int, dict[str, tuple[tuple[int, int], ...]]] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("schema_id") != _SCHEMA_ID:
        raise ValueError("existing chamber JSON has an unsupported schema")
    if Path(payload.get("source_video", "")).resolve() != video.resolve():
        raise ValueError("existing chamber JSON belongs to a different video")
    chambers = _validate_chambers(
        {label: payload["chambers"][label]["polygon_xy"] for label in _LABELS},
        width=metadata.width,
        height=metadata.height,
    )
    index = _validate_frame_index(payload["frame_index"], frame_count=metadata.frame_count)
    return index, chambers


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a two-chamber polygon selector over SSH.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-output", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    video = args.video.resolve()
    metadata = _probe_video(video)
    output = args.output.resolve()
    preview = (
        args.preview_output.resolve()
        if args.preview_output is not None
        else output.with_name(f"{output.stem}_preview.png")
    )
    frame_index = _validate_frame_index(args.frame_index, frame_count=metadata.frame_count)
    chambers = None
    existing = _load_existing(output, video=video, metadata=metadata)
    if existing is not None:
        frame_index, chambers = existing
    state = ChamberSelectorState(
        video=video,
        output_json=output,
        preview_output=preview,
        metadata=metadata,
        initial_frame_index=frame_index,
        initial_chambers=chambers,
    )
    server = ThreadingHTTPServer((args.host, int(args.port)), _handler_class(state))
    print(f"video: {video}")
    print(f"video geometry: {metadata.width}x{metadata.height}, frames={metadata.frame_count}, fps={metadata.fps:.6g}")
    print(f"chamber JSON: {output}")
    print(f"preview PNG: {preview}")
    print(f"selector: http://127.0.0.1:{int(args.port)}")
    print(f"forward with: ssh -N -L {int(args.port)}:127.0.0.1:{int(args.port)} <same-workstation-ssh-target>")
    print("press Ctrl-C to stop")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nstopping selector")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
