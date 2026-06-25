"""HTTP handlers for the group analytics viewer."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

from .query import (
    ViewerContext,
    build_health_report,
    query_chaser_histogram,
    query_chaser_summary,
    query_cra_near_field_object_phase,
    query_cra_near_field_curves,
    query_cra_near_field_summary,
    query_cra_object_phase,
    query_cra_quadrant_occupancy_density,
    query_cra_specificity,
    query_cra_summary,
    query_egocentric_histogram,
    query_egocentric_summary,
    query_epoch_center_distance_histogram,
    query_epoch_speed_summary,
    query_export_summary,
    query_group_statistics,
    query_options,
    query_provenance,
    query_recordings,
    query_speed_distance_bins,
    query_spatial_occupancy,
)

_CONTENT_TYPES: Mapping[str, str] = MappingProxyType(
    {
        ".css": "text/css; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".txt": "text/plain; charset=utf-8",
    }
)


def _param_first(query_params: Mapping[str, list[str]], key: str) -> str | None:
    values = query_params.get(key)
    if not values:
        return None
    value = values[0].strip()
    return value or None


def _parse_bool_param(
    query_params: Mapping[str, list[str]],
    key: str,
    *,
    default: bool = False,
) -> bool:
    value = _param_first(query_params, key)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean for '{key}': {value}")


def _parse_int_param(query_params: Mapping[str, list[str]], key: str) -> int | None:
    value = _param_first(query_params, key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for '{key}': {value}") from exc


def build_handler(*, context: ViewerContext, static_dir: Path) -> type[BaseHTTPRequestHandler]:
    static_root = static_dir.resolve()

    class GroupAnalyticsRequestHandler(BaseHTTPRequestHandler):
        server_version = "PaletteGroupAnalyticsViewer/0.1"
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
            data = json.dumps(payload, sort_keys=True).encode("utf-8")
            self._write_bytes(data, status=status, content_type="application/json; charset=utf-8")

        def _write_error(self, status: HTTPStatus, *, error: str, details: str | None = None) -> None:
            payload: dict[str, Any] = {"ok": False, "error": error}
            if details:
                payload["details"] = details
            self._write_json(payload, status=status)

        def _write_not_found(self, message: str = "Not found") -> None:
            self._write_error(HTTPStatus.NOT_FOUND, error=message)

        def _serve_static(self, relative_path: str) -> None:
            candidate = (static_root / relative_path).resolve()
            if not candidate.is_relative_to(static_root) or not candidate.is_file():
                self._write_not_found("Static asset not found")
                return
            content_type = _CONTENT_TYPES.get(candidate.suffix.lower(), "application/octet-stream")
            self._write_bytes(candidate.read_bytes(), content_type=content_type)

        def _handle_api_get(self, path: str, query_params: Mapping[str, list[str]]) -> bool:
            if path == "/api/export/summary":
                self._write_json({"ok": True, "summary": query_export_summary(context)})
                return True

            if path == "/api/options":
                self._write_json({"ok": True, "options": query_options(context)})
                return True

            if path == "/api/goodcopbadcop/spatial":
                metric = _param_first(query_params, "metric") or "time_s"
                value_mode = _param_first(query_params, "value_mode") or "auto"
                zone_set_id = _param_first(query_params, "zone_set_id")
                include_recordings = _parse_bool_param(query_params, "include_recordings", default=False)
                self._write_json(
                    {
                        "ok": True,
                        "spatial": query_spatial_occupancy(
                            context,
                            metric=metric,
                            value_mode=value_mode,
                            zone_set_id=zone_set_id,
                            include_recordings=include_recordings,
                        ),
                    }
                )
                return True

            if path == "/api/goodcopbadcop/chaser-summary":
                metric = _param_first(query_params, "metric") or "p50_distance_mm"
                stat = _param_first(query_params, "stat") or "mean"
                include_recordings = _parse_bool_param(query_params, "include_recordings", default=False)
                self._write_json(
                    {
                        "ok": True,
                        "chaser_summary": query_chaser_summary(
                            context,
                            metric=metric,
                            stat=stat,
                            include_recordings=include_recordings,
                        ),
                    }
                )
                return True

            if path == "/api/goodcopbadcop/chaser-histogram":
                window_label = _param_first(query_params, "window_label")
                chaser_index = _parse_int_param(query_params, "chaser_index")
                self._write_json(
                    {
                        "ok": True,
                        "histogram": query_chaser_histogram(
                            context,
                            window_label=window_label,
                            chaser_index=chaser_index,
                        ),
                    }
                )
                return True

            if path == "/api/goodcopbadcop/epoch-speed":
                metric = _param_first(query_params, "metric") or "mean_speed_mm_s"
                stat = _param_first(query_params, "stat") or "mean"
                include_recordings = _parse_bool_param(query_params, "include_recordings", default=False)
                self._write_json(
                    {
                        "ok": True,
                        "epoch_speed": query_epoch_speed_summary(
                            context,
                            metric=metric,
                            stat=stat,
                            include_recordings=include_recordings,
                        ),
                    }
                )
                return True

            if path == "/api/goodcopbadcop/speed-distance":
                window_label = _param_first(query_params, "window_label")
                chaser_index = _parse_int_param(query_params, "chaser_index")
                self._write_json(
                    {
                        "ok": True,
                        "speed_distance": query_speed_distance_bins(
                            context,
                            window_label=window_label,
                            chaser_index=chaser_index,
                        ),
                    }
                )
                return True

            if path == "/api/goodcopbadcop/cra-object-phase":
                metric = _param_first(query_params, "metric") or "median_distance_mm"
                stat = _param_first(query_params, "stat") or "mean"
                object_role = _param_first(query_params, "object_role")
                include_recordings = _parse_bool_param(query_params, "include_recordings", default=False)
                self._write_json(
                    {
                        "ok": True,
                        "cra_object_phase": query_cra_object_phase(
                            context,
                            metric=metric,
                            stat=stat,
                            object_role=object_role,
                            include_recordings=include_recordings,
                        ),
                    }
                )
                return True

            if path == "/api/goodcopbadcop/epoch-center-distance-histogram":
                window_label = _param_first(query_params, "window_label")
                self._write_json(
                    {
                        "ok": True,
                        "center_distance_histogram": query_epoch_center_distance_histogram(
                            context,
                            window_label=window_label,
                        ),
                    }
                )
                return True

            if path == "/api/goodcopbadcop/cra-summary":
                metric = _param_first(query_params, "metric")
                endpoint_status = _param_first(query_params, "endpoint_status")
                include_rows = _parse_bool_param(query_params, "include_rows", default=True)
                self._write_json(
                    {
                        "ok": True,
                        "cra_summary": query_cra_summary(
                            context,
                            metric=metric,
                            endpoint_status=endpoint_status,
                            include_rows=include_rows,
                        ),
                    }
                )
                return True

            if path == "/api/goodcopbadcop/cra-specificity":
                self._write_json(
                    {
                        "ok": True,
                        "cra_specificity": query_cra_specificity(context),
                    }
                )
                return True

            if path == "/api/goodcopbadcop/cra-quadrant-occupancy-density":
                self._write_json(
                    {
                        "ok": True,
                        "cra_quadrant_occupancy_density": query_cra_quadrant_occupancy_density(context),
                    }
                )
                return True

            if path == "/api/goodcopbadcop/cra-near-field-object-phase":
                metric = _param_first(query_params, "metric") or "near_zone_occupancy_fraction"
                stat = _param_first(query_params, "stat") or "mean"
                object_role = _param_first(query_params, "object_role")
                include_recordings = _parse_bool_param(query_params, "include_recordings", default=False)
                self._write_json(
                    {
                        "ok": True,
                        "cra_near_field_object_phase": query_cra_near_field_object_phase(
                            context,
                            metric=metric,
                            stat=stat,
                            object_role=object_role,
                            include_recordings=include_recordings,
                        ),
                    }
                )
                return True

            if path == "/api/goodcopbadcop/cra-near-field-curves":
                self._write_json(
                    {
                        "ok": True,
                        "cra_near_field_curves": query_cra_near_field_curves(context),
                    }
                )
                return True

            if path == "/api/goodcopbadcop/cra-near-field-summary":
                metric = _param_first(query_params, "metric")
                endpoint_status = _param_first(query_params, "endpoint_status")
                include_rows = _parse_bool_param(query_params, "include_rows", default=True)
                self._write_json(
                    {
                        "ok": True,
                        "cra_near_field_summary": query_cra_near_field_summary(
                            context,
                            metric=metric,
                            endpoint_status=endpoint_status,
                            include_rows=include_rows,
                        ),
                    }
                )
                return True

            if path == "/api/goodcopbadcop/egocentric-summary":
                metric = _param_first(query_params, "metric") or "mean_alignment_cos"
                stat = _param_first(query_params, "stat") or "mean"
                include_recordings = _parse_bool_param(query_params, "include_recordings", default=False)
                self._write_json(
                    {
                        "ok": True,
                        "egocentric_summary": query_egocentric_summary(
                            context,
                            metric=metric,
                            stat=stat,
                            include_recordings=include_recordings,
                        ),
                    }
                )
                return True

            if path == "/api/goodcopbadcop/egocentric-histogram":
                window_label = _param_first(query_params, "window_label")
                chaser_index = _parse_int_param(query_params, "chaser_index")
                self._write_json(
                    {
                        "ok": True,
                        "histogram": query_egocentric_histogram(
                            context,
                            window_label=window_label,
                            chaser_index=chaser_index,
                        ),
                    }
                )
                return True

            if path == "/api/goodcopbadcop/statistics":
                self._write_json(
                    {
                        "ok": True,
                        "statistics": query_group_statistics(
                            context,
                            metric_family=_param_first(query_params, "metric_family"),
                            metric_name=_param_first(query_params, "metric_name"),
                            contrast_name=_param_first(query_params, "contrast_name"),
                            status=_param_first(query_params, "status"),
                        ),
                    }
                )
                return True

            if path == "/api/goodcopbadcop/recordings":
                self._write_json({"ok": True, "recordings": query_recordings(context)})
                return True

            if path == "/api/goodcopbadcop/provenance":
                self._write_json({"ok": True, "provenance": query_provenance(context)})
                return True

            return False

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path or "/"
            query_params = parse_qs(parsed.query, keep_blank_values=False)

            if path == "/":
                self._serve_static("index.html")
                return
            if path == "/healthz":
                report = build_health_report(context)
                status = HTTPStatus.OK if report.ok else HTTPStatus.SERVICE_UNAVAILABLE
                self._write_json(report.to_dict(), status=status)
                return
            if path.startswith("/static/"):
                rel = path.removeprefix("/static/").lstrip("/")
                if not rel:
                    self._write_not_found()
                    return
                self._serve_static(rel)
                return
            if path.startswith("/api/"):
                try:
                    handled = self._handle_api_get(path, query_params)
                except ValueError as exc:
                    self._write_error(HTTPStatus.BAD_REQUEST, error="invalid_request", details=str(exc))
                    return
                except Exception as exc:
                    self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, error="internal_error", details=str(exc))
                    return
                if handled:
                    return
            self._write_not_found()

        def do_HEAD(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path or "/"
            if path in {"/", "/healthz"} or path.startswith("/api/"):
                self.send_response(HTTPStatus.OK)
                self.end_headers()
                return
            if path.startswith("/static/"):
                rel = path.removeprefix("/static/").lstrip("/")
                candidate = (static_root / rel).resolve() if rel else static_root
                if rel and candidate.is_relative_to(static_root) and candidate.is_file():
                    self.send_response(HTTPStatus.OK)
                    self.end_headers()
                    return
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

    return GroupAnalyticsRequestHandler
