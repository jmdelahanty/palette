"""Atomic sandbox publication of grouped-statistics cohort figures."""

from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Mapping, Sequence

import matplotlib.pyplot as plt

from fisheye.analytics_exports.publication import safe_component, sha256_file
from fisheye.group_statistics.validated_behavior_views import (
    ValidatedBehaviorStatisticsViewSource,
    available_statistics_views,
    build_statistics_view_payload,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.visualization.validated_behavior_group_statistics import (
    PLOT_DPI,
    ROBUST_HISTOGRAM_QUANTILE,
    ROBUST_OCCUPANCY_QUANTILE,
    render_statistics_view,
)

SCHEMA_ID = "palette.analytics.validated_behavior.group_statistics.static_report"
SCHEMA_VERSION = 1
METHOD_ID = "shared_payload_matplotlib_static_report_v1"
STATUS = "selector_ineligible_exploratory_candidate"


class ValidatedBehaviorStatisticsReportError(ValueError):
    """Raised when a grouped-statistics report is incomplete or stale."""


def _fail(message: str) -> None:
    raise ValidatedBehaviorStatisticsReportError(message)


def _html_document(
    *,
    report_run_id: str,
    statistics_run_id: str,
    statistics_digest: str,
    artifacts: Sequence[Mapping[str, object]],
) -> str:
    tiles = "\n".join(
        (
            '<section class="tile">'
            f"<h2>{escape(str(record['label']))}</h2>"
            f"<p>{escape(str(record['description']))}</p>"
            f'<a href="{escape(str(record["path"]))}">'
            f'<img src="{escape(str(record["path"]))}" '
            f'alt="{escape(str(record["label"]))}"></a>'
            f"<code>payload {escape(str(record['view_payload_sha256']))}</code>"
            "</section>"
        )
        for record in artifacts
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Validated behavior cohort report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; background: #fafafa; }}
    header {{ max-width: 90rem; margin: auto; }}
    main {{ display: grid; gap: 2rem; max-width: 100rem; margin: 2rem auto; }}
    .tile {{ background: white; padding: 1.2rem; border: 1px solid #ddd; border-radius: .5rem; }}
    img {{ width: 100%; height: auto; display: block; }}
    code {{ overflow-wrap: anywhere; color: #555; }}
  </style>
</head>
<body>
  <header>
    <h1>Validated behavior cohort report</h1>
    <p>Report <code>{escape(report_run_id)}</code></p>
    <p>Exploratory, equal recording weight, no acquisition-batch adjustment.</p>
    <p>Statistics <code>{escape(statistics_run_id)}</code> · <code>{escape(statistics_digest)}</code></p>
  </header>
  <main>{tiles}</main>
</body>
</html>
"""


def _validate_report_directory(root: Path, manifest: Mapping[str, object]) -> None:
    if (
        manifest.get("schema_id") != SCHEMA_ID
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        _fail("Static statistics report schema is unsupported")
    if manifest.get("method_id") != METHOD_ID or manifest.get("status") != STATUS:
        _fail("Static statistics report method or status is unsupported")
    body = {key: value for key, value in manifest.items() if key != "record_sha256"}
    if manifest.get("record_sha256") != canonical_json_sha256(body):
        _fail("Static statistics report manifest digest is stale")
    for field in (
        "selector_eligible",
        "production_authority",
        "selector_activation",
        "registry_update",
        "source_mutation",
    ):
        if manifest.get(field) is not False:
            _fail(f"Static statistics report safety flag is not false: {field}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        _fail("Static statistics report contains no artifacts")
    expected_names = {"manifest.json", "index.html"}
    for record in artifacts:
        if not isinstance(record, Mapping):
            _fail("Static statistics report artifact record is malformed")
        name = str(record.get("path"))
        if Path(name).name != name or not name.endswith(".png"):
            _fail("Static statistics report artifact path is unsafe")
        path = root / name
        if not path.is_file():
            _fail(f"Static statistics report artifact is missing: {name}")
        if path.stat().st_size != record.get("size_bytes"):
            _fail(f"Static statistics report artifact size is stale: {name}")
        if sha256_file(path) != record.get("file_sha256"):
            _fail(f"Static statistics report artifact digest is stale: {name}")
        expected_names.add(name)
    index_record = manifest.get("index")
    if not isinstance(index_record, Mapping):
        _fail("Static statistics report lacks its HTML index record")
    index_path = root / "index.html"
    if (
        not index_path.is_file()
        or index_path.stat().st_size != index_record.get("size_bytes")
        or sha256_file(index_path) != index_record.get("file_sha256")
    ):
        _fail("Static statistics report HTML index is missing or stale")
    if {path.name for path in root.iterdir()} != expected_names:
        _fail("Static statistics report contains an unrecorded file")


def read_validated_behavior_statistics_report(
    output_dir: str | Path,
) -> Mapping[str, object]:
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatedBehaviorStatisticsReportError(
            f"Cannot read statistics report manifest: {manifest_path}"
        ) from exc
    if not isinstance(value, dict):
        _fail("Static statistics report manifest must be one object")
    _validate_report_directory(root, value)
    return MappingProxyType({**value, "manifest_path": str(manifest_path)})


def render_validated_behavior_statistics_report(
    source: ValidatedBehaviorStatisticsViewSource,
    *,
    report_run_id: str,
    output_dir: str | Path,
    view_ids: Sequence[str] | None = None,
    dpi: int = PLOT_DPI,
) -> Mapping[str, object]:
    safe_component(report_run_id, label="report_run_id")
    if type(dpi) is not int or dpi < 72 or dpi > 600:
        raise ValueError("dpi must be one integer from 72 through 600")
    available = {item.view_id: item for item in available_statistics_views(source)}
    selected = tuple(view_ids or available)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Report view IDs must be one nonempty unique sequence")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unavailable report views: {unknown}")

    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite statistics report: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    )
    try:
        assert temporary is not None
        artifacts: list[dict[str, object]] = []
        payload_records: list[dict[str, object]] = []
        for view_id in selected:
            definition = available[view_id]
            payload = build_statistics_view_payload(source, view_id)
            figure = render_statistics_view(payload)
            filename = f"{view_id}.png"
            path = temporary / filename
            try:
                figure.savefig(
                    path,
                    format="png",
                    dpi=dpi,
                    bbox_inches="tight",
                    facecolor="white",
                )
            finally:
                plt.close(figure)
            artifacts.append(
                {
                    "artifact_id": view_id,
                    "label": definition.label,
                    "description": definition.description,
                    "path": filename,
                    "media_type": "image/png",
                    "size_bytes": path.stat().st_size,
                    "file_sha256": sha256_file(path),
                    "view_payload_sha256": payload["payload_sha256"],
                    "metric_family": definition.metric_family,
                }
            )
            payload_records.append(
                {
                    "view_id": view_id,
                    "view_payload_sha256": payload["payload_sha256"],
                    "metric_ids": [
                        record["metric_id"] for record in payload["metric_catalog"]
                    ],
                    "recording_row_count": len(payload["recording_rows"]),
                    "descriptive_row_count": len(payload["descriptive_rows"]),
                    "contrast_row_count": len(payload["contrast_rows"]),
                }
            )

        html = _html_document(
            report_run_id=report_run_id,
            statistics_run_id=source.statistics_run_id,
            statistics_digest=source.cache_identity,
            artifacts=artifacts,
        )
        index_path = temporary / "index.html"
        index_path.write_text(html, encoding="utf-8")
        index_record = {
            "path": "index.html",
            "media_type": "text/html",
            "size_bytes": index_path.stat().st_size,
            "file_sha256": sha256_file(index_path),
        }
        body: dict[str, object] = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "status": STATUS,
            "report_run_id": report_run_id,
            "source_statistics": {
                "path": str(source.root),
                "statistics_run_id": source.statistics_run_id,
                "statistics_manifest_sha256": source.cache_identity,
                "source_export_manifest_sha256": source.manifest["source_export"][
                    "export_manifest_record_sha256"
                ],
            },
            "renderer": {
                "backend": "matplotlib_agg",
                "dpi": dpi,
                "occupancy_color_scale_quantile": ROBUST_OCCUPANCY_QUANTILE,
                "histogram_color_scale_quantile": ROBUST_HISTOGRAM_QUANTILE,
                "condition_color_semantics": "epoch_role_not_stimulus_raw_color",
                "behavior_role_color_semantics": "semantic_role_not_stimulus_raw_color",
                "provider_distinction": "line_style_and_explicit_label",
                "renderer_module_sha256": sha256_file(Path(__file__)),
                "figure_module_sha256": sha256_file(
                    Path(render_statistics_view.__code__.co_filename)
                ),
            },
            "view_payloads": payload_records,
            "artifacts": artifacts,
            "index": index_record,
            "scientific_claim": "exploratory_descriptive_and_paired_visualization",
            "experimental_unit": "recording_id",
            "cohort_weighting": "equal_weight_per_finite_recording",
            "acquisition_batch_adjustment": "not_performed_identity_unavailable",
            "selector_eligible": False,
            "production_authority": False,
            "selector_activation": False,
            "registry_update": False,
            "source_mutation": False,
        }
        manifest = {**body, "record_sha256": canonical_json_sha256(body)}
        write_json_atomic(
            temporary / "manifest.json",
            manifest,
            overwrite=False,
        )
        _validate_report_directory(temporary, manifest)
        os.replace(temporary, target)
        temporary = None
        return MappingProxyType(
            {**manifest, "manifest_path": str(target / "manifest.json")}
        )
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


__all__ = [
    "METHOD_ID",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "STATUS",
    "ValidatedBehaviorStatisticsReportError",
    "read_validated_behavior_statistics_report",
    "render_validated_behavior_statistics_report",
]
