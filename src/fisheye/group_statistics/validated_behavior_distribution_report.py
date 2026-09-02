"""Atomic static report for exact validated-behavior distributions."""

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
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.visualization.validated_behavior_distributions import (
    PLOT_DPI,
    render_distribution_figure,
)

from .validated_behavior_distribution_views import (
    DEFAULT_COHORT_STATISTIC,
    DEFAULT_DISPLAY_RANGE,
    DISPLAY_RANGE_LABELS,
    ValidatedBehaviorDistributionViewSource,
    available_distribution_metrics,
    build_distribution_view_payload,
    resolve_distribution_display_range,
    validate_distribution_display_range,
)

SCHEMA_ID = "palette.analytics.validated_behavior.distributions.static_report"
SCHEMA_VERSION = 2
METHOD_ID = "shared_payload_matplotlib_distribution_report_v2"
LEGACY_METHOD_ID = "shared_payload_matplotlib_distribution_report_v1"
STATUS = "selector_ineligible_exploratory_candidate"


class ValidatedBehaviorDistributionReportError(ValueError):
    """Raised when a distribution report is incomplete or stale."""


def _fail(message: str) -> None:
    raise ValidatedBehaviorDistributionReportError(message)


def _html_document(
    *,
    report_run_id: str,
    distribution_run_id: str,
    distribution_digest: str,
    display_range_label: str,
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
  <title>Validated behavior distributions</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; background: #fafafa; }}
    header {{ max-width: 100rem; margin: auto; }}
    main {{ display: grid; gap: 2rem; max-width: 110rem; margin: 2rem auto; }}
    .tile {{ background: white; padding: 1.2rem; border: 1px solid #ddd; border-radius: .5rem; }}
    img {{ width: 100%; height: auto; display: block; }}
    code {{ overflow-wrap: anywhere; color: #555; }}
  </style>
</head>
<body>
  <header>
    <h1>Validated behavior distributions</h1>
    <p>Report <code>{escape(report_run_id)}</code></p>
    <p>Whole-session, pre, training, and post views. Default curves use equal recording weight.</p>
    <p>X-axis view: <strong>{escape(display_range_label)}</strong>. Central views retain complete
       bins and leave all tail evidence sealed in the source distribution.</p>
    <p>Distribution <code>{escape(distribution_run_id)}</code> · <code>{escape(distribution_digest)}</code></p>
  </header>
  <main>{tiles}</main>
</body>
</html>
"""


def _validate_report_directory(root: Path, manifest: Mapping[str, object]) -> None:
    version_method = (manifest.get("schema_version"), manifest.get("method_id"))
    if (
        manifest.get("schema_id") != SCHEMA_ID
        or version_method not in {(1, LEGACY_METHOD_ID), (SCHEMA_VERSION, METHOD_ID)}
        or manifest.get("status") != STATUS
    ):
        _fail("Distribution report schema, method, or status is unsupported")
    body = {key: value for key, value in manifest.items() if key != "record_sha256"}
    if manifest.get("record_sha256") != canonical_json_sha256(body):
        _fail("Distribution report manifest digest is stale")
    for field in (
        "selector_eligible",
        "production_authority",
        "selector_activation",
        "registry_update",
        "source_mutation",
    ):
        if manifest.get(field) is not False:
            _fail(f"Distribution report safety flag is not false: {field}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        _fail("Distribution report contains no figures")
    expected = {"manifest.json", "index.html"}
    renderer = manifest.get("renderer")
    if not isinstance(renderer, Mapping):
        _fail("Distribution report renderer record is malformed")
    report_display_range = renderer.get("display_range_id")
    if manifest.get("schema_version") == SCHEMA_VERSION and (
        report_display_range not in DISPLAY_RANGE_LABELS
        or renderer.get("display_range_policy")
        != "whole_bin_minimum_per_series_equal_recording_mass_v1"
    ):
        _fail("Distribution report display-range contract is unsupported")
    for record in artifacts:
        if not isinstance(record, Mapping):
            _fail("Distribution report artifact is malformed")
        name = str(record.get("path"))
        path = root / name
        if Path(name).name != name or not name.endswith(".png") or not path.is_file():
            _fail("Distribution report artifact path is unsafe or absent")
        if path.stat().st_size != record.get("size_bytes") or sha256_file(
            path
        ) != record.get("file_sha256"):
            _fail(f"Distribution report artifact bytes are stale: {name}")
        if manifest.get("schema_version") == SCHEMA_VERSION:
            display_range = record.get("display_range")
            if record.get(
                "requested_display_range_id"
            ) != report_display_range or not isinstance(display_range, Mapping):
                _fail("Distribution report artifact display range is absent or stale")
            try:
                validate_distribution_display_range(display_range)
            except ValueError as exc:
                raise ValidatedBehaviorDistributionReportError(
                    "Distribution report artifact display range is invalid"
                ) from exc
        expected.add(name)
    index = manifest.get("index")
    path = root / "index.html"
    if not isinstance(index, Mapping) or not path.is_file():
        _fail("Distribution report HTML index is absent")
    if path.stat().st_size != index.get("size_bytes") or sha256_file(path) != index.get(
        "file_sha256"
    ):
        _fail("Distribution report HTML index bytes are stale")
    if {path.name for path in root.iterdir()} != expected:
        _fail("Distribution report contains an unrecorded file")


def read_validated_behavior_distribution_report(
    output_dir: str | Path,
) -> Mapping[str, object]:
    root = Path(output_dir).expanduser().resolve()
    path = root / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatedBehaviorDistributionReportError(
            f"Cannot read distribution report manifest: {path}"
        ) from exc
    if not isinstance(manifest, dict):
        _fail("Distribution report manifest must be one object")
    _validate_report_directory(root, manifest)
    return MappingProxyType({**manifest, "manifest_path": str(path)})


def render_validated_behavior_distribution_report(
    source: ValidatedBehaviorDistributionViewSource,
    *,
    report_run_id: str,
    output_dir: str | Path,
    metric_ids: Sequence[str] | None = None,
    dpi: int = PLOT_DPI,
    display_range_id: str = DEFAULT_DISPLAY_RANGE,
) -> Mapping[str, object]:
    """Render one immutable PNG per selected metric and supported weighting."""

    safe_component(report_run_id, label="report_run_id")
    if type(dpi) is not int or not 72 <= dpi <= 600:
        raise ValueError("dpi must be one integer from 72 through 600")
    if display_range_id not in DISPLAY_RANGE_LABELS:
        raise ValueError(
            f"Unknown display range {display_range_id!r}; choose one of "
            f"{tuple(DISPLAY_RANGE_LABELS)}"
        )
    catalog = {
        str(record["metric_id"]): record
        for record in available_distribution_metrics(source)
    }
    selected = tuple(metric_ids or catalog)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Metric IDs must be one nonempty unique sequence")
    unknown = sorted(set(selected) - set(catalog))
    if unknown:
        raise ValueError(f"Unknown distribution metrics: {unknown}")

    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite distribution report: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    )
    try:
        assert temporary is not None
        artifacts: list[dict[str, object]] = []
        for metric_id in selected:
            metric = catalog[metric_id]
            for weighting_id in metric["weighting_ids"]:
                payload = build_distribution_view_payload(
                    source, metric_id, str(weighting_id)
                )
                display_range = resolve_distribution_display_range(
                    payload, display_range_id=display_range_id
                )
                figure = render_distribution_figure(
                    payload,
                    cohort_statistic=DEFAULT_COHORT_STATISTIC,
                    display_range_id=display_range_id,
                )
                artifact_id = f"{metric_id.replace('.', '_')}__{weighting_id}"
                filename = f"{artifact_id}.png"
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
                        "artifact_id": artifact_id,
                        "metric_id": metric_id,
                        "metric_family": metric["metric_family"],
                        "weighting_id": weighting_id,
                        "label": metric["interpretation"],
                        "description": (
                            f"{str(weighting_id).title()}-weighted whole-session and "
                            "exact semantic-epoch distributions; "
                            f"{DISPLAY_RANGE_LABELS[display_range_id]} x-axis."
                        ),
                        "path": filename,
                        "media_type": "image/png",
                        "size_bytes": path.stat().st_size,
                        "file_sha256": sha256_file(path),
                        "view_payload_sha256": payload["payload_sha256"],
                        "histogram_recipe_sha256": payload["histogram_recipe"][
                            "histogram_recipe_sha256"
                        ],
                        "requested_display_range_id": display_range_id,
                        "display_range": dict(display_range),
                    }
                )

        html = _html_document(
            report_run_id=report_run_id,
            distribution_run_id=source.distribution_run_id,
            distribution_digest=source.cache_identity,
            display_range_label=DISPLAY_RANGE_LABELS[display_range_id],
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
        source_export = source.manifest["source_export"]
        body: dict[str, object] = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "status": STATUS,
            "report_run_id": report_run_id,
            "source_distribution": {
                "path": str(source.root),
                "distribution_run_id": source.distribution_run_id,
                "distribution_manifest_sha256": source.cache_identity,
                "source_export_manifest_sha256": source_export[
                    "export_manifest_record_sha256"
                ],
            },
            "renderer": {
                "backend": "matplotlib_agg",
                "dpi": dpi,
                "cohort_statistic": DEFAULT_COHORT_STATISTIC,
                "display_range_id": display_range_id,
                "display_range_label": DISPLAY_RANGE_LABELS[display_range_id],
                "display_range_policy": (
                    "whole_bin_minimum_per_series_equal_recording_mass_v1"
                ),
                "scope_layout": "whole_pre_training_post_shared_axis_v1",
                "pooled_observations": "not_used_in_default_report",
                "role_style": "semantic_glyph_independent_protocol_color_policy",
                "provider_style": "explicit_color_and_line_style",
                "renderer_module_sha256": sha256_file(Path(__file__)),
                "figure_module_sha256": sha256_file(
                    Path(render_distribution_figure.__code__.co_filename)
                ),
            },
            "artifacts": artifacts,
            "index": index_record,
            "scientific_claim": "exploratory_recording_normalized_distributions",
            "experimental_unit": "recording_id",
            "selector_eligible": False,
            "production_authority": False,
            "selector_activation": False,
            "registry_update": False,
            "source_mutation": False,
        }
        manifest = {**body, "record_sha256": canonical_json_sha256(body)}
        write_json_atomic(temporary / "manifest.json", manifest, overwrite=False)
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
    "ValidatedBehaviorDistributionReportError",
    "read_validated_behavior_distribution_report",
    "render_validated_behavior_distribution_report",
]
