"""Bounded renderer for persisted bout-kinematics visualization contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from fisheye.shared.zarr_io import open_zarr_root
from fisheye.utils.view_zarr_visualization import load_png_artifact_bytes
from fisheye.visualization.bout_kinematics_interactive import (
    BOUT_SCHEMA_TO_ANALYSIS_ID,
)

from .common import join_path, normalize_path, png_bytes_to_markdown_image
from .provenance import build_spec_provenance_panel
from .registry import InteractiveSpecOption, discover_recording_explorer_spec_options


MAX_BOUT_SNAPSHOT_BYTES = 25_000_000
_ANALYSIS_ORDER = ("heading", "movement", "eye_gaze")


@dataclass(frozen=True)
class BoutSnapshot:
    analysis_id: str
    option: InteractiveSpecOption
    artifact_path: str
    png_bytes: bytes


def bout_options_for_run(
    zarr_path: Path | str,
    selected_option: InteractiveSpecOption,
) -> tuple[InteractiveSpecOption, ...]:
    options = discover_recording_explorer_spec_options(
        zarr_path,
        run_path_filter=selected_option.run_path,
    )
    bout_options = [
        option for option in options if option.schema_id in BOUT_SCHEMA_TO_ANALYSIS_ID
    ]
    if (
        selected_option.schema_id in BOUT_SCHEMA_TO_ANALYSIS_ID
        and selected_option.artifact_path not in {item.artifact_path for item in bout_options}
    ):
        bout_options.append(selected_option)
    return tuple(sorted(bout_options, key=lambda item: item.artifact_name))


def available_bout_analysis_ids(
    zarr_path: Path | str,
    selected_option: InteractiveSpecOption,
) -> tuple[str, ...]:
    available = {
        BOUT_SCHEMA_TO_ANALYSIS_ID[option.schema_id]
        for option in bout_options_for_run(zarr_path, selected_option)
        if option.schema_id in BOUT_SCHEMA_TO_ANALYSIS_ID
    }
    ordered = [analysis_id for analysis_id in _ANALYSIS_ORDER if analysis_id in available]
    if available:
        ordered.append("provenance")
    return tuple(ordered)


def _option_for_analysis(
    options: Iterable[InteractiveSpecOption],
    analysis_id: str,
) -> Optional[InteractiveSpecOption]:
    for option in options:
        if BOUT_SCHEMA_TO_ANALYSIS_ID.get(str(option.schema_id or "")) == analysis_id:
            return option
    return None


def resolve_bout_snapshot_path(option: InteractiveSpecOption) -> str:
    raw_snapshot = str(option.attrs.get("snapshot_artifact") or "").strip()
    if not raw_snapshot:
        raise ValueError(f"Bout spec has no snapshot_artifact: {option.artifact_path}")
    snapshot = normalize_path(raw_snapshot)
    if not snapshot:
        raise ValueError(f"Bout spec has an empty snapshot_artifact: {option.artifact_path}")
    if "/" not in snapshot:
        return join_path(option.run_path, "visualizations", snapshot)
    if snapshot.startswith("visualizations/"):
        return join_path(option.run_path, snapshot)
    return snapshot


def load_bout_snapshot(
    zarr_path: Path | str,
    option: InteractiveSpecOption,
    *,
    analysis_id: str,
    max_bytes: int = MAX_BOUT_SNAPSHOT_BYTES,
) -> BoutSnapshot:
    artifact_path = resolve_bout_snapshot_path(option)
    root = open_zarr_root(Path(zarr_path), mode="r")
    try:
        artifact = root[artifact_path]
    except Exception as exc:
        raise ValueError(f"Bout snapshot is missing: {artifact_path}") from exc
    byte_length = int(getattr(artifact, "attrs", {}).get("byte_length") or 0)
    if byte_length > int(max_bytes):
        raise ValueError(
            f"Bout snapshot is {byte_length:,} bytes, above the {int(max_bytes):,}-byte display limit."
        )
    resolved_path, png_bytes = load_png_artifact_bytes(root, artifact_path)
    if len(png_bytes) > int(max_bytes):
        raise ValueError(
            f"Bout snapshot is {len(png_bytes):,} bytes, above the {int(max_bytes):,}-byte display limit."
        )
    return BoutSnapshot(
        analysis_id=analysis_id,
        option=option,
        artifact_path=resolved_path,
        png_bytes=png_bytes,
    )


def build_bout_kinematics_output(
    mo: Any,
    *,
    zarr_path: Path | str,
    selected_option: InteractiveSpecOption,
    analysis_id: str,
) -> Any:
    options = bout_options_for_run(zarr_path, selected_option)
    if analysis_id == "provenance":
        sections: dict[str, Any] = {}
        for option in options:
            option_analysis = BOUT_SCHEMA_TO_ANALYSIS_ID.get(str(option.schema_id or ""), "bout")
            sections[option_analysis.replace("_", " ").title()] = build_spec_provenance_panel(
                mo,
                spec=option.spec,
                artifact_attrs=option.attrs,
                option=option,
            )
        return mo.accordion(sections) if sections else mo.md("No bout provenance is available.")

    option = _option_for_analysis(options, analysis_id)
    if option is None:
        return mo.callout(
            f"No persisted `{analysis_id}` bout visualization is present in this run.",
            kind="warn",
        )
    try:
        snapshot = load_bout_snapshot(
            zarr_path,
            option,
            analysis_id=analysis_id,
        )
    except Exception as exc:
        return mo.callout(
            f"Bout visualization could not be loaded: `{type(exc).__name__}: {exc}`",
            kind="danger",
        )

    details: Mapping[str, Any] = {
        "run": option.run_name,
        "schema_id": option.schema_id,
        "renderer": option.renderer,
        "snapshot_artifact": snapshot.artifact_path,
        "source_paths": option.spec.get("source_paths", {}),
        "parameters": option.spec.get("parameters", {}),
    }
    return mo.vstack(
        [
            mo.md(f"### {option.title}"),
            png_bytes_to_markdown_image(
                mo,
                snapshot.png_bytes,
                alt_text=option.title or analysis_id,
            ),
            mo.accordion({"Contract and sources": mo.tree(dict(details))}),
        ]
    )
