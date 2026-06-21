"""Spec and provenance panels for Palette marimo explorers."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def build_spec_provenance_panel(
    mo: Any,
    *,
    spec: Mapping[str, Any],
    artifact_attrs: Optional[Mapping[str, Any]] = None,
    option: Optional[Any] = None,
    extra_sections: Optional[Mapping[str, Any]] = None,
) -> Any:
    sections: dict[str, Any] = {}
    if option is not None:
        sections["Selected artifact"] = mo.tree(
            {
                "artifact_path": getattr(option, "artifact_path", None),
                "run_path": getattr(option, "run_path", None),
                "artifact_name": getattr(option, "artifact_name", None),
                "renderer": getattr(option, "renderer", None),
                "schema_id": getattr(option, "schema_id", None),
                "title": getattr(option, "title", None),
                "supported": getattr(option, "is_supported", None),
            }
        )
    source_paths = spec.get("source_paths")
    source_runs = spec.get("source_runs")
    if isinstance(source_runs, Mapping):
        sections["Source runs"] = mo.tree(dict(source_runs))
    if isinstance(source_paths, Mapping):
        sections["Source paths"] = mo.tree(dict(source_paths))
    sections["Interactive spec"] = mo.tree(dict(spec))
    if artifact_attrs is not None:
        sections["Artifact attrs"] = mo.tree(dict(artifact_attrs))
    if extra_sections:
        sections.update(dict(extra_sections))
    return mo.accordion(sections)

