"""Read-only discovery of copied image artifacts bound to analytics exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from fisheye.reporting.export import (
    REPORT_EXPORT_SCHEMA_ID,
    verify_report_manifest_sha256,
)
from fisheye.reporting.montage_report import SEMANTIC_MONTAGE_ARTIFACT_CONTRACT_ID


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class PublishedArtifactDiagnostic:
    manifest_path: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class PublishedImageArtifact:
    export_root: str
    export_run_id: str
    report_id: str
    report_manifest_path: str
    visualization_id: str
    label: str
    artifact_role: str
    media_type: str
    visualization_contract_id: str
    source_visualization_contract_id: str | None
    artifact_path: str
    content_sha256: str
    byte_length: int
    width_px: int | None
    height_px: int | None

    @property
    def selection_label(self) -> str:
        return f"{self.label} · {self.report_id}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selection_label"] = self.selection_label
        return payload


@dataclass(frozen=True)
class PublishedArtifactCatalog:
    export_root: str
    export_run_id: str
    artifacts: tuple[PublishedImageArtifact, ...]
    diagnostics: tuple[PublishedArtifactDiagnostic, ...]


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be an object")
    return payload


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def discover_published_image_artifacts(
    export_root: Path,
    export_run_id: str,
) -> PublishedArtifactCatalog:
    """Discover copied PNG report artifacts without opening source Zarrs."""

    root = Path(export_root).expanduser().resolve()
    run_id = str(export_run_id).strip()
    if not _SAFE_ID.fullmatch(run_id):
        raise ValueError(f"Invalid export_run_id: {run_id!r}")
    report_root = root / "v1" / "reports" / f"export_run_id={run_id}"
    diagnostics: list[PublishedArtifactDiagnostic] = []
    artifacts: list[PublishedImageArtifact] = []
    if not report_root.exists():
        return PublishedArtifactCatalog(str(root), run_id, (), ())
    resolved_report_root = report_root.resolve()
    if not _within(resolved_report_root, root):
        raise PermissionError(
            f"Analytics report directory resolves outside the authorized root: {report_root}"
        )

    for manifest_path in sorted(report_root.glob("report_id=*/report_manifest.json")):
        resolved_manifest = manifest_path.resolve()
        if not _within(resolved_manifest, root):
            diagnostics.append(
                PublishedArtifactDiagnostic(
                    str(manifest_path),
                    "manifest_outside_root",
                    "Report manifest resolves outside the authorized export root.",
                )
            )
            continue
        try:
            manifest = _load_json_object(resolved_manifest)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            diagnostics.append(
                PublishedArtifactDiagnostic(
                    str(manifest_path), "invalid_manifest", str(exc)
                )
            )
            continue
        if not verify_report_manifest_sha256(manifest):
            diagnostics.append(
                PublishedArtifactDiagnostic(
                    str(manifest_path),
                    "manifest_sha256_mismatch",
                    "Report manifest content hash does not verify.",
                )
            )
            continue
        if (
            manifest.get("schema_id") != REPORT_EXPORT_SCHEMA_ID
            or _safe_int(manifest.get("schema_version")) != 1
        ):
            diagnostics.append(
                PublishedArtifactDiagnostic(
                    str(manifest_path),
                    "unsupported_report_schema",
                    "Report manifest does not use the supported immutable report schema.",
                )
            )
            continue
        report_id = str(manifest.get("report_id", ""))
        directory_report_id = manifest_path.parent.name.removeprefix("report_id=")
        if report_id != directory_report_id or not _SAFE_ID.fullmatch(report_id):
            diagnostics.append(
                PublishedArtifactDiagnostic(
                    str(manifest_path),
                    "report_id_mismatch",
                    "Report ID does not match its canonical directory.",
                )
            )
            continue
        binding = manifest.get("analytics_export")
        if not isinstance(binding, Mapping) or binding.get("export_run_id") != run_id:
            diagnostics.append(
                PublishedArtifactDiagnostic(
                    str(manifest_path),
                    "analytics_export_mismatch",
                    "Report is not bound to the selected analytics export.",
                )
            )
            continue

        for index, item in enumerate(manifest.get("artifacts", [])):
            if not isinstance(item, Mapping):
                continue
            materialized = item.get("materialized")
            if (
                item.get("media_type") != "image/png"
                or not isinstance(materialized, Mapping)
                or not isinstance(materialized.get("relative_path"), str)
            ):
                continue
            relative_path = Path(str(materialized["relative_path"]))
            artifact_path = (manifest_path.parent / relative_path).resolve()
            if (
                relative_path.is_absolute()
                or not _within(artifact_path, manifest_path.parent.resolve())
                or not _within(artifact_path, root)
            ):
                diagnostics.append(
                    PublishedArtifactDiagnostic(
                        str(manifest_path),
                        "artifact_outside_report",
                        f"Artifact {index} resolves outside its report directory.",
                    )
                )
                continue
            if not artifact_path.is_file():
                diagnostics.append(
                    PublishedArtifactDiagnostic(
                        str(manifest_path),
                        "artifact_missing",
                        f"Copied image artifact is missing: {relative_path}",
                    )
                )
                continue
            expected_sha256 = materialized.get("content_sha256")
            if not isinstance(expected_sha256, str) or not expected_sha256:
                diagnostics.append(
                    PublishedArtifactDiagnostic(
                        str(manifest_path),
                        "artifact_hash_missing",
                        f"Copied image artifact has no content hash: {relative_path}",
                    )
                )
                continue
            artifacts.append(
                PublishedImageArtifact(
                    export_root=str(root),
                    export_run_id=run_id,
                    report_id=report_id,
                    report_manifest_path=str(resolved_manifest),
                    visualization_id=str(item.get("visualization_id", "")),
                    label=str(item.get("label") or item.get("visualization_id") or "Image"),
                    artifact_role=str(item.get("artifact_role", "image")),
                    media_type="image/png",
                    visualization_contract_id=str(
                        item.get("visualization_contract_id", "")
                    ),
                    source_visualization_contract_id=(
                        str(item["source_visualization_contract_id"])
                        if item.get("source_visualization_contract_id") is not None
                        else None
                    ),
                    artifact_path=str(artifact_path),
                    content_sha256=expected_sha256,
                    byte_length=(
                        _safe_int(materialized.get("byte_length"))
                        or artifact_path.stat().st_size
                    ),
                    width_px=_safe_int(materialized.get("width_px")),
                    height_px=_safe_int(materialized.get("height_px")),
                )
            )

    artifacts.sort(key=lambda item: (item.report_id, item.visualization_id))
    return PublishedArtifactCatalog(
        str(root), run_id, tuple(artifacts), tuple(diagnostics)
    )


def load_published_image_bytes(
    artifact: PublishedImageArtifact,
    *,
    max_bytes: int = 7_000_000,
) -> bytes:
    """Load one selected copied image and verify its manifest-declared hash."""

    root = Path(artifact.export_root).resolve()
    path = Path(artifact.artifact_path).resolve()
    if not _within(path, root):
        raise PermissionError(f"Published artifact resolves outside its export root: {path}")
    size = path.stat().st_size
    if size > int(max_bytes):
        raise ValueError(
            f"Published artifact is {size:,} bytes; viewer limit is {int(max_bytes):,} bytes"
        )
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != artifact.content_sha256:
        raise ValueError(
            f"Published artifact hash mismatch: expected {artifact.content_sha256}, "
            f"calculated {actual_sha256}"
        )
    return payload


def has_semantic_montage_artifacts(catalog: PublishedArtifactCatalog) -> bool:
    return any(
        item.visualization_contract_id == SEMANTIC_MONTAGE_ARTIFACT_CONTRACT_ID
        and item.artifact_role == "cohort_montage"
        for item in catalog.artifacts
    )


__all__ = [
    "PublishedArtifactCatalog",
    "PublishedArtifactDiagnostic",
    "PublishedImageArtifact",
    "discover_published_image_artifacts",
    "has_semantic_montage_artifacts",
    "load_published_image_bytes",
]
