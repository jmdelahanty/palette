"""Immutable Zarr-owned arena-geometry fit and review evidence.

The blind fitter may use node-local or campaign scratch while it is running, but
its complete review package is imported into the canonical analysis Zarr before
the scratch package is considered disposable.  Publication remains
selector-ineligible and cannot select geometry or gate detections.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.json_safety import json_attr_safe, strict_json_dumps
from fisheye.shared.plot_artifacts import (
    PNG_ARTIFACT_SCHEMA_ID,
    write_png_visualization_artifact,
)
from fisheye.shared.run_provenance import (
    build_writer_run_provenance,
    validate_run_provenance,
)
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)

FIT_REVIEW_RUNS_PARENT = "arena_geometry_fit_runs"
FIT_REVIEW_RECORD_SCHEMA_ID = "palette.arena_geometry_fit_review_record"
FIT_REVIEW_RECORD_SCHEMA_VERSION = 1
FIT_REVIEW_RUN_SCHEMA_ID = "palette.arena_geometry_fit_review_run"
FIT_REVIEW_RUN_SCHEMA_VERSION = 1
FIT_REVIEW_PUBLISH_SCHEMA_ID = "palette.arena_geometry_fit_review_publish"
FIT_REVIEW_PUBLISH_ALGORITHM_VERSION = 1
FIT_REPORT_ARRAY = "fit_report_json"
REVIEW_PACKAGE_ARRAY = "review_package_json"
ACQUISITION_REVEAL_ARRAY = "acquisition_reveal_json"
MONTAGE_ARTIFACT = "dish_rim_review_montage_png"
JSON_BYTES_SCHEMA_ID = "palette.artifact.json_bytes.v1"
PROBE_SCHEMA_ID = "palette.diagnostics.recording_dish_rim_probe"
REVIEW_PACKAGE_SCHEMA_ID = f"{PROBE_SCHEMA_ID}.review_package"
ACQUISITION_REVEAL_SCHEMA_ID = f"{PROBE_SCHEMA_ID}.acquisition_reveal"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FitReviewArtifact:
    role: str
    source_path: Path
    zarr_path: str
    media_type: str
    content_sha256: str
    byte_length: int

    def to_json(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "source_name": self.source_path.name,
            "zarr_path": self.zarr_path,
            "media_type": self.media_type,
            "content_sha256": self.content_sha256,
            "byte_length": self.byte_length,
        }


@dataclass(frozen=True)
class ArenaGeometryFitReviewPlan:
    source_zarr: Path
    package_dir: Path
    run_name: str
    target_run_path: Path
    review_record: Mapping[str, Any]
    review_record_sha256: str
    artifacts: tuple[FitReviewArtifact, ...]
    run_provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ArenaGeometryFitReviewEvidence:
    source_zarr: Path
    run_name: str
    run_path: Path
    review_record: Mapping[str, Any]
    review_record_sha256: str
    fit_report_bytes: bytes
    review_package_bytes: bytes
    montage_bytes: bytes
    acquisition_reveal_bytes: bytes | None
    fit_report_ref: str
    montage_ref: str
    acquisition_reveal_ref: str | None


def _canonical_copy(value: Any) -> Any:
    return json.loads(strict_json_dumps(value))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _payload_sha256(payload: Any) -> str:
    return _sha256_bytes(strict_json_dumps(payload).encode("utf-8"))


def _read_json_bytes(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON.") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one JSON object.")
    return value


def _safe_source_file(package_dir: Path, value: object, *, label: str) -> Path:
    name = str(value or "").strip()
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"{label} must be one package-local filename.")
    path = (package_dir / name).resolve()
    try:
        path.relative_to(package_dir)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the review package.") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _required_sha256(value: object, *, label: str) -> str:
    digest = str(value or "").strip().lower().removeprefix("sha256:")
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return digest


def _bound_artifact(
    *,
    role: str,
    source_path: Path,
    zarr_path: str,
    media_type: str,
    expected_sha256: object | None = None,
) -> FitReviewArtifact:
    payload = source_path.read_bytes()
    if not payload:
        raise ValueError(f"{role} is empty: {source_path}")
    digest = _sha256_bytes(payload)
    if expected_sha256 is not None and digest != _required_sha256(
        expected_sha256, label=f"{role}.sha256"
    ):
        raise ValueError(f"{role} changed after review-package creation.")
    if media_type == "image/png" and not payload.startswith(_PNG_SIGNATURE):
        raise ValueError(f"{role} is not a PNG byte stream.")
    return FitReviewArtifact(
        role=role,
        source_path=source_path,
        zarr_path=zarr_path,
        media_type=media_type,
        content_sha256=digest,
        byte_length=len(payload),
    )


def _panel_artifact_name(index: int, source_name: str) -> str:
    stem = Path(source_name).stem
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or f"panel_{index}"
    return f"source_panel_{index:02d}_{safe}_png"


def build_arena_geometry_fit_review_plan(
    source_zarr: str | Path,
    *,
    review_package_dir: str | Path,
) -> ArenaGeometryFitReviewPlan:
    """Validate one frozen probe package and plan a content-addressed import."""

    archive = Path(source_zarr).expanduser().resolve()
    package_dir = Path(review_package_dir).expanduser().resolve()
    if not (archive / "zarr.json").is_file():
        raise FileNotFoundError(f"Analysis target is not Zarr v3: {archive}")
    if not package_dir.is_dir():
        raise FileNotFoundError(f"Review package directory is missing: {package_dir}")

    receipt_path = package_dir / "review_package.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt_bytes = receipt_path.read_bytes()
    receipt = _read_json_bytes(receipt_bytes, label="review_package.json")
    if (
        receipt.get("schema_id") != REVIEW_PACKAGE_SCHEMA_ID
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "awaiting_explicit_human_review"
    ):
        raise ValueError("Unsupported arena-geometry review package contract.")

    fit_binding = receipt.get("fit_report")
    montage_binding = receipt.get("montage")
    panel_bindings = receipt.get("source_panels")
    if not isinstance(fit_binding, Mapping) or not isinstance(montage_binding, Mapping):
        raise ValueError("Review package lacks fit-report or montage bindings.")
    if not isinstance(panel_bindings, list) or len(panel_bindings) != 3:
        raise ValueError(
            "Review package must bind exactly three temporal source panels."
        )

    fit_path = _safe_source_file(
        package_dir, fit_binding.get("path"), label="fit_report.path"
    )
    montage_path = _safe_source_file(
        package_dir, montage_binding.get("path"), label="montage.path"
    )
    artifacts: list[FitReviewArtifact] = [
        _bound_artifact(
            role="fit_report",
            source_path=fit_path,
            zarr_path=FIT_REPORT_ARRAY,
            media_type="application/json",
            expected_sha256=fit_binding.get("sha256"),
        ),
        _bound_artifact(
            role="review_package",
            source_path=receipt_path,
            zarr_path=REVIEW_PACKAGE_ARRAY,
            media_type="application/json",
        ),
        _bound_artifact(
            role="review_montage",
            source_path=montage_path,
            zarr_path=f"visualizations/{MONTAGE_ARTIFACT}",
            media_type="image/png",
            expected_sha256=montage_binding.get("sha256"),
        ),
    ]
    seen_panel_names: set[str] = set()
    for index, raw in enumerate(panel_bindings):
        if not isinstance(raw, Mapping):
            raise ValueError(f"source_panels[{index}] is not an object.")
        panel_path = _safe_source_file(
            package_dir, raw.get("path"), label=f"source_panels[{index}].path"
        )
        if panel_path.name in seen_panel_names:
            raise ValueError("Review package source-panel names are not unique.")
        seen_panel_names.add(panel_path.name)
        artifact_name = _panel_artifact_name(index, panel_path.name)
        artifacts.append(
            _bound_artifact(
                role=f"source_panel_{index}",
                source_path=panel_path,
                zarr_path=f"visualizations/{artifact_name}",
                media_type="image/png",
                expected_sha256=raw.get("sha256"),
            )
        )

    reveal_path = package_dir / "acquisition_reveal.json"
    if reveal_path.is_file():
        reveal_artifact = _bound_artifact(
            role="acquisition_reveal",
            source_path=reveal_path,
            zarr_path=ACQUISITION_REVEAL_ARRAY,
            media_type="application/json",
        )
        reveal = _read_json_bytes(
            reveal_path.read_bytes(), label="acquisition_reveal.json"
        )
        reveal_fit = reveal.get("fit_report")
        if (
            reveal.get("schema_id") != ACQUISITION_REVEAL_SCHEMA_ID
            or reveal.get("schema_version") != 1
            or not isinstance(reveal_fit, Mapping)
            or _required_sha256(
                reveal_fit.get("sha256"), label="acquisition_reveal.fit_report.sha256"
            )
            != artifacts[0].content_sha256
        ):
            raise ValueError("Acquisition reveal does not bind the frozen fit report.")
        artifacts.append(reveal_artifact)

    fit_report = _read_json_bytes(fit_path.read_bytes(), label="fit_report.json")
    if (
        fit_report.get("schema_id") != PROBE_SCHEMA_ID
        or fit_report.get("schema_version") != 1
        or fit_report.get("status") != "provisional_visual_review_required"
        or fit_report.get("fit_frozen_before_acquisition_reveal") is not True
    ):
        raise ValueError("Unsupported frozen arena-geometry fit report.")
    if set((fit_report.get("windows") or {}).keys()) != {"early", "middle", "late"}:
        raise ValueError("Fit report lacks exact early/middle/late evidence.")
    source = fit_report.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Fit report lacks source identity.")

    record = _canonical_copy(
        {
            "schema_id": FIT_REVIEW_RECORD_SCHEMA_ID,
            "schema_version": FIT_REVIEW_RECORD_SCHEMA_VERSION,
            "review_status": "awaiting_explicit_human_review",
            "fit_report_schema_id": fit_report.get("schema_id"),
            "fit_report_schema_version": fit_report.get("schema_version"),
            "fit_method": fit_report.get("fit_method"),
            "fit_frozen_before_acquisition_reveal": True,
            "source": {
                "camera_serial": source.get("camera_serial"),
                "video_path": source.get("video_path"),
                "video_sha256": source.get("video_sha256"),
                "summary_sha256": source.get("summary_sha256"),
                "keyframe_summary_sha256": source.get("keyframe_summary_sha256"),
                "frame_count": source.get("frame_count"),
                "image_shape_px": source.get("image_shape_px"),
                "pixel_contract": source.get("pixel_contract"),
            },
            "artifacts": {artifact.role: artifact.to_json() for artifact in artifacts},
            "policy": {
                "candidate_published": False,
                "candidate_selected": False,
                "detection_gate_applied": False,
                "raw_detections_mutated": False,
                "human_review_required": True,
                "external_package_disposable_after_verified_publication": True,
            },
            "canonicalization": "canonical_json_sort_keys_v1",
        }
    )
    digest = _payload_sha256(record)
    run_name = f"arena-geometry-fit-review-{digest[:24]}"
    provenance = build_writer_run_provenance(
        command="fisheye.utils.publish_arena_geometry_fit_review",
        params={
            "algorithm_version": FIT_REVIEW_PUBLISH_ALGORITHM_VERSION,
            "run_name": run_name,
            "review_record_sha256": digest,
            "publication_role": "immutable_pre_review_evidence",
        },
        input_run_ids={},
        input_artifacts=tuple(
            {
                "role": artifact.role,
                "path": str(artifact.source_path),
                "sha256": artifact.content_sha256,
            }
            for artifact in artifacts
        ),
        include_system_context=False,
    )
    provenance_check = validate_run_provenance(provenance)
    if not provenance_check.valid:
        raise RuntimeError(
            f"Arena-geometry fit-review provenance is invalid: {provenance_check.errors}"
        )
    return ArenaGeometryFitReviewPlan(
        source_zarr=archive,
        package_dir=package_dir,
        run_name=run_name,
        target_run_path=archive / "analysis" / FIT_REVIEW_RUNS_PARENT / run_name,
        review_record=record,
        review_record_sha256=digest,
        artifacts=tuple(artifacts),
        run_provenance=provenance,
    )


def _run_attrs(plan: ArenaGeometryFitReviewPlan) -> dict[str, Any]:
    return {
        "schema_id": FIT_REVIEW_RUN_SCHEMA_ID,
        "schema_version": FIT_REVIEW_RUN_SCHEMA_VERSION,
        "fit_review_run_id": plan.run_name,
        "review_record": _canonical_copy(plan.review_record),
        "review_record_sha256": plan.review_record_sha256,
        "review_status": "awaiting_explicit_human_review",
        "stage_selector_eligible": False,
        "candidate_published": False,
        "candidate_selected": False,
        "detection_gate_applied": False,
        "run_provenance": _canonical_copy(plan.run_provenance),
    }


def _write_json_array(group: zarr.Group, artifact: FitReviewArtifact) -> None:
    payload = artifact.source_path.read_bytes()
    data = np.frombuffer(payload, dtype=np.uint8)
    array = group.create_array(
        artifact.zarr_path,
        data=data,
        chunks=(max(1, min(len(data), 1_048_576)),),
        overwrite=False,
    )
    array.attrs.update(
        {
            "artifact_schema_id": JSON_BYTES_SCHEMA_ID,
            "artifact_type": "evidence",
            "artifact_role": artifact.role,
            "media_type": "application/json",
            "storage_encoding": "utf8_json_bytes_uint8",
            "content_sha256": artifact.content_sha256,
            "byte_length": artifact.byte_length,
        }
    )


def _materialize_local_run(plan: ArenaGeometryFitReviewPlan, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing existing local fit-review run: {path}")
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.attrs.update(json_attr_safe(_run_attrs(plan)))
    mark_run_started(group, run_name=plan.run_name, stage="arena_geometry_fit_review")
    for artifact in plan.artifacts:
        if artifact.media_type == "application/json":
            _write_json_array(group, artifact)
            continue
        artifact_name = artifact.zarr_path.removeprefix("visualizations/")
        write_png_visualization_artifact(
            group,
            artifact_name,
            artifact.source_path.read_bytes(),
            description=(
                "Early/middle/late acquisition-versus-Palette geometry review montage"
                if artifact.role == "review_montage"
                else f"Arena-geometry review source panel {artifact.role}"
            ),
            created_by="fisheye.diagnostics.probe_recording_dish_rim_fit",
            role="review_snapshot",
            source_paths={"fit_report": FIT_REPORT_ARRAY},
            extra_attrs={"evidence_role": artifact.role},
            overwrite=False,
        )
    validation = validate_arena_geometry_fit_review_run(path, expected_plan=plan)
    if not validation["valid"]:
        raise RuntimeError(f"Local fit-review validation failed: {validation}")


def _array_bytes(group: zarr.Group, path: str) -> bytes:
    try:
        array = group[path]
    except KeyError as exc:
        raise ValueError(f"Fit-review artifact is missing: {path}") from exc
    if np.dtype(array.dtype) != np.dtype(np.uint8) or len(array.shape) != 1:
        raise ValueError(f"Fit-review artifact {path!r} must be rank-1 uint8.")
    return np.asarray(array[:], dtype=np.uint8).tobytes()


def _validate_group(
    group: zarr.Group,
    *,
    expected_plan: ArenaGeometryFitReviewPlan | None,
    require_complete: bool,
) -> tuple[list[str], Mapping[str, Any] | None]:
    errors: list[str] = []
    attrs = dict(group.attrs)
    if (
        attrs.get("schema_id") != FIT_REVIEW_RUN_SCHEMA_ID
        or attrs.get("schema_version") != FIT_REVIEW_RUN_SCHEMA_VERSION
    ):
        errors.append("unsupported fit-review run schema")
    record = attrs.get("review_record")
    if not isinstance(record, Mapping):
        errors.append("review_record missing")
        record = None
    else:
        digest = _payload_sha256(record)
        if attrs.get("review_record_sha256") != digest:
            errors.append("review record digest mismatch")
        if record.get("schema_id") != FIT_REVIEW_RECORD_SCHEMA_ID:
            errors.append("review record schema mismatch")
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, Mapping):
            errors.append("review record artifacts missing")
        else:
            expected_visualizations: dict[str, Mapping[str, Any]] = {}
            for role, raw in artifacts.items():
                if not isinstance(raw, Mapping):
                    errors.append(f"artifact {role} is not an object")
                    continue
                path = str(raw.get("zarr_path") or "")
                try:
                    payload = _array_bytes(group, path)
                except Exception as exc:
                    errors.append(str(exc))
                    continue
                digest = _sha256_bytes(payload)
                if digest != raw.get("content_sha256"):
                    errors.append(f"artifact {role} digest mismatch")
                if len(payload) != raw.get("byte_length"):
                    errors.append(f"artifact {role} byte length mismatch")
                try:
                    node = group[path]
                    node_attrs = dict(node.attrs)
                except Exception as exc:
                    errors.append(f"artifact {role} attrs unavailable: {exc}")
                    continue
                if raw.get("media_type") == "image/png":
                    if not payload.startswith(_PNG_SIGNATURE):
                        errors.append(f"artifact {role} is not PNG")
                    if node_attrs.get("artifact_schema_id") != PNG_ARTIFACT_SCHEMA_ID:
                        errors.append(f"artifact {role} visualization schema mismatch")
                    expected_visualizations[path.removeprefix("visualizations/")] = raw
                elif node_attrs.get("artifact_schema_id") != JSON_BYTES_SCHEMA_ID:
                    errors.append(f"artifact {role} JSON schema mismatch")
            manifest = attrs.get("visualizations")
            if not isinstance(manifest, Mapping):
                errors.append("visualization manifest missing")
            else:
                if set(manifest) != set(expected_visualizations):
                    errors.append("visualization manifest coverage mismatch")
                for name, raw in expected_visualizations.items():
                    entry = manifest.get(name)
                    if not isinstance(entry, Mapping) or (
                        entry.get("content_sha256") != raw.get("content_sha256")
                        or entry.get("path") != raw.get("zarr_path")
                    ):
                        errors.append(f"visualization manifest mismatch for {name}")
    provenance = validate_run_provenance(attrs.get("run_provenance"))
    if not provenance.valid:
        errors.extend(f"run provenance: {item}" for item in provenance.errors)
    if expected_plan is not None:
        expected = _run_attrs(expected_plan)
        for name, value in expected.items():
            if name != "run_provenance" and attrs.get(name) != value:
                errors.append(f"{name} mismatch")
        if record is not None and _canonical_copy(record) != _canonical_copy(
            expected_plan.review_record
        ):
            errors.append("review record differs from expected plan")
    status = attrs.get("palette_run_completion_status")
    if require_complete and status != "complete":
        errors.append("fit-review run is not complete")
    elif status not in {"running", "complete"}:
        errors.append("fit-review run has invalid completion status")
    if attrs.get("stage_selector_eligible") is not False:
        errors.append("fit-review evidence must remain selector-ineligible")
    for name in ("candidate_published", "candidate_selected", "detection_gate_applied"):
        if attrs.get(name) is not False:
            errors.append(f"{name} must be false")
    return errors, record


def validate_arena_geometry_fit_review_run(
    run_path: str | Path,
    *,
    expected_plan: ArenaGeometryFitReviewPlan | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    path = Path(run_path).expanduser().resolve()
    errors: list[str] = []
    record: Mapping[str, Any] | None = None
    try:
        group = open_zarr_root(path, mode="r")
        errors, record = _validate_group(
            group, expected_plan=expected_plan, require_complete=require_complete
        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "valid": not errors,
        "errors": errors,
        "run_name": path.name,
        "review_record_sha256": _payload_sha256(record) if record is not None else None,
        "run_path": str(path),
    }


def load_arena_geometry_fit_review_evidence(
    source_zarr: str | Path,
    *,
    run_name: str,
) -> ArenaGeometryFitReviewEvidence:
    archive = Path(source_zarr).expanduser().resolve()
    name = str(run_name).strip()
    if not name or Path(name).name != name:
        raise ValueError("fit-review run_name must be one safe group name.")
    run_path = archive / "analysis" / FIT_REVIEW_RUNS_PARENT / name
    validation = validate_arena_geometry_fit_review_run(run_path, require_complete=True)
    if not validation["valid"]:
        raise ValueError(f"Fit-review run is invalid: {validation}")
    root = zarr.open_group(str(archive), mode="r", zarr_format=3, use_consolidated=True)
    group = root[f"analysis/{FIT_REVIEW_RUNS_PARENT}/{name}"]
    record = _canonical_copy(group.attrs["review_record"])
    artifacts = record["artifacts"]
    reveal = artifacts.get("acquisition_reveal")
    reveal_ref = str(reveal["zarr_path"]) if isinstance(reveal, Mapping) else None
    return ArenaGeometryFitReviewEvidence(
        source_zarr=archive,
        run_name=name,
        run_path=run_path,
        review_record=record,
        review_record_sha256=str(group.attrs["review_record_sha256"]),
        fit_report_bytes=_array_bytes(group, str(artifacts["fit_report"]["zarr_path"])),
        review_package_bytes=_array_bytes(
            group, str(artifacts["review_package"]["zarr_path"])
        ),
        montage_bytes=_array_bytes(
            group, str(artifacts["review_montage"]["zarr_path"])
        ),
        acquisition_reveal_bytes=(
            _array_bytes(group, reveal_ref) if reveal_ref is not None else None
        ),
        fit_report_ref=(
            f"analysis/{FIT_REVIEW_RUNS_PARENT}/{name}/"
            f"{artifacts['fit_report']['zarr_path']}"
        ),
        montage_ref=(
            f"analysis/{FIT_REVIEW_RUNS_PARENT}/{name}/"
            f"{artifacts['review_montage']['zarr_path']}"
        ),
        acquisition_reveal_ref=(
            f"analysis/{FIT_REVIEW_RUNS_PARENT}/{name}/{reveal_ref}"
            if reveal_ref is not None
            else None
        ),
    )


def publish_arena_geometry_fit_review(
    plan: ArenaGeometryFitReviewPlan,
    *,
    scratch_root: str | Path,
    copy_backend: str = "python",
) -> dict[str, Any]:
    """Publish one complete, immutable, selector-ineligible review package."""

    if plan.target_run_path.exists():
        existing = validate_arena_geometry_fit_review_run(
            plan.target_run_path, expected_plan=plan, require_complete=True
        )
        if not existing["valid"]:
            raise FileExistsError(
                f"Existing fit-review path is not the expected run: {existing}"
            )
        return {"published": False, "status": "already_complete", **existing}
    scratch = Path(scratch_root).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"palette-{plan.run_name}-", dir=scratch
    ) as temporary:
        local_run = Path(temporary) / plan.run_name
        _materialize_local_run(plan, local_run)

        def validate(path: Path) -> dict[str, Any]:
            return validate_arena_geometry_fit_review_run(path, expected_plan=plan)

        def prepare(root: zarr.Group) -> tuple[zarr.Group]:
            analysis = root.require_group("analysis")
            return (require_runs_parent(analysis, FIT_REVIEW_RUNS_PARENT),)

        def complete(_root: zarr.Group, parent: zarr.Group, run: zarr.Group) -> None:
            mark_run_complete(
                run,
                parent_group=parent,
                run_name=plan.run_name,
                run_provenance=plan.run_provenance,
            )

        def verify(root: zarr.Group) -> None:
            parent = root[f"analysis/{FIT_REVIEW_RUNS_PARENT}"]
            run = parent[plan.run_name]
            if (
                run.attrs.get("palette_run_completion_status") != "complete"
                or run.attrs.get("stage_selector_eligible") is not False
                or parent.attrs.get("latest") == plan.run_name
                or parent.attrs.get("latest_complete") == plan.run_name
            ):
                raise RuntimeError("Fit-review evidence became selector-visible.")

        def consolidate(
            _root: zarr.Group, _parent: zarr.Group, _run: zarr.Group
        ) -> None:
            consolidate_metadata_capture_expected_warnings(plan.source_zarr)
            consolidated = zarr.open_group(
                str(plan.source_zarr),
                mode="r",
                zarr_format=3,
                use_consolidated=True,
            )
            run = consolidated[f"analysis/{FIT_REVIEW_RUNS_PARENT}/{plan.run_name}"]
            errors, _record = _validate_group(
                run, expected_plan=plan, require_complete=True
            )
            if errors:
                raise RuntimeError(
                    f"Consolidated fit-review evidence is invalid: {errors}"
                )

        def repair(_target: Path) -> None:
            consolidate_metadata_capture_expected_warnings(plan.source_zarr)

        publication = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=plan.source_zarr,
                local_run_path=local_run,
                target_run_path=plan.target_run_path,
                run_name=plan.run_name,
                lock_suffix="arena-geometry-fit-review-publish",
                publish_schema_id=FIT_REVIEW_PUBLISH_SCHEMA_ID,
                policy="embedded_geometry_review_package_atomic_publish_v1",
                rollback_policy=(
                    "retain_failed_public_tombstone_leave_geometry_selectors_untouched"
                ),
                content_checksum=True,
            ),
            copy_backend=copy_backend,
            validate_run=validate,
            prepare_parents=prepare,
            complete_run=complete,
            verify_pointers=verify,
            activate_run=consolidate,
            repair_failed_publication_visibility=repair,
            accept_persisted_activation_on_callback_error=False,
            payload_metadata={
                "algorithm_version": FIT_REVIEW_PUBLISH_ALGORITHM_VERSION,
                "review_record_sha256": plan.review_record_sha256,
                "human_review_required": True,
                "selection_performed": False,
            },
        )
    final = validate_arena_geometry_fit_review_run(
        plan.target_run_path, expected_plan=plan, require_complete=True
    )
    if not final["valid"]:
        raise RuntimeError(f"Published fit-review run failed validation: {final}")
    return {
        "published": True,
        "status": "complete_awaiting_explicit_human_review",
        "publication": publication,
        **final,
    }


__all__ = [
    "ACQUISITION_REVEAL_ARRAY",
    "ArenaGeometryFitReviewEvidence",
    "ArenaGeometryFitReviewPlan",
    "FIT_REPORT_ARRAY",
    "FIT_REVIEW_RUNS_PARENT",
    "MONTAGE_ARTIFACT",
    "REVIEW_PACKAGE_ARRAY",
    "build_arena_geometry_fit_review_plan",
    "load_arena_geometry_fit_review_evidence",
    "publish_arena_geometry_fit_review",
    "validate_arena_geometry_fit_review_run",
]
