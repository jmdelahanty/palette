"""Fail-closed, read-only geometry evidence helpers for the Marimo reviewer."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis_workflows.materializers.arena_geometry_fit_review import (
    FIT_REVIEW_RECORD_SCHEMA_ID,
    FIT_REVIEW_RUN_SCHEMA_ID,
    FIT_REVIEW_RUNS_PARENT,
    JSON_BYTES_SCHEMA_ID,
    PROBE_SCHEMA_ID,
)
from fisheye.analysis_workflows.materializers.arena_geometry_candidates import (
    ACQUISITION_CANDIDATE_KIND,
)
from fisheye.registry.geometry_review_approval import detection_source_binding
from fisheye.shared.json_safety import strict_json_dumps
from fisheye.shared.plot_artifacts import PNG_ARTIFACT_SCHEMA_ID
from fisheye.shared.zarr_io import open_zarr_root

from .zarr_workspace import ZarrExplorationWorkspace

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_PNG_BYTES = 50 * 1024 * 1024
MAX_JSON_BYTES = 5 * 1024 * 1024
REQUIRED_ARTIFACT_MEDIA_TYPES = {
    "review_montage": "image/png",
    "source_panel_0": "image/png",
    "source_panel_1": "image/png",
    "source_panel_2": "image/png",
    "fit_report": "application/json",
}
OPTIONAL_ARTIFACT_MEDIA_TYPES = {
    "acquisition_reveal": "application/json",
    "review_package": "application/json",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class GeometryEvidenceError(ValueError):
    """Selected immutable evidence is absent, ambiguous, or invalid."""


class GeometryRunSelectionRequired(GeometryEvidenceError):
    """More than one exact immutable run can satisfy the current selection."""

    def __init__(self, run_ids: Sequence[str]) -> None:
        self.run_ids = tuple(run_ids)
        super().__init__(
            "Multiple complete pending fit-review runs exist; choose one exact "
            f"immutable run: {', '.join(self.run_ids)}"
        )


@dataclass(frozen=True)
class FitReviewRunOption:
    run_id: str
    review_record_sha256: str
    review_status: str


@dataclass(frozen=True)
class VerifiedEvidenceArtifact:
    role: str
    zarr_path: str
    media_type: str
    byte_length: int
    content_sha256: str
    payload: bytes
    json_value: Mapping[str, Any] | None


@dataclass(frozen=True)
class GeometryLifecycleEvidence:
    candidates: tuple[Mapping[str, Any], ...]
    comparisons: tuple[Mapping[str, Any], ...]
    selections: tuple[Mapping[str, Any], ...]
    gates: tuple[Mapping[str, Any], ...]
    gate_consumers: tuple[Mapping[str, Any], ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class GeometryReviewEvidence:
    zarr_path: Path
    archive_attrs: Mapping[str, Any]
    run_id: str
    review_record_sha256: str
    review_record: Mapping[str, Any]
    run_attrs: Mapping[str, Any]
    artifacts: Mapping[str, VerifiedEvidenceArtifact]
    fit_report: Mapping[str, Any]
    acquisition_reveal: Mapping[str, Any] | None
    lifecycle: GeometryLifecycleEvidence

    @property
    def montage(self) -> bytes:
        return self.artifacts["review_montage"].payload

    @property
    def source_panels(self) -> tuple[bytes, bytes, bytes]:
        return tuple(
            self.artifacts[f"source_panel_{index}"].payload for index in range(3)
        )  # type: ignore[return-value]


@dataclass(frozen=True)
class GeometryApprovalCandidateOption:
    run_id: str
    candidate_kind: str
    candidate_record_sha256: str


@dataclass(frozen=True)
class GeometryApprovalDetectionOption:
    group_path: str
    run_id: str
    row_count: int
    binding_sha256: str


def dropdown_label_for_value(
    options: Mapping[str, str],
    *,
    selected_value: str,
) -> str:
    """Resolve the display label Marimo requires for one mapped option value."""

    matches = [label for label, value in options.items() if value == selected_value]
    if len(matches) != 1:
        raise GeometryEvidenceError(
            f"Expected one dropdown label for {selected_value!r}; found {len(matches)}."
        )
    return matches[0]


def open_published_geometry_workspace(
    zarr_path: str | Path,
) -> ZarrExplorationWorkspace:
    """Open one immutable canonical archive through consolidated metadata only."""

    path = Path(zarr_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Canonical analysis Zarr was not found: {path}")
    try:
        root = open_zarr_root(path, mode="r", use_consolidated=True)
    except Exception as exc:
        raise GeometryEvidenceError(
            "Published consolidated metadata is missing, stale, or unreadable; "
            "the reviewer will not fall back to unconsolidated traversal: "
            f"{exc}"
        ) from exc
    return ZarrExplorationWorkspace(zarr_path=path, _root=root)


def _safe_run_id(value: object) -> str:
    text = str(value or "").strip()
    if not text or Path(text).name != text or text in {".", ".."}:
        raise GeometryEvidenceError("fit-review run ID must be one safe group name.")
    return text


def _group_names(group: object) -> list[str]:
    group_keys = getattr(group, "group_keys", None)
    if callable(group_keys):
        return sorted(str(name) for name in group_keys())
    keys = getattr(group, "keys", None)
    if callable(keys):
        return sorted(str(name) for name in keys())
    return []


def discover_fit_review_runs(
    workspace: ZarrExplorationWorkspace,
) -> list[FitReviewRunOption]:
    parent_path = f"analysis/{FIT_REVIEW_RUNS_PARENT}"
    try:
        parent = workspace.handle(parent_path)
    except (KeyError, TypeError) as exc:
        raise GeometryEvidenceError(
            f"Canonical Zarr has no embedded fit-review runs at {parent_path}."
        ) from exc
    options: list[FitReviewRunOption] = []
    for name in _group_names(parent):
        run_id = _safe_run_id(name)
        try:
            attrs = dict(workspace.handle(f"{parent_path}/{run_id}").attrs)
        except Exception:
            continue
        if (
            attrs.get("schema_id") != FIT_REVIEW_RUN_SCHEMA_ID
            or attrs.get("schema_version") != 1
            or attrs.get("palette_run_completion_status") != "complete"
            or attrs.get("stage_selector_eligible") is not False
            or attrs.get("review_status") != "awaiting_explicit_human_review"
        ):
            continue
        digest = str(attrs.get("review_record_sha256") or "").strip().lower()
        if _SHA256_RE.fullmatch(digest) is None:
            continue
        options.append(
            FitReviewRunOption(
                run_id=run_id,
                review_record_sha256=digest,
                review_status=str(attrs["review_status"]),
            )
        )
    return options


def resolve_fit_review_run(
    options: Sequence[FitReviewRunOption],
    *,
    requested_run_id: str | None,
) -> FitReviewRunOption:
    if requested_run_id is not None:
        requested = _safe_run_id(requested_run_id)
        selected = next(
            (option for option in options if option.run_id == requested), None
        )
        if selected is None:
            raise GeometryEvidenceError(
                f"Requested fit-review run {requested!r} is not one complete pending "
                "immutable run visible in consolidated metadata."
            )
        return selected
    if not options:
        raise GeometryEvidenceError(
            "No complete pending immutable fit-review run is available."
        )
    if len(options) > 1:
        raise GeometryRunSelectionRequired([option.run_id for option in options])
    return options[0]


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(strict_json_dumps(value).encode("utf-8")).hexdigest()


def _safe_relative_path(value: object, *, role: str) -> str:
    text = str(value or "").strip().strip("/")
    parts = text.split("/") if text else []
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise GeometryEvidenceError(
            f"review_record artifact {role!r} has an unsafe Zarr path."
        )
    return "/".join(parts)


def _required_digest(value: object, *, label: str) -> str:
    digest = str(value or "").strip().lower().removeprefix("sha256:")
    if _SHA256_RE.fullmatch(digest) is None:
        raise GeometryEvidenceError(f"{label} is not a SHA-256 digest.")
    return digest


def _verify_bound_artifact(
    workspace: ZarrExplorationWorkspace,
    *,
    run_path: str,
    role: str,
    binding: Mapping[str, Any],
    expected_media_type: str,
) -> VerifiedEvidenceArtifact:
    if str(binding.get("role") or "") != role:
        raise GeometryEvidenceError(
            f"review_record artifact key {role!r} disagrees with its bound role."
        )
    media_type = str(binding.get("media_type") or "").strip()
    if media_type != expected_media_type:
        raise GeometryEvidenceError(
            f"Artifact {role!r} media type is {media_type!r}; expected "
            f"{expected_media_type!r}."
        )
    relative_path = _safe_relative_path(binding.get("zarr_path"), role=role)
    full_path = f"{run_path}/{relative_path}"
    try:
        info = workspace.info(full_path)
    except Exception as exc:
        raise GeometryEvidenceError(
            f"Bound artifact {role!r} is missing at {relative_path!r}."
        ) from exc
    shape = tuple(info.get("shape") or ())
    if info.get("kind") != "array" or len(shape) != 1:
        raise GeometryEvidenceError(
            f"Bound artifact {role!r} must be one rank-1 byte array."
        )
    if str(info.get("dtype")) != "uint8":
        raise GeometryEvidenceError(f"Bound artifact {role!r} must use uint8 storage.")
    observed_length = int(info.get("elements") or 0)
    declared_length = binding.get("byte_length")
    if (
        isinstance(declared_length, bool)
        or not isinstance(declared_length, int)
        or declared_length <= 0
    ):
        raise GeometryEvidenceError(
            f"Artifact {role!r} has an invalid declared byte length."
        )
    max_bytes = MAX_PNG_BYTES if media_type == "image/png" else MAX_JSON_BYTES
    if declared_length > max_bytes or observed_length > max_bytes:
        raise GeometryEvidenceError(
            f"Artifact {role!r} exceeds the {max_bytes:,}-byte review limit."
        )
    if observed_length != declared_length:
        raise GeometryEvidenceError(
            f"Artifact {role!r} byte length mismatch: review_record declares "
            f"{declared_length:,}, Zarr metadata reports {observed_length:,}."
        )
    expected_digest = _required_digest(
        binding.get("content_sha256"), label=f"artifact {role!r} digest"
    )
    node_attrs = workspace.attrs(full_path, max_items=100, max_value_chars=10_000)
    if node_attrs.get("media_type") != media_type:
        raise GeometryEvidenceError(
            f"Artifact {role!r} node media type disagrees with review_record."
        )
    if int(node_attrs.get("byte_length") or 0) != declared_length:
        raise GeometryEvidenceError(
            f"Artifact {role!r} node byte length disagrees with review_record."
        )
    if (
        _required_digest(
            node_attrs.get("content_sha256"), label=f"artifact {role!r} node digest"
        )
        != expected_digest
    ):
        raise GeometryEvidenceError(
            f"Artifact {role!r} node digest disagrees with review_record."
        )
    if media_type == "image/png":
        if node_attrs.get("artifact_schema_id") != PNG_ARTIFACT_SCHEMA_ID:
            raise GeometryEvidenceError(
                f"Artifact {role!r} does not use the Palette PNG schema."
            )
        _resolved, payload = workspace.load_png(full_path, max_bytes=max_bytes)
        json_value = None
    else:
        if node_attrs.get("artifact_schema_id") != JSON_BYTES_SCHEMA_ID:
            raise GeometryEvidenceError(
                f"Artifact {role!r} does not use the Palette JSON-byte schema."
            )
        payload = np.asarray(
            workspace.read(full_path, max_elements=max_bytes), dtype=np.uint8
        ).tobytes()
        try:
            parsed = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GeometryEvidenceError(
                f"Artifact {role!r} is not valid UTF-8 JSON."
            ) from exc
        if not isinstance(parsed, Mapping):
            raise GeometryEvidenceError(
                f"Artifact {role!r} must contain one JSON object."
            )
        json_value = parsed
    if len(payload) != declared_length:
        raise GeometryEvidenceError(
            f"Artifact {role!r} payload length changed while reading."
        )
    if media_type == "image/png" and not payload.startswith(PNG_SIGNATURE):
        raise GeometryEvidenceError(f"Artifact {role!r} is not a PNG byte stream.")
    observed_digest = hashlib.sha256(payload).hexdigest()
    if observed_digest != expected_digest:
        raise GeometryEvidenceError(
            f"Artifact {role!r} SHA-256 mismatch; refusing stale or altered evidence."
        )
    return VerifiedEvidenceArtifact(
        role=role,
        zarr_path=full_path,
        media_type=media_type,
        byte_length=declared_length,
        content_sha256=expected_digest,
        payload=payload,
        json_value=json_value,
    )


def _detached_record(
    workspace: ZarrExplorationWorkspace,
    path: str,
    *,
    record_attr: str,
    digest_attr: str,
) -> Mapping[str, Any]:
    attrs = dict(workspace.handle(path).attrs)
    record = attrs.get(record_attr)
    if not isinstance(record, Mapping):
        raise GeometryEvidenceError(f"{path} lacks {record_attr}.")
    expected = _required_digest(attrs.get(digest_attr), label=f"{path}.{digest_attr}")
    if _canonical_sha256(record) != expected:
        raise GeometryEvidenceError(f"{path} has an invalid {record_attr} digest.")
    return json.loads(strict_json_dumps(record))


def _complete_children(
    workspace: ZarrExplorationWorkspace, parent_path: str
) -> list[tuple[str, Mapping[str, Any]]]:
    try:
        parent = workspace.handle(parent_path)
    except Exception:
        return []
    result: list[tuple[str, Mapping[str, Any]]] = []
    for name in _group_names(parent):
        path = f"{parent_path}/{name}"
        try:
            attrs = dict(workspace.handle(path).attrs)
        except Exception:
            continue
        if attrs.get("palette_run_completion_status") == "complete":
            result.append((name, attrs))
    return result


def load_geometry_lifecycle(
    workspace: ZarrExplorationWorkspace,
    *,
    fit_review_run: str,
) -> GeometryLifecycleEvidence:
    """Load metadata-only downstream context bound to one fit-review run."""

    errors: list[str] = []
    candidates: list[Mapping[str, Any]] = []
    for name, _attrs in _complete_children(workspace, "analysis/arena_geometry_runs"):
        path = f"analysis/arena_geometry_runs/{name}"
        try:
            record = _detached_record(
                workspace,
                path,
                record_attr="candidate_record",
                digest_attr="candidate_record_sha256",
            )
            source = record.get("palette_fit_source")
            if (
                isinstance(source, Mapping)
                and source.get("fit_review_run") == fit_review_run
            ):
                candidates.append(
                    {
                        "run_id": name,
                        "digest": workspace.handle(path).attrs.get(
                            "candidate_record_sha256"
                        ),
                        "record": record,
                    }
                )
        except GeometryEvidenceError as exc:
            errors.append(str(exc))
    candidate_ids = {str(item["run_id"]) for item in candidates}

    comparisons: list[Mapping[str, Any]] = []
    for name, _attrs in _complete_children(
        workspace, "analysis/arena_geometry_comparison_runs"
    ):
        path = f"analysis/arena_geometry_comparison_runs/{name}"
        try:
            record = _detached_record(
                workspace,
                path,
                record_attr="comparison_record",
                digest_attr="comparison_record_sha256",
            )
            bindings = record.get("candidate_bindings")
            bound_ids = (
                {
                    str(value.get("candidate_id"))
                    for value in bindings.values()
                    if isinstance(bindings, Mapping) and isinstance(value, Mapping)
                }
                if isinstance(bindings, Mapping)
                else set()
            )
            if candidate_ids & bound_ids:
                comparisons.append(
                    {
                        "run_id": name,
                        "digest": workspace.handle(path).attrs.get(
                            "comparison_record_sha256"
                        ),
                        "record": record,
                    }
                )
        except GeometryEvidenceError as exc:
            errors.append(str(exc))
    comparison_ids = {str(item["run_id"]) for item in comparisons}

    selections: list[Mapping[str, Any]] = []
    for name, _attrs in _complete_children(
        workspace, "analysis/arena_geometry_selection"
    ):
        path = f"analysis/arena_geometry_selection/{name}"
        try:
            record = _detached_record(
                workspace,
                path,
                record_attr="selection_record",
                digest_attr="selection_record_sha256",
            )
            decision = record.get("decision")
            comparison = (
                decision.get("comparison_binding")
                if isinstance(decision, Mapping)
                else None
            )
            selected = record.get("selected_candidate")
            if (
                isinstance(comparison, Mapping)
                and comparison.get("run_name") in comparison_ids
            ) or (
                isinstance(selected, Mapping)
                and selected.get("candidate_id") in candidate_ids
            ):
                selections.append(
                    {
                        "run_id": name,
                        "digest": workspace.handle(path).attrs.get(
                            "selection_record_sha256"
                        ),
                        "record": record,
                    }
                )
        except GeometryEvidenceError as exc:
            errors.append(str(exc))
    selection_ids = {str(item["run_id"]) for item in selections}

    gates: list[Mapping[str, Any]] = []
    for name, attrs in _complete_children(workspace, "analysis/detection_gate_runs"):
        if str(attrs.get("selection_run") or "") in selection_ids:
            gates.append(
                {
                    "run_id": name,
                    "selection_run": attrs.get("selection_run"),
                    "selection_record_sha256": attrs.get("selection_record_sha256"),
                    "comparison_run": attrs.get("comparison_run"),
                    "comparison_record_sha256": attrs.get("comparison_record_sha256"),
                    "selected_candidate_record_sha256": attrs.get(
                        "selected_candidate_record_sha256"
                    ),
                }
            )
    gate_ids = {str(item["run_id"]) for item in gates}

    consumers: list[Mapping[str, Any]] = []
    for name, attrs in _complete_children(workspace, "refined_detect_runs"):
        evidence = attrs.get("registered_detection_gate")
        if (
            isinstance(evidence, Mapping)
            and evidence.get("applied") is True
            and str(evidence.get("gate_run") or "") in gate_ids
            and attrs.get("finalized_recording_authority") is True
        ):
            consumers.append({"run_id": name, "gate_evidence": dict(evidence)})
    return GeometryLifecycleEvidence(
        candidates=tuple(candidates),
        comparisons=tuple(comparisons),
        selections=tuple(selections),
        gates=tuple(gates),
        gate_consumers=tuple(consumers),
        errors=tuple(errors),
    )


def load_geometry_review_evidence(
    workspace: ZarrExplorationWorkspace,
    *,
    run_id: str | None = None,
) -> GeometryReviewEvidence:
    options = discover_fit_review_runs(workspace)
    selected = resolve_fit_review_run(options, requested_run_id=run_id)
    run_path = f"analysis/{FIT_REVIEW_RUNS_PARENT}/{selected.run_id}"
    attrs = dict(workspace.handle(run_path).attrs)
    if attrs.get("fit_review_run_id") != selected.run_id:
        raise GeometryEvidenceError("Fit-review run identity disagrees with its path.")
    record = attrs.get("review_record")
    if not isinstance(record, Mapping):
        raise GeometryEvidenceError("Fit-review run lacks review_record.")
    if (
        record.get("schema_id") != FIT_REVIEW_RECORD_SCHEMA_ID
        or record.get("schema_version") != 1
    ):
        raise GeometryEvidenceError("Fit-review review_record schema is unsupported.")
    observed_record_digest = _canonical_sha256(record)
    if observed_record_digest != selected.review_record_sha256:
        raise GeometryEvidenceError("Fit-review review_record SHA-256 mismatch.")
    for field in (
        "candidate_published",
        "candidate_selected",
        "detection_gate_applied",
    ):
        if attrs.get(field) is not False:
            raise GeometryEvidenceError(
                f"Immutable fit-review evidence has unsafe {field}={attrs.get(field)!r}."
            )
    if record.get("fit_frozen_before_acquisition_reveal") is not True:
        raise GeometryEvidenceError(
            "Fit-review evidence does not prove the fit was frozen before reveal."
        )
    bindings = record.get("artifacts")
    if not isinstance(bindings, Mapping):
        raise GeometryEvidenceError("review_record lacks artifact bindings.")
    missing = sorted(set(REQUIRED_ARTIFACT_MEDIA_TYPES) - set(bindings))
    if missing:
        raise GeometryEvidenceError(
            f"review_record is missing required artifact bindings: {', '.join(missing)}"
        )
    verified: dict[str, VerifiedEvidenceArtifact] = {}
    expected = {**REQUIRED_ARTIFACT_MEDIA_TYPES, **OPTIONAL_ARTIFACT_MEDIA_TYPES}
    unsupported = sorted(set(bindings) - set(expected))
    if unsupported:
        raise GeometryEvidenceError(
            "review_record has unsupported or ambiguous artifact roles: "
            + ", ".join(unsupported)
        )
    for role, media_type in expected.items():
        binding = bindings.get(role)
        if binding is None and role in OPTIONAL_ARTIFACT_MEDIA_TYPES:
            continue
        if not isinstance(binding, Mapping):
            raise GeometryEvidenceError(f"Artifact binding {role!r} is malformed.")
        verified[role] = _verify_bound_artifact(
            workspace,
            run_path=run_path,
            role=role,
            binding=binding,
            expected_media_type=media_type,
        )
    fit_report = verified["fit_report"].json_value
    if not isinstance(fit_report, Mapping):
        raise GeometryEvidenceError("Verified fit report is not a JSON object.")
    if (
        fit_report.get("schema_id") != PROBE_SCHEMA_ID
        or fit_report.get("schema_version") != 1
        or fit_report.get("status") != "provisional_visual_review_required"
        or fit_report.get("fit_frozen_before_acquisition_reveal") is not True
    ):
        raise GeometryEvidenceError(
            "Frozen fit-report schema or status is unsupported."
        )
    windows = fit_report.get("windows")
    if not isinstance(windows, Mapping) or set(windows) != {"early", "middle", "late"}:
        raise GeometryEvidenceError(
            "Fit report does not contain exact early/middle/late evidence."
        )
    reveal_artifact = verified.get("acquisition_reveal")
    reveal = reveal_artifact.json_value if reveal_artifact is not None else None
    if reveal is not None:
        reveal_fit = reveal.get("fit_report")
        if (
            not isinstance(reveal_fit, Mapping)
            or _required_digest(
                reveal_fit.get("sha256"), label="acquisition reveal fit-report digest"
            )
            != verified["fit_report"].content_sha256
        ):
            raise GeometryEvidenceError(
                "Acquisition reveal does not bind the exact frozen fit report."
            )
    lifecycle = load_geometry_lifecycle(workspace, fit_review_run=selected.run_id)
    return GeometryReviewEvidence(
        zarr_path=workspace.zarr_path,
        archive_attrs=json.loads(strict_json_dumps(dict(workspace.handle().attrs))),
        run_id=selected.run_id,
        review_record_sha256=selected.review_record_sha256,
        review_record=json.loads(strict_json_dumps(record)),
        run_attrs=json.loads(strict_json_dumps(attrs)),
        artifacts=verified,
        fit_report=fit_report,
        acquisition_reveal=reveal,
        lifecycle=lifecycle,
    )


def discover_geometry_approval_inputs(
    workspace: ZarrExplorationWorkspace,
    *,
    evidence: GeometryReviewEvidence,
) -> tuple[
    tuple[GeometryApprovalCandidateOption, ...],
    tuple[GeometryApprovalDetectionOption, ...],
]:
    """List exact immutable acquisition candidates and raw detection sources."""

    camera_serial = str(
        evidence.review_record.get("source", {}).get("camera_serial") or ""
    ).strip()
    acquisition: list[GeometryApprovalCandidateOption] = []
    for name, attrs in _complete_children(workspace, "analysis/arena_geometry_runs"):
        if attrs.get("stage_selector_eligible") is not True:
            continue
        path = f"analysis/arena_geometry_runs/{name}"
        try:
            record = _detached_record(
                workspace,
                path,
                record_attr="candidate_record",
                digest_attr="candidate_record_sha256",
            )
        except GeometryEvidenceError:
            continue
        arena = record.get("arena_binding")
        if (
            record.get("candidate_kind") != ACQUISITION_CANDIDATE_KIND
            or not isinstance(arena, Mapping)
            or str(arena.get("camera_serial") or "") != camera_serial
        ):
            continue
        acquisition.append(
            GeometryApprovalCandidateOption(
                run_id=name,
                candidate_kind=ACQUISITION_CANDIDATE_KIND,
                candidate_record_sha256=_required_digest(
                    attrs.get("candidate_record_sha256"),
                    label=f"candidate {name} digest",
                ),
            )
        )

    detections: list[GeometryApprovalDetectionOption] = []
    for name, attrs in _complete_children(workspace, "detect_runs"):
        if attrs.get("palette_run_completion_status") != "complete":
            continue
        path = f"detect_runs/{name}"
        try:
            binding = detection_source_binding(workspace.handle(), path)
        except Exception:
            continue
        detections.append(
            GeometryApprovalDetectionOption(
                group_path=path,
                run_id=name,
                row_count=int(binding["row_count"]),
                binding_sha256=str(binding["binding_sha256"]),
            )
        )
    return tuple(acquisition), tuple(detections)


def numerical_fit_rows(evidence: GeometryReviewEvidence) -> list[dict[str, Any]]:
    """Return compact exact fit metrics for early/middle/late display."""

    rows: list[dict[str, Any]] = []
    windows = evidence.fit_report["windows"]
    reveal_files = (
        evidence.acquisition_reveal.get("files", {})
        if isinstance(evidence.acquisition_reveal, Mapping)
        else {}
    )
    for name in ("early", "middle", "late"):
        window = windows[name]
        fit = window.get("fit") if isinstance(window, Mapping) else None
        geometry = fit.get("geometry") if isinstance(fit, Mapping) else None
        center = geometry.get("center_px") if isinstance(geometry, Mapping) else None
        reveal = reveal_files.get(name) if isinstance(reveal_files, Mapping) else None
        dx = reveal.get("delta_center_x_px") if isinstance(reveal, Mapping) else None
        dy = reveal.get("delta_center_y_px") if isinstance(reveal, Mapping) else None
        displacement = (
            float(np.hypot(float(dx), float(dy)))
            if dx is not None and dy is not None
            else None
        )
        rows.append(
            {
                "window": name,
                "center_frame": (
                    window.get("center_frame") if isinstance(window, Mapping) else None
                ),
                "center_x_px": center.get("x") if isinstance(center, Mapping) else None,
                "center_y_px": center.get("y") if isinstance(center, Mapping) else None,
                "radius_px": (
                    geometry.get("radius_px") if isinstance(geometry, Mapping) else None
                ),
                "angular_support": (
                    fit.get("angular_support_fraction")
                    if isinstance(fit, Mapping)
                    else None
                ),
                "radial_residual_px": (
                    fit.get("radial_residual_px") if isinstance(fit, Mapping) else None
                ),
                "median_radial_gradient": (
                    fit.get("median_radial_gradient")
                    if isinstance(fit, Mapping)
                    else None
                ),
                "observed_feature": (
                    fit.get("observed_feature_classification")
                    if isinstance(fit, Mapping)
                    else None
                ),
                "center_displacement_px": displacement,
                "acquisition_reveal_delta_radius_px_diagnostic_only": (
                    reveal.get("delta_radius_px")
                    if isinstance(reveal, Mapping)
                    else None
                ),
            }
        )
    return rows


__all__ = [
    "GeometryEvidenceError",
    "GeometryApprovalCandidateOption",
    "GeometryApprovalDetectionOption",
    "GeometryLifecycleEvidence",
    "GeometryReviewEvidence",
    "GeometryRunSelectionRequired",
    "VerifiedEvidenceArtifact",
    "discover_fit_review_runs",
    "discover_geometry_approval_inputs",
    "dropdown_label_for_value",
    "load_geometry_lifecycle",
    "load_geometry_review_evidence",
    "numerical_fit_rows",
    "open_published_geometry_workspace",
    "resolve_fit_review_run",
]
