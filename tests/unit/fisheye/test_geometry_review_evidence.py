from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import apps.marimo.components.geometry_review as geometry_review
from apps.marimo.components.geometry_review import (
    GeometryEvidenceError,
    GeometryRunSelectionRequired,
    discover_fit_review_runs,
    discover_geometry_approval_inputs,
    dropdown_label_for_value,
    load_geometry_review_evidence,
    numerical_fit_rows,
    open_published_geometry_workspace,
    resolve_fit_review_run,
)
from apps.marimo.components.zarr_workspace import ZarrExplorationWorkspace
from fisheye.analysis_workflows.materializers.arena_geometry_fit_review import (
    FIT_REVIEW_RECORD_SCHEMA_ID,
    FIT_REVIEW_RUN_SCHEMA_ID,
    JSON_BYTES_SCHEMA_ID,
)
from fisheye.shared.json_safety import strict_json_dumps
from fisheye.shared.plot_artifacts import PNG_ARTIFACT_SCHEMA_ID


class _FakeArray:
    def __init__(self, payload: bytes, *, attrs: dict[str, object]) -> None:
        self.values = np.frombuffer(payload, dtype=np.uint8).copy()
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.chunks = self.shape
        self.nbytes = self.values.nbytes
        self.attrs = attrs
        self.reads: list[object] = []

    def __getitem__(self, selection):
        self.reads.append(selection)
        return self.values[selection]


class _FakeGroup:
    def __init__(self, members=None, *, attrs=None) -> None:
        self.members = dict(members or {})
        self.attrs = dict(attrs or {})

    def keys(self):
        return self.members.keys()

    def group_keys(self):
        return [
            name
            for name, value in self.members.items()
            if isinstance(value, _FakeGroup)
        ]

    def __getitem__(self, path: str):
        node: object = self
        for part in str(path).strip("/").split("/"):
            if not part:
                continue
            node = node.members[part]  # type: ignore[attr-defined]
        return node


def _put(group: _FakeGroup, path: str, node: object) -> None:
    parts = path.split("/")
    parent = group
    for part in parts[:-1]:
        child = parent.members.setdefault(part, _FakeGroup())
        assert isinstance(child, _FakeGroup)
        parent = child
    parent.members[parts[-1]] = node


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _fit_report() -> dict[str, object]:
    windows: dict[str, object] = {}
    for index, name in enumerate(("early", "middle", "late")):
        windows[name] = {
            "center_frame": 100 + index * 300,
            "fit": {
                "geometry": {
                    "type": "circle",
                    "center_px": {"x": 320.0 + index, "y": 240.0},
                    "radius_px": 200.0 + index,
                },
                "observed_feature_classification": "unclassified_concentric_rim_edge",
                "angular_support_fraction": 0.98 - index * 0.01,
                "radial_residual_px": 0.25 + index * 0.1,
                "median_radial_gradient": 700.0 - index * 10,
            },
        }
    return {
        "schema_id": "palette.diagnostics.recording_dish_rim_probe",
        "schema_version": 1,
        "status": "provisional_visual_review_required",
        "fit_frozen_before_acquisition_reveal": True,
        "source": {"camera_serial": "2010093"},
        "fit_evidence_contract": {
            "candidate_feature_classification": "unclassified_concentric_rim_edge"
        },
        "temporal_stability_px": {
            "center_x_range": 2.0,
            "center_y_range": 0.0,
            "radius_range": 2.0,
        },
        "windows": windows,
    }


def _artifact(
    role: str,
    path: str,
    media_type: str,
    payload: bytes,
) -> tuple[dict[str, object], _FakeArray]:
    digest = _digest(payload)
    binding = {
        "role": role,
        "source_name": f"{role}.bin",
        "zarr_path": path,
        "media_type": media_type,
        "content_sha256": digest,
        "byte_length": len(payload),
    }
    attrs = {
        "artifact_schema_id": (
            PNG_ARTIFACT_SCHEMA_ID
            if media_type == "image/png"
            else JSON_BYTES_SCHEMA_ID
        ),
        "media_type": media_type,
        "content_sha256": digest,
        "byte_length": len(payload),
    }
    return binding, _FakeArray(payload, attrs=attrs)


def _run(
    run_id: str = "arena-geometry-fit-review-exact",
    *,
    fit_payload: bytes | None = None,
) -> tuple[_FakeGroup, dict[str, _FakeArray]]:
    fit_payload = fit_payload if fit_payload is not None else _json_bytes(_fit_report())
    reveal = {
        "schema_id": "palette.diagnostics.recording_dish_rim_probe.acquisition_reveal",
        "schema_version": 1,
        "fit_report": {"sha256": _digest(fit_payload)},
        "files": {
            name: {
                "delta_center_x_px": float(index + 1),
                "delta_center_y_px": 0.0,
                "delta_radius_px": float(index) - 1.0,
            }
            for index, name in enumerate(("early", "middle", "late"))
        },
    }
    payloads = {
        "review_montage": (
            "bound/montage-custom",
            "image/png",
            b"\x89PNG\r\n\x1a\nmontage",
        ),
        "source_panel_0": (
            "bound/panel-early-custom",
            "image/png",
            b"\x89PNG\r\n\x1a\nearly",
        ),
        "source_panel_1": (
            "bound/panel-middle-custom",
            "image/png",
            b"\x89PNG\r\n\x1a\nmiddle",
        ),
        "source_panel_2": (
            "bound/panel-late-custom",
            "image/png",
            b"\x89PNG\r\n\x1a\nlate",
        ),
        "fit_report": ("bound/fit-custom", "application/json", fit_payload),
        "acquisition_reveal": (
            "bound/reveal-custom",
            "application/json",
            _json_bytes(reveal),
        ),
    }
    run = _FakeGroup()
    bindings: dict[str, object] = {}
    arrays: dict[str, _FakeArray] = {}
    for role, (path, media_type, payload) in payloads.items():
        binding, array = _artifact(role, path, media_type, payload)
        bindings[role] = binding
        arrays[role] = array
        _put(run, path, array)
    record = {
        "schema_id": FIT_REVIEW_RECORD_SCHEMA_ID,
        "schema_version": 1,
        "review_status": "awaiting_explicit_human_review",
        "fit_frozen_before_acquisition_reveal": True,
        "source": {"camera_serial": "2010093"},
        "artifacts": bindings,
    }
    record_digest = hashlib.sha256(
        strict_json_dumps(record).encode("utf-8")
    ).hexdigest()
    run.attrs.update(
        {
            "schema_id": FIT_REVIEW_RUN_SCHEMA_ID,
            "schema_version": 1,
            "fit_review_run_id": run_id,
            "review_record": record,
            "review_record_sha256": record_digest,
            "review_status": "awaiting_explicit_human_review",
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
            "candidate_published": False,
            "candidate_selected": False,
            "detection_gate_applied": False,
        }
    )
    return run, arrays


def _workspace(*runs: tuple[str, _FakeGroup]) -> ZarrExplorationWorkspace:
    parent = _FakeGroup({name: run for name, run in runs})
    root = _FakeGroup({"analysis": _FakeGroup({"arena_geometry_fit_runs": parent})})
    return ZarrExplorationWorkspace(
        zarr_path=Path("/canonical/recording_analysis.zarr"),
        _root=root,
        max_read_elements=100_000_000,
    )


def _refresh_record_digest(run: _FakeGroup) -> None:
    record = run.attrs["review_record"]
    run.attrs["review_record_sha256"] = hashlib.sha256(
        strict_json_dumps(record).encode("utf-8")
    ).hexdigest()


def test_mapped_marimo_dropdown_defaults_use_display_label() -> None:
    direct_path = "/canonical/recording_analysis.zarr"
    direct_options = {direct_path: "__direct__"}
    registry_options = {
        "recording-1 · fit_evidence_awaiting_review · dataset-1": "dataset-1"
    }

    direct_label = dropdown_label_for_value(direct_options, selected_value="__direct__")
    registry_label = dropdown_label_for_value(
        registry_options, selected_value="dataset-1"
    )

    assert direct_label == direct_path
    assert registry_label.startswith("recording-1")


def test_exact_review_record_bindings_drive_montage_and_three_panels() -> None:
    run, arrays = _run()
    workspace = _workspace(("arena-geometry-fit-review-exact", run))
    attrs_before = strict_json_dumps(run.attrs)
    payloads_before = {role: array.values.tobytes() for role, array in arrays.items()}

    evidence = load_geometry_review_evidence(workspace)

    assert strict_json_dumps(run.attrs) == attrs_before
    assert {
        role: array.values.tobytes() for role, array in arrays.items()
    } == payloads_before
    assert evidence.run_id == "arena-geometry-fit-review-exact"
    assert evidence.montage == b"\x89PNG\r\n\x1a\nmontage"
    assert evidence.source_panels == (
        b"\x89PNG\r\n\x1a\nearly",
        b"\x89PNG\r\n\x1a\nmiddle",
        b"\x89PNG\r\n\x1a\nlate",
    )
    assert evidence.artifacts["review_montage"].zarr_path.endswith(
        "bound/montage-custom"
    )
    assert all(array.reads for array in arrays.values())
    assert numerical_fit_rows(evidence)[0]["center_displacement_px"] == 1.0


def test_multiple_complete_runs_require_exact_operator_selection() -> None:
    first, _ = _run("run-first")
    second, _ = _run("run-second")
    workspace = _workspace(("run-first", first), ("run-second", second))
    options = discover_fit_review_runs(workspace)

    with pytest.raises(GeometryRunSelectionRequired, match="run-first"):
        resolve_fit_review_run(options, requested_run_id=None)
    assert resolve_fit_review_run(options, requested_run_id="run-second").run_id == (
        "run-second"
    )
    assert load_geometry_review_evidence(workspace, run_id="run-first").run_id == (
        "run-first"
    )


def test_missing_bound_evidence_fails_visibly() -> None:
    run, _arrays = _run()
    del run.members["bound"].members["panel-middle-custom"]

    with pytest.raises(GeometryEvidenceError, match="source_panel_1.*missing"):
        load_geometry_review_evidence(
            _workspace(("arena-geometry-fit-review-exact", run))
        )


def test_digest_mismatch_fails_before_rendering() -> None:
    run, arrays = _run()
    arrays["review_montage"].values[-1] ^= np.uint8(1)

    with pytest.raises(GeometryEvidenceError, match="SHA-256 mismatch"):
        load_geometry_review_evidence(
            _workspace(("arena-geometry-fit-review-exact", run))
        )


def test_json_digest_and_declared_length_are_validated() -> None:
    run, arrays = _run()
    changed = arrays["fit_report"].values.tobytes().replace(b"2010093", b"2010094", 1)
    arrays["fit_report"].values[:] = np.frombuffer(changed, dtype=np.uint8)
    with pytest.raises(GeometryEvidenceError, match="fit_report.*SHA-256 mismatch"):
        load_geometry_review_evidence(
            _workspace(("arena-geometry-fit-review-exact", run))
        )

    run, arrays = _run()
    binding = run.attrs["review_record"]["artifacts"]["fit_report"]
    binding["byte_length"] += 1
    arrays["fit_report"].attrs["byte_length"] += 1
    _refresh_record_digest(run)
    with pytest.raises(GeometryEvidenceError, match="fit_report.*byte length mismatch"):
        load_geometry_review_evidence(
            _workspace(("arena-geometry-fit-review-exact", run))
        )


def test_malformed_json_with_valid_binding_fails_visibly() -> None:
    run, arrays = _run(fit_payload=b"not-json")

    with pytest.raises(GeometryEvidenceError, match="not valid UTF-8 JSON"):
        load_geometry_review_evidence(
            _workspace(("arena-geometry-fit-review-exact", run))
        )
    assert arrays["review_montage"].reads


def test_oversized_declared_evidence_fails_before_array_read() -> None:
    run, arrays = _run()
    binding = run.attrs["review_record"]["artifacts"]["review_montage"]
    binding["byte_length"] = geometry_review.MAX_PNG_BYTES + 1
    _refresh_record_digest(run)

    with pytest.raises(GeometryEvidenceError, match="exceeds"):
        load_geometry_review_evidence(
            _workspace(("arena-geometry-fit-review-exact", run))
        )
    assert arrays["review_montage"].reads == []


def test_acquisition_reveal_must_bind_frozen_fit_report() -> None:
    run, arrays = _run()
    reveal = json.loads(arrays["acquisition_reveal"].values.tobytes())
    reveal["fit_report"]["sha256"] = "f" * 64
    payload = _json_bytes(reveal)
    binding, replacement = _artifact(
        "acquisition_reveal", "bound/reveal-custom", "application/json", payload
    )
    run.attrs["review_record"]["artifacts"]["acquisition_reveal"] = binding
    run.members["bound"].members["reveal-custom"] = replacement
    _refresh_record_digest(run)

    with pytest.raises(GeometryEvidenceError, match="exact frozen fit report"):
        load_geometry_review_evidence(
            _workspace(("arena-geometry-fit-review-exact", run))
        )


def test_published_open_is_read_only_consolidated_and_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    calls = []
    root = _FakeGroup()

    def _open(path: Path, mode: str, *, use_consolidated: bool):
        calls.append((path, mode, use_consolidated))
        return root

    monkeypatch.setattr(geometry_review, "open_zarr_root", _open)
    workspace = open_published_geometry_workspace(archive)

    assert workspace.handle() is root
    assert calls == [(archive.resolve(), "r", True)]

    def _stale(*_args, **_kwargs):
        raise ValueError("consolidated metadata missing run")

    monkeypatch.setattr(geometry_review, "open_zarr_root", _stale)
    with pytest.raises(GeometryEvidenceError, match="will not fall back"):
        open_published_geometry_workspace(archive)


def test_approval_input_discovery_returns_exact_candidate_and_detection_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, _arrays = _run()
    workspace = _workspace(("arena-geometry-fit-review-exact", run))
    evidence = load_geometry_review_evidence(workspace)
    candidate_record = {
        "candidate_kind": geometry_review.ACQUISITION_CANDIDATE_KIND,
        "arena_binding": {"camera_serial": "2010093"},
    }
    candidate_digest = hashlib.sha256(
        strict_json_dumps(candidate_record).encode("utf-8")
    ).hexdigest()
    candidate = _FakeGroup(
        attrs={
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "candidate_record": candidate_record,
            "candidate_record_sha256": candidate_digest,
        }
    )
    detection = _FakeGroup(attrs={"palette_run_completion_status": "complete"})
    analysis = workspace.handle("analysis")
    analysis.members["arena_geometry_runs"] = _FakeGroup(
        {"acquisition-exact": candidate}
    )
    workspace.handle().members["detect_runs"] = _FakeGroup({"raw-exact": detection})
    monkeypatch.setattr(
        geometry_review,
        "detection_source_binding",
        lambda _root, path: {
            "group_path": path,
            "run_name": "raw-exact",
            "row_count": 42,
            "binding_sha256": "d" * 64,
        },
    )

    candidates, detections = discover_geometry_approval_inputs(
        workspace, evidence=evidence
    )

    assert [item.run_id for item in candidates] == ["acquisition-exact"]
    assert candidates[0].candidate_record_sha256 == candidate_digest
    assert [item.group_path for item in detections] == ["detect_runs/raw-exact"]
    assert detections[0].row_count == 42
