#!/usr/bin/env python3
"""Read-only Palette registered-dish geometry evidence reviewer."""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import marimo as mo

    from apps.marimo.components.geometry_review import (
        discover_fit_review_runs,
        dropdown_label_for_value,
        load_geometry_review_evidence,
        numerical_fit_rows,
        open_published_geometry_workspace,
    )
    from apps.marimo.components.registry import discover_geometry_review_queue
    from fisheye.registry.geometry_review import (
        REGISTERED_GEOMETRY_STAGES,
    )

    return (
        Path,
        REGISTERED_GEOMETRY_STAGES,
        discover_fit_review_runs,
        discover_geometry_review_queue,
        dropdown_label_for_value,
        load_geometry_review_evidence,
        mo,
        numerical_fit_rows,
        open_published_geometry_workspace,
    )


@app.cell(hide_code=True)
def _(Path, discover_geometry_review_queue, mo):
    cli_args = mo.cli_args()
    direct_raw = cli_args.get("zarr-path")
    registry_raw = cli_args.get("registry")
    if bool(direct_raw) == bool(registry_raw):
        raise ValueError(
            "Launch exactly one mode: --zarr-path <analysis.zarr> or "
            "--registry <palette_registry.sqlite>."
        )
    query = mo.query_params()
    requested_dataset_id = str(
        query.get("dataset_id") or cli_args.get("dataset-id") or ""
    ).strip()
    requested_run_id = str(
        query.get("run_id") or cli_args.get("run-id") or ""
    ).strip()
    if registry_raw:
        registry_path = Path(str(registry_raw)).expanduser().resolve()
        queue_items = discover_geometry_review_queue(
            registry_path, include_inactive=True
        )
        if not queue_items:
            raise ValueError(
                "The registry has no analysis datasets with modern geometry stage rows."
            )
        mode_label = f"Registry queue: `{registry_path}`"
        direct_path = None
    else:
        direct_path = Path(str(direct_raw)).expanduser().resolve()
        if not direct_path.is_dir():
            raise FileNotFoundError(f"Canonical analysis Zarr not found: {direct_path}")
        registry_path = None
        queue_items = []
        mode_label = f"Direct single-Zarr mode: `{direct_path}`"
    return (
        direct_path,
        mode_label,
        queue_items,
        requested_dataset_id,
        requested_run_id,
    )


@app.cell(hide_code=True)
def _(
    direct_path,
    dropdown_label_for_value,
    mo,
    queue_items,
    requested_dataset_id,
):
    if queue_items:
        recording_options = {
            (
                f"{item.recording_id} · {item.geometry_state} · "
                f"{item.dataset_id}"
            ): item.dataset_id
            for item in queue_items
        }
        dataset_ids = {item.dataset_id for item in queue_items}
        if requested_dataset_id and requested_dataset_id not in dataset_ids:
            raise ValueError(
                f"Requested dataset ID is not in the registry geometry queue: "
                f"{requested_dataset_id}"
            )
        initial_dataset = (
            requested_dataset_id
            if requested_dataset_id in dataset_ids
            else queue_items[0].dataset_id
        )
    else:
        recording_options = {str(direct_path): "__direct__"}
        initial_dataset = "__direct__"
    initial_recording_label = dropdown_label_for_value(
        recording_options,
        selected_value=initial_dataset,
    )
    recording_picker = mo.ui.dropdown(
        options=recording_options,
        value=initial_recording_label,
        label="Recording / canonical analysis Zarr",
        full_width=True,
    )
    return (recording_picker,)


@app.cell(hide_code=True)
def _(mo, mode_label, recording_picker):
    mo.vstack(
        [
            mo.md("# Registered-dish geometry evidence review"),
            mo.callout(
                "Read-only interface: this app cannot review, select, publish, "
                "gate, refine, or modify registry/Zarr data.",
                kind="info",
            ),
            mo.md(mode_label),
            recording_picker,
        ]
    )
    return


@app.cell(hide_code=True)
def _(direct_path, queue_items, recording_picker):
    if recording_picker.value == "__direct__":
        selected_queue_item = None
        selected_zarr_path = direct_path
    else:
        selected_queue_item = next(
            item for item in queue_items if item.dataset_id == recording_picker.value
        )
        selected_zarr_path = selected_queue_item.zarr_path.expanduser().resolve()
    return selected_queue_item, selected_zarr_path


@app.cell(hide_code=True)
def _(REGISTERED_GEOMETRY_STAGES, mo, selected_queue_item):
    if selected_queue_item is None:
        registry_state_output = mo.callout(
            "Direct mode has no registry stage context. Evidence remains available "
            "for one explicit canonical Zarr.",
            kind="info",
        )
    else:
        stage_map = {
            stage.step_name: stage for stage in selected_queue_item.stages
        }
        stage_rows = []
        for stage_name in REGISTERED_GEOMETRY_STAGES:
            stage = stage_map.get(stage_name)
            stage_rows.append(
                {
                    "stage": stage_name,
                    "status": stage.status if stage is not None else "not registered",
                    "review state": stage.review_state if stage is not None else None,
                    "exact run": stage.run_name if stage is not None else None,
                    "updated UTC": stage.updated_utc if stage is not None else None,
                }
            )
        state = selected_queue_item.geometry_state
        status_kind = (
            "danger"
            if "error" in state or "failure" in state
            else "warn"
            if "review" in state or "incompatibility" in state
            else "success"
        )
        registry_state_output = mo.vstack(
            [
                mo.callout(f"Registry geometry state: `{state}`", kind=status_kind),
                mo.md("## All six modern registry stages"),
                mo.ui.table(stage_rows, selection=None),
            ]
        )
    registry_state_output
    return


@app.cell(hide_code=True)
def _(
    discover_fit_review_runs,
    open_published_geometry_workspace,
    selected_zarr_path,
):
    try:
        geometry_workspace = open_published_geometry_workspace(selected_zarr_path)
        fit_run_options = discover_fit_review_runs(geometry_workspace)
        workspace_error = None
    except Exception as exc:
        geometry_workspace = None
        fit_run_options = []
        workspace_error = str(exc)
    return fit_run_options, geometry_workspace, workspace_error


@app.cell(hide_code=True)
def _(fit_run_options, mo, requested_run_id):
    run_values = {option.run_id: option.run_id for option in fit_run_options}
    if requested_run_id in run_values:
        initial_run = requested_run_id
    elif len(fit_run_options) == 1 and not requested_run_id:
        initial_run = fit_run_options[0].run_id
    else:
        initial_run = None
    run_picker = mo.ui.dropdown(
        options=run_values,
        value=initial_run,
        label="Exact immutable fit-review run",
        full_width=True,
    )
    return (run_picker,)


@app.cell(hide_code=True)
def _(fit_run_options, mo, requested_run_id, run_picker, workspace_error):
    if workspace_error:
        run_status_output = mo.callout(workspace_error, kind="danger")
    elif not fit_run_options:
        run_status_output = mo.callout(
            "No complete pending fit-review run is visible through published "
            "consolidated metadata.",
            kind="danger",
        )
    elif requested_run_id and requested_run_id not in {
        option.run_id for option in fit_run_options
    }:
        run_status_output = mo.vstack(
            [
                mo.callout(
                    f"Requested immutable run is unavailable or invalid: "
                    f"`{requested_run_id}`. Nothing else was selected.",
                    kind="danger",
                ),
                run_picker,
            ]
        )
    elif len(fit_run_options) > 1 and run_picker.value is None:
        run_status_output = mo.vstack(
            [
                mo.callout(
                    "Ambiguous evidence: more than one complete pending run exists. "
                    "Choose an exact immutable run; no newest run is selected automatically.",
                    kind="warn",
                ),
                run_picker,
            ]
        )
    else:
        run_status_output = run_picker
    run_status_output
    return


@app.cell(hide_code=True)
def _(geometry_workspace, load_geometry_review_evidence, run_picker):
    if geometry_workspace is not None and run_picker.value:
        try:
            geometry_evidence = load_geometry_review_evidence(
                geometry_workspace, run_id=str(run_picker.value)
            )
            evidence_error = None
        except Exception as exc:
            geometry_evidence = None
            evidence_error = str(exc)
    else:
        geometry_evidence = None
        evidence_error = None
    return evidence_error, geometry_evidence


@app.cell(hide_code=True)
def _(
    evidence_error,
    geometry_evidence,
    mo,
    numerical_fit_rows,
    selected_queue_item,
    selected_zarr_path,
):
    if evidence_error:
        evidence_output = mo.callout(
            "Evidence validation failed closed; nothing was rendered.\n\n"
            f"`{evidence_error}`",
            kind="danger",
        )
    elif geometry_evidence is None:
        evidence_output = mo.md("")
    else:
        lifecycle_arena = None
        if geometry_evidence.lifecycle.candidates:
            candidate_record = geometry_evidence.lifecycle.candidates[0].get("record", {})
            if isinstance(candidate_record, dict):
                lifecycle_arena = candidate_record.get("arena_binding")
        source = geometry_evidence.review_record.get("source", {})
        fit_contract = geometry_evidence.fit_report.get("fit_evidence_contract", {})
        identity_rows = [
            {
                "field": "recording_id",
                "value": (
                    selected_queue_item.recording_id
                    if selected_queue_item is not None
                    else geometry_evidence.archive_attrs.get("recording_id")
                    or geometry_evidence.fit_report.get("recording_id")
                    or "not embedded"
                ),
            },
            {"field": "canonical Zarr", "value": str(selected_zarr_path)},
            {
                "field": "dataset_id",
                "value": selected_queue_item.dataset_id if selected_queue_item else "direct mode",
            },
            {
                "field": "camera serial",
                "value": source.get("camera_serial")
                or (selected_queue_item.camera_serial if selected_queue_item else None),
            },
            {
                "field": "arena identity",
                "value": (
                    selected_queue_item.arena_id
                    if selected_queue_item and selected_queue_item.arena_id
                    else str(lifecycle_arena)
                    if lifecycle_arena is not None
                    else "not embedded"
                ),
            },
            {"field": "fit-review run ID", "value": geometry_evidence.run_id},
            {
                "field": "review-record SHA-256",
                "value": geometry_evidence.review_record_sha256,
            },
            {
                "field": "publication completion",
                "value": geometry_evidence.run_attrs.get("palette_run_completion_status"),
            },
            {
                "field": "review status",
                "value": geometry_evidence.run_attrs.get("review_status"),
            },
            {
                "field": "stage_selector_eligible",
                "value": geometry_evidence.run_attrs.get("stage_selector_eligible"),
            },
            {
                "field": "fit-review candidate_published",
                "value": geometry_evidence.run_attrs.get("candidate_published"),
            },
            {
                "field": "fit-review candidate_selected",
                "value": geometry_evidence.run_attrs.get("candidate_selected"),
            },
            {
                "field": "fit-review detection_gate_applied",
                "value": geometry_evidence.run_attrs.get("detection_gate_applied"),
            },
            {
                "field": "fit_frozen_before_acquisition_reveal",
                "value": geometry_evidence.review_record.get(
                    "fit_frozen_before_acquisition_reveal"
                ),
            },
            {
                "field": "observed-feature classification",
                "value": fit_contract.get("candidate_feature_classification"),
            },
            {
                "field": "early/middle/late stability (px)",
                "value": str(geometry_evidence.fit_report.get("temporal_stability_px")),
            },
        ]

        comparison_rows = []
        for item in geometry_evidence.lifecycle.comparisons:
            record = item["record"]
            features = record.get("observed_features", {})
            metrics = record.get("geometry", {})
            same_feature = metrics.get("same_feature_physical_boundary_metrics")
            decision = record.get("decision", {})
            comparison_rows.append(
                {
                    "run": item["run_id"],
                    "digest": item["digest"],
                    "result": decision.get("evidence_outcome"),
                    "workflow action": decision.get("workflow_action"),
                    "semantic compatibility": features.get("semantic_compatibility"),
                    "Palette observed feature": features.get("palette"),
                    "center displacement px": metrics.get("center_displacement_native_px"),
                    "center displacement mm": metrics.get(
                        "center_displacement_dish_top_rim_mm"
                    ),
                    "signed radius difference px": (
                        same_feature.get("signed_radius_difference_px")
                        if isinstance(same_feature, dict)
                        else "not semantically valid"
                    ),
                    "absolute radius difference px": (
                        same_feature.get("absolute_radius_difference_px")
                        if isinstance(same_feature, dict)
                        else "not semantically valid"
                    ),
                }
            )

        lifecycle_rows = [
            {
                "surface": "reviewed candidate",
                "count": len(geometry_evidence.lifecycle.candidates),
                "exact runs": ", ".join(
                    str(item["run_id"]) for item in geometry_evidence.lifecycle.candidates
                ),
            },
            {
                "surface": "comparison",
                "count": len(geometry_evidence.lifecycle.comparisons),
                "exact runs": ", ".join(
                    str(item["run_id"]) for item in geometry_evidence.lifecycle.comparisons
                ),
            },
            {
                "surface": "selected geometry",
                "count": len(geometry_evidence.lifecycle.selections),
                "exact runs": ", ".join(
                    str(item["run_id"]) for item in geometry_evidence.lifecycle.selections
                ),
            },
            {
                "surface": "registered detection gate",
                "count": len(geometry_evidence.lifecycle.gates),
                "exact runs": ", ".join(
                    str(item["run_id"]) for item in geometry_evidence.lifecycle.gates
                ),
            },
            {
                "surface": "gate/refinement consumption",
                "count": len(geometry_evidence.lifecycle.gate_consumers),
                "exact runs": ", ".join(
                    str(item["run_id"])
                    for item in geometry_evidence.lifecycle.gate_consumers
                ),
            },
        ]

        panels = geometry_evidence.source_panels
        comparison_block = (
            mo.ui.table(comparison_rows, selection=None)
            if comparison_rows
            else mo.callout(
                "No comparison bound to this exact fit-review run is present. "
                "The pipeline must publish comparison evidence after explicit review.",
                kind="warn",
            )
        )
        lifecycle_errors = (
            mo.callout(
                "Downstream metadata validation warnings:\n\n"
                + "\n".join(f"- {message}" for message in geometry_evidence.lifecycle.errors),
                kind="danger",
            )
            if geometry_evidence.lifecycle.errors
            else mo.md("")
        )
        evidence_output = mo.vstack(
            [
                mo.md("## Exact identity and immutable policy state"),
                mo.ui.table(identity_rows, selection=None),
                mo.md("## Acquisition-versus-Palette montage"),
                mo.image(
                    geometry_evidence.montage,
                    alt=f"Geometry review montage for {geometry_evidence.run_id}",
                    width="100%",
                ),
                mo.md("## Early, middle, and late source panels"),
                mo.hstack(
                    [
                        mo.image(panels[index], alt=label, width="100%")
                        for index, label in enumerate(("early", "middle", "late"))
                    ],
                    widths="equal",
                    gap=1,
                ),
                mo.md("## Numerical fit evidence"),
                mo.callout(
                    "Per-window acquisition-reveal deltas are diagnostic overlays. "
                    "Signed/absolute physical radius differences are shown only in "
                    "the comparison table when semantic same-feature evidence permits them.",
                    kind="info",
                ),
                mo.ui.table(numerical_fit_rows(geometry_evidence), selection=None),
                mo.md("## Comparison and semantic compatibility"),
                comparison_block,
                mo.md("## Selection, gate, and refinement consumption"),
                mo.ui.table(lifecycle_rows, selection=None),
                lifecycle_errors,
                mo.callout(
                    "Operator handoff: use the exact run IDs and digests above in "
                    "the pipeline's explicit review/publication commands. This page "
                    "does not write the decision or advance any stage.",
                    kind="info",
                ),
            ],
            gap=2,
        )
    evidence_output
    return


if __name__ == "__main__":
    app.run()
