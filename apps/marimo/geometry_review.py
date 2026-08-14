#!/usr/bin/env python3
"""Palette registered-dish geometry evidence reviewer and approval launcher."""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    from datetime import datetime, timezone
    import json
    from pathlib import Path

    import marimo as mo

    from apps.marimo.components.geometry_review import (
        discover_fit_review_runs,
        discover_geometry_approval_inputs,
        dropdown_label_for_value,
        load_geometry_review_evidence,
        numerical_fit_rows,
        open_published_geometry_workspace,
    )
    from apps.marimo.components.registry import discover_geometry_review_queue
    from fisheye.registry.geometry_review import (
        REGISTERED_GEOMETRY_STAGES,
    )
    from fisheye.utils.submit_geometry_review_approval import (
        prepare_geometry_review_approval_submission,
    )

    return (
        Path,
        REGISTERED_GEOMETRY_STAGES,
        datetime,
        discover_fit_review_runs,
        discover_geometry_approval_inputs,
        discover_geometry_review_queue,
        dropdown_label_for_value,
        json,
        load_geometry_review_evidence,
        mo,
        numerical_fit_rows,
        open_published_geometry_workspace,
        prepare_geometry_review_approval_submission,
        timezone,
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
    requested_run_id = str(query.get("run_id") or cli_args.get("run-id") or "").strip()
    include_inactive = str(
        cli_args.get("include-inactive") or "false"
    ).strip().lower() in {"1", "true", "yes", "on"}
    approval_mode = str(cli_args.get("approval-mode") or "disabled").strip().lower()
    if approval_mode not in {"disabled", "dry-run", "submit"}:
        raise ValueError("--approval-mode must be disabled, dry-run, or submit.")
    required_ci_success = str(
        cli_args.get("required-ci-success") or "false"
    ).strip().lower() in {"1", "true", "yes", "on"}
    palette_repo_raw = str(cli_args.get("palette-repo") or "").strip()
    approval_root_raw = str(cli_args.get("approval-root") or "").strip()
    submit_host = str(cli_args.get("submit-host") or "").strip() or None
    reviewer_default = str(cli_args.get("reviewer") or "").strip()
    if approval_mode != "disabled":
        if not registry_raw:
            raise ValueError("Approval modes require registry-backed queue mode.")
        if not palette_repo_raw or not approval_root_raw:
            raise ValueError(
                "Approval modes require --palette-repo and --approval-root."
            )
    if approval_mode == "submit" and not required_ci_success:
        raise ValueError(
            "Submit mode requires --required-ci-success true after required CI passes."
        )
    palette_repo = (
        Path(palette_repo_raw).expanduser().resolve() if palette_repo_raw else None
    )
    approval_root = (
        Path(approval_root_raw).expanduser().resolve() if approval_root_raw else None
    )
    is_registry_mode = bool(registry_raw)
    if registry_raw:
        registry_path = Path(str(registry_raw)).expanduser().resolve()
        queue_items = discover_geometry_review_queue(
            registry_path, include_inactive=include_inactive
        )
        queue_scope = (
            "all geometry states" if include_inactive else "waiting for approval"
        )
        mode_label = f"Registry queue ({queue_scope}): `{registry_path}`"
        direct_path = None
    else:
        direct_path = Path(str(direct_raw)).expanduser().resolve()
        if not direct_path.is_dir():
            raise FileNotFoundError(f"Canonical analysis Zarr not found: {direct_path}")
        registry_path = None
        queue_items = []
        mode_label = f"Direct single-Zarr mode: `{direct_path}`"
    return (
        approval_mode,
        approval_root,
        direct_path,
        is_registry_mode,
        mode_label,
        palette_repo,
        queue_items,
        registry_path,
        requested_dataset_id,
        requested_run_id,
        required_ci_success,
        reviewer_default,
        submit_host,
    )


@app.cell(hide_code=True)
def _(
    direct_path,
    dropdown_label_for_value,
    is_registry_mode,
    mo,
    queue_items,
    requested_dataset_id,
):
    if is_registry_mode and queue_items:
        recording_options = {
            (
                f"{item.recording_id} · {item.geometry_state} · " f"{item.dataset_id}"
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
    elif is_registry_mode:
        if requested_dataset_id:
            raise ValueError(
                "Requested dataset ID is not waiting in the registry geometry queue: "
                f"{requested_dataset_id}"
            )
        recording_options = {
            "No recordings are currently waiting for geometry approval": "__empty__"
        }
        initial_dataset = "__empty__"
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
def _(approval_mode, mo, mode_label, recording_picker):
    if approval_mode == "disabled":
        authority_callout = mo.callout(
            "Read-only mode: this process cannot record a decision, submit jobs, "
            "or modify registry/Zarr data.",
            kind="info",
        )
    elif approval_mode == "dry-run":
        authority_callout = mo.callout(
            "Approval dry-run mode: an explicit decision can persist an immutable "
            "request and LSF plan, but no job is submitted and no canonical Zarr "
            "or registry state is changed.",
            kind="warn",
        )
    else:
        authority_callout = mo.callout(
            "Approval submit mode: a confirmed decision submits a commit-pinned LSF "
            "publication and required-gate postprocessing DAG. The browser never "
            "writes the canonical Zarr directly.",
            kind="warn",
        )
    mo.vstack(
        [
            mo.md("# Registered-dish geometry evidence review"),
            authority_callout,
            mo.md(mode_label),
            recording_picker,
        ]
    )
    return


@app.cell(hide_code=True)
def _(direct_path, queue_items, recording_picker):
    if recording_picker.value == "__empty__":
        selected_queue_item = None
        selected_zarr_path = None
    elif recording_picker.value == "__direct__":
        selected_queue_item = None
        selected_zarr_path = direct_path
    else:
        selected_queue_item = next(
            item for item in queue_items if item.dataset_id == recording_picker.value
        )
        selected_zarr_path = selected_queue_item.zarr_path.expanduser().resolve()
    return selected_queue_item, selected_zarr_path


@app.cell(hide_code=True)
def _(REGISTERED_GEOMETRY_STAGES, is_registry_mode, mo, selected_queue_item):
    if is_registry_mode and selected_queue_item is None:
        registry_state_output = mo.callout(
            "No registry-backed recordings match this queue scope. In the default "
            "scope, that means no recordings are currently waiting for approval.",
            kind="success",
        )
    elif selected_queue_item is None:
        registry_state_output = mo.callout(
            "Direct mode has no registry stage context. Evidence remains available "
            "for one explicit canonical Zarr.",
            kind="info",
        )
    else:
        stage_map = {stage.step_name: stage for stage in selected_queue_item.stages}
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
            else (
                "warn" if "review" in state or "incompatibility" in state else "success"
            )
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
    if selected_zarr_path is None:
        geometry_workspace = None
        fit_run_options = []
        workspace_error = None
    else:
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
def _(
    fit_run_options,
    mo,
    requested_run_id,
    run_picker,
    selected_zarr_path,
    workspace_error,
):
    if selected_zarr_path is None:
        run_status_output = mo.md("")
    elif workspace_error:
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
            candidate_record = geometry_evidence.lifecycle.candidates[0].get(
                "record", {}
            )
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
                "value": (
                    selected_queue_item.dataset_id
                    if selected_queue_item
                    else "direct mode"
                ),
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
                    else (
                        str(lifecycle_arena)
                        if lifecycle_arena is not None
                        else "not embedded"
                    )
                ),
            },
            {"field": "fit-review run ID", "value": geometry_evidence.run_id},
            {
                "field": "review-record SHA-256",
                "value": geometry_evidence.review_record_sha256,
            },
            {
                "field": "publication completion",
                "value": geometry_evidence.run_attrs.get(
                    "palette_run_completion_status"
                ),
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
                    "center displacement px": metrics.get(
                        "center_displacement_native_px"
                    ),
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
                    str(item["run_id"])
                    for item in geometry_evidence.lifecycle.candidates
                ),
            },
            {
                "surface": "comparison",
                "count": len(geometry_evidence.lifecycle.comparisons),
                "exact runs": ", ".join(
                    str(item["run_id"])
                    for item in geometry_evidence.lifecycle.comparisons
                ),
            },
            {
                "surface": "selected geometry",
                "count": len(geometry_evidence.lifecycle.selections),
                "exact runs": ", ".join(
                    str(item["run_id"])
                    for item in geometry_evidence.lifecycle.selections
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
                + "\n".join(
                    f"- {message}" for message in geometry_evidence.lifecycle.errors
                ),
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
                    "The approval section below remains bound to this exact dataset, "
                    "fit-review digest, acquisition candidate, and raw detection run. "
                    "It fails closed if any source or registry binding changes.",
                    kind="info",
                ),
            ],
            gap=2,
        )
    evidence_output
    return


@app.cell(hide_code=True)
def _(
    approval_mode,
    discover_geometry_approval_inputs,
    geometry_evidence,
    geometry_workspace,
    selected_queue_item,
):
    approval_candidates = ()
    approval_detections = ()
    approval_blocker = None
    if approval_mode == "disabled":
        approval_blocker = "Approval controls are disabled for this server process."
    elif selected_queue_item is None:
        approval_blocker = "Approval requires one registry-backed queue item."
    elif not selected_queue_item.actionable:
        approval_blocker = (
            "This recording is visible for diagnostics but is not waiting for approval."
        )
    elif geometry_evidence is None or geometry_workspace is None:
        approval_blocker = "Validated immutable fit-review evidence is not loaded."
    elif (
        geometry_evidence.lifecycle.candidates
        or geometry_evidence.lifecycle.comparisons
        or geometry_evidence.lifecycle.selections
        or geometry_evidence.lifecycle.gates
        or geometry_evidence.lifecycle.gate_consumers
    ):
        approval_blocker = (
            "This fit-review run already has downstream lifecycle artifacts. Browser "
            "approval cannot overwrite or repair a partial or completed publication."
        )
    else:
        try:
            approval_candidates, approval_detections = (
                discover_geometry_approval_inputs(
                    geometry_workspace,
                    evidence=geometry_evidence,
                )
            )
        except Exception as exc:
            approval_blocker = f"Approval input discovery failed closed: {exc}"
        if approval_blocker is None and not approval_candidates:
            approval_blocker = (
                "No complete acquisition geometry candidate matches this camera."
            )
        if approval_blocker is None and not approval_detections:
            approval_blocker = "No complete canonical raw detection run is available."
    return approval_blocker, approval_candidates, approval_detections


@app.cell(hide_code=True)
def _(
    approval_blocker,
    approval_candidates,
    approval_detections,
    approval_mode,
    datetime,
    dropdown_label_for_value,
    mo,
    reviewer_default,
    timezone,
):
    if approval_blocker is not None:
        approval_form = None
        approval_controls_output = mo.vstack(
            [
                mo.md("## Operator approval"),
                mo.callout(approval_blocker, kind="info"),
            ]
        )
    else:
        candidate_options = {
            (
                f"{candidate.run_id} · "
                f"sha256:{candidate.candidate_record_sha256[:12]}…"
            ): candidate.run_id
            for candidate in approval_candidates
        }
        detection_options = {
            (
                f"{detection.run_id} · {detection.row_count:,} rows · "
                f"binding sha256:{detection.binding_sha256[:12]}…"
            ): detection.group_path
            for detection in approval_detections
        }
        first_candidate = approval_candidates[0].run_id
        first_detection = approval_detections[0].group_path
        choice_options = {
            "Use Palette fit as the centroid gate": "palette",
            "Use acquisition fit as the centroid gate": "acquisition",
        }
        semantic_options = {
            "Different projected edge confirmed": "different_feature_confirmed",
            "Same physical feature confirmed": "same_feature_confirmed",
            "Projected edge identity remains unresolved": "projected_edges_unresolved",
        }
        reviewed_at_default = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )

        def _validate_approval(value):
            if not isinstance(value, dict):
                return "Complete the approval form."
            reviewer = str(value.get("reviewer") or "").strip()
            reason = str(value.get("decision_reason") or "").strip()
            selected = str(value.get("selected_candidate_kind") or "").strip()
            confirmation = str(value.get("confirmation") or "").strip()
            if not reviewer:
                return "Reviewer identity is required."
            if not reason:
                return "A decision reason is required."
            expected = f"SELECT {selected.upper()}"
            if confirmation != expected:
                return f"Type {expected} exactly to confirm this decision."
            return None

        approval_fields = mo.ui.dictionary(
            {
                "selected_candidate_kind": mo.ui.radio(
                    options=choice_options,
                    value="Use Palette fit as the centroid gate",
                    label="Geometry to use for bounding-box-centroid gating",
                ),
                "semantic_compatibility": mo.ui.dropdown(
                    options=semantic_options,
                    value="Different projected edge confirmed",
                    label="Relationship between visible edges",
                    full_width=True,
                ),
                "acquisition_candidate_run": mo.ui.dropdown(
                    options=candidate_options,
                    value=dropdown_label_for_value(
                        candidate_options, selected_value=first_candidate
                    ),
                    label="Exact acquisition candidate",
                    full_width=True,
                ),
                "source_detection_group_path": mo.ui.dropdown(
                    options=detection_options,
                    value=dropdown_label_for_value(
                        detection_options, selected_value=first_detection
                    ),
                    label="Exact immutable raw detection source",
                    full_width=True,
                ),
                "reviewer": mo.ui.text(
                    value=reviewer_default,
                    placeholder="operator identity",
                    label="Reviewer",
                    full_width=True,
                ),
                "reviewed_at_utc": mo.ui.text(
                    value=reviewed_at_default,
                    label="Review time (UTC)",
                    disabled=True,
                    full_width=True,
                ),
                "decision_reason": mo.ui.text_area(
                    placeholder=(
                        "Explain why the selected circle is the better operational "
                        "centroid gate for this recording."
                    ),
                    label="Decision reason",
                    rows=4,
                    full_width=True,
                ),
                "confirmation": mo.ui.text(
                    placeholder="SELECT PALETTE or SELECT ACQUISITION",
                    label="Typed confirmation",
                    full_width=True,
                ),
            }
        )
        submit_label = (
            "Freeze approval and submit LSF workflow"
            if approval_mode == "submit"
            else "Freeze approval and build dry-run plan"
        )
        approval_form = mo.ui.form(
            approval_fields,
            submit_button_label=submit_label,
            clear_on_submit=False,
            validate=_validate_approval,
            label="Geometry approval",
        )
        approval_controls_output = mo.vstack(
            [
                mo.md("## Operator approval"),
                mo.callout(
                    "This decision selects only the post-detection bounding-box-centroid "
                    "gate. It does not reinterpret the physical inner-rim raster mask. "
                    "The exact request is immutable and all source bindings are checked "
                    "again by the publication job.",
                    kind="warn",
                ),
                approval_form,
            ],
            gap=2,
        )
    approval_controls_output
    return (approval_form,)


@app.cell(hide_code=True)
def _(
    approval_form,
    approval_mode,
    approval_root,
    geometry_evidence,
    json,
    mo,
    palette_repo,
    prepare_geometry_review_approval_submission,
    registry_path,
    required_ci_success,
    selected_queue_item,
    selected_zarr_path,
    submit_host,
):
    if approval_form is None or approval_form.value is None:
        approval_result_output = mo.md("")
    else:
        form_value = approval_form.value
        try:
            approval_result = prepare_geometry_review_approval_submission(
                registry_path=registry_path,
                dataset_id=selected_queue_item.dataset_id,
                recording_id=selected_queue_item.recording_id,
                analysis_zarr=selected_zarr_path,
                fit_review_run=geometry_evidence.run_id,
                acquisition_candidate_run=form_value["acquisition_candidate_run"],
                source_detection_group_path=form_value["source_detection_group_path"],
                selected_candidate_kind=form_value["selected_candidate_kind"],
                semantic_compatibility=form_value["semantic_compatibility"],
                reviewer=form_value["reviewer"],
                reviewed_at_utc=form_value["reviewed_at_utc"],
                decision_reason=form_value["decision_reason"],
                palette_repo=palette_repo,
                approval_root=approval_root,
                submit=approval_mode == "submit",
                submit_host=submit_host,
                required_ci_success=required_ci_success,
            )
            status = str(approval_result["status"])
            kind = (
                "success"
                if status in {"planned", "submitted", "already_submitted"}
                else "warn"
            )
            approval_result_output = mo.vstack(
                [
                    mo.callout(f"Approval operation status: `{status}`", kind=kind),
                    mo.md(
                        "```json\n"
                        + json.dumps(
                            approval_result, indent=2, sort_keys=True, default=str
                        )
                        + "\n```"
                    ),
                ]
            )
        except Exception as exc:
            approval_result_output = mo.callout(
                "Approval failed closed. No automatic fallback was attempted.\n\n"
                f"`{type(exc).__name__}: {exc}`",
                kind="danger",
            )
    approval_result_output
    return


if __name__ == "__main__":
    app.run()
