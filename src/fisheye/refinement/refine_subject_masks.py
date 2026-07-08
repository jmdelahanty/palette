"""Scheduler-aware non-UI apply path for refined subject-mask runs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import dask
import numpy as np
from dask import delayed
from dask.diagnostics import ProgressBar
from rich.console import Console

try:
    from dask.distributed import Client, LocalCluster

    HAVE_DISTRIBUTED = True
except ImportError:  # pragma: no cover - depends on optional dependency
    Client = None  # type: ignore
    LocalCluster = None  # type: ignore
    HAVE_DISTRIBUTED = False

from ..tune.refined_subject_mask_review import (
    RefinedSubjectMaskRun,
    _component_sync_state,
    _capture_component_sync_states,
    _component_sync_states_changed,
    _compute_refined_subject_component_apply_rows,
    _default_refined_run_name,
    _finalize_refined_subject_apply,
    _load_refined_component_source_runs,
    _load_source_subject_mask_run,
    _normalize_component_list,
    _normalize_refined_component_names,
    _normalize_roi_indices,
    _open_existing_refined_subject_run,
    _write_refined_component_last_update,
    _write_refined_subject_component_apply_rows,
    prepare_refined_subject_run,
)
from ..shared.mask_store import MaskStoreError, open_mask_store
from ..shared.subject_mask_registry_status import emit_refined_subject_mask_stage_completion
from ..shared.zarr_io import open_zarr_root

_REFINED_SUBJECT_MASKS_STATUS_SOURCE = "runtime_refine_subject_masks"


def _parse_roi_index_spec(text: str) -> list[int]:
    raw = str(text or "").replace(" ", "")
    if not raw:
        raise argparse.ArgumentTypeError("ROI indices must not be empty.")
    values: list[int] = []
    for token in raw.split(","):
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid ROI range '{token}'. Expected syntax like '10-20'."
                ) from exc
            if end < start:
                raise argparse.ArgumentTypeError(
                    f"Invalid ROI range '{token}'. Range end must be greater than or equal to start."
                )
            values.extend(range(start, end + 1))
            continue
        try:
            values.append(int(token))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid ROI index '{token}'. Expected integers or ranges like '10-20'."
            ) from exc
    if not values:
        raise argparse.ArgumentTypeError("ROI indices must not be empty.")
    return values


def _parse_scheduler_arg(text: str) -> str:
    scheduler_key = _normalize_scheduler(text)
    normalized_raw = str(text or "").strip().lower()
    if scheduler_key == "processes" and normalized_raw not in {"", "processes"}:
        raise argparse.ArgumentTypeError(
            "Invalid scheduler. Expected one of: threads, processes, distributed, "
            "single-threaded, single-thread, single_thread."
        )
    return scheduler_key


def _normalize_scheduler(scheduler: str) -> str:
    scheduler_key = str(scheduler or "processes").lower()
    if scheduler_key in {"single-thread", "single_thread"}:
        scheduler_key = "single-threaded"
    if scheduler_key not in {"threads", "processes", "distributed", "single-threaded"}:
        scheduler_key = "processes"
    return scheduler_key


def _chunk_roi_indices(roi_indices: Sequence[int], chunk_size: int) -> list[tuple[int, ...]]:
    row_list = [int(idx) for idx in roi_indices]
    return [
        tuple(row_list[start : start + chunk_size])
        for start in range(0, len(row_list), max(1, int(chunk_size)))
    ]


def _resolve_refined_run_name(
    root,
    *,
    source_run: str,
    refined_run: Optional[str],
) -> tuple[str, str, bool]:
    refined_parent = root.get("refined_subject_masks_runs")
    if refined_run is not None:
        target_run = str(refined_run)
        return target_run, "explicit", refined_parent is not None and target_run in refined_parent

    if refined_parent is not None:
        latest = refined_parent.attrs.get("latest")
        if latest and str(latest) in refined_parent:
            candidate = refined_parent[str(latest)]
            if str(candidate.attrs.get("source_subject_mask_run") or "") == source_run:
                return str(latest), "inferred_latest", True

    return _default_refined_run_name(), "inferred_new", False


def _open_existing_refined_subject_run_for_plan(
    root,
    refined_run: str,
) -> tuple[RefinedSubjectMaskRun, int, str, str, str]:
    """Inspect an existing refined run without materializing dense masks.

    The normal review/apply opener is an edit boundary and may create
    ``masks_roi`` from compact ``mask_rle``. Planning runs in read-only mode, so
    it must use the logical mask-store surface instead.
    """

    refined_parent = root.get("refined_subject_masks_runs")
    if refined_parent is None or str(refined_run) not in refined_parent:
        raise RuntimeError(f"refined_subject_masks_runs/{refined_run} not found.")

    run_group = refined_parent[str(refined_run)]
    labels_raw = run_group.attrs.get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)) or not labels_raw:
        raise RuntimeError(f"refined_subject_masks_runs/{refined_run} missing usable mask_labels attr.")
    component_names = tuple(str(item) for item in labels_raw)
    try:
        mask_store = open_mask_store(
            run_group,
            source_path=f"refined_subject_masks_runs/{refined_run}",
            prefer="dense",
        )
    except (MaskStoreError, ValueError) as exc:
        raise RuntimeError(
            f"refined_subject_masks_runs/{refined_run} has no usable dense or compact mask store."
        ) from exc
    if tuple(mask_store.mask_labels) != component_names:
        raise RuntimeError(
            f"refined_subject_masks_runs/{refined_run} mask-store labels "
            f"{list(mask_store.mask_labels)!r} do not match mask_labels {list(component_names)!r}."
        )

    return (
        RefinedSubjectMaskRun(
            run_name=str(refined_run),
            parent=refined_parent,
            group=run_group,
            component_names=component_names,
            component_to_index={name: idx for idx, name in enumerate(component_names)},
        ),
        int(mask_store.n_rows),
        str(mask_store.encoding),
        str(mask_store.storage_surface),
        str(mask_store.storage_path),
    )


def _collect_component_args(
    component_groups: Optional[Sequence[Sequence[str]]],
    component_values: Optional[Sequence[str]],
) -> Optional[list[str]]:
    merged: list[str] = []
    for group in component_groups or ():
        merged.extend(str(value) for value in group)
    for value in component_values or ():
        merged.append(str(value))
    return merged or None


def _collect_roi_index_args(
    roi_specs: Optional[Sequence[Sequence[int]]],
    roi_indices: Optional[Sequence[int]],
) -> Optional[list[int]]:
    merged: list[int] = []
    for spec in roi_specs or ():
        merged.extend(int(value) for value in spec)
    for value in roi_indices or ():
        merged.append(int(value))
    return sorted(merged) if merged else None


def _build_refined_subject_apply_plan(
    zarr_path: str | Path,
    *,
    subject_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    components: Optional[Sequence[str]] = None,
    roi_indices: Optional[Sequence[int]] = None,
    chunk_size: int = 512,
    scheduler: str = "processes",
    num_workers: Optional[int] = None,
) -> dict[str, object]:
    zarr_path = str(Path(zarr_path))
    chunk_size = max(1, int(chunk_size))
    scheduler_key = _normalize_scheduler(scheduler)

    root = open_zarr_root(zarr_path, mode="r")
    explicit_source = _load_source_subject_mask_run(root, subject_run) if subject_run is not None else None
    source_run_name = explicit_source.run_name if explicit_source is not None else ""

    if refined_run is not None:
        refined_parent = root.get("refined_subject_masks_runs")
        target_run = str(refined_run)
        selection_mode = "explicit"
        target_exists = refined_parent is not None and target_run in refined_parent
    else:
        inferred_source = explicit_source or _load_source_subject_mask_run(root, None)
        target_run, selection_mode, target_exists = _resolve_refined_run_name(
            root,
            source_run=inferred_source.run_name,
            refined_run=None,
        )
        source_run_name = inferred_source.run_name

    if target_exists:
        (
            refined,
            total_rois,
            mask_store_encoding,
            mask_storage_surface,
            mask_store_path,
        ) = _open_existing_refined_subject_run_for_plan(root, target_run)
        selected_components = _normalize_refined_component_names(refined, components)
        available_components = list(refined.component_names)
        if not source_run_name:
            source_run_name = str(refined.group.attrs.get("source_subject_mask_run") or "")
    else:
        source = explicit_source or _load_source_subject_mask_run(root, subject_run)
        selected_components = _normalize_component_list(components)
        available_components = [str(label) for label in source.mask_labels]
        total_rois = int(source.masks_roi.shape[0])
        source_run_name = source.run_name
        mask_store_encoding = None
        mask_storage_surface = None
        mask_store_path = None

    selected_rows = (
        tuple(_normalize_roi_indices(roi_indices, total_rois))
        if roi_indices is not None
        else tuple(range(total_rois))
    )
    row_chunks = _chunk_roi_indices(selected_rows, chunk_size)

    return {
        "status": "planned",
        "zarr_path": zarr_path,
        "source_subject_mask_run": str(source_run_name),
        "refined_run": target_run,
        "refined_run_selection": selection_mode,
        "refined_run_exists": bool(target_exists),
        "would_create_refined_run": not bool(target_exists),
        "mutates_archive": False,
        "component_names": list(selected_components),
        "available_component_names": available_components,
        "roi_indices": [int(roi_idx) for roi_idx in selected_rows],
        "roi_count": int(len(selected_rows)),
        "total_roi_count": int(total_rois),
        "refined_mask_store_encoding": mask_store_encoding,
        "refined_mask_storage_surface": mask_storage_surface,
        "refined_mask_store_path": mask_store_path,
        "chunk_count": int(len(row_chunks)),
        "chunk_size": int(chunk_size),
        "scheduler": scheduler_key,
        "num_workers": int(num_workers) if num_workers is not None else None,
    }


def _print_refined_subject_apply_plan(
    console: Console,
    plan: dict[str, object],
    *,
    dry_run: bool,
) -> None:
    target_run = str(plan["refined_run"])
    selection_mode = str(plan["refined_run_selection"])
    if selection_mode == "explicit":
        target_label = "explicit"
    elif selection_mode == "inferred_latest":
        target_label = "inferred latest"
    else:
        target_label = "inferred new"

    if bool(plan["would_create_refined_run"]):
        mutation_line = (
            f"  Mutation: create [cyan]refined_subject_masks_runs/{target_run}[/cyan] "
            "and populate selected rows/components"
        )
    else:
        mutation_line = (
            f"  Mutation: update [cyan]refined_subject_masks_runs/{target_run}[/cyan] "
            "in place for selected rows/components"
        )

    console.print(
        f"Resolved refined subject-mask apply target: [cyan]{target_run}[/cyan] ({target_label})"
    )
    console.print(f"  Source run: [cyan]{plan['source_subject_mask_run']}[/cyan]")
    console.print(mutation_line)
    console.print(
        f"  Components: [cyan]{list(plan['component_names'])}[/cyan] "
        f"| ROI rows: [cyan]{plan['roi_count']}[/cyan]/[cyan]{plan['total_roi_count']}[/cyan]"
    )
    console.print(
        f"  Scheduler: [cyan]{plan['scheduler']}[/cyan] | Chunk size: [cyan]{plan['chunk_size']}[/cyan]"
        + (
            f" | Workers: [cyan]{int(plan['num_workers'])}[/cyan]"
            if plan.get("num_workers") is not None and str(plan["scheduler"]) != "single-threaded"
            else ""
        )
    )
    if dry_run:
        console.print("[yellow]Dry run only. No archive data will be modified.[/yellow]")


def _compute_refined_subject_apply_chunk(
    zarr_path: str,
    *,
    refined_run: str,
    component_names: Sequence[str],
    roi_indices: Sequence[int],
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="r")
    refined = _open_existing_refined_subject_run(root, refined_run)
    _primary_source, component_sources = _load_refined_component_source_runs(root, refined)
    edited_masks_batch = np.stack(
        [np.asarray(refined.group["masks_roi"][int(roi_idx)], dtype=np.uint8) for roi_idx in roi_indices],
        axis=0,
    )
    component_updates = {
        str(component_name): _compute_refined_subject_component_apply_rows(
            component_sources=component_sources,
            refined=refined,
            component_name=str(component_name),
            roi_indices=tuple(int(roi_idx) for roi_idx in roi_indices),
            edited_masks_batch=edited_masks_batch,
        )
        for component_name in component_names
    }
    return {
        "roi_indices": [int(roi_idx) for roi_idx in roi_indices],
        "edited_masks_batch": edited_masks_batch,
        "component_updates": component_updates,
    }


def _row_component_state(
    refined,
    component_names: Sequence[str],
    roi_idx: int,
) -> tuple[object, ...]:
    run_group = refined.group
    components_parent = run_group.require_group("components")
    state: list[object] = []
    for component_name in component_names:
        component_group = components_parent.require_group(str(component_name))
        state.extend(
            _component_sync_state(
                run_group,
                component_group,
                comp_idx=int(refined.component_to_index[str(component_name)]),
                roi_idx=int(roi_idx),
            )
        )
    return tuple(state)


def refine_subject_masks(
    zarr_path: str | Path,
    *,
    subject_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    components: Optional[Sequence[str]] = None,
    roi_indices: Optional[Sequence[int]] = None,
    chunk_size: int = 512,
    scheduler: str = "processes",
    num_workers: Optional[int] = None,
    console: Optional[Console] = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Recompute refined subject-mask metadata in chunked non-UI mode."""

    stage_start = time.perf_counter()
    plan = _build_refined_subject_apply_plan(
        zarr_path,
        subject_run=subject_run,
        refined_run=refined_run,
        components=components,
        roi_indices=roi_indices,
        chunk_size=chunk_size,
        scheduler=scheduler,
        num_workers=num_workers,
    )
    if console is not None:
        _print_refined_subject_apply_plan(console, plan, dry_run=dry_run)
    if dry_run:
        return plan

    zarr_path = str(plan["zarr_path"])
    selected_components = tuple(str(name) for name in plan["component_names"])
    selected_rows = tuple(int(roi_idx) for roi_idx in plan["roi_indices"])
    chunk_size = int(plan["chunk_size"])
    scheduler_key = str(plan["scheduler"])
    resolved_refined_run = str(plan["refined_run"])

    root = open_zarr_root(zarr_path, mode="a")
    default_source = _load_source_subject_mask_run(root, subject_run) if subject_run is not None else None
    if bool(plan["refined_run_exists"]):
        refined = _open_existing_refined_subject_run(root, resolved_refined_run)
        primary_source, component_sources = _load_refined_component_source_runs(
            root,
            refined,
            default_source=default_source,
        )
    else:
        source = default_source or _load_source_subject_mask_run(root, subject_run)
        refined = prepare_refined_subject_run(
            root,
            subject_run=source.run_name,
            refined_run=resolved_refined_run,
            components=selected_components,
        )[1]
        primary_source, component_sources = _load_refined_component_source_runs(
            root,
            refined,
            default_source=source,
        )

    before_states = {
        int(roi_idx): _row_component_state(refined, selected_components, int(roi_idx))
        for roi_idx in selected_rows
    }
    before_component_states = {
        str(component_name): _capture_component_sync_states(
            refined.group,
            str(component_name),
            selected_rows,
            comp_idx=int(refined.component_to_index[str(component_name)]),
        )
        for component_name in selected_components
    }

    row_chunks = _chunk_roi_indices(selected_rows, chunk_size)
    tasks = [
        delayed(_compute_refined_subject_apply_chunk)(
            zarr_path,
            refined_run=refined.run_name,
            component_names=selected_components,
            roi_indices=chunk_rows,
        )
        for chunk_rows in row_chunks
    ]

    results_list: list[dict[str, object]] = []
    cluster = None
    client = None
    if tasks:
        try:
            if scheduler_key == "distributed":
                if not HAVE_DISTRIBUTED:
                    raise RuntimeError(
                        "Dask distributed is not available. Install dask[distributed] "
                        "or choose a different scheduler (e.g. 'processes' or 'threads')."
                    )
                cluster_kwargs: Dict[str, object] = {}
                if num_workers is not None:
                    cluster_kwargs["n_workers"] = int(num_workers)
                cluster = LocalCluster(**cluster_kwargs)
                client = Client(cluster)
                futures = client.compute(tasks)
                results_list = list(client.gather(futures))
            else:
                compute_kwargs: Dict[str, object] = {"scheduler": scheduler_key}
                if num_workers is not None and scheduler_key != "single-threaded":
                    compute_kwargs["num_workers"] = int(num_workers)
                if console is not None:
                    with ProgressBar():
                        compute_result = dask.compute(*tasks, **compute_kwargs)
                else:
                    compute_result = dask.compute(*tasks, **compute_kwargs)
                results_list = list(compute_result) if isinstance(compute_result, tuple) else list(compute_result)
        finally:
            if client is not None:
                client.close()
            if cluster is not None:
                cluster.close()

    for result in results_list:
        chunk_rows = tuple(int(roi_idx) for roi_idx in result["roi_indices"])
        edited_masks_batch = np.asarray(result["edited_masks_batch"], dtype=np.uint8)
        component_updates = dict(result["component_updates"])
        for component_name in selected_components:
            _write_refined_subject_component_apply_rows(
                refined=refined,
                component_name=str(component_name),
                roi_indices=chunk_rows,
                edited_masks_batch=edited_masks_batch,
                component_updates=component_updates[str(component_name)],
                row_update_reason="batch_refined_subject_mask_edit",
            )

    updated_at_utc = _finalize_refined_subject_apply(
        refined,
        updated_components=selected_components,
        updated_rows=selected_rows,
        stale_reason="batch_refined_subject_mask_edit",
    )
    refined.group.attrs["dask_scheduler"] = scheduler_key
    refined.group.attrs["dask_num_workers"] = int(num_workers) if num_workers is not None else None
    refined.group.attrs["dask_chunk_size"] = int(chunk_size)
    refined.group.attrs["dask_version"] = getattr(dask, "__version__", "unknown")
    for component_name in selected_components:
        comp_idx = int(refined.component_to_index[str(component_name)])
        if _component_sync_states_changed(
            refined.group,
            str(component_name),
            selected_rows,
            comp_idx=comp_idx,
            before_states=before_component_states[str(component_name)],
        ):
            _write_refined_component_last_update(
                refined.group,
                component_name=str(component_name),
                updated_at_utc=updated_at_utc,
                update_mode="batch",
                update_method="fisheye.refinement.refine_subject_masks",
            )

    changed_count = 0
    noop_count = 0
    for roi_idx in selected_rows:
        before = before_states[int(roi_idx)]
        after = _row_component_state(refined, selected_components, int(roi_idx))
        if before == after:
            noop_count += 1
        else:
            changed_count += 1

    summary = {
        "status": "updated",
        "zarr_path": zarr_path,
        "refined_run": refined.run_name,
        "source_subject_mask_run": primary_source.run_name,
        "refined_run_selection": str(plan["refined_run_selection"]),
        "refined_run_exists": bool(plan["refined_run_exists"]),
        "would_create_refined_run": bool(plan["would_create_refined_run"]),
        "mutates_archive": True,
        "component_names": list(selected_components),
        "available_component_names": list(plan["available_component_names"]),
        "roi_indices": [int(roi_idx) for roi_idx in selected_rows],
        "roi_count": int(len(selected_rows)),
        "total_roi_count": int(plan["total_roi_count"]),
        "chunk_count": int(len(row_chunks)),
        "chunk_size": int(chunk_size),
        "scheduler": scheduler_key,
        "num_workers": int(num_workers) if num_workers is not None else None,
        "changed_roi_count": int(changed_count),
        "noop_roi_count": int(noop_count),
        "updated_at_utc": str(updated_at_utc),
        "duration_seconds": float(time.perf_counter() - stage_start),
    }
    emit_refined_subject_mask_stage_completion(
        root,
        zarr_path,
        run_group=refined.group,
        run_name=refined.run_name,
        source=_REFINED_SUBJECT_MASKS_STATUS_SOURCE,
        console=console,
        invalidate_on_ok=True,
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", help="Path to the Palette zarr archive.")
    parser.add_argument(
        "--subject-run",
        "--source-run",
        dest="subject_run",
        help="Source subject_mask_runs/<run> override (default: refined run lineage or latest source run).",
    )
    parser.add_argument(
        "--refined-run",
        "--run-name",
        dest="refined_run",
        help=(
            "Target refined_subject_masks_runs/<run> to update in place or create. "
            "If omitted, the latest matching refined run is reused when possible."
        ),
    )
    parser.add_argument(
        "--components",
        nargs="+",
        action="append",
        help="Optional refined components to recompute (space-separated, repeatable).",
    )
    parser.add_argument(
        "--component",
        action="append",
        dest="component_values",
        help="Optional single refined component selector. Repeat to add more components.",
    )
    parser.add_argument(
        "--roi-indices",
        type=_parse_roi_index_spec,
        action="append",
        help="Optional ROI row selector spec. Accepts comma-separated values and ranges like '0,2,5-8'.",
    )
    parser.add_argument(
        "--roi-index",
        type=int,
        action="append",
        dest="roi_index_values",
        help="Optional single ROI row selector. Repeat to add more rows.",
    )
    parser.add_argument("--chunk-size", type=int, default=512, help="Number of ROI rows per compute chunk.")
    parser.add_argument(
        "--scheduler",
        type=_parse_scheduler_arg,
        default="processes",
        help=(
            "Dask scheduler to use for chunked recompute. Accepted values: threads, processes, "
            "distributed, single-threaded, single-thread, single_thread."
        ),
    )
    parser.add_argument("--num-workers", type=int, help="Optional Dask worker count.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the apply, print the resolved target and selections, and exit without mutating the archive.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the apply summary as JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    console = None if args.json else Console()
    components = _collect_component_args(args.components, args.component_values)
    roi_indices = _collect_roi_index_args(args.roi_indices, args.roi_index_values)
    summary = refine_subject_masks(
        args.zarr_path,
        subject_run=args.subject_run,
        refined_run=args.refined_run,
        components=components,
        roi_indices=roi_indices,
        chunk_size=args.chunk_size,
        scheduler=args.scheduler,
        num_workers=args.num_workers,
        console=console,
        dry_run=bool(args.dry_run),
    )
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        Console().print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
