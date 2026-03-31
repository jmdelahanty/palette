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
    _component_sync_state,
    _compute_refined_subject_component_apply_rows,
    _finalize_refined_subject_apply,
    _load_source_subject_mask_run,
    _normalize_refined_component_names,
    _normalize_roi_indices,
    _open_existing_refined_subject_run,
    _write_refined_subject_component_apply_rows,
    prepare_refined_subject_run,
)
from ..utils.zarr_io import open_zarr_root


def _parse_roi_indices(text: str) -> list[int]:
    raw = str(text or "").replace(" ", "")
    if not raw:
        raise argparse.ArgumentTypeError("ROI indices must not be empty.")
    try:
        return [int(token) for token in raw.split(",") if token]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ROI indices '{text}'. Expected comma-separated integers.") from exc


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


def _compute_refined_subject_apply_chunk(
    zarr_path: str,
    *,
    source_subject_mask_run: str,
    refined_run: str,
    component_names: Sequence[str],
    roi_indices: Sequence[int],
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="r")
    source = _load_source_subject_mask_run(root, source_subject_mask_run)
    refined = _open_existing_refined_subject_run(root, refined_run)
    edited_masks_batch = np.stack(
        [np.asarray(refined.group["masks_roi"][int(roi_idx)], dtype=np.uint8) for roi_idx in roi_indices],
        axis=0,
    )
    component_updates = {
        str(component_name): _compute_refined_subject_component_apply_rows(
            source=source,
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
) -> dict[str, object]:
    """Recompute refined subject-mask metadata in chunked non-UI mode."""

    stage_start = time.perf_counter()
    zarr_path = str(Path(zarr_path))
    chunk_size = max(1, int(chunk_size))
    scheduler_key = _normalize_scheduler(scheduler)

    root = open_zarr_root(zarr_path, mode="a")
    source, refined = prepare_refined_subject_run(
        root,
        subject_run=subject_run,
        refined_run=refined_run,
        components=components,
    )
    selected_components = _normalize_refined_component_names(refined, components)
    total_rois = int(refined.group["masks_roi"].shape[0])
    if total_rois <= 0:
        raise RuntimeError("refined_subject_masks run has no ROI rows.")
    selected_rows = (
        tuple(_normalize_roi_indices(roi_indices, total_rois))
        if roi_indices is not None
        else tuple(range(total_rois))
    )

    if console is not None:
        console.print(
            f"Applying refined subject masks for [cyan]{refined.run_name}[/cyan] "
            f"(components={list(selected_components)}, rois={len(selected_rows)})"
        )
        console.print(
            f"  Scheduler: [cyan]{scheduler_key}[/cyan] | Chunk size: [cyan]{chunk_size}[/cyan]"
            + (
                f" | Workers: [cyan]{int(num_workers)}[/cyan]"
                if num_workers is not None and scheduler_key != "single-threaded"
                else ""
            )
        )

    before_states = {
        int(roi_idx): _row_component_state(refined, selected_components, int(roi_idx))
        for roi_idx in selected_rows
    }

    row_chunks = _chunk_roi_indices(selected_rows, chunk_size)
    tasks = [
        delayed(_compute_refined_subject_apply_chunk)(
            zarr_path,
            source_subject_mask_run=source.run_name,
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
            )

    _finalize_refined_subject_apply(refined)
    refined.group.attrs["dask_scheduler"] = scheduler_key
    refined.group.attrs["dask_num_workers"] = int(num_workers) if num_workers is not None else None
    refined.group.attrs["dask_chunk_size"] = int(chunk_size)
    refined.group.attrs["dask_version"] = getattr(dask, "__version__", "unknown")

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
        "source_subject_mask_run": source.run_name,
        "component_names": list(selected_components),
        "roi_indices": [int(roi_idx) for roi_idx in selected_rows],
        "roi_count": int(len(selected_rows)),
        "chunk_count": int(len(row_chunks)),
        "chunk_size": int(chunk_size),
        "scheduler": scheduler_key,
        "num_workers": int(num_workers) if num_workers is not None else None,
        "changed_roi_count": int(changed_count),
        "noop_roi_count": int(noop_count),
        "updated_at_utc": str(refined.group.attrs.get("updated_at_utc") or ""),
        "duration_seconds": float(time.perf_counter() - stage_start),
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", help="Path to the Palette zarr archive.")
    parser.add_argument("--subject-run", help="Source subject_mask_runs/<run> override (default: refined run lineage).")
    parser.add_argument("--refined-run", help="Target refined_subject_masks_runs/<run> to open or create.")
    parser.add_argument("--components", nargs="+", help="Optional refined components to recompute (default: all).")
    parser.add_argument(
        "--roi-indices",
        type=_parse_roi_indices,
        help="Optional comma-separated ROI rows to recompute (default: all rows).",
    )
    parser.add_argument("--chunk-size", type=int, default=512, help="Number of ROI rows per compute chunk.")
    parser.add_argument(
        "--scheduler",
        default="processes",
        choices=["threads", "processes", "distributed", "single-threaded"],
        help="Dask scheduler to use for chunked recompute.",
    )
    parser.add_argument("--num-workers", type=int, help="Optional Dask worker count.")
    parser.add_argument("--json", action="store_true", help="Emit the apply summary as JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    console = None if args.json else Console()
    summary = refine_subject_masks(
        args.zarr_path,
        subject_run=args.subject_run,
        refined_run=args.refined_run,
        components=args.components,
        roi_indices=args.roi_indices,
        chunk_size=args.chunk_size,
        scheduler=args.scheduler,
        num_workers=args.num_workers,
        console=console,
    )
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        Console().print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
