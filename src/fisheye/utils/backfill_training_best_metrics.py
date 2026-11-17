"""Rebind historical detect/pose training metrics to the registered best.pt.

Before #465/#466 (2026-11-17) the detect and pose trainers recorded the LAST
epoch's validation metrics while registering ``best.pt``; under early stopping
those are different epochs. This backfill rewrites, for each detect/pose
``training_runs`` row (and its ``training_models`` row), the metric keys with
the values Ultralytics embedded in ``best.pt`` itself (``train_metrics``) and
adds the same binding fields the fixed trainers now write
(``metrics_source``, ``best_epoch``, ``fitness``, ``last_epoch``,
``last_epoch_metrics``), plus a ``metrics_backfill`` block holding the previous
values. Jeremy approved rewriting rows whose metrics are not valid for the
registered weights (2026-11-17).

Classification is an enforcement/data correction of historical evidence:

- ``corrected``: stored values differ from best.pt's (they were the last epoch's);
- ``filled``: the row had no metric values at all;
- ``confirmed``: stored values already equal best.pt's (best epoch == last
  epoch); values are unchanged and only the binding fields are added;
- ``already_bound``: the row already carries ``metrics_source`` from the fixed
  trainers; untouched.

A row is planned only when its registered ``model_sha256`` and
``metrics_sha256`` still match the files on disk, so the metrics are read from
exactly the registered artifacts. Anything else is reported as skipped.

Dry run by default. ``--apply`` writes through ``publish_registry_shadow``
(backup, local candidate, publish only if the canonical registry did not
change). Inside the mutation every row's ``final_metrics_json`` must still be
byte-identical to the planned "before" value, or the whole publication fails.
A receipt holding every before/after payload is written to ``--receipt-dir``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Mapping, Optional, Sequence

from fisheye.training.training_run_shared import (
    BEST_CHECKPOINT_METRICS_SOURCE,
    ULTRALYTICS_BOX_METRIC_COLUMNS,
    ULTRALYTICS_POSE_METRIC_COLUMNS,
    best_checkpoint_validation_metrics,
    results_row_epoch,
    results_row_metrics,
)

TOOL = "fisheye.utils.backfill_training_best_metrics"
RECEIPT_SCHEMA = "palette.training_best_metrics_backfill.v1"
TASK_COLUMNS: dict[str, dict[str, str]] = {
    "detect": ULTRALYTICS_BOX_METRIC_COLUMNS,
    "pose": ULTRALYTICS_POSE_METRIC_COLUMNS,
}
_VALUE_TOLERANCE = 5e-6
APPLY_CLASSIFICATIONS = ("corrected", "filled", "confirmed")


@dataclass
class TableUpdate:
    table: str
    before_final_metrics_json: Optional[str]
    after_final_metrics_json: str
    before_metadata_json: Optional[str] = None
    after_metadata_json: Optional[str] = None


@dataclass
class PlannedRow:
    run_id: str
    task_type: str
    classification: str
    model_path: str
    model_sha256: str
    metrics_path: str
    metrics_sha256: str
    best_epoch: Optional[int]
    last_epoch: Optional[int]
    previous_values: dict[str, Optional[float]]
    bound_values: dict[str, float]
    updates: list[TableUpdate] = field(default_factory=list)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dict(raw: Optional[str]) -> dict[str, Any]:
    if raw is None or str(raw).strip() == "":
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object, got {type(payload).__name__}")
    return payload


def _values_match(stored: Mapping[str, Any], bound: Mapping[str, float]) -> bool:
    for key, value in bound.items():
        current = stored.get(key)
        if current is None:
            return False
        try:
            if abs(float(current) - float(value)) > _VALUE_TOLERANCE:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _classify(stored: Mapping[str, Any], bound: Mapping[str, float]) -> str:
    if stored.get("metrics_source") == BEST_CHECKPOINT_METRICS_SOURCE:
        return "already_bound"
    if all(stored.get(key) is None for key in bound):
        return "filled"
    return "confirmed" if _values_match(stored, bound) else "corrected"


def rebound_final_metrics(
    stored: Mapping[str, Any],
    *,
    bound: Mapping[str, Any],
    columns: Mapping[str, str],
    last_epoch: Optional[int],
    last_epoch_metrics: Mapping[str, Optional[float]],
    classification: str,
    model_sha256: str,
    applied_utc: str,
) -> dict[str, Any]:
    """Return ``stored`` with its metric keys rebound to best.pt's values."""
    updated = dict(stored)
    previous = {key: stored.get(key) for key in columns}
    for key in columns:
        updated[key] = float(bound[key])
    updated["fitness"] = bound.get("fitness")
    updated["best_epoch"] = bound.get("best_epoch")
    updated["metrics_checkpoint"] = bound.get("metrics_checkpoint", "best.pt")
    updated["metrics_source"] = bound.get("metrics_source", BEST_CHECKPOINT_METRICS_SOURCE)
    updated["last_epoch"] = last_epoch
    updated["last_epoch_metrics"] = dict(last_epoch_metrics)
    updated["metrics_backfill"] = {
        "schema_id": RECEIPT_SCHEMA,
        "tool": TOOL,
        "applied_utc": applied_utc,
        "classification": classification,
        "previous_values": previous,
        "model_sha256": model_sha256,
    }
    return updated


def plan(
    registry: Path,
    *,
    tasks: Sequence[str] = ("detect", "pose"),
    run_ids: Optional[Sequence[str]] = None,
    applied_utc: Optional[str] = None,
) -> tuple[list[PlannedRow], list[dict[str, Any]]]:
    """Plan the rebinding. Reads the registry read-only and the run files."""
    import pandas as pd

    stamp = applied_utc or datetime.now(timezone.utc).isoformat()
    wanted = set(run_ids or [])
    planned: list[PlannedRow] = []
    skipped: list[dict[str, Any]] = []
    con = sqlite3.connect(f"file:{registry}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT run_id, task_type, model_path, metrics_path, model_sha256,
                   metrics_sha256, final_metrics_json
            FROM training_runs ORDER BY created_utc, run_id
            """
        ).fetchall()
        for row in rows:
            run_id = str(row["run_id"])
            task_type = str(row["task_type"] or "")
            if wanted and run_id not in wanted:
                continue
            if task_type not in tasks or task_type not in TASK_COLUMNS:
                continue
            columns = TASK_COLUMNS[task_type]

            def skip(reason: str, **extra: Any) -> None:
                skipped.append({"run_id": run_id, "task_type": task_type, "reason": reason, **extra})

            model_path = Path(str(row["model_path"] or ""))
            metrics_path = Path(str(row["metrics_path"] or ""))
            if not row["model_path"] or not model_path.is_file():
                skip("model_missing", model_path=row["model_path"])
                continue
            if not row["metrics_path"] or not metrics_path.is_file():
                skip("metrics_missing", metrics_path=row["metrics_path"])
                continue
            model_sha = _sha256_file(model_path)
            if not row["model_sha256"] or model_sha != row["model_sha256"]:
                skip("model_sha256_mismatch", registered=row["model_sha256"], on_disk=model_sha)
                continue
            metrics_sha = _sha256_file(metrics_path)
            if not row["metrics_sha256"] or metrics_sha != row["metrics_sha256"]:
                skip("metrics_sha256_mismatch", registered=row["metrics_sha256"], on_disk=metrics_sha)
                continue
            try:
                stored = _json_dict(row["final_metrics_json"])
                results_df = pd.read_csv(metrics_path)
                results_df.columns = results_df.columns.str.strip()
                bound = best_checkpoint_validation_metrics(
                    weights_path=model_path, results_df=results_df, columns=columns
                )
            except Exception as exc:  # reported, never applied
                skip("binding_failed", error=f"{type(exc).__name__}: {exc}")
                continue

            classification = _classify(stored, {key: bound[key] for key in columns})
            last_row = results_df.iloc[-1]
            last_epoch = results_row_epoch(last_row)
            last_metrics = results_row_metrics(last_row, columns)
            item = PlannedRow(
                run_id=run_id,
                task_type=task_type,
                classification=classification,
                model_path=str(model_path),
                model_sha256=model_sha,
                metrics_path=str(metrics_path),
                metrics_sha256=metrics_sha,
                best_epoch=bound.get("best_epoch"),
                last_epoch=last_epoch,
                previous_values={key: stored.get(key) for key in columns},
                bound_values={key: float(bound[key]) for key in columns},
            )
            if classification in APPLY_CLASSIFICATIONS:
                rebound_kwargs = dict(
                    bound=bound,
                    columns=columns,
                    last_epoch=last_epoch,
                    last_epoch_metrics=last_metrics,
                    classification=classification,
                    model_sha256=model_sha,
                    applied_utc=stamp,
                )
                item.updates.append(
                    TableUpdate(
                        table="training_runs",
                        before_final_metrics_json=row["final_metrics_json"],
                        after_final_metrics_json=json.dumps(
                            rebound_final_metrics(stored, **rebound_kwargs), sort_keys=True
                        ),
                    )
                )
                model_row = con.execute(
                    "SELECT final_metrics_json, metadata_json, model_sha256 FROM training_models WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if model_row is not None:
                    if model_row["model_sha256"] and model_row["model_sha256"] != model_sha:
                        skip("training_models_sha256_mismatch", registered=model_row["model_sha256"])
                        continue
                    model_metrics = _json_dict(model_row["final_metrics_json"])
                    metadata = _json_dict(model_row["metadata_json"])
                    if bound.get("best_epoch") is not None:
                        metadata["best_epoch"] = bound["best_epoch"]
                    item.updates.append(
                        TableUpdate(
                            table="training_models",
                            before_final_metrics_json=model_row["final_metrics_json"],
                            after_final_metrics_json=json.dumps(
                                rebound_final_metrics(model_metrics, **rebound_kwargs), sort_keys=True
                            ),
                            before_metadata_json=model_row["metadata_json"],
                            after_metadata_json=json.dumps(metadata, sort_keys=True),
                        )
                    )
            planned.append(item)
    finally:
        con.close()
    return planned, skipped


class StaleBackfillPlanError(RuntimeError):
    """A registry row changed between planning and the shadow mutation."""


def apply_plan(candidate: Path, planned: Sequence[PlannedRow]) -> dict[str, Any]:
    """Apply planned updates to the shadow candidate; fail on any drift."""
    con = sqlite3.connect(str(candidate))
    try:
        with con:
            applied = 0
            for item in planned:
                for update in item.updates:
                    current = con.execute(
                        f"SELECT final_metrics_json, "
                        f"{'metadata_json' if update.table == 'training_models' else 'NULL'} "
                        f"FROM {update.table} WHERE run_id = ?",
                        (item.run_id,),
                    ).fetchone()
                    if current is None:
                        raise StaleBackfillPlanError(f"{update.table} row {item.run_id!r} disappeared")
                    if current[0] != update.before_final_metrics_json:
                        raise StaleBackfillPlanError(
                            f"{update.table}.final_metrics_json for {item.run_id!r} changed since planning"
                        )
                    if update.table == "training_models":
                        if current[1] != update.before_metadata_json:
                            raise StaleBackfillPlanError(
                                f"training_models.metadata_json for {item.run_id!r} changed since planning"
                            )
                        con.execute(
                            "UPDATE training_models SET final_metrics_json = ?, metadata_json = ? WHERE run_id = ?",
                            (update.after_final_metrics_json, update.after_metadata_json, item.run_id),
                        )
                    else:
                        con.execute(
                            "UPDATE training_runs SET final_metrics_json = ? WHERE run_id = ?",
                            (update.after_final_metrics_json, item.run_id),
                        )
                    applied += 1
    finally:
        con.close()
    return {"operation": "backfill_training_best_metrics", "row_updates": applied}


def _summary(planned: Sequence[PlannedRow], skipped: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for item in planned:
        counts[item.classification] = counts.get(item.classification, 0) + 1
    return {"planned": counts, "skipped": len(skipped)}


def build_receipt(
    *,
    registry: Path,
    planned: Sequence[PlannedRow],
    skipped: Sequence[Mapping[str, Any]],
    applied_utc: str,
    publication: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    from fisheye.shared.system_metadata import get_git_info

    return {
        "schema_id": RECEIPT_SCHEMA,
        "tool": TOOL,
        "applied_utc": applied_utc,
        "registry": str(registry),
        "code": get_git_info(),
        "summary": _summary(planned, skipped),
        "rows": [asdict(item) for item in planned],
        "skipped": list(skipped),
        "publication": dict(publication) if publication is not None else None,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--task", choices=("detect", "pose", "all"), default="all")
    parser.add_argument("--run-id", action="append", default=None, help="Limit to these run ids.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup", type=Path, help="Registry backup path (required with --apply).")
    parser.add_argument("--receipt-dir", type=Path, help="Receipt directory (required with --apply).")
    args = parser.parse_args(argv)

    from fisheye.registry.write_policy import running_inside_lsf_job

    tasks = ("detect", "pose") if args.task == "all" else (args.task,)
    applied_utc = datetime.now(timezone.utc).isoformat()
    planned, skipped = plan(args.registry, tasks=tasks, run_ids=args.run_id, applied_utc=applied_utc)
    report = {
        "summary": _summary(planned, skipped),
        "rows": [
            {
                "run_id": item.run_id,
                "task_type": item.task_type,
                "classification": item.classification,
                "best_epoch": item.best_epoch,
                "last_epoch": item.last_epoch,
                "previous": item.previous_values,
                "bound": item.bound_values,
                "tables": [update.table for update in item.updates],
            }
            for item in planned
        ],
        "skipped": skipped,
    }
    print(json.dumps(report, indent=1, default=str))
    if not args.apply:
        return 0
    if args.backup is None or args.receipt_dir is None:
        parser.error("--apply requires --backup and --receipt-dir")
    if running_inside_lsf_job():
        print("refusing to write the registry from inside an LSF job", file=sys.stderr)
        return 2
    to_apply = [item for item in planned if item.updates]
    if not to_apply:
        print("nothing to apply", file=sys.stderr)
        return 0

    stamp = applied_utc.replace(":", "").replace("-", "").split(".")[0]
    plan_sha = _write_json(
        args.receipt_dir / f"plan_{stamp}.json",
        build_receipt(registry=args.registry, planned=planned, skipped=skipped,
                      applied_utc=applied_utc, publication=None),
    )
    print(f"plan written: plan_{stamp}.json sha256={plan_sha}", file=sys.stderr)

    from fisheye.registry.shadow_publish import publish_registry_shadow

    publication = publish_registry_shadow(
        canonical_registry=args.registry,
        backup_path=args.backup,
        mutate=lambda candidate: apply_plan(candidate, to_apply),
    )
    receipt_sha = _write_json(
        args.receipt_dir / f"receipt_{stamp}.json",
        build_receipt(registry=args.registry, planned=planned, skipped=skipped,
                      applied_utc=applied_utc, publication=publication.to_json()),
    )
    print(json.dumps(dict(publication.mutation_result)), file=sys.stderr)
    print(f"receipt written: receipt_{stamp}.json sha256={receipt_sha}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
